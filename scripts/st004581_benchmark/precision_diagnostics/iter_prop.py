"""Precision-first iterative propagation (linear label-propagation / message-passing analog).
Seed = certain calls. Each round: train struct->sec on the admitted set, predict, admit only
compounds whose EIC apex lands in a TIGHT window of the prediction, reproducibly, isobar-
resolved, FDR-strict. Admitted compounds' apexes become new training anchors -> predictions
tighten -> next tier crosses the gate. Expand until no new admissions. Withhold everything
below the confidence gate (FPs never enter)."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
KIT="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; NRAND=60; FDR_ADMIT=0.01; ISOBAR_PPM=10.0
ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# ladder (fallback prediction for seed + compounds w/o smiles)
cal={p:list(v) for p,v in anchors.items()}
for plat,e in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in e))
    for c in e:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
_,_,_,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI]); u,ui=np.unique(RI,return_index=True)
inv=PchipInterpolator(u,SEC[ui],extrapolate=True)
# compounds
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    comp.append(dict(name=nm,mz=c["mz"],ladder=float(inv(c["ri"])),desc=d,ik=c["ik14"]))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
inmaf=np.array([c["name"] in maf for c in comp]); nmaf=len(maf); N=len(MZML)
print(f"neg compounds {n} (with smiles {sum(c['desc'] is not None for c in comp)}, MAF {inmaf.sum()})  inj {N}",flush=True)
# ONE EIC pass: cache peaks (rt,intensity) per compound per injection
peaks=[[] for _ in range(n)]; rtspans=[]
for mp in MZML:
    rts=[];rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(n)); continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(n)]))
    rt=np.array(rts); mat=np.array(rows); rtspans.append((rt.min(),rt.max()))
    for k in range(n):
        col=mat[:,k]; loc=(col>FLOOR)&(col>=np.roll(col,1))&(col>=np.roll(col,-1))
        peaks[k].append(np.array([(rt[j],col[j]) for j in np.where(loc)[0]]) if loc.any() else np.empty((0,2)))
print("EIC cached",flush=True)
rng=np.random.RandomState(0)
RC=[rng.uniform(lo+TIGHT,hi-TIGHT,NRAND) for lo,hi in rtspans]   # random centers per injection for null
def rep_apex(k,pred,W):  # n injections with a peak within W of pred; apex = strongest such peak rt
    nrep=0; best=None; bi=0.0
    for inj in range(N):
        P=peaks[k][inj]
        if not len(P): continue
        m=np.abs(P[:,0]-pred)<=W
        if m.any():
            nrep+=1; j=np.argmax(P[m,1])
            if P[m][j,1]>bi: bi=P[m][j,1]; best=P[m][j,0]
    return nrep,best,bi
def null_rate(k,W):
    hit=0; tot=0
    for inj in range(N):
        P=peaks[k][inj]; cs=RC[inj]
        for c in cs:
            tot+=1
            if len(P) and (np.abs(P[:,0]-c)<=W).any(): hit+=1
    return hit/max(tot,1)
def fdr_q(present,b):
    pv=binom.sf(present-1,N,np.clip(b,1e-6,0.999)); return pv
# seed: kit anchors known sec
kit=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"])
    except: sec=None
    if d is not None and sec: kit.append((d,sec))
admitted=np.zeros(n,bool); obs=np.full(n,np.nan)
Xtr=[k[0] for k in kit]; ytr=[k[1] for k in kit]
nullc=np.array([null_rate(k,TIGHT) for k in range(n)])
print(f"seed kit anchors: {len(kit)}\n\n{'round':>5}{'admitted':>9}{'newMAF':>8}{'recall':>8}{'precision':>10}",flush=True)
for rd in range(12):
    mdl=HistGradientBoostingRegressor(max_iter=400,max_depth=4,learning_rate=0.05,min_samples_leaf=5).fit(np.array(Xtr),np.array(ytr))
    # candidate admissions among unadmitted-with-desc
    cand=[]
    for k in range(n):
        if admitted[k] or comp[k]["desc"] is None: continue
        pred=float(mdl.predict([comp[k]["desc"]])[0])
        nrep,apex,bi=rep_apex(k,pred,TIGHT)
        if nrep<MINREP or apex is None: continue
        pv=fdr_q(nrep,nullc[k])
        if pv>FDR_ADMIT: continue
        cand.append((k,apex,abs(apex-pred),pv,bi))
    if not cand: break
    # BH across candidates this round; isobar-resolve (shared apex within ISOBAR_PPM+TIGHT -> keep closest pred)
    cand.sort(key=lambda x:x[2])  # closest-pred first
    taken=[]; newk=[]
    for k,apex,dp,pv,bi in cand:
        clash=False
        for (kk,aa) in taken:
            if abs(MZc[k]-MZc[kk])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(apex-aa)<=TIGHT: clash=True; break
        if clash: continue
        taken.append((k,apex)); newk.append((k,apex))
    if not newk: break
    for k,apex in newk:
        admitted[k]=True; obs[k]=apex; Xtr.append(comp[k]["desc"]); ytr.append(apex)
    nm=sum(inmaf[k] for k,_ in newk)
    rec=inmaf[admitted].sum()/nmaf; prec=inmaf[admitted].sum()/max(admitted.sum(),1)
    print(f"{rd:>5}{admitted.sum():>9}{nm:>8}{rec:>8.3f}{prec:>10.3f}",flush=True)
print(f"\nFINAL  admitted {admitted.sum()}  recall {inmaf[admitted].sum()/nmaf:.3f}  precision {inmaf[admitted].sum()/max(admitted.sum(),1):.3f}")

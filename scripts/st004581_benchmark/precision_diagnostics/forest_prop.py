"""Forest of precision-first propagation chains (the user's xgboost-forest-in-tandem idea).
K chains, each = bootstrapped seed + random descriptor subset, run independently with
EARLY-CLIP (stop the chain the moment held-out RT-prediction error traverses upward = drift
onset). Aggregate by CONSENSUS: admit a compound only if >=M of K chains independently admit
it. Drift FPs are idiosyncratic to whichever bad anchor a chain let in -> they don't reach
consensus -> voted out. M is the precision-first knob (higher M = stricter)."""
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
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; NRAND=60; FDR_ADMIT=0.01
ISOBAR_PPM=10.0; K=10; NFEAT=15; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None)))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
inmaf=np.array([c["name"] in maf for c in comp]); nmaf=len(maf); N=len(MZML)
DDIM=len(_descriptors("CCO"))
print(f"neg compounds {n} (smiles {sum(c['desc'] is not None for c in comp)}, MAF {inmaf.sum()})  descdim {DDIM}  K {K}",flush=True)
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
grng=np.random.RandomState(0); RC=[grng.uniform(lo+TIGHT,hi-TIGHT,NRAND) for lo,hi in rtspans]
def rep_apex(k,pred,W):
    nrep=0; best=None; bi=0.0
    for inj in range(N):
        P=peaks[k][inj]
        if not len(P): continue
        m=np.abs(P[:,0]-pred)<=W
        if m.any():
            nrep+=1; j=np.argmax(P[m,1])
            if P[m][j,1]>bi: bi=P[m][j,1]; best=P[m][j,0]
    return nrep,best
nullc=np.array([sum((len(peaks[k][inj]) and (np.abs(peaks[k][inj][:,0]-c)<=TIGHT).any()) for inj in range(N) for c in RC[inj])/(N*NRAND) for k in range(n)])
withdesc=[k for k in range(n) if comp[k]["desc"] is not None]
DX=np.array([comp[k]["desc"] for k in withdesc])   # descriptor matrix for predictable compounds
# seed kit
kitX=[]; kitY=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"])
    except: sec=None
    if d is not None and sec: kitX.append(np.array(d)); kitY.append(sec)
kitX=np.array(kitX); kitY=np.array(kitY)
def run_chain(cs):
    rng=np.random.RandomState(cs)
    fsub=np.sort(rng.choice(DDIM,NFEAT,replace=False))
    bidx=rng.choice(len(kitX),len(kitX),replace=True)
    vmask=~np.isin(np.arange(len(kitX)),np.unique(bidx))   # OOB seed = validation for early-clip
    Xtr=list(kitX[bidx][:,fsub]); ytr=list(kitY[bidx])
    vX=kitX[vmask][:,fsub]; vY=kitY[vmask]
    admitted=np.zeros(n,bool); best_val=np.inf; bad=0
    for rd in range(12):
        mdl=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=5).fit(np.array(Xtr),np.array(ytr))
        if len(vY):
            ve=np.median(np.abs(mdl.predict(vX)-vY))
            if ve>best_val*1.05: bad+=1
            else: best_val=min(best_val,ve); bad=0
            if bad>=2: break          # EARLY-CLIP: held-out error traversing upward = drift
        pr=mdl.predict(DX[:,fsub]); cand=[]
        for ii,k in enumerate(withdesc):
            if admitted[k]: continue
            nrep,apex=rep_apex(k,pr[ii],TIGHT)
            if nrep<MINREP or apex is None: continue
            if binom.sf(nrep-1,N,np.clip(nullc[k],1e-6,0.999))>FDR_ADMIT: continue
            cand.append((k,apex,abs(apex-pr[ii])))
        if not cand: break
        cand.sort(key=lambda x:x[2]); taken=[]; new=[]
        for k,apex,dp in cand:
            if any(abs(MZc[k]-MZc[kk])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(apex-aa)<=TIGHT for kk,aa in taken): continue
            taken.append((k,apex)); new.append((k,apex))
        if not new: break
        for k,apex in new: admitted[k]=True; Xtr.append(comp[k]["desc"][fsub]); ytr.append(apex)
    return admitted
votes=np.zeros(n)
for c in range(K):
    votes+=run_chain(c); print(f"  chain {c} done (admitted {int(votes.sum() if c==0 else 0) or ''})",flush=True) if c==0 else None
print(f"\n{'consensus>=M':>12}{'admitted':>9}{'recall':>8}{'precision':>10}")
for Mv in range(1,K+1):
    sel=votes>=Mv; a=int(sel.sum())
    if not a: continue
    print(f"{Mv:>12}{a:>9}{inmaf[sel].sum()/nmaf:>8.3f}{inmaf[sel].sum()/a:>10.3f}")

"""Phase 2: score each NEG call by EIC family shape-coherence (max over injections of the
M<->M+1 peak-shape correlation near the call's expected RT). Does it separate TP from FP at
scale -> a per-call confidence for FDR-controlled output?"""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SMI="/tmp/dd_pubchem_smiles.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:5]
MZ_PPM=20.0; MIN_REP=1; ISO=1.003355; PLAT="lc/ms neg"
maf=defaultdict(dict)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    try: ri=float(r["rt"])
    except: continue
    plat=(r.get("platform") or "").strip().lower(); nm=M._norm(r.get("name") or "")
    if nm: maf[plat].setdefault(nm,ri)
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
cal={p:list(v) for p,v in anchors.items()}
for plat,ent in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in ent))
    for c in ent:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
lib=M.expand_library_adducts(lib0)
ladders,pooled,cov,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
slope=M.ri_per_sec(pp); ri_tol={p:10.0*s for p,s in slope.items()}
anchor_ris={p:np.array(sorted(ri for _,ri in v)) for p,v in anchors.items()}
df=M.normalise_ri(df,ladders,pooled); cons=M.consensus_features(df,MZ_PPM,ri_tol,MIN_REP)
smi=M.load_smiles_map(Path(SMI)); qm=M.fit_qsrr_ri(cons,lib,smi,MZ_PPM); M.attach_qsrr_ri(lib,qm,smi)
qsrr_by_id={}
for cand in lib.values():
    for c in cand:
        if c.get("is_primary",True) and c.get("ri_qsrr") is not None: qsrr_by_id.setdefault(M.cid_of(c,"name"),c["ri_qsrr"])
win=M.make_ri_win_fn(slope,anchor_ris,"fixed",30.0,10.0,0.5,60.0,{})
ann=M.match(cons,lib,MZ_PPM,win,True,"gated",id_key="name"); ann=M.ordinal_reassign(ann,lib,MZ_PPM,win,"name")
# neg calls: name -> (mz, ri_norm, is_tp)
gri=maf.get(PLAT,{}); calls={}
sub=ann[(ann.platform==PLAT)&(ann.match_id.astype(bool))]
for mid,rn,inten,mz in zip(sub.match_id.to_numpy(),sub.ri_norm.to_numpy(),sub.intensity.to_numpy(),sub.mz.to_numpy()):
    if mid not in calls or inten>calls[mid][0]:
        corr=False; ri=gri.get(mid)
        if ri is not None:
            w=win(PLAT,ri); q=qsrr_by_id.get(mid); corr=abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w)
        calls[mid]=[float(inten),float(mz),float(rn),mid in gri and corr]
names=list(calls); MZ=np.array([calls[n][1] for n in names]); RN=np.array([calls[n][2] for n in names])
TP=np.array([calls[n][3] for n in names])
# inverse pooled neg ladder RI->sec
pr=pp[PLAT]; agg=defaultdict(list)
for sec,ri in pr: agg[round(ri,1)].append(sec)
xs=np.array(sorted(agg)); ys=np.array([np.median(agg[x]) for x in xs])
inv=PchipInterpolator(xs,ys,extrapolate=True)
esec=np.array([float(inv(r)) for r in RN])
# EIC extraction (M and M+1) per injection, one pass
targets=np.concatenate([MZ, MZ+ISO]); tol=targets*MZ_PPM*1e-6
print(f"neg calls {len(names)} (TP {TP.sum()})  injections {len(MZML)}  EIC targets {len(targets)}", flush=True)
best=np.full(len(names),np.nan)
for mp in MZML:
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(smz): rows.append(np.zeros(len(targets))); continue
        lo=np.searchsorted(smz,targets-tol); hi=np.searchsorted(smz,targets+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(targets))]))
    rt=np.array(rts); mat=np.array(rows); nN=len(names)
    for i in range(nN):
        m=np.abs(rt-esec[i])<=20
        if not m.any(): continue
        seg0=mat[:,i]; segt=seg0[m]
        if segt.max()<=0: continue
        jj=np.where(m)[0][np.argmax(segt)]; w=np.abs(rt-rt[jj])<=12
        a=mat[w,i]; b=mat[w,i+nN]
        if a.std()>0 and b.std()>0 and (b>0).sum()>=3:
            c=np.corrcoef(a,b)[0,1]
            if np.isnan(best[i]) or c>best[i]: best[i]=c
    print(f"  done {Path(mp).name}", flush=True)
ok=~np.isnan(best)
def auc(sig,pos):
    o=np.argsort(sig); r=np.empty(len(sig)); r[o]=np.arange(1,len(sig)+1)
    n1=pos.sum(); n0=len(sig)-n1; return (r[pos].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"\ncalls with computable EIC shape-corr: {ok.sum()}/{len(names)}")
print(f"EIC M<->M+1 shape-corr AUC (TP vs FP): {auc(best[ok],TP[ok]):.3f}")
print(f"  median: TP {np.median(best[ok&TP]):.3f}  FP {np.median(best[ok&~TP]):.3f}")
nm=sum(len(v) for v in maf.values()); negmaf=len(maf[PLAT])
print(f"\nrank neg calls by shape-corr -> precision/recall (recall denom = {negmaf} neg MAF):")
o=np.argsort(-np.nan_to_num(best,nan=-2)); t=TP[o]; cum=np.cumsum(t); called=np.arange(1,len(o)+1)
for thr in [0.95,0.9,0.8,0.6,0.4,0.0]:
    keep=ok&(best>=thr); c=keep.sum(); tp=(keep&TP).sum()
    print(f"  shape-corr>={thr}:  called {c:>4}  TP {tp:>3}  precision {tp/c if c else 0:.3f}  recall {tp/negmaf:.3f}")

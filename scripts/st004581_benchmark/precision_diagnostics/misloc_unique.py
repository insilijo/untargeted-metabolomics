"""Of the 148 present-but-mislocated MAF compounds, how many are MASS-UNIQUE (recoverable by
m/z alone, RT only confirms) vs isobaric (need structure model to disambiguate)? Mass-unique
mislocated = directly recoverable recall with NO null-inflation cost."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
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
ent=lib0.get(PLAT,[]); allmz=np.array(sorted(c["mz"] for c in ent))
def uniq(mz):
    t=mz*MZ_PPM*1e-6; return (np.searchsorted(allmz,mz+t)-np.searchsorted(allmz,mz-t))==1
comp=[dict(name=M._norm(c["name"]),mz=c["mz"],pred=float(inv(c["ri"])),uq=uniq(c["mz"])) for c in ent if M._norm(c["name"]) in maf]
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6; pred=np.array([c["pred"] for c in comp])
uq=np.array([c["uq"] for c in comp])
peaks=[[] for _ in range(n)]
for mp in MZML:
    rts=[];rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(n)); continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(n)]))
    rt=np.array(rts); mat=np.array(rows)
    for k in range(n):
        col=mat[:,k]; loc=(col>FLOOR)&(col>=np.roll(col,1))&(col>=np.roll(col,-1))
        peaks[k].append(np.array([(rt[j],col[j]) for j in np.where(loc)[0]]) if loc.any() else np.empty((0,2)))
N=len(MZML)
def at_pred(k):
    r=0
    for inj in range(N):
        P=peaks[k][inj]
        if len(P) and (np.abs(P[:,0]-pred[k])<=TIGHT).any() and P[np.abs(P[:,0]-pred[k])<=TIGHT,1].max()>2*FLOOR: r+=1
    return r>=MINREP
def anywhere(k):
    cand=defaultdict(int); inten=defaultdict(float)
    for inj in range(N):
        for rt_,it in peaks[k][inj]:
            if it>2*FLOOR:
                b=round(rt_/(2*TIGHT)); cand[b]+=1; inten[b]=max(inten[b],it)
    good=[b for b in cand if cand[b]>=MINREP]
    return (max(good,key=lambda b:inten[b]) if good else None)
misloc=[k for k in range(n) if not at_pred(k) and anywhere(k) is not None]
mu=sum(uq[k] for k in misloc); iso=len(misloc)-mu
at=sum(at_pred(k) for k in range(n))
print(f"\nMAF mislocated-but-present     : {len(misloc)}")
print(f"  MASS-UNIQUE (recover by m/z, RT just confirms): {mu}")
print(f"  isobaric (need struct model to disambiguate)  : {iso}")
print(f"\nrecall now (at predicted RT)                    : {at}/{n} = {at/n:.3f}")
print(f"+ mass-unique mislocated (low-risk recovery)    : {at+mu}/{n} = {(at+mu)/n:.3f}")
print(f"+ all mislocated (full ceiling)                 : {at+len(misloc)}/{n} = {(at+len(misloc))/n:.3f}")

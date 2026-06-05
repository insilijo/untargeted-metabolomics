"""Are the 190 MAF compounds 'undetectable at predicted RT' actually ABSENT, or present-but-
MISLOCATED (RT mispredicted/OOB)? For each, search the WHOLE run at its m/z for a strong
reproducible peak (any RT). If found -> present, just RT-mislocated = recoverable. Report the
mislocation distance (|peak - predicted RT|)."""
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
comp=[dict(name=M._norm(c["name"]),mz=c["mz"],pred=float(inv(c["ri"]))) for c in lib0.get(PLAT,[]) if M._norm(c["name"])]
comp=[c for c in comp if c["name"] in maf]   # MAF only
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6; pred=np.array([c["pred"] for c in comp])
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
def anywhere(k):  # strongest reproducible peak at ANY RT (consensus apex across inj)
    cand=defaultdict(int); inten=defaultdict(float)
    for inj in range(N):
        P=peaks[k][inj]
        for rt_,it in P:
            if it>2*FLOOR:
                b=round(rt_/ (2*TIGHT))   # bucket by ~2*TIGHT
                cand[b]+=1; inten[b]=max(inten[b],it)
    good=[(b,cand[b]) for b in cand if cand[b]>=MINREP]
    if not good: return None
    b=max(good,key=lambda x:inten[x[0]])[0]; return b*(2*TIGHT)
nopred=[k for k in range(n) if not at_pred(k)]
misloc=0; absent=0; dists=[]
for k in nopred:
    ap=anywhere(k)
    if ap is None: absent+=1
    else: misloc+=1; dists.append(abs(ap-pred[k]))
print(f"\nMAF compounds                              : {n}")
print(f"  detectable AT predicted RT               : {n-len(nopred)}")
print(f"  NOT at predicted RT                      : {len(nopred)}")
print(f"     -> PRESENT but RT-MISLOCATED (peak elsewhere): {misloc}")
print(f"     -> truly absent (no peak at m/z anywhere)    : {absent}")
if dists:
    d=np.array(dists)
    print(f"\n  mislocation |peak - predicted RT|: median {np.median(d):.0f}s  p90 {np.percentile(d,90):.0f}s  max {d.max():.0f}s")
print(f"\nTRUE recoverable recall ceiling (at-pred + mislocated) = {(n-len(nopred)+misloc)}/{n} = {(n-len(nopred)+misloc)/n:.3f}")
print(f"  vs ladder-only-at-predicted-RT          = {(n-len(nopred))}/{n} = {(n-len(nopred))/n:.3f}")

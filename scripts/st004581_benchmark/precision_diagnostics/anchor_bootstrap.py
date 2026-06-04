"""Anchor-contribution permutation: bootstrap the sparse anchor set -> ensemble of RI->sec
ladders -> per-compound RT-prediction confidence (mean +- spread). VALIDATE: do the
46s-mislocated recall misses show HIGH spread (-> ensemble flags them -> adaptive search
recovers them)? vs recovered compounds (should be LOW spread)."""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZ_PPM=7.0; PLAT="lc/ms neg"; B=60
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
cal={p:list(v) for p,v in anchors.items()}
for plat,ent in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in ent))
    for c in ent:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
ladders,pooled,cov,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
pairs=pp[PLAT]                        # detected anchor (sec, RI)
# aggregate to unique RI->sec
agg=defaultdict(list)
for sec,ri in pairs: agg[round(ri,1)].append(sec)
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI])
rng=np.random.RandomState(0)
# bootstrap ensemble of inverse ladders
rows=[r for r in csv.DictReader(open("/tmp/ew30.csv"))]
ric=np.array([float(r["ri"]) for r in rows])
preds=np.full((B,len(rows)),np.nan)
for b in range(B):
    idx=np.sort(rng.choice(len(RI),len(RI),replace=True))
    ri_b,sec_b=RI[idx],SEC[idx]
    u,ui=np.unique(ri_b,return_index=True)       # PCHIP needs strictly increasing
    if len(u)<4: continue
    f=PchipInterpolator(u,sec_b[ui],extrapolate=True)
    preds[b]=f(ric)
std=np.nanstd(preds,axis=0)                       # per-compound RT-prediction spread (s)
# categorize each compound
def cat(r):
    nm=M._norm(r["name"]); pres=int(r["present"]); q=float(r["qvalue"]); prim=int(r["primary"])
    if nm not in maf: return "non-MAF"
    if q<0.05 and pres>=2 and prim: return "CALLED"
    if pres==0: return "MISS:no-peak"
    if pres>=2 and prim==0: return "MISS:isobar"
    if pres>=2 and q>=0.05: return "MISS:not-sig"
    return "MISS:weak"
cats=np.array([cat(r) for r in rows])
print(f"anchor bootstrap B={B}; neg anchors={len(RI)}\n")
print(f"per-compound RT-prediction spread (s), median by category:")
for c in ["CALLED","MISS:no-peak","MISS:isobar","MISS:not-sig","MISS:weak"]:
    m=cats==c
    if m.sum(): print(f"  {c:<16} n={int(m.sum()):>4}  median spread {np.nanmedian(std[m]):>5.1f}s  P90 {np.nanpercentile(std[m],90):>5.1f}s")
# the key test: are no-peak misses higher-spread than CALLED?
print(f"\nAUC (spread separates MISS:no-peak from CALLED): ",end="")
a=std[cats=='CALLED']; bb=std[cats=='MISS:no-peak']; a=a[~np.isnan(a)]; bb=bb[~np.isnan(bb)]
allv=np.concatenate([a,bb]); lab=np.concatenate([np.zeros(len(a)),np.ones(len(bb))])
o=np.argsort(allv); rank=np.empty(len(allv)); rank[o]=np.arange(1,len(allv)+1)
n1=lab.sum(); auc=(rank[lab==1].sum()-n1*(n1+1)/2)/(n1*(len(allv)-n1))
print(f"{auc:.3f}  (>0.5 = ensemble flags the mislocated misses as uncertain)")

"""Integrated per-compound PERMUTATION FDR (DD RI is predicted, not truth -> calibrate each
call against a LOCAL null). For each neg compound: is its peak-presence at the expected RT
(DD RI->sec) above the presence rate at RANDOM RTs for the same m/z (local m/z-density null)?
Binomial p per compound -> BH q-value -> FDR-thresholded annotation set."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:6]
MZ_PPM=7.0; PLAT="lc/ms neg"; FLOOR=5e4; W=5.0; R=40
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
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
xs=np.array(sorted(agg)); ys=np.array([np.median(agg[x]) for x in xs]); inv=PchipInterpolator(xs,ys,extrapolate=True)
comp={}
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if nm and nm not in comp: comp[nm]=[c["mz"],c["ri"]]
names=list(comp); MZc=np.array([comp[n][0] for n in names]); RIc=np.array([comp[n][1] for n in names])
esec=np.array([float(inv(r)) for r in RIc]); tol=MZc*MZ_PPM*1e-6
N=len(MZML); rng=np.random.RandomState(0)
present=np.zeros(len(names)); bg_hits=np.zeros(len(names)); bg_tot=np.zeros(len(names))
rtlo,rthi=None,None
print(f"neg compounds {len(names)} (MAF {sum(n in maf for n in names)})  injections {N}", flush=True)
for mp in MZML:
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(smz): rows.append(np.zeros(len(names))); continue
        lo=np.searchsorted(smz,MZc-tol); hi=np.searchsorted(smz,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(names))]))
    rt=np.array(rts); mat=np.array(rows); lo_t,hi_t=rt.min()+W,rt.max()-W
    centers=rng.uniform(lo_t,hi_t,R)
    for k in range(len(names)):
        col=mat[:,k]
        if col[np.abs(rt-esec[k])<=W].max(initial=0)>FLOOR: present[k]+=1
        for c0 in centers:
            bg_tot[k]+=1
            if col[np.abs(rt-c0)<=W].max(initial=0)>FLOOR: bg_hits[k]+=1
    print(f"  done {Path(mp).name.split('_')[-1]}", flush=True)
b=bg_hits/np.maximum(bg_tot,1)                       # local null presence rate per compound
pval=binom.sf(present-1,N,np.clip(b,1e-6,0.999))     # P(>= present | null)
# BH-FDR
order=np.argsort(pval); m=len(pval); q=np.empty(m)
qm=1.0
for r_,i in enumerate(order[::-1]):
    rank=m-r_; qm=min(qm,pval[i]*m/rank); q[i]=qm
inmaf=np.array([n in maf for n in names])
print(f"\nlocal-null presence rate b: median {np.median(b):.2f} (this is the m/z-density floor)\n")
print(f"{'FDR<':>6}{'called':>8}{'TP(MAF)':>9}{'precision':>11}{'recall':>9}")
nmaf=len(maf)
for lvl in [0.01,0.05,0.10,0.20,0.50]:
    keep=q<lvl; c=keep.sum(); tp=(keep&inmaf).sum()
    print(f"{lvl:>6.2f}{c:>8}{tp:>9}{(tp/c if c else 0):>11.3f}{tp/nmaf:>9.3f}")

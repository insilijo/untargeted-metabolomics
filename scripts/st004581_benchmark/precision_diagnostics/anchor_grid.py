"""Sparse-anchor grid: extract EICs ONCE, then vary the NUMBER of anchors (evenly tiled by
RI) -> rebuild the RI->sec ladder -> re-predict RT -> recall/precision at FDR<5%. Finds
where adding anchors stops helping. Plus the full precision/recall tradeoff curve."""
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
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
MZ_PPM=7.0; PLAT="lc/ms neg"; FLOOR=50000.0; W=5.0; MINREP=2; NRAND=40
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
# full anchor pool (detected sec,RI), aggregated, sorted by RI
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI])
comp={}
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if nm and nm not in comp: comp[nm]=(c["mz"],c["ri"])
names=list(comp); MZc=np.array([comp[n][0] for n in names]); RIc=np.array([comp[n][1] for n in names])
inmaf=np.array([n in maf for n in names]); nmaf=len(maf); tol=MZc*MZ_PPM*1e-6
# --- extract EIC ONCE: present-count at any candidate RT needs the matrix; store rt+mat per inj ---
print(f"extracting EICs ({len(names)} m/z x {len(MZML)} inj) once ...", flush=True)
RT=[]; MAT=[]
for mp in MZML:
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(smz): rows.append(np.zeros(len(names))); continue
        lo=np.searchsorted(smz,MZc-tol); hi=np.searchsorted(smz,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(names))]))
    RT.append(np.array(rts)); MAT.append(np.array(rows))
N=len(MZML); rng=np.random.RandomState(0)
# local null b (anchor-independent): peak rate at random RTs
bg_hits=np.zeros(len(names)); bg_tot=np.zeros(len(names))
for rt,mat in zip(RT,MAT):
    centers=rng.uniform(rt.min()+W,rt.max()-W,NRAND)
    for k in range(len(names)):
        col=mat[:,k]
        for c0 in centers:
            bg_tot[k]+=1
            if col[np.abs(rt-c0)<=W].max(initial=0)>FLOOR: bg_hits[k]+=1
b=np.clip(bg_hits/np.maximum(bg_tot,1),1e-6,0.999)
def evaluate(esec, fdr=0.05):
    present=np.zeros(len(names))
    for rt,mat in zip(RT,MAT):
        for k in range(len(names)):
            if mat[:,k][np.abs(rt-esec[k])<=W].max(initial=0)>FLOOR: present[k]+=1
    pval=binom.sf(present-1,N,b)
    o=np.argsort(pval); m=len(pval); q=np.empty(m); cur=1.0
    for r_,i in enumerate(o[::-1]):
        cur=min(cur,pval[i]*m/(m-r_)); q[i]=cur
    return present,q
def ladder_from(nanc):
    idx=np.linspace(0,len(RI)-1,nanc).round().astype(int); idx=np.unique(idx)
    u,ui=np.unique(RI[idx],return_index=True)
    return PchipInterpolator(u,SEC[idx][ui],extrapolate=True)
print(f"\nANCHOR-COUNT GRID (recall/precision at FDR<5%, fixed 5s window):")
print(f"{'#anchors':>9}{'called':>8}{'recall':>9}{'precision':>11}")
for nanc in [5,8,12,20,30,50,80,len(RI)]:
    if nanc>len(RI): continue
    f=ladder_from(nanc); esec=np.array([float(f(r)) for r in RIc])
    present,q=evaluate(esec)
    keep=(q<0.05)&(present>=MINREP); c=keep.sum(); tp=(keep&inmaf).sum()
    print(f"{nanc:>9}{c:>8}{tp/nmaf:>9.3f}{(tp/c if c else 0):>11.3f}")
# full tradeoff curve at full anchors
f=ladder_from(len(RI)); esec=np.array([float(f(r)) for r in RIc]); present,q=evaluate(esec)
print(f"\nFULL PRECISION/RECALL TRADEOFF (all {len(RI)} anchors):")
print(f"{'FDR<':>6}{'called':>8}{'recall':>9}{'precision':>11}")
for lvl in [0.001,0.01,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.01]:
    keep=(q<lvl)&(present>=MINREP); c=keep.sum(); tp=(keep&inmaf).sum()
    print(f"{lvl:>6.3f}{c:>8}{tp/nmaf:>9.3f}{(tp/c if c else 0):>11.3f}")

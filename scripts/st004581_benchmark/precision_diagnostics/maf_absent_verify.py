"""Verify the 'truly absent' MAF misses: is there REALLY no peak, or is the peak picker
(strict local-max > 2*FLOOR, 7ppm) hiding it? Pull each MAF compound's raw EIC at WIDER m/z
(15ppm) and report actual max intensity + a lenient peak check, bucketed."""
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
PLAT="lc/ms neg"; FLOOR=50000.0; TIGHT=6.0; MINREP=2; WIDE_PPM=15.0; NARROW_PPM=7.0
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
        t=c["mz"]*NARROW_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
_,_,_,pp=M.build_batch_ladders(df,cal,NARROW_PPM,1,True)
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI]); u,ui=np.unique(RI,return_index=True)
inv=PchipInterpolator(u,SEC[ui],extrapolate=True)
mafc=[(M._norm(c["name"]),c["mz"],float(inv(c["ri"]))) for c in lib0.get(PLAT,[]) if M._norm(c["name"]) in maf]
# dedup by m/z (synonyms)
seen=set(); mafc=[(nm,mz,pr) for nm,mz,pr in mafc if (round(mz,4) not in seen and not seen.add(round(mz,4)))]
n=len(mafc); MZc=np.array([m for _,m,_ in mafc]); tolW=MZc*WIDE_PPM*1e-6
maxint=np.zeros(n); strongrep=np.zeros(n)  # reproducible peaks >2*FLOOR (current 'strong' def)
for mp in MZML:
    rts=[];rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(n)); continue
        lo=np.searchsorted(sm,MZc-tolW); hi=np.searchsorted(sm,MZc+tolW)
        rows.append(np.array([si[lo[k]:hi[k]].max() if hi[k]>lo[k] else 0.0 for k in range(n)]))
    rt=np.array(rts); mat=np.array(rows)
    for k in range(n):
        col=mat[:,k]; maxint[k]=max(maxint[k],col.max())
        if col.max()>2*FLOOR: strongrep[k]+=1
absent=[k for k in range(n) if strongrep[k]<MINREP]   # current 'truly absent / weak' set (wide-mz)
print(f"MAF compounds (m/z-dedup): {n}   not-strong (current 'absent/weak' bucket, 15ppm): {len(absent)}")
mi=maxint[absent]
def pct(lo,hi): return int(((mi>=lo)&(mi<hi)).sum())
print("\nmax EIC intensity (WIDE 15ppm, any RT, max over 8 inj) of the 'absent' set:")
print(f"  >100k (2*FLOOR, SHOULD have been 'strong' -> peak-picker/threshold MISS): {pct(100000,1e18)}")
print(f"  50k-100k (FLOOR..2*FLOOR, real peak below the 'strong' bar)            : {pct(50000,100000)}")
print(f"  10k-50k  (sub-FLOOR signal, faint but present)                         : {pct(10000,50000)}")
print(f"  2k-10k   (trace)                                                       : {pct(2000,10000)}")
print(f"  <2k      (genuinely absent)                                            : {pct(0,2000)}")
# how many gained signal from WIDE vs NARROW (mass drift)?
print(f"\nmedian max-int of 'absent' set: {int(np.median(mi))}")

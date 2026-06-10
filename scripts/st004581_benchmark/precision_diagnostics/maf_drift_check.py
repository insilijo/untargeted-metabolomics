"""Of the 'absent' MAF compounds visible only at 15ppm: are they TRUE mass-drift (peak at
narrow 7ppm too, just sub-threshold) or NEAR-ISOBARS (peak only off-mass)? Compare max
intensity at 5ppm / 7ppm / 15ppm per compound."""
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
PLAT="lc/ms neg"; FLOOR=50000.0
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"])
lib0=M.load_library(Path(DD))
mafc=[(M._norm(c["name"]),c["mz"]) for c in lib0.get(PLAT,[]) if M._norm(c["name"]) in maf]
seen=set(); mafc=[(nm,mz) for nm,mz in mafc if (round(mz,4) not in seen and not seen.add(round(mz,4)))]
n=len(mafc); MZc=np.array([m for _,m in mafc])
m5=np.zeros(n); m7=np.zeros(n); m15=np.zeros(n)
for mp in MZML:
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i)
        if not len(sm): continue
        for ppm,arr in ((5e-6,m5),(7e-6,m7),(15e-6,m15)):
            t=MZc*ppm; lo=np.searchsorted(sm,MZc-t); hi=np.searchsorted(sm,MZc+t)
            for k in range(n):
                if hi[k]>lo[k]: arr[k]=max(arr[k],si[lo[k]:hi[k]].max())
# the 'absent at strong=2*FLOOR/15ppm but >100k at 15ppm' set
drift=[k for k in range(n) if m15[k]>2*FLOOR and m7[k]<2*FLOOR]
print(f"compounds 'recovered' only by 15ppm (>100k at 15ppm, <100k at 7ppm): {len(drift)}")
true_drift=sum(1 for k in drift if m5[k]>FLOOR or m7[k]>FLOOR)  # real signal at tight mass too
nearisobar=len(drift)-true_drift
print(f"  TRUE mass-drift (also has signal >FLOOR at 5/7ppm): {true_drift}")
print(f"  NEAR-ISOBAR (only off-mass, 7ppm<FLOOR): {nearisobar}")
for k in drift[:12]:
    print(f"    mz={MZc[k]:.4f}  5ppm={int(m5[k]):>8}  7ppm={int(m7[k]):>8}  15ppm={int(m15[k]):>9}")

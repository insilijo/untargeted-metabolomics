"""Size the MAF false-negative pool: of the non-MAF DD neg compounds, how many have a STRONG
REPRODUCIBLE peak at their predicted RT (ladder)? That's the pool 'novel_real' is a subset of
-- i.e. real compounds the MAF didn't report. Splits by isobaric-MAF-present (could be
substitution) vs clean (no MAF compound at that m/z = true MAF FN / novel metabolite)."""
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
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; ISOBAR_PPM=10.0
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
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    comp.append(dict(name=nm,mz=c["mz"],pred=float(inv(c["ri"]))))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
inmaf=np.array([c["name"] in maf for c in comp]); pred=np.array([c["pred"] for c in comp])
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
N=len(MZML); maf_mz=np.array([MZc[j] for j in range(n) if inmaf[j]])
def strong_at_pred(k):
    nrep=0; mx=0.0
    for inj in range(N):
        P=peaks[k][inj]
        if len(P) and (np.abs(P[:,0]-pred[k])<=TIGHT).any():
            nrep+=1; mx=max(mx,P[np.abs(P[:,0]-pred[k])<=TIGHT,1].max())
    return nrep>=MINREP and mx>2*FLOOR
nonmaf=[k for k in range(n) if not inmaf[k]]
hits=[k for k in nonmaf if strong_at_pred(k)]
clean=[k for k in hits if not (np.abs(maf_mz-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6).any()]
print(f"MAF present (neg)                              : {int(inmaf.sum())}")
print(f"non-MAF DD compounds                           : {len(nonmaf)}")
print(f"  ...with a STRONG reproducible peak at pred RT: {len(hits)}  (MAF-FN candidate pool)")
print(f"  ...of those, NO isobaric MAF compound (clean): {len(clean)}  (true MAF-FN / novel metabolite)")
print(f"\nMAF FN pool as fraction of MAF-present         : {len(hits)/max(inmaf.sum(),1):.2f}x")
print(f"if these are real, the MAF itself misses ~{len(hits)} neg compounds it could have called")

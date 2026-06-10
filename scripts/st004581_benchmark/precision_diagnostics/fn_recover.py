"""Recover FNs FIRST: does a missed MAF compound's peak get stolen by a 'novel' call?
Answer-key compounds get first claim on every peak; only unclaimed peaks may be novel.
Quantify: (1) MAF compounds detectable at predicted RT = recall ceiling; (2) currently-
uncalled-but-detectable = recoverable FNs; (3) recoverable FNs whose peak is ALSO claimed by
a non-MAF candidate (isobaric+co-eluting) = the contaminated-novel overlap."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; ISOBAR_PPM=10.0; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
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
    comp.append(dict(name=nm,mz=c["mz"],pred=float(inv(c["ri"])),ismaf=nm in maf))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
pred=np.array([c["pred"] for c in comp]); ismaf=np.array([c["ismaf"] for c in comp])
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
def detect(k):  # strong reproducible peak at predicted RT -> apex
    nrep=0; mx=0.0; apex=None
    for inj in range(N):
        P=peaks[k][inj]
        if len(P):
            m=np.abs(P[:,0]-pred[k])<=TIGHT
            if m.any():
                nrep+=1; j=P[m,1].argmax()
                if P[m][j,1]>mx: mx=P[m][j,1]; apex=P[m][j,0]
    return (nrep>=MINREP and mx>2*FLOOR), apex
maf_idx=np.where(ismaf)[0]; nmaf=len(maf_idx)
det={}; 
for k in range(n):
    ok,ap=detect(k); det[k]=(ok,ap)
maf_det=[k for k in maf_idx if det[k][0]]
print(f"\nMAF present (recall denom)              : {nmaf}")
print(f"MAF compounds DETECTABLE at predicted RT: {len(maf_det)}  = recall CEILING {len(maf_det)/nmaf:.3f}")
print(f"MAF NOT detectable (truly absent/weak)  : {nmaf-len(maf_det)}")
# of detectable MAF, how many share their peak with a non-MAF candidate (contested)?
nonmaf_det=[k for k in range(n) if not ismaf[k] and det[k][0]]
contested=0; clean_novel=0
for k in nonmaf_det:
    ap=det[k][1]
    rival=[j for j in maf_det if abs(MZc[j]-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(det[j][1]-ap)<=TIGHT]
    if rival: contested+=1
    else: clean_novel+=1
print(f"\nnon-MAF DD detectable at predicted RT    : {len(nonmaf_det)}")
print(f"  CONTESTED (share peak w/ a detectable MAF compound = FN-stealing risk): {contested}")
print(f"  CLEAN novel (no MAF compound claims this peak)                        : {clean_novel}")
print(f"\n-> answer-key-first assignment: the {contested} contested go to the MAF compound (recover FN),")
print(f"   leaving {clean_novel} genuinely-unclaimed peaks eligible to be called novel.")

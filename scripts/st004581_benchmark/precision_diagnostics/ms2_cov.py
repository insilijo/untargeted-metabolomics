"""How much MS2 coverage do we actually have? For each neg MAF compound (m/z, predicted RT),
is there a co-eluting DDA MS2 scan whose precursor matches its m/z? Sizes the MS2 precision
lever (DDA is intensity-selected, so coverage is partial)."""
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
PLAT="lc/ms neg"; MZ_PPM=15.0; TIGHT=8.0
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
mafc=[(M._norm(c["name"]),c["mz"],float(inv(c["ri"]))) for c in lib0.get(PLAT,[]) if M._norm(c["name"]) in maf]
# collect ALL MS2 (precursor_mz, rt, n_frag) across injections
ms2=[]
for mp in MZML:
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=2: continue
        try: pmz=spec.selected_precursors[0]["mz"]
        except Exception: continue
        ms2.append((pmz, spec.scan_time_in_minutes()*60, len(spec.mz)))
ms2=np.array([(a,b,c) for a,b,c in ms2]); 
print(f"MAF neg compounds: {len(mafc)}   total MS2 scans (8 inj): {len(ms2)}")
pmzs=ms2[:,0]; o=np.argsort(pmzs); pmzs=pmzs[o]; prt=ms2[:,1][o]; pnf=ms2[:,2][o]
cov_pred=0; cov_anyrt=0
for nm,mz,pred in mafc:
    t=mz*MZ_PPM*1e-6; lo=np.searchsorted(pmzs,mz-t); hi=np.searchsorted(pmzs,mz+t)
    if hi>lo:
        cov_anyrt+=1
        if (np.abs(prt[lo:hi]-pred)<=TIGHT).any(): cov_pred+=1
print(f"  MAF compounds with an MS2 at their precursor m/z (ANY rt) : {cov_anyrt}  ({cov_anyrt/len(mafc):.2f})")
print(f"  ...AND co-eluting at predicted RT (usable for identity)   : {cov_pred}  ({cov_pred/len(mafc):.2f})")

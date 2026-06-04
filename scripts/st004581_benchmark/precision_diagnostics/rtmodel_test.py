"""Does a structure->seconds model (ladder-independent), trained on the sparse KIT anchors,
predict RT better than the RI ladder -- especially for the ladder-mislocated compounds?
Compare |rtmodel - EICapex| vs |ladder - EICapex|."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
FEAT="/tmp/feat_colu.parquet"; ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
ANCNEG="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:6]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; ik14=lambda s:(s or "")[:14]
# train structure->sec on KIT anchors
Xa=[]; ya=[]
for r in csv.DictReader(open(ANCNEG)):
    d=_descriptors(r.get("smiles","")); 
    try: sec=float(r["observed_rt_sec"])
    except: sec=None
    if d is not None and sec: Xa.append(d); ya.append(sec)
print(f"kit anchors trained on: {len(Xa)}", flush=True)
mdl=HistGradientBoostingRegressor(max_iter=300,max_depth=3,learning_rate=0.05,min_samples_leaf=3).fit(np.array(Xa),np.array(ya))
# ladder + compounds
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
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI]); u,ui=np.unique(RI,return_index=True)
inv=PchipInterpolator(u,SEC[ui],extrapolate=True)
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"]); s=smi.get(c["ik14"])
    if nm and s:
        d=_descriptors(s)
        if d is not None: comp.append((nm,c["mz"],float(inv(c["ri"])),float(mdl.predict([d])[0])))
names=[x[0] for x in comp]; MZc=np.array([x[1] for x in comp]); lad=np.array([x[2] for x in comp]); rtm=np.array([x[3] for x in comp])
tol=MZc*MZ_PPM*1e-6
print(f"neg compounds with smiles: {len(comp)}  -- EIC wide-scan apex ...",flush=True)
maxint=np.zeros(len(comp)); apex=np.full(len(comp),np.nan)
for mp in MZML:
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(len(comp))); continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(comp))]))
    rt=np.array(rts); mat=np.array(rows)
    for k in range(len(comp)):
        j=np.argmax(mat[:,k])
        if mat[j,k]>maxint[k]: maxint[k]=mat[j,k]; apex[k]=rt[j]
ok=(maxint>FLOOR)&~np.isnan(apex)
dl=np.abs(lad[ok]-apex[ok]); dm=np.abs(rtm[ok]-apex[ok])
print(f"\ncompounds with a clear peak: {ok.sum()}")
print(f"  |ladder - EICapex|   median {np.median(dl):.1f}s")
print(f"  |rtmodel - EICapex|  median {np.median(dm):.1f}s")
far=dl>20    # the ladder-mislocated subset
print(f"\n  ladder-mislocated (|ladder-apex|>20s): {int(far.sum())}")
print(f"    there, |ladder-apex| median {np.median(dl[far]):.0f}s  vs  |rtmodel-apex| median {np.median(dm[far]):.0f}s")
print(f"    rtmodel closer than ladder on {int((dm[far]<dl[far]).sum())}/{int(far.sum())} of them")

"""Kit-design map: bootstrap the sparse KIT -> per-RI-region RT-prediction uncertainty.
High-uncertainty regions = the gaps where the kit is starving prediction. Cross with where
the mislocated FNs live (do they cluster in the gaps?) and whether the library-densify pool
can fill those gaps (vs needing a physical standard). The 'informational space' map."""
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
KIT="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
PLAT="lc/ms neg"; B=80
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
# KIT anchor (RI, sec) from the panel
kit=[]
for r in csv.DictReader(open(KIT)):
    try: kit.append((float(r["ri"]),float(r["observed_rt_sec"])))
    except: pass
kit=sorted(set(kit)); KRI=np.array([k[0] for k in kit]); KSEC=np.array([k[1] for k in kit])
print(f"kit anchors: {len(kit)}  RI span [{KRI.min():.0f},{KRI.max():.0f}]")
# densify pool: mass-unique detected library compounds (RI) -> can fill gaps without a standard
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD))
densify_ri=[]
for plat,ent in lib0.items():
    if plat!=PLAT: continue
    mzs=np.array(sorted(c["mz"] for c in ent))
    for c in ent:
        t=c["mz"]*7e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: densify_ri.append(c["ri"])
densify_ri=np.array(densify_ri)
# compound RIs (neg DD) + which are MAF
crI=[]; cmaf=[]
for c in lib0.get(PLAT,[]):
    crI.append(c["ri"]); cmaf.append(M._norm(c["name"]) in maf)
crI=np.array(crI); cmaf=np.array(cmaf)
# bootstrap kit -> per-compound RT-prediction spread
rng=np.random.RandomState(0); preds=np.full((B,len(crI)),np.nan)
for b in range(B):
    idx=np.sort(rng.choice(len(KRI),len(KRI),replace=True))
    ri_b,sec_b=KRI[idx],KSEC[idx]; uu,ui=np.unique(ri_b,return_index=True)
    if len(uu)<4: continue
    f=PchipInterpolator(uu,sec_b[ui],extrapolate=True); preds[b]=f(crI)
spread=np.nanstd(preds,axis=0)
# bin by RI
bins=np.linspace(KRI.min(),KRI.max(),11)
print(f"\n{'RI bin':>14}{'kitAnc':>7}{'predSpread':>11}{'MAFcmpds':>9}{'densifyAvail':>13}")
for i in range(10):
    lo,hi=bins[i],bins[i+1]; m=(crI>=lo)&(crI<hi)
    ka=int(((KRI>=lo)&(KRI<hi)).sum()); da=int(((densify_ri>=lo)&(densify_ri<hi)).sum())
    sp=np.nanmedian(spread[m]) if m.any() else 0; nm=int((m&cmaf).sum())
    flag="  <-- GAP" if ka<=1 and nm>=8 else ""
    print(f"{lo:>6.0f}-{hi:<6.0f}{ka:>7}{sp:>11.0f}{nm:>9}{da:>13}{flag}")
print("\nGAP = kit has <=1 anchor but >=8 MAF compounds there -> add a standard (or densify if pool has compounds)")

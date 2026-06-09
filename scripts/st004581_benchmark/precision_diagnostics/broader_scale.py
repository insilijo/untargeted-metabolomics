"""Fixed: universe [M-H] m/z from SMILES (RDKit ExactMolWt). How many broader-universe isobaric
competitors per MAF compound vs DD -> the open-world discrimination challenge."""
import sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
csv.field_size_limit(10**7)
UNI="/mnt/volume-hel1-1/data/processed/compound_universe_broad.csv"
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; PLAT="lc/ms neg"; PPM=10.0; PROTON=1.0072765
maf_ik=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf_ik.add((r.get("inchikey") or "")[:14])
maf_ik={x for x in maf_ik if len(x)>=14}
lib0=M.load_library(Path(DD))
maf_mz=np.array(sorted(set(round(c["mz"],4) for c in lib0.get(PLAT,[]) if (c.get("ik14") or "")[:14] in maf_ik)))
ddmz=np.array(sorted(c["mz"] for c in lib0.get(PLAT,[])))
uni_mz=[]; n=0
for r in csv.DictReader(open(UNI)):
    n+=1; s=r.get("smiles")
    if not s: continue
    try:
        m=Chem.MolFromSmiles(s)
        if m is None: continue
        mw=Descriptors.ExactMolWt(m)
        if 50<mw<1500: uni_mz.append(mw-PROTON)
    except Exception: pass
    if n%40000==0: print(f"  ...{n} parsed, {len(uni_mz)} m/z",flush=True)
uni_mz=np.array(sorted(uni_mz))
print(f"universe: {n} compounds, {len(uni_mz)} with [M-H] m/z (50-1500 Da)",flush=True)
dd_iso=[]; uni_iso=[]
for mz in maf_mz:
    t=mz*PPM*1e-6
    dd_iso.append(np.searchsorted(ddmz,mz+t)-np.searchsorted(ddmz,mz-t))
    uni_iso.append(np.searchsorted(uni_mz,mz+t)-np.searchsorted(uni_mz,mz-t))
dd_iso=np.array(dd_iso); uni_iso=np.array(uni_iso)
print(f"\nisobaric competitors per MAF m/z (within {PPM}ppm):")
print(f"  DD (1718 cmpds) : median {int(np.median(dd_iso))}  mean {dd_iso.mean():.1f}  max {dd_iso.max()}")
print(f"  UNIVERSE (172k) : median {int(np.median(uni_iso))}  mean {uni_iso.mean():.0f}  max {uni_iso.max()}")
print(f"  competition multiplier: {uni_iso.mean()/max(dd_iso.mean(),1):.0f}x")
print(f"\n=> RT prediction must discriminate the MAF compound among ~{int(np.median(uni_iso))} co-isobaric")
print(f"   universe candidates per peak (vs ~{int(np.median(dd_iso))} in the DD). That is the open-world ask.")

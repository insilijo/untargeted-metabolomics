"""Why is recall lower than expected? Decompose the recall ceiling by what's STRUCTURALLY
reachable: SMILES-gating (propagation needs SMILES) x detection. How much of the MAF can the
propagation even reach?"""
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
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
lib0=M.load_library(Path(DD))
maf_total=0; maf_smiles=0; maf_nosmiles=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if nm not in maf: continue
    maf_total+=1
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    if d is not None: maf_smiles+=1
    else: maf_nosmiles.append(nm)
print(f"MAF neg compounds (in DD)         : {maf_total}")
print(f"  with usable SMILES (reachable)  : {maf_smiles}  ({maf_smiles/maf_total:.3f})")
print(f"  NO SMILES (propagation CANNOT reach, auto-FN): {maf_total-maf_smiles}")
print(f"\n=> SMILES-gated recall CEILING = {maf_smiles}/{maf_total} = {maf_smiles/maf_total:.3f}")
print(f"   current recall 0.532 as fraction of REACHABLE = {0.532*maf_total/maf_smiles:.3f}")

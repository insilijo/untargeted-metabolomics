s=open("/tmp/forest_sweep_stereo.py").read()

# 1) docstring + remove squid_inc/library_match imports, inline _norm + _descriptors
old_imports='''"""Slim sweep: kit (subsampled) drives BOTH the RI->sec ladder AND the structure-RT seed.
args: KIT_SIZE(0=all) KIT_SEED K  -> outputs cluster P/R per consensus-M (threshold curve)
+ cardinality recall + cross-platform precision."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor'''
new_imports='''"""Sparse-anchor metabolite annotation (self-contained; no external squid_inc/library_match deps).

A sparse spike-in anchor kit calibrates an RI->seconds ladder + a structure->RT model; forest
propagation + cardinality assignment annotate raw mzML against a candidate library, scored on
distinct compounds (full InChIKey, stereo/geometric isomers differentiated).

USAGE:  KIT_SIZE KIT_SEED K  (positional; 0 0 10 = full kit, 10 chains)
CONFIG (env vars):
  LIBRARY_CSV     candidate library. Columns (case-insensitive): name/BIOCHEMICAL, inchikey/INCHIKEY,
                  mz/MASS, ri/RI, and optionally smiles/SMILES and platform/PLATFORM.
  KIT_CSV         anchor kit. Columns: smiles, observed_rt_sec, ri.
  MZML_GLOB       glob for raw mzML files (e.g. "/data/*.mzML").
  PLATFORM        platform label to select rows from LIBRARY_CSV/ANSWER_KEY_CSV (default "lc/ms neg").
  ANSWER_KEY_CSV  optional. If given, scores recall/precision + isomer recovery. Columns:
                  platform, name, inchikey, rt, optional unannotatable.
  SMILES_CSV      optional ik14->smiles fallback (columns: inchikey, smiles) if LIBRARY_CSV lacks SMILES.
  OUTPUT_TSV      optional. Write per-annotation table (compound, inchikey, mz, rt, intensity, reps).
  FLOOR (50000), KIT_MODE (chem|random|spread|gap), ROUNDS (10), CACHE_DIR (./.eic_cache), MAX_MZML (8).
"""
import sys, csv, glob, os
from pathlib import Path
from collections import defaultdict
import numpy as np, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor
_norm = lambda s: "".join(ch for ch in (s or "").lower() if ch.isalnum())
def _descriptors(smiles):
    """19 RDKit descriptors for RT prediction; None if RDKit unavailable / SMILES invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol=Chem.MolFromSmiles(smiles)
        if mol is None: return None
        nc=nn=no=ns=npp=nh=0
        for a in mol.GetAtoms():
            sy=a.GetSymbol()
            if sy=="C": nc+=1
            elif sy=="N": nn+=1
            elif sy=="O": no+=1
            elif sy=="S": ns+=1
            elif sy=="P": npp+=1
            elif sy in ("F","Cl","Br","I"): nh+=1
        return [Descriptors.ExactMolWt(mol),Descriptors.MolLogP(mol),Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            float(rdMolDescriptors.CalcNumHeteroatoms(mol)),float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
            float(nc),float(nn),float(no),float(ns),float(npp),float(nh)]
    except Exception: return None'''
assert old_imports in s; s=s.replace(old_imports,new_imports)

# 2) env-driven paths (replace the _METH/hardcoded-path block)
old_paths='''_meth,_kitf,ADD=_METH[PLATFORM]
PLAT=PLATFORM; DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
KIT="/root/untargeted-metabolomics/data/anchor_panels/"+_kitf; SMI="/tmp/dd_pubchem_smiles.csv"
UNI="/mnt/volume-hel1-1/data/processed/compound_universe_broad.csv"; CACHE="/mnt/volume-hel1-1/cache"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/%s_*COLU*.mzML"%_meth))[:8]'''
new_paths='''PLAT=PLATFORM
ADD=-1.007276 if "neg" in PLATFORM.lower() else 1.007276   # [M-H] / [M+H]
_meth=_osf.environ.get("PLATFORM_TAG", _norm(PLATFORM))     # cache tag
DD=_osf.environ["LIBRARY_CSV"]
GT=_osf.environ.get("ANSWER_KEY_CSV","")
KIT=_osf.environ["KIT_CSV"]
SMI=_osf.environ.get("SMILES_CSV","")
UNI=_osf.environ.get("UNIVERSE_CSV","")
CACHE=_osf.environ.get("CACHE_DIR","./.eic_cache")
MZML=sorted(glob.glob(_osf.environ["MZML_GLOB"]))[:int(_osf.environ.get("MAX_MZML","8"))]'''
assert old_paths in s; s=s.replace(old_paths,new_paths)

# 3) optional answer key: guard the MAF-loading loop
s=s.replace("for r in csv.DictReader(open(GT)):\n    if r.get(\"unannotatable\",\"\")==\"true\": continue",
            "for r in (csv.DictReader(open(GT)) if GT else []):\n    if r.get(\"unannotatable\",\"\")==\"true\": continue")

# 4) DD loader: flexible columns + library-provided SMILES + lib0 shim from DDFULL (drops M.load_library)
old_dd='''lib0=M.load_library(Path(DD)); smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
from collections import defaultdict as _dd
DDFULL=_dd(list)  # platform -> [{name,mz,ri,ik(full)}]  (full InChIKey retained)
for _r in csv.DictReader(open(DD,encoding="utf-8-sig")):
    _P=(_r.get("PLATFORM") or "").strip().lower()
    try: _mz=float(_r.get("MASS")); _ri=float(_r.get("RI"))
    except (TypeError,ValueError): continue
    if _mz<=0 or _ri<=0: continue
    DDFULL[_P].append(dict(name=_r.get("BIOCHEMICAL") or "",mz=_mz,ri=_ri,ik=(_r.get("INCHIKEY") or "").strip()))'''
new_dd='''smi={}
if SMI:
    for r in csv.DictReader(open(SMI)):
        _i=ik14(r.get("inchikey") or r.get("INCHIKEY") or ""); _s=r.get("smiles") or r.get("SMILES")
        if _i and _s: smi[_i]=_s
def _col(r,*names):
    for nme in names:
        if nme in r and r[nme] not in (None,""): return r[nme]
    return ""
from collections import defaultdict as _dd
DDFULL=_dd(list)  # platform -> [{name,mz,ri,ik(full),smi}]
for _r in csv.DictReader(open(DD,encoding="utf-8-sig")):
    _P=(_col(_r,"PLATFORM","platform","method")).strip().lower()
    try: _mz=float(_col(_r,"MASS","mz","mass"))
    except (TypeError,ValueError): continue
    if _mz<=0: continue
    try: _ri=float(_col(_r,"RI","ri","retention_index"))
    except (TypeError,ValueError): _ri=0.0
    _ik=_col(_r,"INCHIKEY","inchikey").strip(); _sm=_col(_r,"SMILES","smiles")
    DDFULL[_P].append(dict(name=_col(_r,"BIOCHEMICAL","name","compound"),mz=_mz,ri=_ri,ik=_ik,smi=_sm))
    if _sm and ik14(_ik) not in smi: smi[ik14(_ik)]=_sm
lib0={p:[{"name":c["name"],"ik14":ik14(c["ik"]),"mz":c["mz"],"ri":c["ri"]} for c in v] for p,v in DDFULL.items()}'''
assert old_dd in s; s=s.replace(old_dd,new_dd)

# 5) candidate SMILES fallback to library-provided
s=s.replace('s=smi.get(ik14(c["ik"])); d=_descriptors(s) if s else None',
            's=(c.get("smi") or smi.get(ik14(c["ik"]))); d=_descriptors(s) if s else None')

# 5b) STRUCT_RT: override comp pred with kit-trained structure model (no library RI) -- for no-RI libs
s=s.replace('n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6',
'''n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
if _osf.environ.get("STRUCT_RT"):
    _km=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=3).fit(kitX,kitY)
    for _c in comp:
        if _c["desc"] is not None: _c["pred"]=float(_km.predict(np.asarray(_c["desc"])[None,:])[0])''')

# 6) M._norm -> _norm everywhere
s=s.replace("M._norm","_norm")

# 6b) neutralize any dormant squid_inc import line (standalone is RT/RI+MS1; MS2 needs full pipeline)
_lines=[]
for ln in s.split("\n"):
    if "squid_inc" in ln and "import" in ln:
        ind=ln[:len(ln)-len(ln.lstrip())]
        _lines.append(ind+"raise RuntimeError('MS2 mode is not available in the standalone build')")
    else:
        _lines.append(ln)
s="\n".join(_lines)

# 7) scoring blocks: only when answer key present
s=s.replace("_rec=len(card_recall_set(_cov))/nmaf","_rec=len(card_recall_set(_cov))/max(nmaf,1)")
s=s.replace('print("H2H %s | %s | n=%d nmaf=%d |','nmaf and print("H2H %s | %s | n=%d nmaf=%d |')
s=s.replace("if True:\n    # isomer-pair recovery","if GT and nmaf:\n    # isomer-pair recovery")

# 8) annotation output (works with or without answer key) -- inserted before DUMP block
out_block='''if os.environ.get("OUTPUT_TSV"):
    _oc=cluster(votes>=1)
    with open(os.environ["OUTPUT_TSV"],"w") as _of:
        _of.write("compound\\tinchikey\\tmz\\trt_sec\\tlog10_intensity\\tn_injections\\tn_isobaric_candidates\\tin_answer_key\\n")
        for cl in _oc:
            kk=max(cl,key=lambda k:strength[k]); nr,_=rep_apex(kk,comp[kk]["pred"],TIGHT)
            _ink=("yes" if inmaf[kk] else "no") if nmaf else ""
            _of.write("%s\\t%s\\t%.4f\\t%.1f\\t%.2f\\t%d\\t%d\\t%s\\n"%(comp[kk]["name"],comp[kk]["ik"],MZc[kk],gapex[kk] if not np.isnan(gapex[kk]) else -1,np.log10(strength[kk]+1),nr,len(cl),_ink))
    print("wrote %d annotations -> %s"%(len(_oc),os.environ["OUTPUT_TSV"]),flush=True)
if os.environ.get('DUMP'):'''
s=s.replace("if os.environ.get('DUMP'):", out_block, 1)

open("/tmp/sparse_anchor_annotate.py","w").write(s)
import ast; ast.parse(s)
# report any lingering external deps (don't die — show them)
for bad in ["squid_inc","library_match_rtri","/root/SQuID-INC","import library_match","M.load_library"]:
    if bad in s:
        print("LINGERING:",bad)
        for i,ln in enumerate(s.split("\n"),1):
            if bad in ln: print("   %d: %s"%(i,ln))
print("sparse_anchor_annotate.py built, lines:",s.count(chr(10)))

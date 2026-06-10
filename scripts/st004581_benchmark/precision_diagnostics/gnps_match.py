"""GNPS-style spectral-library matching head-to-head: for each observed DDA MS2, among DD
candidates at the precursor m/z, pick the one whose EXPERIMENTAL reference spectrum (MoNA+
MassBank+GNPS) best matches (entropy). Annotate, score recall/precision vs MAF -- same scoring
as our RT/RI+MS1 method (which gets ~0.84/0.99)."""
import sys, csv, glob, json
from pathlib import Path
import numpy as np, pymzml
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.ms2_similarity import entropy_similarity
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; P="/mnt/volume-hel1-1/data/processed/"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=15.0; ik14=lambda s:(s or "")[:14]
# MAF (scoring)
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(ik14(r.get("inchikey")))
maf={x for x in maf if len(x)>=14}; nmaf=len(maf)
# DD compounds: ik14, m/z
lib0=M.load_library(Path(DD))
dd=[(c.get("ik14") or "", c["mz"]) for c in lib0.get(PLAT,[]) if (c.get("ik14") or "")]
ddik=np.array([ik14(i) for i,_ in dd]); ddmz=np.array([m for _,m in dd])
o=np.argsort(ddmz); ddik=ddik[o]; ddmz=ddmz[o]
our=set(ddik)
# reference library (combined neg), only our DD iks
ref={}
for fn in ["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]:
    d=json.load(open(P+fn))
    for k,v in d.items():
        ik=k[:14]
        if ik in our: ref.setdefault(ik,[]).append([(float(a),float(b)) for a,b in v])
print(f"DD neg with reference spectrum: {len(ref)}  MAF compounds: {nmaf}",flush=True)
# observed MS2
ms2=[]
for mp in MZML:
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=2: continue
        try: pmz=spec.selected_precursors[0]["mz"]
        except Exception: continue
        mz=np.asarray(spec.mz); ii=np.asarray(spec.i)
        if len(mz): ms2.append((float(pmz),[(float(a),float(b)) for a,b in zip(mz.tolist(),ii.tolist())]))
print(f"observed DDA MS2 scans: {len(ms2)}",flush=True)
# match: best DD candidate (m/z within ppm, has ref) by entropy
best={}  # ik -> best entropy score seen
for pmz,pk in ms2:
    t=pmz*MZ_PPM*1e-6; lo=np.searchsorted(ddmz,pmz-t); hi=np.searchsorted(ddmz,pmz+t)
    cand=set(ddik[lo:hi]) & set(ref)
    for ik in cand:
        sc=max(entropy_similarity(pk,r,mz_tol=0.02) for r in ref[ik])
        if sc>best.get(ik,0): best[ik]=sc
print(f"\nGNPS-style spectral-library matching (recall/precision vs MAF):")
print(f"{'entropy>=':>10}{'annot':>7}{'recall':>8}{'precision':>11}")
for tau in [0.3,0.4,0.5,0.6,0.7,0.8]:
    annot=[ik for ik,s in best.items() if s>=tau]
    tp=sum(1 for ik in annot if ik in maf)
    if annot: print(f"{tau:>10}{len(annot):>7}{tp/nmaf:>8.3f}{tp/len(annot):>11.3f}")

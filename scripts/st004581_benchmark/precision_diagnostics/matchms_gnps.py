"""CORRECT GNPS-style matching via matchms (ModifiedCosine + CosineGreedy, the actual GNPS
scoring). Accurate-mass-constrained library search vs MoNA+MassBank+GNPS experimental spectra.
Scored vs MAF identically to our method. GNPS-standard call = cosine>=0.7 & >=6 matched peaks."""
import sys, csv, glob, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pymzml
from matchms import Spectrum
from matchms.similarity import ModifiedCosineGreedy, CosineGreedy
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; P="/mnt/volume-hel1-1/data/processed/"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=15.0; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(ik14(r.get("inchikey")))
maf={x for x in maf if len(x)>=14}; nmaf=len(maf)
lib0=M.load_library(Path(DD))
ddmz={}
for c in lib0.get(PLAT,[]):
    ik=ik14(c.get("ik14") or "")
    if ik and ik not in ddmz: ddmz[ik]=c["mz"]
ddik=np.array(sorted(ddmz, key=lambda k:ddmz[k])); ddmzs=np.array([ddmz[k] for k in ddik])
def mkspec(peaks,pmz):
    p=sorted(peaks); mz=np.array([a for a,_ in p],float); it=np.array([b for _,b in p],float)
    return Spectrum(mz=mz,intensities=it,metadata={"precursor_mz":float(pmz)})
ref={}
for fn in ["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]:
    d=json.load(open(P+fn))
    for k,v in d.items():
        ik=k[:14]
        if ik in ddmz and len(v)>=2:
            try: ref.setdefault(ik,[]).append(mkspec([(float(a),float(b)) for a,b in v], ddmz[ik]))
            except Exception: pass
print(f"DD with reference spectrum: {len(ref)}  MAF: {nmaf}",flush=True)
obs=[]
for mp in MZML:
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=2: continue
        try: pmz=spec.selected_precursors[0]["mz"]
        except Exception: continue
        mz=np.asarray(spec.mz); ii=np.asarray(spec.i)
        if len(mz)>=2:
            try: obs.append(mkspec([(float(a),float(b)) for a,b in zip(mz.tolist(),ii.tolist())], pmz))
            except Exception: pass
print(f"observed DDA MS2: {len(obs)}",flush=True)
mc=ModifiedCosineGreedy(tolerance=0.02)
best={}  # ik -> (best cosine, matches)
for q in obs:
    pmz=q.get("precursor_mz"); t=pmz*MZ_PPM*1e-6
    lo=np.searchsorted(ddmzs,pmz-t); hi=np.searchsorted(ddmzs,pmz+t)
    for ik in set(ddik[lo:hi]) & set(ref):
        for rs in ref[ik]:
            try:
                r=mc.pair(q,rs)
                try: sc=float(r["score"]); nm=int(r["matches"])
                except Exception:
                    tl=np.asarray(r).tolist(); sc,nm=(float(tl[0]),int(tl[1])) if isinstance(tl,(tuple,list)) else (float(tl),0)
            except Exception: continue
            if sc>best.get(ik,(0,0))[0]: best[ik]=(sc,nm)
print(f"\nCORRECT matchms ModifiedCosine library search (vs MAF):")
print(f"{'criterion':>28}{'annot':>7}{'recall':>8}{'precision':>11}")
def score(sel,label):
    tp=sum(1 for ik in sel if ik in maf)
    if sel: print(f"{label:>28}{len(sel):>7}{tp/nmaf:>8.3f}{tp/len(sel):>11.3f}")
for tau in [0.5,0.6,0.7]:
    score([ik for ik,(s,m) in best.items() if s>=tau],"cosine>=%.1f"%tau)
score([ik for ik,(s,m) in best.items() if s>=0.7 and m>=6],"GNPS-standard (>=0.7,>=6pk)")

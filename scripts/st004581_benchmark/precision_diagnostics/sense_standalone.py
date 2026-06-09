"""Independent sense-check: do MAF compounds' observed EIC apexes (sec) track the MAF's reported
RT (RI scale, never used by the method)? High Spearman => our peaks are the real compounds at
consistent locations => recall is sensible, not lucky coincidences."""
import sys, csv, glob
import numpy as np, pymzml
from scipy.stats import spearmanr
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from pathlib import Path
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=15000.0
# MAF neg: name->(mz from DD, rt)
mafrt={}
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT:
        try: mafrt[M._norm(r.get("name") or "")]=float(r.get("rt"))
        except: pass
lib0=M.load_library(Path(DD))
comp=[(M._norm(c["name"]),c["mz"]) for c in lib0.get(PLAT,[]) if M._norm(c["name"]) in mafrt]
seen=set(); comp=[(nm,mz) for nm,mz in comp if nm not in seen and not seen.add(nm)]
n=len(comp); MZc=np.array([m for _,m in comp]); tol=MZc*MZ_PPM*1e-6
apex=np.full(n,np.nan); mx=np.zeros(n)
for mp in MZML:
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rt=spec.scan_time_in_minutes()*60
        if not len(sm): continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        for k in range(n):
            if hi[k]>lo[k]:
                v=si[lo[k]:hi[k]].max()
                if v>mx[k]: mx[k]=v; apex[k]=rt
det=[k for k in range(n) if mx[k]>FLOOR and not np.isnan(apex[k])]
osec=[apex[k] for k in det]; mrt=[mafrt[comp[k][0]] for k in det]
rho,p=spearmanr(osec,mrt)
print(f"MAF neg compounds with a detectable peak (>{int(FLOOR)}): {len(det)}/{n}")
print(f">>> Spearman(our observed apex sec, MAF reported rt) = {rho:.3f}  (p={p:.1e}) <<<")
print("    high (~>0.9) => observed peaks track the MAF's ground-truth elution order = recoveries are real")
# bucketed: fraction of compounds whose apex is consistent (within rank tolerance)
print("\nexamples (MAF_rt, our_apex_sec) sorted by MAF_rt:")
order=sorted(range(len(det)),key=lambda i:mrt[i])
for i in order[::max(1,len(order)//12)][:12]:
    print(f"    {mrt[i]:7.0f}   {osec[i]:6.0f}")

"""Real OpenMS (pyopenms) untargeted feature detection (FeatureFinderMetabo = mass-trace +
elution-peak + feature finding, the OpenMS/XCMS-family pipeline) -> accurate-mass match to DD
library -> score vs MAF. Tests gold-standard detection + accurate-mass annotation."""
import sys, csv, glob
from pathlib import Path
import numpy as np, pyopenms as oms
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:4]
PLAT="lc/ms neg"; MZ_PPM=10.0; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(ik14(r.get("inchikey")))
maf={x for x in maf if len(x)>=14}; nmaf=len(maf)
lib0=M.load_library(Path(DD))
ddik=[]; ddmz=[]
for c in lib0.get(PLAT,[]):
    ik=ik14(c.get("ik14") or "")
    if ik: ddik.append(ik); ddmz.append(c["mz"])
ddik=np.array(ddik); ddmz=np.array(ddmz); o=np.argsort(ddmz); ddik=ddik[o]; ddmz=ddmz[o]
def detect(mzml):
    exp=oms.MSExperiment(); oms.MzMLFile().load(mzml,exp)
    exp.updateRanges()
    mtd=oms.MassTraceDetection(); p=mtd.getParameters(); p.setValue("noise_threshold_int",1000.0); mtd.setParameters(p)
    mts=[]; mtd.run(exp,mts,0)
    epd=oms.ElutionPeakDetection(); p=epd.getParameters(); p.setValue("width_filtering","fixed"); epd.setParameters(p)
    mts_split=[]; epd.detectPeaks(mts,mts_split)
    ffm=oms.FeatureFindingMetabo(); p=ffm.getParameters(); p.setValue("remove_single_traces","true"); ffm.setParameters(p)
    feats=oms.FeatureMap(); chash=[]; ffm.run(mts_split,feats,chash)
    return [(f.getMZ(),f.getRT()) for f in feats]
allf=[]
for mp in MZML:
    try: allf+=detect(mp)
    except Exception as e: print("detect fail",mp.split("/")[-1],e)
print(f"OpenMS features detected (4 inj): {len(allf)}  DD compounds: {len(ddik)}  MAF: {nmaf}",flush=True)
fmz=np.array(sorted(m for m,_ in allf))
# annotate: a DD compound is 'present' if a feature exists within ppm of its [M-H] (accurate-mass match)
called=set()
for i in range(len(ddik)):
    t=ddmz[i]*MZ_PPM*1e-6; lo=np.searchsorted(fmz,ddmz[i]-t); hi=np.searchsorted(fmz,ddmz[i]+t)
    if hi>lo: called.add(ddik[i])
tp=sum(1 for ik in called if ik in maf)
print(f"\nOpenMS feature-detect + accurate-mass match (m/z only, no RT):")
print(f"  called {len(called)}  recall {tp/nmaf:.3f}  precision {tp/max(len(called),1):.3f}")

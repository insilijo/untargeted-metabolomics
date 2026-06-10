"""Build per-platform RT-anchor panels that COVER the RI space.
Selection: compounds that are (a) mass-unique in the data dictionary
(unambiguous detection), (b) reproducibly detected (>=MIN_REP injections),
then greedily tiled across the RI gradient (one strongest anchor per RI bin)
so the RI->sec calibration has no large gaps. Emits panel CSVs usable by
build_anchor_rt_observations / build_rt_calibration.
"""
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
OUTDIR="/root/squid_anchor_panels"
MASS_PPM=20.0; MIN_REP=3; N_BINS=30
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)
import os; os.makedirs(OUTDIR, exist_ok=True)
DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
MAF=list(csv.DictReader(open(MAFCSV)))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]
PLATS=[("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]

def maxgap(vals):
    v=sorted(vals); return max((v[i+1]-v[i] for i in range(len(v)-1)), default=0)

all_rows=[]
for plat,method in PLATS:
    dd_mz=np.array(sorted(ff(r["MASS"]) for r in DD if r["PLATFORM"]==plat and ff(r["MASS"]) and ff(r["RI"])))
    cuniq=lambda mz:(np.searchsorted(dd_mz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(dd_mz,mz-mz*MASS_PPM*1e-6))==1
    fs=feat[feat.method==method]
    fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy(); fsrc=fs.source_file.to_numpy()
    so=np.argsort(fmz); fmz,frt,fint,fsrc=fmz[so],frt[so],fint[so],fsrc[so]
    def detect(mz):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it,sr=frt[lo:hi],fint[lo:hi],fsrc[lo:hi]
        c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20; rt,it,sr=rt[m],it[m],sr[m]
        if len(set(sr))<MIN_REP: return None
        return float(np.median(rt)), float(it.sum())
    # candidate anchors: mass-unique + detected, carry smiles/ik/pubchem from MAF
    cands=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri) or not cuniq(mz): continue
        d=detect(mz)
        if d is None: continue
        rt_sec,inten=d
        cands.append({"name":r["name"],"inchikey":r["inchikey"],"smiles":r.get("smiles",""),
                      "pubchem":r.get("pubchem",""),"mz":mz,"ri":ri,"observed_rt_sec":round(rt_sec,2),
                      "intensity":inten})
    if len(cands)<8:
        print(f"{plat:<16} only {len(cands)} candidate anchors — skipped"); continue
    ris=[c["ri"] for c in cands]; lo,hi=min(ris),max(ris)
    # even RI coverage: nearest unused candidate to each of K evenly-spaced targets
    K=min(N_BINS, len(cands))
    targets=np.linspace(lo,hi,K)
    used=set(); chosen=[]
    for tgt in targets:
        order=sorted(range(len(cands)), key=lambda i:abs(cands[i]["ri"]-tgt))
        for i in order:
            if i not in used: used.add(i); chosen.append(cands[i]); break
    chosen.sort(key=lambda c:c["ri"])
    # residual gaps where the panel is thin (kit should add a standard here)
    span=hi-lo; gapthr=2.0*span/max(K,1)
    crisorted=[c["ri"] for c in chosen]
    gaps=[(round(crisorted[i],0),round(crisorted[i+1],0)) for i in range(len(crisorted)-1)
          if crisorted[i+1]-crisorted[i]>gapthr]
    for c in chosen:
        c["compound_id"]=f"pubchem:{c['pubchem']}" if str(c['pubchem']).strip() else ik14(c["inchikey"])
        c["platform"]=plat; c["method"]=method
    cov_after=maxgap([c["ri"] for c in chosen])
    print(f"{plat:<16} candidates={len(cands):3d} -> anchors={len(chosen):2d}  "
          f"RI[{lo:.0f},{hi:.0f}]  max RI-gap {cov_after:.0f}  thin-regions(kit-supplement): {gaps if gaps else 'none'}")
    cols=["compound_id","name","inchikey","smiles","platform","method","pubchem","mz","ri","observed_rt_sec","intensity"]
    safe=plat.replace("/","_").replace(" ","_")
    with open(f"{OUTDIR}/anchors_{safe}.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for c in chosen: w.writerow({k:c.get(k,"") for k in cols})
    all_rows+=[{k:c.get(k,"") for k in cols} for c in chosen]
cols=["compound_id","name","inchikey","smiles","platform","method","pubchem","mz","ri","observed_rt_sec","intensity"]
with open(f"{OUTDIR}/anchors_all_platforms.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(all_rows)
print(f"\nwrote {len(all_rows)} anchors across platforms -> {OUTDIR}/anchors_*.csv")

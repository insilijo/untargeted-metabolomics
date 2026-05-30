"""Tiered RT/RI -> MS1 matching of detected neg features against the public
Metabolon DATA DICTIONARY (PMC-OA subset) as the reference library, scored
per-feature TP/FP/FN vs the ST004581 MAF ground truth.

Candidate pool = full DD neg (1,718 compounds incl. no-InChIKey confusers),
independent of this study's observed ions -> non-circular library match.
GT = MAF neg compounds. Matching ability is reported on GT compounds that
exist in the DD; the DD library-coverage gap is reported separately.
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd

PLATFORM="lc/ms neg"; MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=3; MIN_ANCHORS=10
ik14=lambda s:(s or "")[:14]
f=lambda x:(float(x) if str(x).strip() not in ("","None") else None)

# ---- candidate library = DD neg ----
DD="/home/jgardner/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
cand=[]
for r in csv.DictReader(open(DD, encoding="utf-8-sig")):
    if r["PLATFORM"]!=PLATFORM: continue
    mz=f(r["MASS"]); ri=f(r["RI"])
    if not(mz and ri): continue
    cand.append({"name":r["BIOCHEMICAL"],"ik14":ik14(r["INCHIKEY"]),"mz":mz,"ri":ri})
cand_mz=np.array([c["mz"] for c in cand]); o=np.argsort(cand_mz)
cand=[cand[i] for i in o]; cand_mz=cand_mz[o]
print(f"DD-neg candidate library: {len(cand)} (with ik14: {sum(1 for c in cand if c['ik14'])})")

# ---- GT = MAF neg ----
gt=[]
dd_ik={c["ik14"] for c in cand if c["ik14"]}
for r in csv.DictReader(open("/home/jgardner/squid_results/st004581/annotations_repaired.csv")):
    if r["platform"]!=PLATFORM: continue
    mz=f(r["mz"]); ri=f(r["rt"])
    if not(mz and ri): continue
    if r.get("unannotatable","")=="true": continue
    g={"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri}
    g["in_dd"]= g["ik14"] in dd_ik
    gt.append(g)
print(f"MAF-neg annotatable GT: {len(gt)}  (present in DD library: {sum(g['in_dd'] for g in gt)})")

# ---- detected features (samples) ----
df=pd.read_parquet("data/st004581_work/features_all.parquet")
s=df[df.sample_type=="sample"]
fmz=s.mz.to_numpy(); frt=s.rt.to_numpy(); fint=s.intensity.to_numpy(); fsrc=s.source_file.to_numpy()
o=np.argsort(fmz); fmz,frt,fint,fsrc=fmz[o],frt[o],fint[o],fsrc[o]
def cons(mz,rc=None,rw=None):
    t=mz*MASS_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
    if hi<=lo: return None
    rt,it,sr=frt[lo:hi],fint[lo:hi],fsrc[lo:hi]
    if rc is not None:
        m=np.abs(rt-rc)<=rw
        if not m.any(): return None
        rt,it,sr=rt[m],it[m],sr[m]
    else:
        c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20; rt,it,sr=rt[m],it[m],sr[m]
    if len(set(sr))<MIN_REP: return None
    return float(np.median(rt))

# ---- calibration RI->sec from mass-unique-in-DD detected GT anchors ----
def n_cand_within(mz):
    t=mz*MASS_PPM*1e-6
    return np.searchsorted(cand_mz,mz+t)-np.searchsorted(cand_mz,mz-t)
anc=[]
for g in gt:
    if n_cand_within(g["mz"])!=1: continue   # mass-unique vs the big DD pool
    rt=cons(g["mz"])
    if rt is not None: anc.append((g["ri"],rt))
anc.sort()
r2=defaultdict(list)
for ri,rt in anc: r2[round(ri,1)].append(rt)
ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
print(f"calibration anchors (mass-unique in DD, detected): {len(anc)}  RI[{ris[0]:.0f},{ris[-1]:.0f}]")
cand_rtsec=np.array([float(pchip(c["ri"])) for c in cand])

def match(mz,rt,use_rt):
    t=mz*MASS_PPM*1e-6; lo=np.searchsorted(cand_mz,mz-t); hi=np.searchsorted(cand_mz,mz+t)
    best=None; bs=9e9
    for i in range(lo,hi):
        if use_rt and abs(cand_rtsec[i]-rt)>RT_WIN: continue
        sc=abs(cand_mz[i]-mz)/t+(abs(cand_rtsec[i]-rt)/RT_WIN if use_rt else 0)
        if sc<bs: bs=sc; best=i
    return best

def evaluate(use_rt, subset):
    tp=fp=fn=0; conf=defaultdict(int)
    for g in gt:
        if subset=="in_dd" and not g["in_dd"]: continue
        er=float(pchip(g["ri"]))
        rt=cons(g["mz"], er if use_rt else None, RT_WIN if use_rt else None)
        if rt is None: fn+=1; continue
        j=match(g["mz"],rt,use_rt)
        if j is None: fn+=1; continue
        if cand[j]["ik14"] and cand[j]["ik14"]==g["ik14"]: tp+=1
        else: fp+=1; conf[(g["name"],cand[j]["name"])]+=1
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    F1=2*P*R/(P+R) if P+R else 0
    return tp,fp,fn,P,R,F1,conf

for subset,label in [("in_dd","GT present in DD library (matching ability)"),
                     ("all","all annotatable GT (incl. library gaps)")]:
    n=sum(1 for g in gt if subset=="all" or g["in_dd"])
    print(f"\n=== {label}: n={n} ===")
    print(f"{'method':<16}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for use_rt,nm in [(False,"MS1 only"),(True,"RT/RI + MS1")]:
        tp,fp,fn,P,R,F1,conf=evaluate(use_rt,subset)
        print(f"{nm:<16}{tp:>5}{fp:>5}{fn:>5}{P:>8.3f}{R:>8.3f}{F1:>8.3f}")
        if use_rt and subset=="in_dd":
            print("   top isobaric confusions (true -> assigned):")
            for (a,b),c in sorted(conf.items(),key=lambda x:-x[1])[:10]:
                print(f"     {a[:32]:<32} -> {b[:32]}")

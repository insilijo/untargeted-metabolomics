"""Quantify FP fixes + recall lever on the answer-in-library full-dataset match.
 baseline      : current RT/RI+MS1
 +drop_noik    : structureless DD candidates (no InChIKey) cannot win  [library hygiene, no GT used]
 +presence(orc): candidate pool restricted to compounds present in the study [ORACLE upper bound for an anchor/graph presence prior]
Recall: MIN_REP sweep (3->2->1) on the not-detected FNs.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
MASS_PPM=20.0; RT_WIN=30.0
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)
MAF=list(csv.DictReader(open(MAFCSV)))
study_present=set()
for r in MAF:
    if r.get("unannotatable","")!="true" and r["inchikey"]: study_present.add(ik14(r["inchikey"]))
DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]

def run(plat, method, MIN_REP=3):
    cand=[]
    for r in DD:
        if r["PLATFORM"]!=plat: continue
        mz=ff(r["MASS"]); ri=ff(r["RI"])
        if not(mz and ri): continue
        cand.append({"ik14":ik14(r["INCHIKEY"]),"mz":mz,"ri":ri})
    o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]; cmz=np.array([c["mz"] for c in cand])
    dd_ik={c["ik14"] for c in cand if c["ik14"]}
    gt=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri) or ik14(r["inchikey"]) not in dd_ik: continue
        gt.append({"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri})
    fs=feat[feat.method==method]
    fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy(); fsrc=fs.source_file.to_numpy()
    so=np.argsort(fmz); fmz,frt,fint,fsrc=fmz[so],frt[so],fint[so],fsrc[so]
    def cons(mz,rc,rw,minrep):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it,sr=frt[lo:hi],fint[lo:hi],fsrc[lo:hi]
        if rc is not None:
            m=np.abs(rt-rc)<=rw
            if not m.any(): return None
            rt,sr=rt[m],sr[m]
        else:
            c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20; rt,sr=rt[m],sr[m]
        if len(set(sr))<minrep: return None
        return float(np.median(rt))
    pmz=np.array(sorted(g["mz"] for g in gt))
    puniq=lambda mz:(np.searchsorted(pmz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(pmz,mz-mz*MASS_PPM*1e-6))==1
    anc=sorted((g["ri"],cons(g["mz"],None,None,MIN_REP)) for g in gt if puniq(g["mz"]) and cons(g["mz"],None,None,MIN_REP) is not None)
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    def evalmode(mode, minrep):
        tp=fp=fn=0
        for g in gt:
            er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN, minrep)
            if rt is None: fn+=1; continue
            t=g["mz"]*MASS_PPM*1e-6; lo=np.searchsorted(cmz,g["mz"]-t); hi=np.searchsorted(cmz,g["mz"]+t)
            best=None; bs=9e9
            for i in range(lo,hi):
                if abs(crt[i]-rt)>RT_WIN: continue
                if mode in("noik","presence") and not cand[i]["ik14"]: continue
                if mode=="presence" and cand[i]["ik14"] not in study_present: continue
                sc=abs(cmz[i]-g["mz"])/t+abs(crt[i]-rt)/RT_WIN
                if sc<bs: bs=sc; best=i
            if best is None: fn+=1; continue
            if cand[best]["ik14"]==g["ik14"]: tp+=1
            else: fp+=1
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F1=2*P*R/(P+R) if P+R else 0
        return tp,fp,fn,P,R,F1
    return gt,{m:evalmode(m,MIN_REP) for m in ("base","noik","presence")}, \
           {mr:evalmode("noik",mr) for mr in (3,2,1)}

plats=[("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]
agg={m:[0,0,0] for m in ("base","noik","presence")}
print("PRECISION FIXES (answer-in-library, RT/RI+MS1):")
print(f"{'platform':<16}{'base F1':>9}{'+noik':>9}{'+presence':>11}")
for plat,method in plats:
    gt,modes,_=run(plat,method)
    print(f"{plat:<16}{modes['base'][5]:>9.3f}{modes['noik'][5]:>9.3f}{modes['presence'][5]:>11.3f}")
    for m in agg:
        agg[m][0]+=modes[m][0]; agg[m][1]+=modes[m][1]; agg[m][2]+=modes[m][2]
print("POOLED:")
for m in ("base","noik","presence"):
    tp,fp,fn=agg[m]; P=tp/(tp+fp); R=tp/(tp+fn); F1=2*P*R/(P+R)
    print(f"  {m:<10} TP{tp} FP{fp} FN{fn}  P{P:.3f} R{R:.3f} F1{F1:.3f}")

print("\nRECALL LEVER (MIN_REP sweep, +noik fix), pooled:")
for mr in (3,2,1):
    T=F=N=0
    for plat,method in plats:
        _,_,sweep=run(plat,method,MIN_REP=mr)
        tp,fp,fn,_,_,_=sweep[mr]; T+=tp; F+=fp; N+=fn
    P=T/(T+F); R=T/(T+N); print(f"  MIN_REP={mr}: TP{T} FP{F} FN{N}  P{P:.3f} R{R:.3f} F1{2*P*R/(P+R):.3f}")

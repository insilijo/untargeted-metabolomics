"""Non-oracle presence prior built from the matcher's OWN cross-platform calls.
supported = ik14 that is the top-1 call (with InChIKey) on >=1 platform where it
is MASS-UNIQUE there (unambiguous detection -> vouches for presence study-wide).
Then on platforms where it's an isobaric competitor, require winner in supported.
Compare: base / +noik / +xpresence(data-derived) / +presence(oracle).
"""
from __future__ import annotations
import csv
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=2
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)
MAF=list(csv.DictReader(open(MAFCSV)))
study_present=set(ik14(r["inchikey"]) for r in MAF if r.get("unannotatable","")!="true" and r["inchikey"])
DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]
PLATS=[("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]

def setup(plat, method):
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
    def cons(mz,rc,rw):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it,sr=frt[lo:hi],fint[lo:hi],fsrc[lo:hi]
        if rc is not None:
            m=np.abs(rt-rc)<=rw
            if not m.any(): return None
            rt,sr=rt[m],sr[m]
        else:
            c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20; rt,sr=rt[m],sr[m]
        if len(set(sr))<MIN_REP: return None
        return float(np.median(rt))
    pmz=np.array(sorted(g["mz"] for g in gt))
    puniq=lambda mz:(np.searchsorted(pmz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(pmz,mz-mz*MASS_PPM*1e-6))==1
    cuniq=lambda mz:(np.searchsorted(cmz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(cmz,mz-mz*MASS_PPM*1e-6))==1
    anc=sorted((g["ri"],cons(g["mz"],None,None)) for g in gt if puniq(g["mz"]) and cons(g["mz"],None,None) is not None)
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    return dict(cand=cand,cmz=cmz,crt=crt,gt=gt,cons=cons,pchip=pchip,cuniq=cuniq)

S={p:setup(p,m) for p,m in PLATS}
# data-derived presence: compound mass-unique on its platform AND detected (top-1 trivially) -> supported
supported=set()
for p,_ in PLATS:
    s=S[p]
    for g in s["gt"]:
        if s["cuniq"](g["mz"]):
            rt=s["cons"](g["mz"], float(s["pchip"](g["ri"])), RT_WIN)
            if rt is not None: supported.add(g["ik14"])
print(f"data-derived 'supported present' (mass-unique+detected on some platform): {len(supported)}")

def evalmode(s, mode):
    cand,cmz,crt,gt,cons,pchip=s["cand"],s["cmz"],s["crt"],s["gt"],s["cons"],s["pchip"]
    tp=fp=fn=0
    for g in gt:
        er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN)
        if rt is None: fn+=1; continue
        t=g["mz"]*MASS_PPM*1e-6; lo=np.searchsorted(cmz,g["mz"]-t); hi=np.searchsorted(cmz,g["mz"]+t)
        cands=[]
        for i in range(lo,hi):
            if abs(crt[i]-rt)>RT_WIN: continue
            ik=cand[i]["ik14"]
            if mode!="base" and not ik: continue
            if mode=="oracle" and ik not in study_present: continue
            sc=abs(cmz[i]-g["mz"])/t+abs(crt[i]-rt)/RT_WIN
            cands.append((sc,i))
        if not cands: fn+=1; continue
        cands.sort()
        if mode=="xpresence":
            # soft: prefer best-scoring SUPPORTED candidate; fall back to best overall
            sup=[c for c in cands if cand[c[1]]["ik14"] in supported]
            best=(sup[0] if sup else cands[0])[1]
        else:
            best=cands[0][1]
        if cand[best]["ik14"]==g["ik14"]: tp+=1
        else: fp+=1
    return np.array([tp,fp,fn])

print(f"\n(MIN_REP={MIN_REP})  pooled answer-in-library:")
print(f"{'mode':<12}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}")
for mode,lbl in [("base","base"),("noik","+noik"),("xpresence","+xpresence(data)"),("oracle","+presence(oracle)")]:
    tot=sum(evalmode(S[p],mode) for p,_ in PLATS); tp,fp,fn=tot
    P=tp/(tp+fp); R=tp/(tp+fn); print(f"{lbl:<16}{tp:>5}{fp:>5}{fn:>5}{P:>8.3f}{R:>8.3f}{2*P*R/(P+R):>8.3f}")

"""Per-platform tiered RT/RI -> MS1 (-> MS2) library match vs the public
Metabolon data dictionary, for all four ST004581 LC-MS platforms.

Platform <- mzML method prefix:
  Method1 lc/ms pos early | Method2 lc/ms pos late | Method3 lc/ms neg | Method4 lc/ms polar
Features: features_all.parquet (neg) + features_pospolar.parquet (pos/polar).
Calibration: per-platform PCHIP RI->sec on panel-mass-unique detected anchors.
MS2: squid-inc get_ms2, GNPS experimental (pos/neg) preferred + RDKit predicted.
"""
from __future__ import annotations
import csv, sys, os
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
from pathlib import Path
sys.path.insert(0,"/home/jgardner/SQuID-INC")
from squid_inc.features.ms2_predict import get_ms2, load_library

MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=3; MS2_MZ_TOL=0.02; MS2_MIN_COS=0.2
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)
PREFIX={"Method1":"lc/ms pos early","Method2":"lc/ms pos late","Method3":"lc/ms neg","Method4":"lc/ms polar"}
MODE={"lc/ms pos early":"pos","lc/ms pos late":"pos","lc/ms neg":"neg","lc/ms polar":"pos"}

# smiles map
smi={}
for r in csv.DictReader(open("data/raw/metabolon_annotations.csv")):
    if r.get("smiles","").strip(): smi.setdefault(ik14(r["inchikey"]), r["smiles"].strip())
uni="/home/jgardner/SQuID-INC/data/processed/compound_universe.csv"
if os.path.exists(uni):
    for r in csv.DictReader(open(uni)):
        k=ik14(r.get("inchikey","")); sm=r.get("smiles","").strip()
        if k and sm: smi.setdefault(k,sm)
# GNPS experimental libs (pos+neg) into squid-inc caches
for m,p in [("neg","data/st004581_work/gnps_neg_ms2_ik14.json"),("pos","data/st004581_work/gnps_pos_ms2_ik14.json")]:
    if os.path.exists(p): load_library(Path(p), mode=m)

# DD candidates + MAF GT (all platforms)
DD=list(csv.DictReader(open("/home/jgardner/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv", encoding="utf-8-sig")))
MAF=list(csv.DictReader(open("/home/jgardner/squid_results/st004581/annotations_repaired.csv")))
# features
feat=pd.concat([pd.read_parquet("data/st004581_work/features_all.parquet"),
                pd.read_parquet("data/st004581_work/features_pospolar.parquet")], ignore_index=True)
feat=feat[feat.sample_type=="sample"].copy()
feat["method"]=feat.source_file.str.split("_").str[0]
# MS2 observed
ms2=pd.concat([pd.read_parquet("data/st004581_work/ms2_observed.parquet"),
               pd.read_parquet("data/st004581_work/ms2_pospolar.parquet")], ignore_index=True)
ms2["method"]=ms2.source_file.str.split("_").str[0]

_ref={}
def cosine(ma,ia,mb,ib):
    if len(ma)==0 or len(mb)==0: return 0.0
    ia=ia/(np.linalg.norm(ia) or 1); ib=ib/(np.linalg.norm(ib) or 1)
    i=j=0; sc=0.0
    while i<len(ma) and j<len(mb):
        d=ma[i]-mb[j]
        if abs(d)<=MS2_MZ_TOL: sc+=ia[i]*ib[j]; i+=1; j+=1
        elif d<0: i+=1
        else: j+=1
    return float(sc)

def run_platform(plat, method):
    mode=MODE[plat]
    cand=[]
    for r in DD:
        if r["PLATFORM"]!=plat: continue
        mz=ff(r["MASS"]); ri=ff(r["RI"])
        if not(mz and ri): continue
        k=ik14(r["INCHIKEY"]); cand.append({"name":r["BIOCHEMICAL"],"ik14":k,"mz":mz,"ri":ri,"smiles":smi.get(k,"")})
    if not cand: return None
    o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]; cmz=np.array([c["mz"] for c in cand])
    dd_ik={c["ik14"] for c in cand if c["ik14"]}
    gt=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri): continue
        gt.append({"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri,"in_dd":ik14(r["inchikey"]) in dd_ik})
    fs=feat[feat.method==method]
    fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy(); fsrc=fs.source_file.to_numpy()
    so=np.argsort(fmz); fmz,frt,fint,fsrc=fmz[so],frt[so],fint[so],fsrc[so]
    ms=ms2[ms2.method==method]; mo=np.argsort(ms.precursor_mz.to_numpy())
    o_mz=ms.precursor_mz.to_numpy()[mo]; o_rt=ms.rt.to_numpy()[mo]
    o_ma=ms.mz_array.to_numpy()[mo]; o_ia=ms.intensity_array.to_numpy()[mo]
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
    def obs(mz,rt):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(o_mz,mz-t); hi=np.searchsorted(o_mz,mz+t)
        best=None; bd=RT_WIN
        for i in range(lo,hi):
            d=abs(o_rt[i]-rt)
            if d<=bd: bd=d; best=i
        return (np.asarray(o_ma[best],float),np.asarray(o_ia[best],float)) if best is not None else None
    def refms2(c,prec):
        key=(c["ik14"],round(prec,2),mode)
        if key in _ref: return _ref[key]
        pk=get_ms2(smiles=c["smiles"],inchikey=c["ik14"],precursor_mz=prec,mode=mode)
        v=None
        if pk:
            a=np.array(pk,float); a=a[np.argsort(a[:,0])]; v=(a[:,0],a[:,1])
        _ref[key]=v; return v
    # calibration: panel-unique anchors
    pmz=np.array(sorted(g["mz"] for g in gt))
    def puniq(mz):
        t=mz*MASS_PPM*1e-6; return (np.searchsorted(pmz,mz+t)-np.searchsorted(pmz,mz-t))==1
    anc=sorted((g["ri"],cons(g["mz"])) for g in gt if puniq(g["mz"]) and cons(g["mz"]) is not None)
    if len(anc)<8: return {"plat":plat,"err":f"only {len(anc)} calib anchors"}
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    def cand_at(mz,rt,use_rt):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(cmz,mz-t); hi=np.searchsorted(cmz,mz+t)
        out=[]
        for i in range(lo,hi):
            if use_rt and abs(crt[i]-rt)>RT_WIN: continue
            out.append((abs(cmz[i]-mz)/t+(abs(crt[i]-rt)/RT_WIN if use_rt else 0),i))
        return out
    def ev(tier, indd=False):
        use_rt=tier in("rtri","ms2"); tp=fp=fn=0
        for g in gt:
            if indd and not g["in_dd"]: continue
            er=float(pchip(g["ri"])); rt=cons(g["mz"], er if use_rt else None, RT_WIN if use_rt else None)
            if rt is None: fn+=1; continue
            C=cand_at(g["mz"],rt,use_rt)
            if not C: fn+=1; continue
            C.sort(); win=C[0][1]
            if tier=="ms2":
                om=obs(g["mz"],rt)
                if om is not None:
                    sc=[(cosine(om[0],om[1],*refms2(cand[i],g["mz"])),i) for _,i in C if refms2(cand[i],g["mz"]) is not None]
                    if sc:
                        sc.sort(reverse=True)
                        if sc[0][0]>=MS2_MIN_COS: win=sc[0][1]
            if cand[win]["ik14"] and cand[win]["ik14"]==g["ik14"]: tp+=1
            else: fp+=1
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F1=2*P*R/(P+R) if P+R else 0
        return tp,fp,fn,P,R,F1
    return {"plat":plat,"n_cand":len(cand),"n_gt":len(gt),"n_indd":sum(g["in_dd"] for g in gt),
            "anchors":len(anc),
            "ms1":ev("ms1",True),"rtri":ev("rtri",True),"ms2":ev("ms2",True),
            "ms1_all":ev("ms1"),"rtri_all":ev("rtri"),"ms2_all":ev("ms2")}

print("ANSWER-IN-LIBRARY (matching ability, GT present in public DD) — the Metabolon paradigm")
print(f"{'platform':<16}{'cand':>6}{'GTinDD':>7}{'anc':>5}   {'MS1':>7}{'RT/RI':>7}{'+MS2':>7}")
allrows=[]
for plat,method in [("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),
                    ("lc/ms neg","Method3"),("lc/ms polar","Method4")]:
    r=run_platform(plat,method)
    if r is None or "err" in r: print(f"{plat:<16} {r.get('err') if r else 'no candidates'}"); continue
    allrows.append(r)
    print(f"{plat:<16}{r['n_cand']:>6}{r['n_indd']:>7}{r['anchors']:>5}   "
          f"{r['ms1'][5]:>7.3f}{r['rtri'][5]:>7.3f}{r['ms2'][5]:>7.3f}")
    t=r['rtri']; print(f"     RT/RI+MS1:  TP{t[0]:>4} FP{t[1]:>4} FN{t[2]:>4}  P{t[3]:.3f} R{t[4]:.3f}")
print("\nEND-TO-END (all annotatable GT, incl. public-subset library gaps)")
print(f"{'platform':<16}{'GT':>5}   {'MS1':>7}{'RT/RI':>7}{'+MS2':>7}")
for r in allrows:
    print(f"{r['plat']:<16}{r['n_gt']:>5}   {r['ms1_all'][5]:>7.3f}{r['rtri_all'][5]:>7.3f}{r['ms2_all'][5]:>7.3f}")

"""(1) no-ik candidates win only if they're the only option in the window.
(2) backfill DD InChIKeys by matching BIOCHEMICAL name -> MAF inchikey
    (proxy for Metabolon's full structured library; imports study structures
    by NAME, not by mz/RT, so flag as upper-bound like the oracle).
Eval on a FIXED denominator (compounds structured in the ORIGINAL DD) for
comparability; separately report coverage gained by backfill.
"""
import csv, re
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
norm=lambda s:re.sub(r"\s+"," ",(s or "").strip().lower())
MAF=list(csv.DictReader(open(MAFCSV)))
name2ik={norm(r["name"]):ik14(r["inchikey"]) for r in MAF
         if r.get("unannotatable","")!="true" and (r.get("inchikey") or "").strip()}
DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]
PLATS=[("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]

bf_total=0; newcov=0
def setup(plat, method):
    global bf_total,newcov
    cand=[]
    for r in DD:
        if r["PLATFORM"]!=plat: continue
        mz=ff(r["MASS"]); ri=ff(r["RI"])
        if not(mz and ri): continue
        ikorig=ik14(r["INCHIKEY"]); bf=""
        if not ikorig:
            cand_ik=name2ik.get(norm(r["BIOCHEMICAL"]))
            if cand_ik: bf=cand_ik
        cand.append({"ik":ikorig,"bf":bf,"mz":mz,"ri":ri})
    o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]; cmz=np.array([c["mz"] for c in cand])
    nbf=sum(1 for c in cand if c["bf"]); bf_total+=nbf
    ik_orig={c["ik"] for c in cand if c["ik"]}
    ik_fixed=ik_orig|{c["bf"] for c in cand if c["bf"]}
    newcov+=len(ik_fixed-ik_orig)
    gt=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri) or ik14(r["inchikey"]) not in ik_orig: continue   # FIXED denom = original DD
        gt.append({"ik":ik14(r["inchikey"]),"mz":mz,"ri":ri})
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
    anc=sorted((g["ri"],cons(g["mz"],None,None)) for g in gt if puniq(g["mz"]) and cons(g["mz"],None,None) is not None)
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    return dict(cand=cand,cmz=cmz,crt=crt,gt=gt,cons=cons,pchip=pchip)

S={p:setup(p,m) for p,m in PLATS}
def effective_ik(c, mode):
    # which ik does this candidate carry under each mode?
    if mode=="ddfix": return c["ik"] or c["bf"]
    return c["ik"]
def evalmode(s, mode):
    cand,cmz,crt,gt,cons,pchip=s["cand"],s["cmz"],s["crt"],s["gt"],s["cons"],s["pchip"]
    tp=fp=fn=0
    for g in gt:
        er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN)
        if rt is None: fn+=1; continue
        t=g["mz"]*MASS_PPM*1e-6; lo=np.searchsorted(cmz,g["mz"]-t); hi=np.searchsorted(cmz,g["mz"]+t)
        struct=[]; nostruct=[]
        for i in range(lo,hi):
            if abs(crt[i]-rt)>RT_WIN: continue
            sc=abs(cmz[i]-g["mz"])/t+abs(crt[i]-rt)/RT_WIN
            eik=effective_ik(cand[i],mode)
            (struct if eik else nostruct).append((sc,i,eik))
        if mode=="base":
            allc=sorted(struct+nostruct)
            if not allc: fn+=1; continue
            wik=allc[0][2]
        else:  # noik / onlyif / ddfix: prefer structured; no-ik only if no structured (onlyif/ddfix); noik never
            if struct:
                wik=sorted(struct)[0][2]
            elif mode in("onlyif","ddfix") and nostruct:
                wik=sorted(nostruct)[0][2]   # only option -> structureless family (FP for scoring)
            else:
                fn+=1; continue
        if wik==g["ik"]: tp+=1
        else: fp+=1
    return np.array([tp,fp,fn])

print(f"backfilled DD candidates (name->MAF ik): {bf_total};  new compounds addressable: {newcov}")
print(f"\n(MIN_REP={MIN_REP}, fixed denom = original-DD-structured GT)  pooled:")
print(f"{'mode':<22}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}")
for mode,lbl in [("base","base"),("noik","+noik(hard)"),("onlyif","+noik-only-if-alone"),("ddfix","+ddfix backfill")]:
    tot=sum(evalmode(S[p],mode) for p,_ in PLATS); tp,fp,fn=tot
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    print(f"{lbl:<22}{tp:>5}{fp:>5}{fn:>5}{P:>8.3f}{R:>8.3f}{2*P*R/(P+R) if P+R else 0:>8.3f}")

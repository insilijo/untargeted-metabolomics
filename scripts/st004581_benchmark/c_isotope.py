"""Isotope deconvolution of the isobaric FPs.
Same-formula isomers -> identical isotope pattern (isotopes CANNOT help).
Different-formula isobars -> distinct M+1/M+2 envelope (isotopes CAN help, esp.
neg sulfates: 34S adds ~4.25% at M+2).
For each noik-FP where true(T) and assigned(A) both have InChIKey, classify by
formula; for diff-formula ones extract observed M+1/M+2 ratios from the feature
table and re-rank window candidates by isotope-pattern match -> does it flip to T?
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import re
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
UNI="/root/SQuID-INC/data/processed/compound_universe.csv"
MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=2; ISO_DMZ=0.01; RT_ISO=6.0
C13,N15,O17,O18,S33,S34=0.0108,0.00369,0.00038,0.00205,0.0079,0.0425
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)
smi={}
for r in csv.DictReader(open(UNI)):
    k=ik14(r.get("inchikey","")); s=(r.get("smiles") or "").strip()
    if k and s: smi.setdefault(k,s)
_F={}
def formula(ik):
    if ik in _F: return _F[ik]
    s=smi.get(ik,""); f=None
    if s:
        m=Chem.MolFromSmiles(s)
        if m is not None:
            try: f=CalcMolFormula(m)
            except Exception: f=None
    _F[ik]=f; return f
def counts(fm):
    d=defaultdict(int)
    for el,n in re.findall(r"([A-Z][a-z]?)(\d*)",fm or ""):
        if el: d[el]+=int(n) if n else 1
    return d
def pred_iso(fm):  # predicted (M+1/M0, M+2/M0)
    c=counts(fm); nC=c.get("C",0)
    m1=nC*C13+c.get("N",0)*N15+c.get("O",0)*O17+c.get("S",0)*S33+c.get("H",0)*0.000115
    m2=c.get("O",0)*O18+c.get("S",0)*S34+c.get("Cl",0)*0.32+c.get("Br",0)*0.97+(nC*C13)**2/2
    return m1,m2

DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
MAF=list(csv.DictReader(open(MAFCSV)))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]

def run(plat, method):
    cand=[]
    for r in DD:
        if r["PLATFORM"]!=plat: continue
        mz=ff(r["MASS"]); ri=ff(r["RI"])
        if not(mz and ri): continue
        cand.append({"name":r["BIOCHEMICAL"],"ik14":ik14(r["INCHIKEY"]),"mz":mz,"ri":ri})
    o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]; cmz=np.array([c["mz"] for c in cand])
    dd_ik={c["ik14"] for c in cand if c["ik14"]}
    gt=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri) or ik14(r["inchikey"]) not in dd_ik: continue
        gt.append({"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri})
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
    def isum(mz,rt):  # summed intensity near (mz,rt)
        t=mz*ISO_DMZ if False else 0.01; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return 0.0
        sel=np.abs(frt[lo:hi]-rt)<=RT_ISO
        return float(fint[lo:hi][sel].sum())
    def obs_iso(mz,rt):
        i0=isum(mz,rt)
        if i0<=0: return None
        return isum(mz+1.003355,rt)/i0, isum(mz+2.00671,rt)/i0
    pmz=np.array(sorted(g["mz"] for g in gt))
    puniq=lambda mz:(np.searchsorted(pmz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(pmz,mz-mz*MASS_PPM*1e-6))==1
    anc=sorted((g["ri"],cons(g["mz"],None,None)) for g in gt if puniq(g["mz"]) and cons(g["mz"],None,None) is not None)
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    res=Counter()
    for g in gt:
        er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN)
        if rt is None: continue
        t=g["mz"]*MASS_PPM*1e-6; lo=np.searchsorted(cmz,g["mz"]-t); hi=np.searchsorted(cmz,g["mz"]+t)
        W=[i for i in range(lo,hi) if abs(crt[i]-rt)<=RT_WIN and cand[i]["ik14"]]
        if not W: continue
        best=min(W, key=lambda i: abs(cmz[i]-g["mz"])/t+abs(crt[i]-rt)/RT_WIN)
        if cand[best]["ik14"]==g["ik14"]: continue   # TP under noik
        res["FP_noik"]+=1
        fT=formula(g["ik14"]); fA=formula(cand[best]["ik14"])
        if not fT or not fA: res["FP_no_formula"]+=1; continue
        # distinct formulas among window candidates?
        wf={i:formula(cand[i]["ik14"]) for i in W}
        if fT==fA: res["isomer_sameformula"]+=1; continue
        res["isobar_diffformula"]+=1
        oi=obs_iso(g["mz"],rt)
        if oi is None: res["diff_no_isodata"]+=1; continue
        res["diff_with_isodata"]+=1
        # re-rank window candidates (with formula) by isotope-pattern distance
        scored=[]
        for i in W:
            f=wf[i]
            if not f: continue
            p1,p2=pred_iso(f)
            d=abs(p1-oi[0])+abs(p2-oi[1])
            scored.append((d,i))
        if not scored: continue
        scored.sort()
        if cand[scored[0][1]]["ik14"]==g["ik14"]: res["iso_FIXED"]+=1
    return res

tot=Counter()
for plat,method in [("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]:
    r=run(plat,method); tot+=r
    print(f"{plat:<16} FP_noik={r['FP_noik']:3d}  isomer(sameF)={r['isomer_sameformula']:3d}  "
          f"isobar(diffF)={r['isobar_diffformula']:3d}  w/isodata={r['diff_with_isodata']:3d}  iso_FIXED={r['iso_FIXED']:3d}")
print("\nPOOLED:")
for k in ["FP_noik","isomer_sameformula","isobar_diffformula","diff_with_isodata","iso_FIXED","FP_no_formula"]:
    print(f"  {k}: {tot[k]}")

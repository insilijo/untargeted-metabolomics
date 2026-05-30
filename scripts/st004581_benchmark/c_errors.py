"""Error analysis of the full-dataset RT/RI+MS1 match (answer-in-library set).
FP: is the wrong assignment an ISOMER (same formula) / isobar (<3 mDa) of the
    truth (unresolvable by MS1) or a genuinely different-mass mistake?
FN: not detected at all (no feature at m/z) vs RT-window rejection (feature
    exists but calibrated RT missed it) vs no candidate.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict, Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
sys.path.insert(0,"/root/SQuID-INC")
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
UNI="/root/SQuID-INC/data/processed/compound_universe.csv"
MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=3
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)

smi={}
for r in csv.DictReader(open(UNI)):
    k=ik14(r.get("inchikey","")); s=(r.get("smiles") or "").strip()
    if k and s: smi.setdefault(k,s)
_formula={}
def formula(ik):
    if ik in _formula: return _formula[ik]
    s=smi.get(ik,""); f=""
    if s:
        m=Chem.MolFromSmiles(s)
        if m is not None:
            try: f=CalcMolFormula(m)
            except Exception: f=""
    _formula[ik]=f; return f

DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
MAF=list(csv.DictReader(open(MAFCSV)))
feat=pd.read_csv(FEATTSV, sep="\t", usecols=["source_file","mz","rt","intensity"])
feat=feat[feat.source_file.str.contains("COLU", na=False)].copy()
feat["method"]=feat.source_file.str.split("_").str[0]

def analyze(plat, method):
    cand=[]
    for r in DD:
        if r["PLATFORM"]!=plat: continue
        mz=ff(r["MASS"]); ri=ff(r["RI"])
        if not(mz and ri): continue
        k=ik14(r["INCHIKEY"]); cand.append({"name":r["BIOCHEMICAL"],"ik14":k,"mz":mz,"ri":ri})
    o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]; cmz=np.array([c["mz"] for c in cand])
    dd_ik={c["ik14"] for c in cand if c["ik14"]}
    gt=[]
    for r in MAF:
        if r["platform"]!=plat or r.get("unannotatable","")=="true": continue
        mz=ff(r["mz"]); ri=ff(r["rt"])
        if not(mz and ri) or ik14(r["inchikey"]) not in dd_ik: continue   # answer-in-library set
        gt.append({"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri})
    fs=feat[feat.method==method]
    fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy(); fsrc=fs.source_file.to_numpy()
    so=np.argsort(fmz); fmz,frt,fint,fsrc=fmz[so],frt[so],fint[so],fsrc[so]
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
    pmz=np.array(sorted(g["mz"] for g in gt))
    def puniq(mz):
        t=mz*MASS_PPM*1e-6; return (np.searchsorted(pmz,mz+t)-np.searchsorted(pmz,mz-t))==1
    anc=sorted((g["ri"],cons(g["mz"])) for g in gt if puniq(g["mz"]) and cons(g["mz"]) is not None)
    r2=defaultdict(list)
    for ri,rt in anc: r2[round(ri,1)].append(rt)
    ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
    pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
    crt=np.array([float(pchip(c["ri"])) for c in cand])
    def cand_at(mz,rt):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(cmz,mz-t); hi=np.searchsorted(cmz,mz+t)
        out=[]
        for i in range(lo,hi):
            if abs(crt[i]-rt)>RT_WIN: continue
            out.append((abs(cmz[i]-mz)/t+abs(crt[i]-rt)/RT_WIN,i))
        return out
    fp=Counter(); fp_ex=[]; fn=Counter(); rt_offsets=[]
    for g in gt:
        er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN)
        if rt is None:                          # FN
            rt0=cons(g["mz"])
            if rt0 is None: fn["not_detected"]+=1
            else:
                fn["rt_window_reject"]+=1; rt_offsets.append(rt0-er)
            continue
        C=cand_at(g["mz"],rt)
        if not C: fn["no_candidate"]+=1; continue
        C.sort(); w=cand[C[0][1]]
        if w["ik14"]==g["ik14"]: continue       # TP
        # FP: classify
        dmz=abs(w["mz"]-g["mz"])*1000  # mDa
        ftrue=formula(g["ik14"]); fass=formula(w["ik14"])
        if ftrue and fass and ftrue==fass: cat="isomer(same formula)"
        elif dmz<3.0: cat="isobar(<3mDa)"
        else: cat="diff-mass mistake"
        fp[cat]+=1
        fp_ex.append((cat,g["name"],w["name"],dmz,w["ri"]-g["ri"]))
    return dict(plat=plat,n=len(gt),fp=fp,fp_ex=fp_ex,fn=fn,
                rt_off=(np.median(np.abs(rt_offsets)) if rt_offsets else 0,len(rt_offsets)))

for plat,method in [("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]:
    r=analyze(plat,method)
    print(f"\n===== {plat}  (answer-in-lib n={r['n']}) =====")
    print(f"  FP {sum(r['fp'].values())}: "+", ".join(f"{k}={v}" for k,v in r['fp'].most_common()))
    print(f"  FN {sum(r['fn'].values())}: "+", ".join(f"{k}={v}" for k,v in r['fn'].most_common())
          +f"   [rt_reject median |offset| {r['rt_off'][0]:.0f}s over {r['rt_off'][1]}]")
    dm=[e for e in r['fp_ex'] if e[0]=="diff-mass mistake"]
    if dm:
        print("   diff-mass mistakes (potential real errors):")
        for c,tn,an,dmz,dri in dm[:6]: print(f"     {tn[:30]:<30} -> {an[:30]:<30} dmz {dmz:.1f}mDa dRI {dri:+.0f}")
    iso=[e for e in r['fp_ex'] if e[0].startswith("isomer")]
    if iso:
        print("   isomer confusions (unresolvable by MS1):")
        for c,tn,an,dmz,dri in iso[:5]: print(f"     {tn[:30]:<30} -> {an[:30]:<30} dRI {dri:+.0f}")

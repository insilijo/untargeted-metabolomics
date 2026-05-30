"""Can cross-method observation + presence priors pull apart the isobaric FPs?

For each FP (true T mis-assigned to A on platform P), classify the WRONG
candidate A:
  decoy_no_ik       : DD entry has no structure (pure library noise)
  decoy_not_in_study: A is in NO platform's MAF -> not actually present anywhere
  decoy_other_plat  : A is in the study but NOT on platform P
  co_present_thisplat: A is genuinely annotated on P -> real co-eluting isobar (hard)
Also: how many true compounds are MULTI-PLATFORM (cross-method evidence available)
and the dRI(sec) gap of the FP (co-elution vs separable-by-better-RT).
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict, Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEATTSV="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=3
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)

MAF=list(csv.DictReader(open(MAFCSV)))
# ik14 -> set(platforms) annotated in the study (presence map)
plats_of=defaultdict(set)
for r in MAF:
    if r.get("unannotatable","")=="true": continue
    if r["platform"] and r["inchikey"]: plats_of[ik14(r["inchikey"])].add(r["platform"])
study_present=set(plats_of)
print(f"study compounds (any platform): {len(study_present)};  "
      f"multi-platform: {sum(1 for k,v in plats_of.items() if len(v)>1)}")

DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
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
        m=np.abs(rt-rc)<=rw
        if not m.any(): return None
        rt,sr=rt[m],sr[m]
        if len(set(sr))<MIN_REP: return None
        return float(np.median(rt))
    def cons0(mz):
        t=mz*MASS_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it,sr=frt[lo:hi],fint[lo:hi],fsrc[lo:hi]
        c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20; rt,sr=rt[m],sr[m]
        if len(set(sr))<MIN_REP: return None
        return float(np.median(rt))
    pmz=np.array(sorted(g["mz"] for g in gt))
    puniq=lambda mz:(np.searchsorted(pmz,mz+mz*MASS_PPM*1e-6)-np.searchsorted(pmz,mz-mz*MASS_PPM*1e-6))==1
    anc=sorted((g["ri"],cons0(g["mz"])) for g in gt if puniq(g["mz"]) and cons0(g["mz"]) is not None)
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
    cat=Counter(); dRI=[]; rescue_xmethod=0; fp_co=[]
    for g in gt:
        er=float(pchip(g["ri"])); rt=cons(g["mz"], er, RT_WIN)
        if rt is None: continue
        C=cand_at(g["mz"],rt)
        if not C: continue
        C.sort(); A=cand[C[0][1]]
        if A["ik14"]==g["ik14"]: continue   # TP
        # FP: classify the wrong candidate A
        if not A["ik14"]: c="decoy_no_ik"
        elif A["ik14"] not in study_present: c="decoy_not_in_study"
        elif plat not in plats_of[A["ik14"]]: c="decoy_other_plat"
        else: c="co_present_thisplat"
        cat[c]+=1; dRI.append(abs(crt[C[0][1]]-er))
        # cross-method rescue heuristic: true compound seen on another platform
        # AND the wrong A is NOT co-present on this platform (a decoy/phantom)
        if len(plats_of[g["ik14"]])>1 and c!="co_present_thisplat": rescue_xmethod+=1
        if c=="co_present_thisplat": fp_co.append((g["name"],A["name"]))
    return dict(plat=plat,n=len(gt),cat=cat,dRI=dRI,rescue=rescue_xmethod,
                mp=sum(1 for g in gt if len(plats_of[g["ik14"]])>1),fp_co=fp_co)

tot=Counter(); tot_resc=0; tot_fp=0
for plat,method in [("lc/ms pos early","Method1"),("lc/ms pos late","Method2"),("lc/ms neg","Method3"),("lc/ms polar","Method4")]:
    r=run(plat,method); nfp=sum(r['cat'].values()); tot+=r['cat']; tot_resc+=r['rescue']; tot_fp+=nfp
    dim=np.array(r['dRI']);
    print(f"\n== {plat}  n={r['n']}  multi-platform GT={r['mp']}/{r['n']} ==")
    print(f"  FP {nfp}: "+", ".join(f"{k}={v}" for k,v in r['cat'].most_common()))
    if len(dim): print(f"  FP dRI(sec): median {np.median(dim):.0f}, <5s(co-elute) {int((dim<5).sum())}, >=5s(separable) {int((dim>=5).sum())}")
    print(f"  x-method-rescuable FP (true is multi-platform & A is a decoy): {r['rescue']}/{nfp}")
    if r['fp_co'][:4]: print("   genuine co-present isobars:", "; ".join(f"{a[:18]}->{b[:18]}" for a,b in r['fp_co'][:4]))
print(f"\n===== POOLED FP {tot_fp} =====")
for k,v in tot.most_common(): print(f"  {k}: {v} ({v/tot_fp:.0%})")
print(f"  cross-method-rescuable: {tot_resc}/{tot_fp} ({tot_resc/tot_fp:.0%})")

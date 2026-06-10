"""Add MS2 as a 3rd tier (RT/RI -> MS1 -> MS2) to the neg ST004581 library
match against the public Metabolon data dictionary.

- Calibration WIDENED: PCHIP RI->sec fit on panel-mass-unique anchors (~153)
  rather than DD-unique (~53), to test whether tighter calibration lets the
  RT/RI tier net-help (task 'a').
- MS2 via squid-inc squid_inc.features.ms2_predict.get_ms2: GNPS-neg
  experimental library preferred (loaded into its neg cache), RDKit neg-mode
  fragment prediction fallback for the rest.
- MS2 only RE-RANKS the candidates already retrieved by RT/RI+MS1 (when the
  feature was fragmented), so it can fix isobaric FPs but never costs recall.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
sys.path.insert(0,"/home/jgardner/SQuID-INC")
from squid_inc.features.ms2_predict import get_ms2, load_library, library_size

PLATFORM="lc/ms neg"; MASS_PPM=20.0; RT_WIN=30.0; MIN_REP=3
MS2_MZ_TOL=0.02; MS2_MIN_COS=0.2   # override mass-winner only if a candidate beats this
ik14=lambda s:(s or "")[:14]
ff=lambda x:(float(x) if str(x).strip() not in ("","None") else None)

# ---- smiles map (for predicted MS2 fallback) ----
smi={}
for r in csv.DictReader(open("data/raw/metabolon_annotations.csv")):
    if r.get("smiles","").strip(): smi.setdefault(ik14(r["inchikey"]), r["smiles"].strip())
uni="/home/jgardner/SQuID-INC/data/processed/compound_universe.csv"
import os
if os.path.exists(uni):
    for r in csv.DictReader(open(uni)):
        k=ik14(r.get("inchikey","")); s=r.get("smiles","").strip()
        if k and s: smi.setdefault(k,s)
print(f"ik14->smiles map: {len(smi)}")

# ---- GNPS-neg experimental library into squid-inc neg cache ----
from pathlib import Path
n=load_library(Path("data/st004581_work/gnps_neg_ms2_ik14.json"), mode="neg")
print(f"loaded GNPS-neg experimental spectra: {n} (neg cache size {library_size('neg')})")

# ---- candidate library = DD neg ----
cand=[]
for r in csv.DictReader(open("/home/jgardner/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv", encoding="utf-8-sig")):
    if r["PLATFORM"]!=PLATFORM: continue
    mz=ff(r["MASS"]); ri=ff(r["RI"])
    if not(mz and ri): continue
    k=ik14(r["INCHIKEY"])
    cand.append({"name":r["BIOCHEMICAL"],"ik14":k,"mz":mz,"ri":ri,"smiles":smi.get(k,"")})
o=np.argsort([c["mz"] for c in cand]); cand=[cand[i] for i in o]
cand_mz=np.array([c["mz"] for c in cand])
print(f"DD-neg candidates: {len(cand)} (with smiles: {sum(1 for c in cand if c['smiles'])})")

# ---- GT = MAF neg ----
dd_ik={c["ik14"] for c in cand if c["ik14"]}
gt=[]
for r in csv.DictReader(open("/home/jgardner/squid_results/st004581/annotations_repaired.csv")):
    if r["platform"]!=PLATFORM or r.get("unannotatable","")=="true": continue
    mz=ff(r["mz"]); ri=ff(r["rt"])
    if not(mz and ri): continue
    gt.append({"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri,
               "in_dd":ik14(r["inchikey"]) in dd_ik})
print(f"MAF-neg GT: {len(gt)}  in DD: {sum(g['in_dd'] for g in gt)}")

# ---- detected features ----
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

# ---- observed MS2 indexed by precursor mz ----
m2=pd.read_parquet("data/st004581_work/ms2_observed.parquet")
o2=np.argsort(m2.precursor_mz.to_numpy())
o2_mz=m2.precursor_mz.to_numpy()[o2]; o2_rt=m2.rt.to_numpy()[o2]
o2_marr=m2.mz_array.to_numpy()[o2]; o2_iarr=m2.intensity_array.to_numpy()[o2]
def obs_ms2(mz, rt):
    t=mz*MASS_PPM*1e-6; lo=np.searchsorted(o2_mz,mz-t); hi=np.searchsorted(o2_mz,mz+t)
    if hi<=lo: return None
    best=None; bd=RT_WIN
    for i in range(lo,hi):
        d=abs(o2_rt[i]-rt)
        if d<=bd: bd=d; best=i
    if best is None: return None
    return np.asarray(o2_marr[best],float), np.asarray(o2_iarr[best],float)

def cosine(ma,ia,mb,ib):
    if len(ma)==0 or len(mb)==0: return 0.0
    ia=ia/ (np.linalg.norm(ia) or 1); ib=ib/(np.linalg.norm(ib) or 1)
    i=j=0; sc=0.0
    while i<len(ma) and j<len(mb):
        d=ma[i]-mb[j]
        if abs(d)<=MS2_MZ_TOL: sc+=ia[i]*ib[j]; i+=1; j+=1
        elif d<0: i+=1
        else: j+=1
    return float(sc)

_ref_cache={}
def ref_ms2(c, prec):
    key=(c["ik14"], round(prec,2))
    if key in _ref_cache: return _ref_cache[key]
    peaks=get_ms2(smiles=c["smiles"], inchikey=c["ik14"], precursor_mz=prec, mode="neg")
    if peaks:
        arr=np.array(peaks,float); arr=arr[np.argsort(arr[:,0])]
        v=(arr[:,0],arr[:,1])
    else: v=None
    _ref_cache[key]=v; return v

# ---- candidate retrieval (RT/RI + MS1) ----
cand_rtsec=None
def candidates(mz, rt, use_rt):
    t=mz*MASS_PPM*1e-6; lo=np.searchsorted(cand_mz,mz-t); hi=np.searchsorted(cand_mz,mz+t)
    out=[]
    for i in range(lo,hi):
        if use_rt and abs(cand_rtsec[i]-rt)>RT_WIN: continue
        massrt=abs(cand_mz[i]-mz)/t + (abs(cand_rtsec[i]-rt)/RT_WIN if use_rt else 0)
        out.append((massrt,i))
    return out

def evaluate(mode, subset):  # mode: 'ms1','rtri','ms2'
    use_rt = mode in ("rtri","ms2")
    tp=fp=fn=0; conf=defaultdict(int); n_ms2_used=0; n_ms2_fix=0; n_ms2_avail=0
    for g in gt:
        if subset=="in_dd" and not g["in_dd"]: continue
        er=float(pchip(g["ri"]))
        rt=cons(g["mz"], er if use_rt else None, RT_WIN if use_rt else None)
        if rt is None: fn+=1; continue
        C=candidates(g["mz"], rt, use_rt)
        if not C: fn+=1; continue
        C.sort()
        base_i=C[0][1]                      # mass/rt winner
        win_i=base_i
        if mode=="ms2":
            om=obs_ms2(g["mz"], rt)
            if om is not None:
                n_ms2_avail+=1
                scored=[]
                for _,i in C:
                    rf=ref_ms2(cand[i], g["mz"])
                    if rf is not None:
                        scored.append((cosine(om[0],om[1],rf[0],rf[1]), i))
                if scored:
                    scored.sort(reverse=True)
                    if scored[0][0]>=MS2_MIN_COS and scored[0][1]!=base_i:
                        win_i=scored[0][1]; n_ms2_used+=1
        ok = cand[win_i]["ik14"] and cand[win_i]["ik14"]==g["ik14"]
        if mode=="ms2" and win_i!=base_i:
            base_ok = cand[base_i]["ik14"]==g["ik14"]
            if ok and not base_ok: n_ms2_fix+=1
        if ok: tp+=1
        else: fp+=1; conf[(g["name"],cand[win_i]["name"])]+=1
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    F1=2*P*R/(P+R) if P+R else 0
    return dict(tp=tp,fp=fp,fn=fn,P=P,R=R,F1=F1,conf=conf,
                ms2_avail=n_ms2_avail,ms2_used=n_ms2_used,ms2_fix=n_ms2_fix)

# ---- calibration: WIDENED (panel-mass-unique anchors) ----
panel_mz=np.array(sorted(g["mz"] for g in gt))   # 406-neg panel (GT mz)
def panel_unique(mz):
    t=mz*MASS_PPM*1e-6
    return (np.searchsorted(panel_mz,mz+t)-np.searchsorted(panel_mz,mz-t))==1
anc=sorted((g["ri"],cons(g["mz"])) for g in gt if panel_unique(g["mz"]) and cons(g["mz"]) is not None)
r2=defaultdict(list)
for ri,rt in anc: r2[round(ri,1)].append(rt)
ris=sorted(r2); rts=[float(np.median(r2[r])) for r in ris]
pchip=PchipInterpolator(np.array(ris,float),np.array(rts,float),extrapolate=True)
cand_rtsec=np.array([float(pchip(c["ri"])) for c in cand])
print(f"WIDENED calibration anchors (panel-unique, detected): {len(anc)}  RI[{ris[0]:.0f},{ris[-1]:.0f}]")

for subset,label in [("in_dd","GT present in DD (matching ability)"),
                     ("all","all annotatable GT (incl. library gaps)")]:
    n=sum(1 for g in gt if subset=="all" or g["in_dd"])
    print(f"\n=== {label}: n={n} ===")
    print(f"{'tier':<22}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    res={}
    for mode,nm in [("ms1","MS1 only"),("rtri","RT/RI + MS1"),("ms2","RT/RI + MS1 + MS2")]:
        m=evaluate(mode,subset); res[mode]=m
        print(f"{nm:<22}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['P']:>8.3f}{m['R']:>8.3f}{m['F1']:>8.3f}")
    m=res["ms2"]
    print(f"  MS2: {m['ms2_avail']} retrieved GT had an observed MS2; MS2 changed {m['ms2_used']} calls, "
          f"fixed {m['ms2_fix']} isobaric FP->TP  (FP {res['rtri']['fp']}->{res['ms2']['fp']})")

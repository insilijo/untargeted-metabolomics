"""Train the structure->SECONDS model on the DENSIFY POOL (hundreds of mass-unique compounds
detected in THIS run; observed RT free from the feature table, no MAF) instead of 30 kit
anchors. Does prediction error on the mislocated compounds collapse -> dual-RT recovers FNs?"""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PLAT="lc/ms neg"; MZ_PPM=7.0; FLOOR=50000.0; W=5.0; MINREP=2; NRAND=40; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
fneg=df[df.platform==PLAT]; fmz=fneg.mz.to_numpy(); frt=fneg.rt.to_numpy(); fint=fneg.intensity.to_numpy()
o=np.argsort(fmz); fmz,frt,fint=fmz[o],frt[o],fint[o]
def dom_rt(mz):
    t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
    if hi<=lo: return None
    rt,it=frt[lo:hi],fint[lo:hi]; c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20
    return float(np.median(rt[m])) if m.sum()>=3 else None
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# DENSIFY-POOL training: mass-unique neg compounds with smiles + a detected dominant RT
ent=lib0.get(PLAT,[]); mzs_all=np.array(sorted(c["mz"] for c in ent))
Xtr=[]; ytr=[]
for c in ent:
    t=c["mz"]*MZ_PPM*1e-6
    if (np.searchsorted(mzs_all,c["mz"]+t)-np.searchsorted(mzs_all,c["mz"]-t))!=1: continue  # mass-unique
    s=smi.get(c["ik14"]); rt=dom_rt(c["mz"])
    if s and rt:
        d=_descriptors(s)
        if d is not None: Xtr.append(d); ytr.append(rt)
print(f"densify-pool training compounds: {len(Xtr)} (vs 30 kit anchors)",flush=True)
mdl=HistGradientBoostingRegressor(max_iter=400,max_depth=4,learning_rate=0.05,min_samples_leaf=5).fit(np.array(Xtr),np.array(ytr))
cal={p:list(v) for p,v in anchors.items()}
for plat,e in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in e))
    for c in e:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
ladders,pooled,cov,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
RI=np.array(sorted(agg)); SEC=np.array([np.median(agg[r]) for r in RI]); u,ui=np.unique(RI,return_index=True)
inv=PchipInterpolator(u,SEC[ui],extrapolate=True)
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"]); 
    if not nm: continue
    es=float(inv(c["ri"])); s=smi.get(c["ik14"]); rs=es
    if s:
        d=_descriptors(s)
        if d is not None: rs=float(mdl.predict([d])[0])
    comp.append((nm,c["mz"],es,rs))
names=[x[0] for x in comp]; MZc=np.array([x[1] for x in comp]); esec=np.array([x[2] for x in comp]); rsec=np.array([x[3] for x in comp])
inmaf=np.array([n in maf for n in names]); nmaf=len(maf); tol=MZc*MZ_PPM*1e-6; N=len(MZML)
rng=np.random.RandomState(0)
pres_s=np.zeros(len(comp)); pres_d=np.zeros(len(comp)); bh_s=np.zeros(len(comp)); bh_d=np.zeros(len(comp)); bt=np.zeros(len(comp))
apex=np.full(len(comp),np.nan); maxi=np.zeros(len(comp))
for mp in MZML:
    rts=[];rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(len(comp))); continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(comp))]))
    rt=np.array(rts); mat=np.array(rows); c1=rng.uniform(rt.min()+W,rt.max()-W,NRAND); c2=rng.uniform(rt.min()+W,rt.max()-W,NRAND)
    for k in range(len(comp)):
        col=mat[:,k]; j=np.argmax(col)
        if col[j]>maxi[k]: maxi[k]=col[j]; apex[k]=rt[j]
        ps=col[np.abs(rt-esec[k])<=W].max(initial=0)>FLOOR
        pdd=ps or (col[np.abs(rt-rsec[k])<=W].max(initial=0)>FLOOR)
        if ps: pres_s[k]+=1
        if pdd: pres_d[k]+=1
        for i in range(NRAND):
            bt[k]+=1
            h1=col[np.abs(rt-c1[i])<=W].max(initial=0)>FLOOR
            if h1: bh_s[k]+=1
            if h1 or (col[np.abs(rt-c2[i])<=W].max(initial=0)>FLOOR): bh_d[k]+=1
ok=(maxi>FLOOR)&~np.isnan(apex)
dl=np.abs(esec[ok]-apex[ok]); dm=np.abs(rsec[ok]-apex[ok]); far=dl>20
print(f"\nprediction error vs EIC apex: |ladder| median {np.median(dl):.1f}s   |struct(densify)| median {np.median(dm):.1f}s")
print(f"  on ladder-mislocated ({int(far.sum())}): ladder {np.median(dl[far]):.0f}s  struct {np.median(dm[far]):.0f}s  struct-closer {int((dm[far]<dl[far]).sum())}/{int(far.sum())}")
def fdr(present,bh):
    b=np.clip(bh/np.maximum(bt,1),1e-6,0.999); pv=binom.sf(present-1,N,b)
    o=np.argsort(pv); m=len(pv); q=np.empty(m); cur=1.0
    for r_,i in enumerate(o[::-1]): cur=min(cur,pv[i]*m/(m-r_)); q[i]=cur
    keep=(q<0.05)&(present>=MINREP); c=keep.sum(); tp=(keep&inmaf).sum(); return c,tp/nmaf,(tp/c if c else 0)
cs,rs_,ps=fdr(pres_s,bh_s); cd,rd,pdp=fdr(pres_d,bh_d)
print(f"\n{'mode':<22}{'called':>8}{'recall':>9}{'precision':>11}")
print(f"{'ladder only':<22}{cs:>8}{rs_:>9.3f}{ps:>11.3f}")
print(f"{'dual-RT (densify)':<22}{cd:>8}{rd:>9.3f}{pdp:>11.3f}")

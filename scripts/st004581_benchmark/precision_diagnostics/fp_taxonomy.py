"""Classify every FP call into mechanistic groups by its relation to the nearest PRESENT
(MAF) compound. No gating/rerun — pure taxonomy."""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SMI="/tmp/dd_pubchem_smiles.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZ_PPM=20.0; MIN_REP=1; ISO=1.003355
maf=defaultdict(dict)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    try: ri=float(r["rt"])
    except: continue
    plat=(r.get("platform") or "").strip().lower(); nm=M._norm(r.get("name") or "")
    if nm: maf[plat].setdefault(nm,ri)
total_maf=sum(len(v) for v in maf.values())
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
cal={p:list(v) for p,v in anchors.items()}
for plat,ent in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in ent))
    for c in ent:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
lib=M.expand_library_adducts(lib0)
ladders,pooled,cov,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
slope=M.ri_per_sec(pp); ri_tol={p:10.0*s for p,s in slope.items()}
anchor_ris={p:np.array(sorted(ri for _,ri in v)) for p,v in anchors.items()}
df=M.normalise_ri(df,ladders,pooled); cons=M.consensus_features(df,MZ_PPM,ri_tol,MIN_REP)
smi=M.load_smiles_map(Path(SMI)); qm=M.fit_qsrr_ri(cons,lib,smi,MZ_PPM); M.attach_qsrr_ri(lib,qm,smi)
qsrr_by_id={}
for cand in lib.values():
    for c in cand:
        if c.get("is_primary",True) and c.get("ri_qsrr") is not None: qsrr_by_id.setdefault(M.cid_of(c,"name"),c["ri_qsrr"])
win=M.make_ri_win_fn(slope,anchor_ris,"fixed",30.0,10.0,0.5,60.0,{})
ann=M.match(cons,lib,MZ_PPM,win,True,"gated",id_key="name"); ann=M.ordinal_reassign(ann,lib,MZ_PPM,win,"name")
CI={}
for plat,g in cons.groupby("platform"):
    o=np.argsort(g.mz.to_numpy()); CI[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o])
def iso_ok(plat,mz,ri):
    cm,cr=CI[plat]; m1=mz+ISO; t=m1*MZ_PPM*1e-6; lo=np.searchsorted(cm,m1-t); hi=np.searchsorted(cm,m1+t)
    return hi>lo and bool(np.any(np.abs(cr[lo:hi]-ri)<=ri_tol[plat]))
# best feature per called compound
rec={}
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    for mid,rn,inten,nf,mz in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.intensity.to_numpy(),g.n_files.to_numpy(),g.mz.to_numpy()):
        k=(plat,mid); inten=float(inten)
        if k not in rec or inten>rec[k][2]:
            corr=False; ri=gri.get(mid)
            if ri is not None:
                w=win(plat,ri); q=qsrr_by_id.get(mid)
                corr=abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w)
            rec[k]=[float(mz),float(rn),inten,int(nf),mid in gri,corr]
# present compound features (mz,ri) per platform
PF=defaultdict(list)
for (plat,mid),v in rec.items():
    if v[4] and v[5]: PF[plat].append((v[0],v[1]))
PFa={p:(np.array([x[0] for x in v]),np.array([x[1] for x in v])) for p,v in PF.items()}
ISOS=[1.00336,2.00671]
def offs(plat): return ISOS+([36.9839,46.0055,-18.0106,1.9958] if "neg" in plat else [21.9819,37.9559,17.0265,-18.0106])
def classify(plat,mz,ri,nf,iso):
    pmz,pri=PFa.get(plat,(np.array([]),np.array([])))
    if len(pmz):
        co=np.abs(pri-ri)<=ri_tol[plat]
        # substitution: present compound at SAME m/z, co-eluting
        same=np.abs(pmz-mz)<=mz*MZ_PPM*1e-6
        if (same&co).any(): return "substitution (present co-elutes, same m/z)"
        # duplicate: feature = present + isotope/adduct offset, co-eluting
        if co.any():
            for o in offs(plat):
                if np.any(np.abs((mz-pmz[co])-o)<=0.01): return "duplicate (isotope/adduct of present)"
        # close m/z far RT: present at same m/z but not co-eluting
        if same.any(): return "close m/z, far RT (present elsewhere)"
    if iso and nf>=20: return "novel/real-other (reproducible, not in MAF)"
    return "noise (no M+1, sparse)"
cnt=defaultdict(int); ntp=0
for (plat,mid),v in rec.items():
    mz,ri,inten,nf,ismaf,corr=v
    if ismaf and corr: ntp+=1; continue
    cnt[classify(plat,mz,ri,nf,iso_ok(plat,mz,ri))]+=1
fp=sum(cnt.values())
print(f"called compounds {ntp+fp}   TP {ntp}   FP {fp}   (MAF present {total_maf})\n")
print("FP taxonomy:")
for k in ["substitution (present co-elutes, same m/z)","duplicate (isotope/adduct of present)",
          "close m/z, far RT (present elsewhere)","novel/real-other (reproducible, not in MAF)","noise (no M+1, sparse)"]:
    print(f"  {cnt[k]:>4} ({cnt[k]/fp*100:>3.0f}% of FP)  {k}")

"""Two identity/presence signals from the ion envelope:
 (1) envelope size = # co-eluting adduct/isotope partners (real compound throws a family).
 (2) isotope-ratio consistency = |observed M+1/M0  -  expected nC*1.07%| (IDENTITY-aware:
     a feature really = compound Y must show Y's carbon count in its M+1 ratio)."""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from rdkit import Chem
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SMI="/tmp/dd_pubchem_smiles.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZ_PPM=20.0; MIN_REP=1; ISO=1.003355
ik14=lambda s:(s or "")[:14]
nC={}
for r in csv.DictReader(open(SMI)):
    k=ik14(r.get("inchikey","")); s=(r.get("smiles") or "").strip()
    if k and s and k not in nC:
        m=Chem.MolFromSmiles(s); nC[k]=sum(1 for a in m.GetAtoms() if a.GetSymbol()=="C") if m else None
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
# ik per compound name (for carbon count lookup)
name_ik={}
for plat,ent in lib0.items():
    for c in ent: name_ik.setdefault(M.cid_of(c,"name"),c["ik14"])
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
    o=np.argsort(g.mz.to_numpy()); CI[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o],g.intensity.to_numpy()[o])
def feat_int(plat,mz,ri):
    mzs,ris,its=CI[plat]; t=mz*MZ_PPM*1e-6; lo=np.searchsorted(mzs,mz-t); hi=np.searchsorted(mzs,mz+t)
    if hi<=lo: return 0.0
    m=np.abs(ris[lo:hi]-ri)<=ri_tol[plat]
    return float(its[lo:hi][m].max()) if m.any() else 0.0
def envelope(plat,mz,ri):
    offs=[ISO,2*ISO]+([36.9839,46.0055,-18.0106,1.9958] if "neg" in plat else [21.9819,37.9559,17.0265,-18.0106])
    return sum(1 for o in offs if feat_int(plat,mz+o,ri)>0)
rec={}
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    for mid,rn,inten,mz in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.intensity.to_numpy(),g.mz.to_numpy()):
        k=(plat,mid); inten=float(inten)
        if k not in rec or inten>rec[k][0]:
            corr=False; ri=gri.get(mid)
            if ri is not None:
                w=win(plat,ri); q=qsrr_by_id.get(mid)
                corr=abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w)
            rec[k]=[inten,float(mz),float(rn),mid in gri,corr,mid]
env=[]; isoc=[]; tp=[]; have_iso=[]
for (plat,mid),v in rec.items():
    inten,mz,ri,ismaf,corr,mname=v
    e=envelope(plat,mz,ri); env.append(e); tp.append(ismaf and corr)
    c=nC.get(name_ik.get(mname,""))
    i1=feat_int(plat,mz+ISO,ri)
    if c and c>0 and inten>0 and i1>0:
        obs=i1/inten; exp=c*0.0107; isoc.append(abs(obs-exp)/max(exp,1e-3)); have_iso.append(True)
    else: isoc.append(np.nan); have_iso.append(False)
env=np.array(env); tp=np.array(tp); isoc=np.array(isoc); have_iso=np.array(have_iso)
def auc(sig,pos):
    order=np.argsort(sig); rank=np.empty(len(sig)); rank[order]=np.arange(1,len(sig)+1)
    n1=pos.sum(); n0=len(sig)-n1
    return (rank[pos].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"called {len(tp)}  TP {tp.sum()}\n")
print(f"envelope size (# co-eluting adduct/isotope partners): AUC {auc(env.astype(float),tp):.3f}")
print(f"  median envelope: TP {np.median(env[tp]):.1f}  FP {np.median(env[~tp]):.1f}")
m=have_iso
print(f"\nisotope-ratio consistency (lower=more consistent), on {m.sum()} calls w/ formula+M+1:")
print(f"  AUC (consistency as TP signal): {auc(-isoc[m], tp[m]):.3f}")
print(f"  median |obs-exp|/exp: TP {np.median(isoc[m&tp]):.2f}  FP {np.median(isoc[m&~tp]):.2f}")
# combined ranking: envelope desc then iso-consistency
print(f"\nRanking by envelope size -> closed-world precision:")
o=np.argsort(-env); t=tp[o]; cum=np.cumsum(t); called=np.arange(1,len(o)+1)
for target in [0.1,0.2,0.3,0.4,0.5,0.6]:
    need=int(target*total_maf); idx=np.searchsorted(cum,need)
    if idx<len(o): print(f"  recall {cum[idx]/total_maf:.2f}  precision {cum[idx]/called[idx]:.3f}")

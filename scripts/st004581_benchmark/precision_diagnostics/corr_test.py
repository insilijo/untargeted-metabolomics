"""Cross-sample correlation: does each call's M0 intensity track its M+1 (and adduct
partners) ACROSS injections? Real ion family -> high corr; coincidental neighbor -> low.
The user's 'correlated as adducts' test, done properly (not just 'a partner exists')."""
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
sf_idx={s:i for i,s in enumerate(sorted(df.source_file.unique()))}
df["sfi"]=df.source_file.map(sf_idx); NSAMP=len(sf_idx)
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
# per-platform sorted raw arrays with sample index
RAW={}
for plat,g in df.groupby("platform"):
    o=np.argsort(g.mz.to_numpy()); RAW[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o],g.intensity.to_numpy()[o],g.sfi.to_numpy()[o])
def vec(plat,mz,ri):
    mzs,ris,its,sfi=RAW[plat]; t=mz*MZ_PPM*1e-6; lo=np.searchsorted(mzs,mz-t); hi=np.searchsorted(mzs,mz+t)
    if hi<=lo: return None
    m=np.abs(ris[lo:hi]-ri)<=ri_tol[plat]
    if m.sum()<8: return None
    v=np.zeros(NSAMP); s=sfi[lo:hi][m]; it=its[lo:hi][m]
    for si,iv in zip(s,it): v[si]=max(v[si],iv)
    return v
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
            rec[k]=[inten,float(mz),float(rn),mid in gri,corr]
cors=[]; tp=[]
for (plat,mid),v in rec.items():
    inten,mz,ri,ismaf,corr=v
    v0=vec(plat,mz,ri); v1=vec(plat,mz+ISO,ri)
    if v0 is None or v1 is None: c=np.nan
    else:
        both=(v0>0)&(v1>0)
        c=np.corrcoef(np.log1p(v0[both]),np.log1p(v1[both]))[0,1] if both.sum()>=8 else np.nan
    cors.append(c); tp.append(ismaf and corr)
cors=np.array(cors); tp=np.array(tp); ok=~np.isnan(cors)
def auc(sig,pos):
    order=np.argsort(sig); rank=np.empty(len(sig)); rank[order]=np.arange(1,len(sig)+1)
    n1=pos.sum(); n0=len(sig)-n1
    return (rank[pos].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"called {len(tp)}  TP {tp.sum()}  with computable M0-M+1 correlation: {ok.sum()}\n")
print(f"M0<->M+1 cross-sample correlation: AUC {auc(cors[ok],tp[ok]):.3f}")
print(f"  median corr: TP {np.median(cors[ok&tp]):.3f}  FP {np.median(cors[ok&~tp]):.3f}")
# gate: keep only calls with corr above thresholds, closed-world P/R (uncomputable corr -> excluded)
for thr in [0.0,0.5,0.7,0.8,0.9]:
    keep=ok&(cors>=thr); called=keep.sum(); t=(keep&tp).sum()
    P=t/called if called else 0; R=t/total_maf
    print(f"  corr>={thr}: called {called:>4}  TP {t:>3}  precision {P:.3f}  recall {R:.3f}")

"""Closed-world compound-level precision/recall tradeoff, mirroring Metabolon's
library-bounded annotation. A 'call' = a distinct library compound we annotate >=1
feature to. TP = called compound is in the MAF AND placed at the right RI. FP = called
compound NOT in the MAF, or in MAF but only at wrong RI (a real annotation error in a
closed library universe). Sweep the match-score threshold to trace the tradeoff."""
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
MZ_PPM=20.0; MIN_REP=1
maf=defaultdict(dict)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    try: ri=float(r["rt"])
    except: continue
    plat=(r.get("platform") or "").strip().lower(); nm=M._norm(r.get("name") or "")
    if nm: maf[plat].setdefault(nm,ri)
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file   # injection align default
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
# per called compound: best score, whether any call at right RI, whether in MAF
total_maf=sum(len(v) for v in maf.values())
calls=[]   # (best_score, is_maf, correct_ri)
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    byc=defaultdict(lambda:[1e9,False])
    for mid,rn,sc in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.score.to_numpy()):
        sc=float(sc) if sc==sc else 1e9
        rec=byc[mid]; rec[0]=min(rec[0],sc)
        ri=gri.get(mid)
        if ri is not None:
            w=win(plat,ri); q=qsrr_by_id.get(mid)
            if abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w): rec[1]=True
    for mid,(bs,corr) in byc.items():
        calls.append((bs, mid in gri, corr))
calls.sort()
scores=np.array([c[0] for c in calls]); ismaf=np.array([c[1] for c in calls]); corr=np.array([c[2] for c in calls])
# sweep threshold: keep calls with score<=tau
print(f"total MAF compounds (recall denom): {total_maf}")
print(f"total distinct compounds CALLED (any score): {len(calls)}  of which in MAF: {ismaf.sum()}  correct-RI: {corr.sum()}\n")
print(f"{'score<=':>8}{'called':>8}{'TP':>6}{'recall':>8}{'precision':>10}{'F1':>7}")
taus=np.quantile(scores,[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
prev=None; pts=[]
for tau in taus:
    keep=scores<=tau
    called=keep.sum(); tp=(keep&ismaf&corr).sum()
    R=tp/total_maf; P=tp/called if called else 0; F1=2*P*R/(P+R) if P+R else 0
    pts.append((R,P))
    print(f"{tau:>8.3f}{called:>8}{tp:>6}{R:>8.3f}{P:>10.3f}{F1:>7.3f}")
# PR-AUC (trapezoid over recall)
pts=sorted(set(pts))
R=np.array([p[0] for p in pts]); P=np.array([p[1] for p in pts])
auc=np.trapz(P[np.argsort(R)],np.sort(R))
print(f"\nclosed-world PR-AUC (trapezoid): {auc:.3f}")
# operating point (all calls)
called=len(calls); tp=(ismaf&corr).sum(); R=tp/total_maf; P=tp/called; F1=2*P*R/(P+R)
print(f"operating point (all calls): recall {R:.3f}  precision {P:.3f}  F1 {F1:.3f}  (called={called}, TP={tp})")
print(f"  vs identity-restricted precision (old): 0.988  <- only scored calls that hit a MAF compound")

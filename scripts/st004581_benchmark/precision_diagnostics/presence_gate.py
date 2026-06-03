"""Presence gate: cut the 59%->26% over-calling using GT-free signals the matcher
ignores — isotope (M+1) confirmation, reproducibility (n_files), intensity. Report
precision/recall under each gate vs the ungated baseline. Goal: approach Metabolon's
26%-present discipline (higher precision) at acceptable recall cost."""
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
# isotope-confirm index per platform from consensus (M+1 co-eluting)
CI={}
for plat,g in cons.groupby("platform"):
    o=np.argsort(g.mz.to_numpy()); CI[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o])
def iso_ok(plat,mz,ri):
    cm,cr=CI[plat]; m1=mz+ISO; t=m1*MZ_PPM*1e-6; lo=np.searchsorted(cm,m1-t); hi=np.searchsorted(cm,m1+t)
    return hi>lo and bool(np.any(np.abs(cr[lo:hi]-ri)<=ri_tol[plat]))
# per called compound
rec=defaultdict(lambda:[0.0,0,False,False,False])  # intensity,nfiles,iso,is_maf,correct
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    for mid,rn,inten,nf,mz in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.intensity.to_numpy(),g.n_files.to_numpy(),g.mz.to_numpy()):
        r=rec[(plat,mid)]; r[0]=max(r[0],float(inten)); r[1]=max(r[1],int(nf))
        if iso_ok(plat,float(mz),float(rn)): r[2]=True
        r[3]=mid in gri
        ri=gri.get(mid)
        if ri is not None:
            w=win(plat,ri); q=qsrr_by_id.get(mid)
            if abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w): r[4]=True
arr=list(rec.values())
inten=np.array([r[0] for r in arr]); nf=np.array([r[1] for r in arr]); iso=np.array([r[2] for r in arr])
tp=np.array([r[3] and r[4] for r in arr])
def report(mask,label):
    called=mask.sum(); t=(mask&tp).sum()
    P=t/called if called else 0; R=t/total_maf; F1=2*P*R/(P+R) if P+R else 0
    print(f"  {label:<42} called {called:>4}  TP {t:>3}  precision {P:.3f}  recall {R:.3f}  F1 {F1:.3f}")
allm=np.ones(len(arr),bool)
print(f"called {len(arr)}  TP {tp.sum()}  MAF {total_maf}   (Metabolon present fraction = 26%)\n")
report(allm,"ungated baseline")
report(iso,"isotope-confirmed (M+1 co-elutes)")
report(nf>=20,"n_files >= 20")
report(inten>=np.median(inten),"intensity >= median")
report(iso&(nf>=20),"isotope + n_files>=20")
report(iso&(nf>=20)&(inten>=np.percentile(inten,40)),"isotope + n_files>=20 + intensity>=p40")
report(iso&(nf>=30)&(inten>=np.median(inten)),"isotope + n_files>=30 + intensity>=median")

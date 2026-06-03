"""Does INTENSITY / reproducibility discriminate present (MAF) from absent (non-MAF)
calls? If yes, it's the presence signal the m/z+RI score ignores. Compare closed-world
PR when ranking calls by match-score (current) vs by intensity vs by n_files."""
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
total_maf=sum(len(v) for v in maf.values())
# per called compound: best score, max intensity, max n_files, label
rec=defaultdict(lambda:[1e9,0.0,0,False,False])  # score,intensity,nfiles,is_maf,correct
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    for mid,rn,sc,inten,nf in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.score.to_numpy(),g.intensity.to_numpy(),g.n_files.to_numpy()):
        k=(plat,mid); r=rec[k]
        sc=float(sc) if sc==sc else 1e9
        r[0]=min(r[0],sc); r[1]=max(r[1],float(inten)); r[2]=max(r[2],int(nf)); r[3]=mid in gri
        ri=gri.get(mid)
        if ri is not None:
            w=win(plat,ri); q=qsrr_by_id.get(mid)
            if abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w): r[4]=True
arr=list(rec.values())
score=np.array([r[0] for r in arr]); inten=np.array([r[1] for r in arr]); nfile=np.array([r[2] for r in arr])
ismaf=np.array([r[3] for r in arr]); corr=np.array([r[4] for r in arr]); tp=ismaf&corr
# discrimination: AUC of each signal for separating TP from rest
def auc(sig,pos,higher_better=True):
    s=sig if higher_better else -sig
    order=np.argsort(s); rank=np.empty(len(s)); rank[order]=np.arange(1,len(s)+1)
    n1=pos.sum(); n0=len(s)-n1
    if n1==0 or n0==0: return float("nan")
    return (rank[pos].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"called compounds {len(arr)}  TP(present+right-RI) {tp.sum()}  total MAF {total_maf}\n")
print("Discrimination of TP vs the rest (AUC, higher=better separation):")
print(f"  match-score (lower better): {auc(score,tp,higher_better=False):.3f}")
print(f"  log10 intensity           : {auc(np.log10(inten+1),tp):.3f}")
print(f"  n_files (reproducibility) : {auc(nfile.astype(float),tp):.3f}")
print(f"  intensity median: TP {np.median(inten[tp]):.2e}  FP {np.median(inten[~tp]):.2e}")
print(f"  n_files   median: TP {np.median(nfile[tp]):.0f}    FP {np.median(nfile[~tp]):.0f}\n")
def curve(key,higher_better,label):
    o=np.argsort(-key if higher_better else key)
    t=tp[o]; cum_tp=np.cumsum(t); called=np.arange(1,len(o)+1)
    print(f"  {label}: PR at recall milestones")
    for target in [0.1,0.2,0.3,0.4,0.5,0.6,0.67]:
        need=int(target*total_maf); idx=np.searchsorted(cum_tp,need)
        if idx>=len(o): continue
        P=cum_tp[idx]/called[idx]; R=cum_tp[idx]/total_maf
        print(f"    recall {R:.2f}  precision {P:.3f}  (top {called[idx]} calls)")
print("Ranking calls by different signals -> closed-world precision at each recall:")
curve(score,False,"by MATCH-SCORE (current)")
curve(inten,True,"by INTENSITY")
curve(nfile.astype(float)+np.log10(inten+1)/20,True,"by N_FILES (+tiny intensity tiebreak)")

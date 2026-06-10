"""Does centWave PEAK QUALITY separate present (TP) from absent/noise (FP) where
intensity/isotope couldn't? Metabolon judges presence on real-peak characteristics,
not MS2. Rebuild the feature table WITH quality_overall and test."""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SMI="/tmp/dd_pubchem_smiles.csv"
CW="/mnt/volume-hel1-1/data/processed/features_centwave.tsv"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZ_PPM=20.0; MIN_REP=1
maf=defaultdict(dict)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    try: ri=float(r["rt"])
    except: continue
    plat=(r.get("platform") or "").strip().lower(); nm=M._norm(r.get("name") or "")
    if nm: maf[plat].setdefault(nm,ri)
total_maf=sum(len(v) for v in maf.values())
df=pd.read_csv(CW,sep="\t",usecols=["source_file","mz","rt","intensity","quality_overall"])
df=df[df.source_file.str.contains("COLU")].copy()
df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
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
# per-platform raw arrays (mz sorted) with ri_norm + quality, for per-call quality lookup
RAW={}
for plat,g in df.groupby("platform"):
    o=np.argsort(g.mz.to_numpy()); RAW[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o],g.quality_overall.to_numpy()[o])
def qual_at(plat,mz,ri):
    mzs,ris,qs=RAW[plat]; t=mz*MZ_PPM*1e-6; lo=np.searchsorted(mzs,mz-t); hi=np.searchsorted(mzs,mz+t)
    if hi<=lo: return 0.0
    sub_ri=ris[lo:hi]; sub_q=qs[lo:hi]; m=np.abs(sub_ri-ri)<=ri_tol[plat]
    return float(np.percentile(sub_q[m],90)) if m.any() else 0.0
rec={}
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    gri=maf.get(plat,{})
    for mid,rn,inten,nf,mz in zip(g.match_id.to_numpy(),g.ri_norm.to_numpy(),g.intensity.to_numpy(),g.n_files.to_numpy(),g.mz.to_numpy()):
        k=(plat,mid); inten=float(inten)
        if k not in rec or inten>rec[k][0]:
            corr=False; ri=gri.get(mid)
            if ri is not None:
                w=win(plat,ri); q=qsrr_by_id.get(mid)
                corr=abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w)
            rec[k]=[inten,int(nf),qual_at(plat,float(mz),float(rn)),mid in gri,corr]
arr=list(rec.values())
inten=np.array([r[0] for r in arr]); nf=np.array([r[1] for r in arr]); qual=np.array([r[2] for r in arr])
tp=np.array([r[3] and r[4] for r in arr])
def auc(sig,pos):
    order=np.argsort(sig); rank=np.empty(len(sig)); rank[order]=np.arange(1,len(sig)+1)
    n1=pos.sum(); n0=len(sig)-n1
    return (rank[pos].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"called {len(arr)}  TP {tp.sum()}  MAF {total_maf}\n")
print("Discrimination of TP vs FP (AUC):")
print(f"  log10 intensity   : {auc(np.log10(inten+1),tp):.3f}")
print(f"  n_files           : {auc(nf.astype(float),tp):.3f}")
print(f"  PEAK QUALITY (p90): {auc(qual,tp):.3f}")
print(f"  quality median: TP {np.median(qual[tp]):.2f}  FP {np.median(qual[~tp]):.2f}\n")
o=np.argsort(-qual); t=tp[o]; cum=np.cumsum(t); called=np.arange(1,len(o)+1)
print("Ranking calls by PEAK QUALITY -> closed-world precision at recall milestones:")
for target in [0.1,0.2,0.3,0.4,0.5,0.6,0.67]:
    need=int(target*total_maf); idx=np.searchsorted(cum,need)
    if idx>=len(o): continue
    print(f"  recall {cum[idx]/total_maf:.2f}  precision {cum[idx]/called[idx]:.3f}  (top {called[idx]} calls)")

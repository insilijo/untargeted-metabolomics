"""What ARE the FP calls? For each non-MAF (or wrong-RI) called compound, is there a
MAF compound at the SAME m/z? -> shadow of a present compound (substitution) vs genuine
non-library signal. Assumes MAF is truth."""
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
maf=defaultdict(list)   # plat -> [(mz, ri, name_id)]
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    try: mz=float(r["mz"]); ri=float(r["rt"])
    except: continue
    plat=(r.get("platform") or "").strip().lower(); nm=M._norm(r.get("name") or "")
    if nm: maf[plat].append((mz,ri,nm))
maf_ids={p:set(x[2] for x in v) for p,v in maf.items()}
maf_mz={p:np.sort([x[0] for x in v]) for p,v in maf.items()}
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
# classify each FP feature-call (call whose id not in MAF OR at wrong RI)
buckets=defaultdict(int)
for plat,g in ann[ann.match_id.astype(bool)].groupby("platform"):
    ids=maf_ids.get(plat,set()); mmz=maf_mz.get(plat,np.array([]))
    mlist=maf.get(plat,[])
    mafmz_arr=np.array([x[0] for x in mlist]); mafri_arr=np.array([x[1] for x in mlist]); 
    o=np.argsort(mafmz_arr); mafmz_arr=mafmz_arr[o]; mafri_arr=mafri_arr[o]
    for mid,fmz,rn in zip(g.match_id.to_numpy(),g.mz.to_numpy(),g.ri_norm.to_numpy()):
        # is THIS call a TP (id in MAF and at right RI)?
        is_tp=False
        if mid in ids:
            # find this id's MAF ri
            ri=next((x[1] for x in mlist if x[2]==mid),None)
            if ri is not None:
                w=win(plat,ri); q=qsrr_by_id.get(mid)
                if abs(rn-ri)<=w or (q is not None and abs(rn-q)<=w): is_tp=True
        if is_tp: buckets["TP (present, right RI)"]+=1; continue
        # FP: is there a MAF compound at this feature's m/z?
        t=fmz*MZ_PPM*1e-6; lo=np.searchsorted(mafmz_arr,fmz-t); hi=np.searchsorted(mafmz_arr,fmz+t)
        if hi<=lo: buckets["FP: no MAF cmpd at this m/z (genuine non-library signal)"]+=1
        else:
            # MAF compound shares m/z; is one within RI window (shadowing a present peak)?
            near=any(abs(mafri_arr[k]-rn)<=win(plat,mafri_arr[k]) for k in range(lo,hi))
            if near: buckets["FP: MAF cmpd at same m/z AND RI (mislabeled present peak)"]+=1
            else: buckets["FP: MAF cmpd at same m/z, diff RI (isobaric decoy elsewhere)"]+=1
tot=sum(buckets.values()); fp=tot-buckets["TP (present, right RI)"]
print(f"feature-level calls: {tot}\n")
for k in ["TP (present, right RI)",
          "FP: MAF cmpd at same m/z AND RI (mislabeled present peak)",
          "FP: MAF cmpd at same m/z, diff RI (isobaric decoy elsewhere)",
          "FP: no MAF cmpd at this m/z (genuine non-library signal)"]:
    print(f"  {buckets[k]:>5} ({buckets[k]/tot*100:>3.0f}%)  {k}")
print(f"\n  total FP feature-calls: {fp} ({fp/tot*100:.0f}%)")

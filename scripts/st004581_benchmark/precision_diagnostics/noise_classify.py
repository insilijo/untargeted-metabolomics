"""Classify the 71% 'noise' FPs (no M+1, sparse) for an exploitable PATTERN:
 - satellite/fragment of a STRONGER co-eluting peak (isotope/adduct/neutral-loss)
 - intensity rank, mass defect, isolation, m/z & RT location."""
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
    o=np.argsort(g.mz.to_numpy()); CI[plat]=(g.mz.to_numpy()[o],g.ri_norm.to_numpy()[o],g.intensity.to_numpy()[o])
def iso_ok(plat,mz,ri):
    cm,cr,ci=CI[plat]; m1=mz+ISO; t=m1*MZ_PPM*1e-6; lo=np.searchsorted(cm,m1-t); hi=np.searchsorted(cm,m1+t)
    return hi>lo and bool(np.any(np.abs(cr[lo:hi]-ri)<=ri_tol[plat]))
# stronger co-eluting feature at a chemical offset?
LOSS=[18.0106,17.0265,43.9898,27.9949,30.0106,46.0055,2*ISO,ISO,36.9839,21.9819,42.0106,162.0528,79.9568]
def satellite(plat,mz,ri,inten):
    cm,cr,ci=CI[plat]
    for o in LOSS:
        for cand in (mz-o,mz+o):
            t=cand*MZ_PPM*1e-6; lo=np.searchsorted(cm,cand-t); hi=np.searchsorted(cm,cand+t)
            if hi>lo:
                m=np.abs(cr[lo:hi]-ri)<=ri_tol[plat]
                if m.any() and ci[lo:hi][m].max()>inten*1.5: return True
    return False
def ncoelute(plat,ri):
    cm,cr,ci=CI[plat]; return int(np.sum(np.abs(cr-ri)<=ri_tol[plat]))
# best feature per call
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
            rec[k]=[inten,int(nf),float(mz),float(rn),mid in gri,corr,plat]
# global intensity percentile reference (all calls)
allint=np.array([v[0] for v in rec.values()])
def pct(x): return float((allint<x).mean())
noise=[]; tp_int=[]
for v in rec.values():
    inten,nf,mz,ri,ismaf,corr,plat=v
    if ismaf and corr: tp_int.append(inten); continue
    if iso_ok(plat,mz,ri) or nf>=20: continue   # only the NOISE bucket
    noise.append(v)
print(f"NOISE bucket: {len(noise)} calls\n")
sat=sum(1 for v in noise if satellite(v[6],v[2],v[3],v[0]))
iso_lo=np.array([pct(v[0]) for v in noise])
nco=np.array([ncoelute(v[6],v[3]) for v in noise])
print(f"  satellite/fragment of a STRONGER co-eluting peak: {sat} ({sat/len(noise)*100:.0f}%)")
print(f"  intensity percentile (vs all calls): median {np.median(iso_lo)*100:.0f}th  (<25th: {int((iso_lo<0.25).sum())} = {(iso_lo<0.25).mean()*100:.0f}%)")
print(f"  TP median intensity {np.median(tp_int):.2e}  vs noise median {np.median([v[0] for v in noise]):.2e}")
print(f"  co-eluting features at same RT: median {np.median(nco):.0f}")
# combined removable: satellite OR bottom-quartile intensity
rem=sum(1 for v,p in zip(noise,iso_lo) if satellite(v[6],v[2],v[3],v[0]) or p<0.25)
print(f"\n  noise that are satellite OR bottom-25%-intensity: {rem} ({rem/len(noise)*100:.0f}% of noise)")

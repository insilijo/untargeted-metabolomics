"""Open-world test: sparse-anchor propagation against a BIG candidate library (172k ZINC universe)
vs the DD (1718), same feature-table matching. Does recall hold + precision survive the 9x isobaric
competition? arg: 'dd' or 'uni'."""
import sys, csv, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
csv.field_size_limit(10**7)
LIB=sys.argv[1] if len(sys.argv)>1 else "dd"
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
UNI="/mnt/volume-hel1-1/data/processed/compound_universe_broad.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
KIT="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
PLAT="lc/ms neg"; MZ_PPM=10.0; TIGHT=8.0; MINREP=2; FDR=0.01; ISOBAR_PPM=10.0; PROTON=1.0072765; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(ik14(r.get("inchikey")))
maf={x for x in maf if len(x)>=14}; nmaf=len(maf)
# feature table neg
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
fn=df[df.platform==PLAT]; fmz=fn.mz.to_numpy(); frt=fn.rt.to_numpy(); fin=fn.intensity.to_numpy()
o=np.argsort(fmz); fmz,frt,fin=fmz[o],frt[o],fin[o]
def has_feat(mz,rt):  # reproducible-ish: a feature at m/z within ppm + rt within window
    t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
    if hi<=lo: return None
    m=np.abs(frt[lo:hi]-rt)<=TIGHT
    if not m.any(): return None
    sub=np.arange(lo,hi)[m]; j=sub[np.argmax(fin[sub])]; return frt[j]
def feat_any(mz):
    t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t); return hi>lo
# ladder from all-platform anchors+DD densify (predict RT for compounds w/o structure too)
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
cal={p:list(v) for p,v in anchors.items()}
_,_,_,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
ag=defaultdict(list)
for sec,ri in pp[PLAT]: ag[round(ri,1)].append(sec)
xs=np.array(sorted(ag)); ys=np.array([np.median(ag[x]) for x in xs]); u,ui=np.unique(xs,return_index=True)
inv=PchipInterpolator(u,ys[ui],extrapolate=True)
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# candidate set
comp=[]  # (ik, mz, desc)
if LIB=="dd":
    for c in lib0.get(PLAT,[]):
        ik=ik14(c.get("ik14") or ""); s=smi.get(ik); d=_descriptors(s) if s else None
        comp.append((ik,c["mz"],np.array(d) if d is not None else None))
else:
    # universe: m/z from SMILES; KEEP only those with a feature at their m/z (detectable) -> prunes 170k
    seen=set()
    for r in csv.DictReader(open(UNI)):
        s=r.get("smiles"); ik=ik14(r.get("inchikey"))
        if not s or ik in seen: continue
        mol=Chem.MolFromSmiles(s)
        if mol is None: continue
        try: mw=Descriptors.ExactMolWt(mol)
        except Exception: continue
        if not (50<mw<1500): continue
        mz=mw-PROTON
        if not feat_any(mz): continue
        seen.add(ik); d=_descriptors(s)
        comp.append((ik,mz,np.array(d) if d is not None else None))
    # ensure MAF compounds present (add DD MAF not already in)
    have=set(c[0] for c in comp)
    for c in lib0.get(PLAT,[]):
        ik=ik14(c.get("ik14") or "")
        if ik in maf and ik not in have:
            s=smi.get(ik); d=_descriptors(s) if s else None
            comp.append((ik,c["mz"],np.array(d) if d is not None else None)); have.add(ik)
n=len(comp); MZc=np.array([c[1] for c in comp]); cik=np.array([c[0] for c in comp])
inmaf=np.array([c[0] in maf for c in comp])
print(f"LIB={LIB}  candidates={n}  MAF in candidates={inmaf.sum()}/{nmaf}  (feature-detectable subset)",flush=True)
withd=[k for k in range(n) if comp[k][2] is not None]; DX=np.array([comp[k][2] for k in withd])
# kit seed
kx=[]; ky=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"])
    except: sec=None
    if d is not None and sec: kx.append(np.array(d)); ky.append(sec)
kx=np.array(kx); ky=np.array(ky)
# expansion (single chain, ~6 rounds): predict -> match feature table -> admit -> retrain
admitted=np.zeros(n,bool); apex=np.full(n,np.nan)
Xtr=list(kx); ytr=list(ky)
for rd in range(6):
    mdl=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=4).fit(np.array(Xtr),np.array(ytr))
    pr=mdl.predict(DX); new=[]
    for ii,k in enumerate(withd):
        if admitted[k]: continue
        ap=has_feat(MZc[k],pr[ii])
        if ap is not None and abs(ap-pr[ii])<=TIGHT: new.append((k,ap))
    # isobar one-to-one per round (closest pred)
    new.sort(key=lambda x:abs(x[1]-pr[withd.index(x[0])]) if False else 0)
    if not new: break
    for k,ap in new: admitted[k]=True; apex[k]=ap; Xtr.append(comp[k][2]); ytr.append(ap)
# cluster admitted by m/z+apex; recall + precision
idx=[k for k in np.where(admitted)[0] if not np.isnan(apex[k])]
idx.sort(key=lambda k:-(MZc[k]))  # arbitrary stable
cl=[]
order=sorted(idx,key=lambda k:apex[k])
for k in order:
    placed=False
    for c in cl:
        if abs(MZc[k]-MZc[c[0]])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(apex[k]-apex[c[0]])<=TIGHT: c.append(k); placed=True; break
    if not placed: cl.append([k])
tp=sum(any(inmaf[k] for k in c) for c in cl); cov=len(set(cik[k] for c in cl for k in c if inmaf[k]))
sizes=[len(c) for c in cl]
print(f"  admitted {int(admitted.sum())}  clusters {len(cl)}")
print(f"  RECALL {cov/nmaf:.3f}  closed-world PRECISION {tp/max(len(cl),1):.3f}")
print(f"  cluster size: median {int(np.median(sizes))} mean {np.mean(sizes):.2f} max {max(sizes)}  (specificity: bigger=more ambiguous)")

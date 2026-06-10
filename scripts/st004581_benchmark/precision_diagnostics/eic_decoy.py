"""Proof: with EIC-precise RT + tight window, does the RI-shuffle decoy collapse (= RI
becomes discriminating = precision lever real)? For each neg compound, is there an EIC peak
at its m/z within +-W of its EXPECTED RT (DD RI->sec)? Target vs RI-shuffled decoy."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:6]
MZ_PPM=20.0; PLAT="lc/ms neg"; FLOOR=5e4; MINREP=2
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
cal={p:list(v) for p,v in anchors.items()}
for plat,ent in lib0.items():
    mzs=np.array(sorted(c["mz"] for c in ent))
    for c in ent:
        t=c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(mzs,c["mz"]+t)-np.searchsorted(mzs,c["mz"]-t))==1: cal.setdefault(plat,[]).append((c["mz"],c["ri"]))
ladders,pooled,cov,pp=M.build_batch_ladders(df,cal,MZ_PPM,1,True)
slope=M.ri_per_sec(pp)
# inverse pooled neg ladder RI->sec
agg=defaultdict(list)
for sec,ri in pp[PLAT]: agg[round(ri,1)].append(sec)
xs=np.array(sorted(agg)); ys=np.array([np.median(agg[x]) for x in xs]); inv=PchipInterpolator(xs,ys,extrapolate=True)
# neg compounds (primary ions), unique by name
comp={}
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if nm and nm not in comp: comp[nm]=[c["mz"],c["ri"]]
names=list(comp); MZc=np.array([comp[n][0] for n in names]); RIc=np.array([comp[n][1] for n in names])
esec=np.array([float(inv(r)) for r in RIc])
rng=np.random.RandomState(0); perm=rng.permutation(len(names)); esec_dec=esec[perm]  # decoy = shuffled expected RT
# EIC presence per compound per injection (peak within +-W of esec)
WS=[30,10,5,3]
present_t={w:np.zeros(len(names)) for w in WS}; present_d={w:np.zeros(len(names)) for w in WS}
tol=MZc*MZ_PPM*1e-6
print(f"neg compounds {len(names)} (in MAF {sum(n in maf for n in names)})  injections {len(MZML)}", flush=True)
for mp in MZML:
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(smz): rows.append(np.zeros(len(names))); continue
        lo=np.searchsorted(smz,MZc-tol); hi=np.searchsorted(smz,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(names))]))
    rt=np.array(rts); mat=np.array(rows)
    for k in range(len(names)):
        col=mat[:,k]
        for w in WS:
            if col[np.abs(rt-esec[k])<=w].max(initial=0)>FLOOR: present_t[w][k]+=1
            if col[np.abs(rt-esec_dec[k])<=w].max(initial=0)>FLOOR: present_d[w][k]+=1
    print(f"  done {Path(mp).name.split('_')[-1]}", flush=True)
inmaf=np.array([n in maf for n in names])
print(f"\n{'window':>7}{'target':>8}{'decoy':>7}{'tgt/decoy':>10}{'tgtTP':>7}{'precision':>11}")
for w in WS:
    tcall=present_t[w]>=MINREP; dcall=present_d[w]>=MINREP
    tc=tcall.sum(); dc=dcall.sum(); tp=(tcall&inmaf).sum()
    print(f"{w:>6}s{tc:>8}{dc:>7}{(tc/dc if dc else 0):>10.2f}{tp:>7}{(tp/tc if tc else 0):>11.3f}")
print(f"\n(decoy collapse + precision rise as window tightens = RI becomes discriminating with EIC RT)")

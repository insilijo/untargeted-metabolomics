"""Is the ~11s RT imprecision fixable peak-jitter or intrinsic? Compare cross-injection RT
spread of centWave apex (feature table) vs raw-EIC apex (max-intensity scan) for anchors."""
import csv, glob, numpy as np, pandas as pd, pymzml
FEAT="/tmp/feat_colu.parquet"; ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
PPM=20.0
anc=[r for r in csv.DictReader(open(ANC))][:15]
mzs=np.array([float(r["mz"]) for r in anc]); ort=np.array([float(r["observed_rt_sec"]) for r in anc])
df=pd.read_parquet(FEAT)
# centWave apex per (injection, anchor)
cw=np.full((len(MZML),len(anc)),np.nan)
for ii,mp in enumerate(MZML):
    sf=mp.split("/")[-1]; g=df[df.source_file==sf]
    m=g.mz.to_numpy(); rt=g.rt.to_numpy(); it=g.intensity.to_numpy(); o=np.argsort(m); m,rt,it=m[o],rt[o],it[o]
    for k in range(len(anc)):
        t=mzs[k]*PPM*1e-6; lo=np.searchsorted(m,mzs[k]-t); hi=np.searchsorted(m,mzs[k]+t)
        if hi>lo:
            sub=np.abs(rt[lo:hi]-ort[k])<=20
            if sub.any(): j=np.where(sub)[0][np.argmax(it[lo:hi][sub])]; cw[ii,k]=rt[lo:hi][j]
# EIC apex per (injection, anchor)
eic=np.full((len(MZML),len(anc)),np.nan)
tol=mzs*PPM*1e-6
for ii,mp in enumerate(MZML):
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(smz): rows.append(np.zeros(len(mzs))); continue
        lo=np.searchsorted(smz,mzs-tol); hi=np.searchsorted(smz,mzs+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(mzs))]))
    rt=np.array(rts); mat=np.array(rows)
    for k in range(len(anc)):
        sub=np.abs(rt-ort[k])<=20
        if sub.any() and mat[sub,k].max()>0: eic[ii,k]=rt[sub][np.argmax(mat[sub,k])]
def spread(col):
    v=col[~np.isnan(col)]; return np.percentile(v,90)-np.percentile(v,10) if len(v)>=5 else np.nan
cws=np.array([spread(cw[:,k]) for k in range(len(anc))])
eis=np.array([spread(eic[:,k]) for k in range(len(anc))])
ok=~np.isnan(cws)&~np.isnan(eis)
print(f"anchors {ok.sum()}  injections {len(MZML)}\n")
print(f"cross-injection RT spread (P90-P10), median:")
print(f"  centWave apex : {np.median(cws[ok]):.1f}s")
print(f"  raw-EIC apex  : {np.median(eis[ok]):.1f}s")
print(f"\nper-anchor (centWave / EIC):")
for k in range(len(anc)):
    if ok[k]: print(f"  {anc[k]['name'][:26]:<26} {cws[k]:5.1f}s / {eis[k]:5.1f}s")

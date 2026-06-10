"""Verify the lever: does Python RT-warp alignment (retcor-style) tighten observed RT?
Warp each neg injection to a reference using shared strong peaks (m/z-matched), then
measure known anchors' cross-injection RT spread RAW vs WARPED. If it collapses
(~12s->~few s), sharper observed RI is achievable in Python and justifies the build."""
import csv, numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
PPM=20.0
df=pd.read_parquet(FEAT)
df=df[df.source_file.str.startswith("Method3") & df.source_file.str.contains("COLU")].copy()
injs=sorted(df.source_file.unique())[:12]
# per-injection sorted (mz, rt, intensity)
P={}
for s in injs:
    g=df[df.source_file==s]; o=np.argsort(g.mz.to_numpy())
    P[s]=(g.mz.to_numpy()[o], g.rt.to_numpy()[o], g.intensity.to_numpy()[o])
ref=max(injs, key=lambda s: len(P[s][0]))   # reference = most features
rmz,rrt,rint=P[ref]
# strong reference features (top by intensity) for anchoring the warp
ord_r=np.argsort(-rint)[:4000]; rmz_s=rmz[ord_r]; rrt_s=rrt[ord_r]
so=np.argsort(rmz_s); rmz_s=rmz_s[so]; rrt_s=rrt_s[so]
def warp_fit(s):
    mz,rt,it=P[s]; pairs=[]
    strong=np.argsort(-it)[:4000]
    for k in strong:
        m=mz[k]; t=m*PPM*1e-6; lo=np.searchsorted(rmz_s,m-t); hi=np.searchsorted(rmz_s,m+t)
        if hi>lo:
            j=lo+np.argmin(np.abs(rrt_s[lo:hi]-rt[k]))
            if abs(rrt_s[j]-rt[k])<=25: pairs.append((rt[k],rrt_s[j]))   # (rt_inj, rt_ref)
    if len(pairs)<20: return None
    pairs=sorted(set(pairs)); x=np.array([p[0] for p in pairs]); y=np.array([p[1] for p in pairs])
    # bin to monotone median to denoise
    xb=np.linspace(x.min(),x.max(),60); yb=[]
    xs=[]
    for a,b in zip(xb[:-1],xb[1:]):
        m=(x>=a)&(x<b)
        if m.sum()>=3: xs.append((a+b)/2); yb.append(np.median(y[m]))
    if len(xs)<5: return None
    xs=np.array(xs); yb=np.maximum.accumulate(np.array(yb))  # enforce monotone
    return PchipInterpolator(xs,yb,extrapolate=True)
warps={s:warp_fit(s) for s in injs}
print(f"injections {len(injs)}  warps fit {sum(w is not None for w in warps.values())}  ref={ref.split('_')[-1]}\n")
anc=[r for r in csv.DictReader(open(ANC))]
def detect_rt(s,mz):
    m,rt,it=P[s]; t=mz*PPM*1e-6; lo=np.searchsorted(m,mz-t); hi=np.searchsorted(m,mz+t)
    if hi<=lo: return None
    return float(rt[lo:hi][np.argmax(it[lo:hi])])  # apex
raw_spreads=[]; warp_spreads=[]
for r in anc[:25]:
    mz=float(r["mz"]); raw=[]; wrp=[]
    for s in injs:
        rt=detect_rt(s,mz)
        if rt is None: continue
        raw.append(rt)
        w=warps[s]; wrp.append(float(w(rt)) if w is not None else rt)
    if len(raw)>=6:
        raw_spreads.append(np.percentile(raw,90)-np.percentile(raw,10))
        warp_spreads.append(np.percentile(wrp,90)-np.percentile(wrp,10))
raw_spreads=np.array(raw_spreads); warp_spreads=np.array(warp_spreads)
print(f"anchors measured: {len(raw_spreads)}")
print(f"cross-injection RT spread (P90-P10), median over anchors:")
print(f"  RAW    : {np.median(raw_spreads):.1f}s")
print(f"  WARPED : {np.median(warp_spreads):.1f}s")
print(f"  improvement: {(1-np.median(warp_spreads)/max(np.median(raw_spreads),1e-9))*100:.0f}%")

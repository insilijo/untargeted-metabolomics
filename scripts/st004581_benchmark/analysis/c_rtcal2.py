"""Build a DENSE RI->sec calibration in-env (from ALL mass-unique detected GT, like
SQuID's rt_calibration) vs the SPARSE kit-panel ladder my matcher used. Translate
each GT compound's MAF RI->RT and check features_centwave.tsv for a peak. Shows
whether calibration DENSITY was the limiter behind the 'not_detected' compounds.
"""
import csv
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
NEG=[-1.007276,44.998201,34.969402,-19.017841]; POS=[1.007276,22.989218,18.033823,38.963158,-17.00274]
MODE={'lc/ms pos early':POS,'lc/ms pos late':POS,'lc/ms neg':NEG,'lc/ms polar':POS}
RT_WIN=30.0; ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
feat=pd.read_csv('/mnt/volume-hel1-1/data/processed/features_centwave.tsv', sep='\t', usecols=['source_file','mz','rt'])
feat=feat[feat.source_file.str.contains('COLU', na=False)].copy()
feat['platform']=feat.source_file.str.split('_').str[0].map(PM)
P={}
for p,g in feat.groupby('platform'):
    o=np.argsort(g.mz.values); P[p]=(g.mz.values[o], g.rt.values[o])
MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
def domrt(plat, mz):
    mzs,rts=P[plat]; t=mz*20e-6; lo=np.searchsorted(mzs,mz-t); hi=np.searchsorted(mzs,mz+t)
    if hi<=lo: return None
    r=rts[lo:hi]
    # dominant cluster: mode-ish via median of densest 20s window
    r=np.sort(r); best=r[0]; bestn=0
    for x in r:
        n=((r>=x)&(r<=x+20)).sum()
        if n>bestn: bestn=n; best=x
    return float(np.median(r[(r>=best)&(r<=best+20)]))
def detected(plat,mz,exp):
    mzs,rts=P[plat]; t=mz*15e-6; lo=np.searchsorted(mzs,mz-t); hi=np.searchsorted(mzs,mz+t)
    return hi>lo and (np.abs(rts[lo:hi]-exp)<=RT_WIN).any()
def build_ladder(pairs):
    agg=defaultdict(list)
    for ri,s in pairs: agg[round(ri,1)].append(s)
    xs=sorted(agg)
    if len(xs)<3: return None
    return PchipInterpolator(np.array(xs,float),np.array([np.median(agg[x]) for x in xs],float),extrapolate=True)

for label, dense in [("SPARSE (kit panel ~30)",False),("DENSE (all mass-unique detected)",True)]:
    tot=defaultdict(int)
    for plat in P:
        gtp=[r for r in MAF if r['platform']==plat and ff(r['mz']) and ff(r['rt'])]
        cmz=np.array(sorted(ff(r['mz']) for r in gtp))
        uniq=lambda mz:(np.searchsorted(cmz,mz*1.00002)-np.searchsorted(cmz,mz*0.99998))==1
        # calibration anchor pairs
        if dense:
            pairs=[(ff(r['rt']),domrt(plat,ff(r['mz']))) for r in gtp if uniq(ff(r['mz'])) and domrt(plat,ff(r['mz'])) is not None]
        else:
            # mimic kit panel: even-RI subsample ~30 of the mass-unique-detected
            mu=[(ff(r['rt']),domrt(plat,ff(r['mz']))) for r in gtp if uniq(ff(r['mz'])) and domrt(plat,ff(r['mz'])) is not None]
            mu.sort();
            pairs=[mu[i] for i in np.linspace(0,len(mu)-1,min(30,len(mu))).astype(int)] if mu else []
        lad=build_ladder(pairs)
        if lad is None: continue
        ads=MODE[plat]
        for r in gtp:
            mz=ff(r['mz']); ri=ff(r['rt']); exp=float(lad(ri)); neutral=mz-ads[0]
            tot['gt']+=1
            if detected(plat,mz,exp) or any(detected(plat,neutral+o,exp) for o in ads[1:]): tot['det']+=1
    print(f"{label:<36} n_anchor~{len(pairs)}/plat  detected {tot['det']}/{tot['gt']} = {tot['det']/tot['gt']*100:.0f}%")

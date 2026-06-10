"""Predict predRT from THESE data: train a QSRR (structure->RT) on confident
detections, CV-predict held-out GT compounds, compare RT-prediction error to the
RI-ladder. Does structure-RT beat the 1-D ladder, esp. on the ladder's failures?
"""
import csv, sys
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,'/root/SQuID-INC')
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
feat=pd.read_parquet('/tmp/feat_colu.parquet'); feat['platform']=feat.source_file.str.split('_').str[0].map(PM)
MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
for plat in ['lc/ms pos early','lc/ms pos late','lc/ms neg','lc/ms polar']:
    fs=feat[feat.platform==plat]; fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy()
    o=np.argsort(fmz); fmz,frt,fint=fmz[o],frt[o],fint[o]
    def domrt(mz):
        t=mz*20e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it=frt[lo:hi],fint[lo:hi]; c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20
        return float(np.median(rt[m]))
    rows=[]
    gtp=[r for r in MAF if r['platform']==plat and ff(r['mz']) and ff(r['rt']) and (r.get('smiles') or '').strip()]
    cmz=np.array(sorted(ff(r['mz']) for r in gtp))
    for r in gtp:
        mz=ff(r['mz']); ri=ff(r['rt']); rt=domrt(mz)
        if rt is None: continue
        d=_descriptors(r['smiles'])
        if d is None: continue
        uniq=(np.searchsorted(cmz,mz*1.00002)-np.searchsorted(cmz,mz*0.99998))==1
        rows.append((ri,rt,d,uniq))
    if len(rows)<40: print(f"{plat:<16} few ({len(rows)})"); continue
    ri=np.array([x[0] for x in rows]); y=np.array([x[1] for x in rows])
    X=np.array([x[2] for x in rows]); uq=np.array([x[3] for x in rows])
    # 5-fold CV
    idx=np.arange(len(rows)); rng=np.random.RandomState(0); rng.shuffle(idx)
    folds=np.array_split(idx,5)
    eq=np.full(len(rows),np.nan); el=np.full(len(rows),np.nan)
    for f in range(5):
        te=folds[f]; tr=np.concatenate([folds[k] for k in range(5) if k!=f])
        # QSRR
        m=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.05,min_samples_leaf=8)
        m.fit(X[tr],y[tr]); eq[te]=np.abs(m.predict(X[te])-y[te])
        # ladder from train (RI->sec), predict test
        agg=defaultdict(list)
        for r_,s_ in zip(ri[tr],y[tr]): agg[round(r_,1)].append(s_)
        xs=sorted(agg); ys=[np.median(agg[x]) for x in xs]
        if len(xs)<3: continue
        lad=PchipInterpolator(np.array(xs,float),np.array(ys,float),extrapolate=True)
        el[te]=np.abs(lad(ri[te])-y[te])
    ok=~np.isnan(eq)&~np.isnan(el)
    eq,el,uq=eq[ok],el[ok],uq[ok]
    print(f"{plat:<16} n={ok.sum()}  RT-pred median|err| s: QSRR {np.median(eq):.1f}  ladder {np.median(el):.1f}  "
          f"| ladder-fail(>30s) n={int((el>30).sum())}: there QSRR median {np.median(eq[el>30]) if (el>30).any() else 0:.1f}  "
          f"QSRR<ladder on fails {int((eq[el>30]<el[el>30]).sum())}/{int((el>30).sum())}")

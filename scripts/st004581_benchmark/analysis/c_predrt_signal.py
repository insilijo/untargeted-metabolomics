"""Decisive pre-build test for the predRT idea: does a STRUCTURE-based predRT carry
incremental signal for RI BEYOND observed RT? If the ladder residual
(library_RI - ladder(observed_RT)) correlates with a structure-predRT, a 2-D
(RT, predRT) matcher can recover ri_mismatch. If residual ~ noise, it can't.
Uses the crude predict_coordinates proxy for a quick directional read.
"""
import sys, csv
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,'/root/SQuID-INC')
from squid_inc.features.predict import predict_coordinates
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
ik14=lambda s:(s or '')[:14]; ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
feat=pd.read_parquet('/tmp/feat_colu.parquet'); feat['platform']=feat.source_file.str.split('_').str[0].map(PM)
DD=list(csv.DictReader(open('/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv',encoding='utf-8-sig')))
ddmz=defaultdict(list)
for r in DD:
    m=ff(r['MASS'])
    if m: ddmz[r['PLATFORM']].append(m)
for p in ddmz: ddmz[p]=np.array(sorted(ddmz[p]))
MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
_pc={}
def predrt(smi):
    if smi in _pc: return _pc[smi]
    try: v=predict_coordinates({'smiles':smi,'inchikey':'','compound_id':''}).get('rt')
    except Exception: v=None
    _pc[smi]=v; return v

print(f"{'platform':<16}{'n':>5}{'corr(resid,predRT)':>20}{'R2_resid~predRT':>17}  (resid=libRI-ladder(RT))")
for plat in ['lc/ms pos early','lc/ms pos late','lc/ms neg','lc/ms polar']:
    fs=feat[feat.platform==plat]; fmz=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy()
    o=np.argsort(fmz); fmz,frt,fint=fmz[o],frt[o],fint[o]
    dm=ddmz.get(plat,np.array([]))
    cuniq=lambda mz:(np.searchsorted(dm,mz+mz*20e-6)-np.searchsorted(dm,mz-mz*20e-6))==1 if len(dm) else False
    def domrt(mz):
        t=mz*20e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
        if hi<=lo: return None
        rt,it=frt[lo:hi],fint[lo:hi]; c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20
        return float(np.median(rt[m]))
    # ladder from mass-unique detected anchors
    anc=[]
    for r in MAF:
        if r['platform']!=plat: continue
        mz=ff(r['mz']); ri=ff(r['rt'])
        if mz and ri and cuniq(mz):
            d=domrt(mz)
            if d is not None: anc.append((d,ri))
    if len(anc)<5: print(f"{plat:<16} few anchors"); continue
    agg=defaultdict(list)
    for s,ri in anc: agg[round(s,1)].append(ri)
    xs=sorted(agg); ys=[np.median(agg[x]) for x in xs]
    lad=PchipInterpolator(np.array(xs,float),np.array(ys,float),extrapolate=True)
    resid=[]; pr=[]
    for r in MAF:
        if r['platform']!=plat: continue
        mz=ff(r['mz']); ri=ff(r['rt']); smi=(r.get('smiles') or '').strip()
        if not(mz and ri and smi): continue
        d=domrt(mz)
        if d is None: continue
        p=predrt(smi)
        if p is None: continue
        resid.append(ri-float(lad(d))); pr.append(p)
    resid=np.array(resid); pr=np.array(pr)
    if len(resid)>10 and pr.std()>0:
        c=np.corrcoef(resid,pr)[0,1]; r2=c**2
        print(f"{plat:<16}{len(resid):>5}{c:>20.3f}{r2:>17.3f}")

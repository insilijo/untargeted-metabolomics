"""Do the still-absent compounds have a real chromatographic peak SOMEWHERE in the
run (just not at my predicted RT)? Build the EIC at their m/z across the FULL RT
range per file; a peak = a run of >=3 consecutive MS1 scans >=100 intensity.
If peaks exist, the limit is RT-PREDICTION (ladder/predRT), not detection.
"""
import csv, glob
from collections import defaultdict
import numpy as np, pandas as pd
import pyopenms as oms
from scipy.interpolate import PchipInterpolator
NEG=[-1.007276,44.998201,34.969402,-19.017841]; POS=[1.007276,22.989218,18.033823,38.963158,-17.00274]
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
MODE={'lc/ms pos early':POS,'lc/ms pos late':POS,'lc/ms neg':NEG,'lc/ms polar':POS}
NOISE=100.0; RT_WIN=30.0; ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
feat=pd.read_parquet('/tmp/feat_colu.parquet'); feat['platform']=feat.source_file.str.split('_').str[0].map(PM)
MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
MZDIR='/root/SQuID-INC/data/st004581/mzml'
out=defaultdict(int); rt_err=[]
for meth,plat in PM.items():
    fs=feat[feat.platform==plat]; fmz_s=fs.mz.to_numpy(); frt=fs.rt.to_numpy(); fint=fs.intensity.to_numpy()
    o=np.argsort(fmz_s); fmz_s,frt,fint=fmz_s[o],frt[o],fint[o]
    def cons(mz):
        t=mz*20e-6; lo=np.searchsorted(fmz_s,mz-t); hi=np.searchsorted(fmz_s,mz+t)
        if hi<=lo: return None
        rt,it=frt[lo:hi],fint[lo:hi]; c=rt[np.argmax(it)]; m=np.abs(rt-c)<=20
        return float(np.median(rt[m]))
    gtp=[r for r in MAF if r['platform']==plat and ff(r['mz']) and ff(r['rt'])]
    cmz=np.array(sorted(ff(r['mz']) for r in gtp))
    anc=[(ff(r['rt']),cons(ff(r['mz']))) for r in gtp
         if (np.searchsorted(cmz,ff(r['mz'])*1.00002)-np.searchsorted(cmz,ff(r['mz'])*0.99998))==1 and cons(ff(r['mz'])) is not None]
    agg=defaultdict(list)
    for ri,s in anc: agg[round(ri,1)].append(s)
    xs=sorted(agg); ys=[np.median(agg[x]) for x in xs]
    if len(xs)<3: continue
    lad=PchipInterpolator(np.array(xs,float),np.array(ys,float),extrapolate=True)
    ads=MODE[plat]
    absent=[]
    for r in gtp:
        mz=ff(r['mz']); ersec=float(lad(ff(r['rt']))); neutral=mz-ads[0]
        ions=[mz]+[neutral+off for off in ads[1:]]
        if not any((lambda lo,hi: hi>lo and (np.abs(frt[lo:hi]-ersec)<=RT_WIN).any())(
                np.searchsorted(fmz_s,i-i*20e-6),np.searchsorted(fmz_s,i+i*20e-6)) for i in ions):
            absent.append((r['name'],mz,ersec))
    if not absent: continue
    files=sorted(glob.glob(f"{MZDIR}/{meth}_*COLU*.mzML"))[:3]
    perfile=[]
    for fpath in files:
        exp=oms.MSExperiment(); oms.MzMLFile().load(fpath,exp)
        sp=[(s.getRT(),)+s.get_peaks() for s in exp if s.getMSLevel()==1]
        sp.sort(key=lambda x:x[0]); perfile.append(sp)
    for name,mz,ersec in absent:
        best_apex=None; peak=False
        for sp in perfile:
            t=mz*15e-6; eic=[]
            for rt,mzv,iv in sp:
                a=np.searchsorted(mzv,mz-t); b=np.searchsorted(mzv,mz+t)
                eic.append((rt, float(iv[a:b].max()) if b>a else 0.0))
            ints=np.array([e[1] for e in eic]); rts=np.array([e[0] for e in eic])
            run=0
            for k,v in enumerate(ints):
                if v>=NOISE:
                    run+=1
                    if run>=3:
                        peak=True; apex=rts[k-1]
                        if best_apex is None or ints[k-1]>0: best_apex=apex
                else: run=0
        if peak:
            # confirm real: co-eluting M+1 isotope (0.2-40% of M0) at the apex
            iso=False
            for sp in perfile:
                for rt,mzv,iv in sp:
                    if abs(rt-best_apex)>5: continue
                    t=mz*15e-6; a=np.searchsorted(mzv,mz-t); b=np.searchsorted(mzv,mz+t)
                    m0=float(iv[a:b].max()) if b>a else 0.0
                    if m0<NOISE: continue
                    mi=mz+1.003355; a2=np.searchsorted(mzv,mi-mi*15e-6); b2=np.searchsorted(mzv,mi+mi*15e-6)
                    m1=float(iv[a2:b2].max()) if b2>a2 else 0.0
                    if 0.002*m0<=m1<=0.5*m0: iso=True; break
                if iso: break
            if iso: out['real_peak_isotope_confirmed']+=1
            else: out['peak_no_isotope']+=1
            rt_err.append(abs(best_apex-ersec))
        else: out['no_peak_anywhere']+=1
tot=sum(out.values()); rt_err=np.array(rt_err)
print(f"still-absent probed: {tot}")
print(f"  real peak + M+1 isotope confirmed: {out['real_peak_isotope_confirmed']} ({out['real_peak_isotope_confirmed']/tot*100:.0f}%)")
print(f"  peak but no clean isotope: {out['peak_no_isotope']} ({out['peak_no_isotope']/tot*100:.0f}%)")
print(f"  REAL chromatographic peak somewhere in run: {out['peak_somewhere']} ({out['peak_somewhere']/tot*100:.0f}%)")
print(f"  no peak anywhere at that m/z: {out['no_peak_anywhere']} ({out['no_peak_anywhere']/tot*100:.0f}%)")
if len(rt_err):
    print(f"  of peaks-found: |apex - predicted_RT| median {np.median(rt_err):.0f}s, >30s {int((rt_err>30).sum())}/{len(rt_err)} "
          f"({(rt_err>30).mean()*100:.0f}% were beyond my RT window -> RT-prediction error, not detection)")

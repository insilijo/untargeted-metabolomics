"""Are my 'not_detected' GT peaks actually present in features_centwave.tsv (the same
extraction SQuID located 99% of GT in)? Distinguish: my per-platform+COLU+consensus
restriction dropping findable peaks vs genuine absence.
"""
import csv
from collections import defaultdict, Counter
import numpy as np, pandas as pd
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
print("loading full features_centwave.tsv ...", flush=True)
f=pd.read_csv('/mnt/volume-hel1-1/data/processed/features_centwave.tsv', sep='\t', usecols=['source_file','mz'])
f=f.dropna(subset=['source_file'])
f['platform']=f.source_file.str.split('_').str[0].map(PM)
f['colu']=f.source_file.str.contains('COLU', na=False)
# sorted mz arrays for the lookups
plat_colu={p:np.sort(g.mz.to_numpy()) for p,g in f[f.colu].groupby('platform')}
plat_all ={p:np.sort(g.mz.to_numpy()) for p,g in f.groupby('platform')}
allmz=np.sort(f.mz.to_numpy())          # any platform, any file
def has(a,mz,ppm=20):
    t=mz*ppm*1e-6; return (np.searchsorted(a,mz+t)-np.searchsorted(a,mz-t))>0

MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
cat=Counter(); ex=defaultdict(list)
for r in MAF:
    plat=r['platform']; mz=ff(r['mz'])
    if not mz or plat not in plat_colu: continue
    if has(plat_colu[plat],mz):            # detected where my matcher looks
        cat['detected_platform_COLU']+=1; continue
    # my 'not_detected' — where is it really?
    if has(plat_all.get(plat,np.array([])),mz): c='same_platform_nonCOLU'
    elif any(has(plat_colu[p],mz) for p in plat_colu if p!=plat): c='other_platform'
    elif has(allmz,mz): c='other_platform'
    else: c='truly_absent_20ppm'
    cat[c]+=1; ex[c].append(r['name'])
tot=sum(cat.values()); nd=tot-cat['detected_platform_COLU']
print(f"\nGT compounds: {tot}")
print(f"  detected where matcher looks (platform+COLU): {cat['detected_platform_COLU']} ({cat['detected_platform_COLU']/tot*100:.0f}%)")
print(f"  my 'not_detected': {nd} ({nd/tot*100:.0f}%) — of those:")
for k in ('same_platform_nonCOLU','other_platform','truly_absent_20ppm'):
    print(f"     {k:<24}{cat[k]:4d} ({cat[k]/nd*100:.0f}% of not_detected)   e.g. {', '.join(ex[k][:5])}")
print(f"\n  => peak present SOMEWHERE in centwave: {(nd-cat['truly_absent_20ppm'])}/{nd} "
      f"({(nd-cat['truly_absent_20ppm'])/nd*100:.0f}% of not_detected are findable, just not where matcher looked)")

"""Are same-m/z library isomers actually RT-separated (distinct RI)? If so, design an
ISOMER-AWARE anchor panel: dense anchors bracketing the RI regions where isomers
cluster, so local sec->RI calibration is fine enough for a tight window to resolve them.
"""
import csv
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
PM={'Method1':'lc/ms pos early','Method2':'lc/ms pos late','Method3':'lc/ms neg','Method4':'lc/ms polar'}
MASS_PPM=20.0; ik14=lambda s:(s or '')[:14]; ff=lambda x:(float(x) if str(x).strip() not in ('','None','nan') else None)
DD=list(csv.DictReader(open('/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv',encoding='utf-8-sig')))
MAF=[r for r in csv.DictReader(open('/root/SQuID-INC/data/st004581/annotations_repaired.csv')) if r.get('unannotatable','')!='true']
feat=pd.read_parquet('/tmp/feat_colu.parquet'); feat['platform']=feat.source_file.str.split('_').str[0].map(PM)
# RI/s slope per platform (rough, from MAF span) for converting RI gaps to seconds
SLOPE={'lc/ms pos early':17.5,'lc/ms pos late':31.6,'lc/ms neg':17.3,'lc/ms polar':11.1}
for plat in ['lc/ms pos early','lc/ms pos late','lc/ms neg','lc/ms polar']:
    comp=[(ff(r['MASS']),ff(r['RI']),ik14(r['INCHIKEY'])) for r in DD if r['PLATFORM']==plat and ff(r['MASS']) and ff(r['RI']) and ik14(r['INCHIKEY'])]
    comp.sort()
    # isomer sets: consecutive same-m/z (within ppm), >=2 distinct ik14
    gaps=[]; iso_ris=[]; nsets=0; i=0; n=len(comp)
    while i<n:
        j=i+1
        while j<n and (comp[j][0]-comp[i][0])<=comp[i][0]*MASS_PPM*1e-6: j+=1
        members=comp[i:j]
        iks={m[2] for m in members}
        if len(iks)>=2:
            nsets+=1
            ris=sorted(set(m[1] for m in members)); iso_ris+=ris
            for a,b in zip(ris[:-1],ris[1:]): gaps.append((b-a)/SLOPE[plat])  # adjacent RI gap in sec
        i=j
    gaps=np.array(gaps)
    if len(gaps):
        print(f"{plat:<16} isomer m/z-sets={nsets}  adjacent isomer RT gaps(s): "
              f"median {np.median(gaps):.0f}, <5s {int((gaps<5).sum())} ({(gaps<5).mean()*100:.0f}%), "
              f"5-15s {int(((gaps>=5)&(gaps<15)).sum())}, >=15s {int((gaps>=15).sum())} ({(gaps>=15).mean()*100:.0f}%)")

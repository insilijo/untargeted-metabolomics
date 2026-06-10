import glob, time
from multiprocessing import Pool
import pyopenms as oms, numpy as np, pandas as pd
def run(f):
    exp=oms.MSExperiment(); oms.MzMLFile().load(f, exp)
    rows=[]
    name=f.split('/')[-1]
    for s in exp:
        if s.getMSLevel()!=2: continue
        pre=s.getPrecursors()
        if not pre: continue
        mz,it=s.get_peaks()
        if len(mz)<3: continue
        # keep top 50 peaks by intensity
        if len(mz)>50:
            idx=np.argsort(it)[-50:]; mz,it=mz[idx],it[idx]
        o=np.argsort(mz)
        rows.append({"source_file":name,"precursor_mz":float(pre[0].getMZ()),
                     "rt":float(s.getRT()),"mz_array":mz[o].astype(float),
                     "intensity_array":it[o].astype(float)})
    return rows
files=sorted(glob.glob("data/st004581_raw/*COLU*.mzML"))
print(f"{len(files)} neg sample files", flush=True)
t=time.time()
with Pool(6) as p: res=p.map(run, files)
allr=[r for rs in res for r in rs]
df=pd.DataFrame(allr)
df.to_parquet("data/st004581_work/ms2_observed.parquet", index=False)
print(f"{len(df)} MS2 scans from {len(files)} files in {time.time()-t:.1f}s -> ms2_observed.parquet", flush=True)

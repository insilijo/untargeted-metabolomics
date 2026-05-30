import sys, time, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,"scripts")
import centwave_py, pandas as pd

CFG={"noise_threshold_int":100.0,"mass_error_ppm":10.0,"chrom_fwhm":5.0,
     "chrom_peak_snr":2.0,"min_trace_length":3.0,"max_trace_length":-1.0}
def stype(name):
    if "COLU" in name: return "sample"
    if "PRCS" in name: return "blank"
    if "CMTRX" in name: return "qc"
    return "other"
def run(f):
    f=Path(f); t=time.time()
    df=centwave_py.find_features(f, CFG)
    df["sample_type"]=stype(f.name)
    return f.name, len(df), time.time()-t, df

files=sorted(glob.glob("data/st004581_raw/*.mzML"))
print(f"{len(files)} files", flush=True)
t0=time.time()
with Pool(6) as p:
    res=p.map(run, files)
parts=[]
for name,n,dt,df in res:
    print(f"  {name}: {n} feats {dt:.1f}s", flush=True)
    parts.append(df)
allf=pd.concat(parts, ignore_index=True)
allf.to_parquet("data/st004581_work/features_all.parquet", index=False)
print(f"TOTAL {len(allf)} features from {len(files)} files in {time.time()-t0:.1f}s -> data/st004581_work/features_all.parquet", flush=True)
print(allf.groupby('sample_type').size().to_string(), flush=True)

import glob, time, os, zipfile
from pathlib import Path
from multiprocessing import Pool
import sys; sys.path.insert(0,"scripts")
import centwave_py, pyopenms as oms, numpy as np, pandas as pd

ZIP="/home/jgardner/SQuID-INC/data/example/ST004581_COLU-05-19VW.zip"
SETS=["Method1_Set181428","Method2_Set181433","Method4_Set181311"]
RAW="data/st004581_raw2"; os.makedirs(RAW, exist_ok=True)
CFG={"noise_threshold_int":100.0,"mass_error_ppm":10.0,"chrom_fwhm":5.0,
     "chrom_peak_snr":2.0,"min_trace_length":3.0,"max_trace_length":-1.0}
def stype(n): return "sample" if "COLU" in n else ("blank" if "PRCS" in n else ("qc" if "CMTRX" in n else "other"))

# extract
t=time.time()
with zipfile.ZipFile(ZIP) as z:
    names=[n for n in z.namelist() if any(n.startswith(s) for s in SETS) and n.endswith(".mzML")]
    for n in names:
        out=Path(RAW)/Path(n).name
        if not out.exists():
            with z.open(n) as src, open(out,"wb") as dst: dst.write(src.read())
print(f"extracted {len(names)} files in {time.time()-t:.0f}s", flush=True)

files=sorted(glob.glob(f"{RAW}/*.mzML"))
def ff(f):
    f=Path(f); df=centwave_py.find_features(f,CFG); df["sample_type"]=stype(f.name); return df
def ms2(f):
    exp=oms.MSExperiment(); oms.MzMLFile().load(f,exp); name=f.split("/")[-1]; rows=[]
    for s in exp:
        if s.getMSLevel()!=2: continue
        pre=s.getPrecursors()
        if not pre: continue
        mz,it=s.get_peaks()
        if len(mz)<3: continue
        if len(mz)>50: idx=np.argsort(it)[-50:]; mz,it=mz[idx],it[idx]
        o=np.argsort(mz)
        rows.append({"source_file":name,"precursor_mz":float(pre[0].getMZ()),"rt":float(s.getRT()),
                     "mz_array":mz[o].astype(float),"intensity_array":it[o].astype(float)})
    return rows
t=time.time()
with Pool(6) as p: feats=p.map(ff, files)
pd.concat(feats, ignore_index=True).to_parquet("data/st004581_work/features_pospolar.parquet", index=False)
print(f"FF done {time.time()-t:.0f}s", flush=True)
t=time.time()
with Pool(6) as p: m2=p.map(ms2, files)
pd.DataFrame([r for rs in m2 for r in rs]).to_parquet("data/st004581_work/ms2_pospolar.parquet", index=False)
print(f"MS2 done {time.time()-t:.0f}s", flush=True)
print("OK", flush=True)

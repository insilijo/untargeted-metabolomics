"""Validate memory-efficient streaming EIC extractor against the dense (scans x n) method.
Gate: streaming peaks must match dense peaks for neg DD candidates."""
import sys, csv, glob, time
from pathlib import Path
from collections import defaultdict
import numpy as np, pymzml
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
PLAT="lc/ms neg"; DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
MZ_PPM=7.0; FLOOR=8000.0; TIGHT=6.0
lib0=M.load_library(Path(DD))
MZc=np.array(sorted(c["mz"] for c in lib0.get(PLAT,[])))
n=len(MZc); tol=MZc*MZ_PPM*1e-6
print(f"neg DD candidates n={n}, files={len(MZML)}")

# ---------- DENSE (original forest_sweep method) ----------
def extract_dense(files):
    peaks=[[] for _ in range(n)]; spans=[]
    for mp in files:
        rts=[]; rows=[]
        for spec in pymzml.run.Reader(mp):
            if spec.ms_level!=1: continue
            sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
            if not len(sm): rows.append(np.zeros(n)); continue
            lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
            rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(n)]))
        rt=np.array(rts); mat=np.array(rows); spans.append((rt.min(),rt.max()))
        for k in range(n):
            col=mat[:,k]; loc=(col>FLOOR)&(col>=np.roll(col,1))&(col>=np.roll(col,-1))
            peaks[k].append(np.array([(rt[j],col[j]) for j in np.where(loc)[0]]) if loc.any() else np.empty((0,2)))
    return peaks,spans

# ---------- STREAMING (memory-efficient) ----------
def extract_stream(files, mzc):
    nn=len(mzc); t=mzc*MZ_PPM*1e-6; lo_mz=mzc-t; hi_mz=mzc+t
    peaks=[[] for _ in range(nn)]; spans=[]
    for mp in files:
        pk=[[] for _ in range(nn)]
        prev2=None; prev1=None; rt2=0.0; rt1=0.0; rmin=1e18; rmax=-1e18
        for spec in pymzml.run.Reader(mp):
            if spec.ms_level!=1: continue
            sm=np.asarray(spec.mz); si=np.asarray(spec.i); rt=spec.scan_time_in_minutes()*60
            rmin=min(rmin,rt); rmax=max(rmax,rt)
            if len(sm):
                cs=np.concatenate(([0.0],np.cumsum(si.astype(np.float64))))
                lo=np.searchsorted(sm,lo_mz); hi=np.searchsorted(sm,hi_mz)
                cur=cs[hi]-cs[lo]
            else:
                cur=np.zeros(nn)
            if prev2 is not None:
                ismax=(prev1>FLOOR)&(prev1>=prev2)&(prev1>=cur)
                for k in np.nonzero(ismax)[0]:
                    pk[int(k)].append((rt1,prev1[int(k)]))
            prev2=prev1; prev1=cur; rt2=rt1; rt1=rt
        # last scan can't be a local max (no following) -> matches roll? dense uses wraparound; handle below
        for k in range(nn):
            peaks[k].append(np.array(pk[k]) if pk[k] else np.empty((0,2)))
        spans.append((rmin,rmax))
    return peaks,spans

t0=time.time(); pd,sd=extract_dense(MZML); print(f"dense done {time.time()-t0:.0f}s")
t0=time.time(); ps,ss=extract_stream(MZML,MZc); print(f"stream done {time.time()-t0:.0f}s")

# compare: peaks keyed by RT (0.01 min); matched if RT present in both, intensity within 0.1% rel
npk_d=0; npk_s=0; matched=0; rt_only_d=0; rt_only_s=0; int_mismatch=0
for k in range(n):
    for inj in range(len(MZML)):
        A=pd[k][inj]; B=ps[k][inj]; npk_d+=len(A); npk_s+=len(B)
        da={round(r,2):i for r,i in A}; db={round(r,2):i for r,i in B}
        for rt_,ia in da.items():
            if rt_ in db:
                matched+=1
                if abs(ia-db[rt_])>1e-3*max(abs(ia),1): int_mismatch+=1
            else: rt_only_d+=1
        for rt_ in db:
            if rt_ not in da: rt_only_s+=1
print(f"dense peaks={npk_d}  stream peaks={npk_s}")
print(f"RT-matched peaks={matched}  dense-only={rt_only_d}  stream-only={rt_only_s}  (boundary/float)")
print(f"of matched, intensity mismatch >0.1%: {int_mismatch}")
print(f"RT agreement = {matched/max(npk_d,1):.4f}")

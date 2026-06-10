"""Phase 1: raw-EIC extractor + validation + the Phase-2 bet test.
(1) one-pass multi-m/z EIC extraction from an mzML.
(2) validate: known neg anchors (m/z + observed_rt_sec) -> recover apex + intensity.
(3) BET: for each anchor, are M+1 / adduct ions present in raw with peak shapes that
    CORRELATE to the principal? (the real-CAMERA discriminator we lacked at consensus)."""
import csv, numpy as np, pymzml
MZML="/root/SQuID-INC/data/st004581/mzml/Method3_Set181307_03_COLU-00966.mzML"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
PPM=20.0; ISO=1.003355
def eic_multi(path, mzs, ppm=PPM):
    mzs=np.asarray(mzs,float); tol=mzs*ppm*1e-6
    rts=[]; rows=[]
    for spec in pymzml.run.Reader(path):
        if spec.ms_level!=1: continue
        smz=np.asarray(spec.mz); si=np.asarray(spec.i)
        if not len(smz): rts.append(spec.scan_time_in_minutes()*60); rows.append(np.zeros(len(mzs))); continue
        lo=np.searchsorted(smz,mzs-tol); hi=np.searchsorted(smz,mzs+tol)
        row=np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(len(mzs))])
        rts.append(spec.scan_time_in_minutes()*60); rows.append(row)
    return np.array(rts), np.array(rows)
anc=[r for r in csv.DictReader(open(ANC))][:12]
mzs=[]; labels=[]
for r in anc:
    m=float(r["mz"])
    for off,tag in [(0,"M"),(ISO,"M+1"),(34.969402-(-1.007276),"+Cl"),(44.998201-(-1.007276),"+FA")]:
        mzs.append(m+off); labels.append((r["name"][:22],tag,float(r["observed_rt_sec"])))
print(f"extracting {len(mzs)} EICs from one neg injection ...", flush=True)
rt,mat=eic_multi(MZML,mzs)
print(f"  {len(rt)} MS1 scans, RT range {rt.min():.0f}-{rt.max():.0f}s\n")
def apex(trace, near, halfwin=20):
    m=np.abs(rt-near)<=halfwin
    if not m.any() or trace[m].max()==0: return None
    j=np.where(m)[0][np.argmax(trace[m])]; return j
print(f"{'compound':<24}{'M_apexRT':>9}{'M_int':>11}{'M+1?':>6}{'+Cl?':>6}{'+FA?':>6}{'shapeCorr(M,M+1)':>17}")
for i in range(0,len(mzs),4):
    nm,_,ort=labels[i]
    jm=apex(mat[:,i],ort)
    if jm is None: print(f"{nm:<24}{'no peak':>9}"); continue
    # window around apex for shape corr
    w=np.abs(rt-rt[jm])<=15
    m0=mat[w,i]
    def present(k): 
        j=apex(mat[:,k],rt[jm],10); return j is not None and mat[j,k]>0
    c1=present(i+1); ccl=present(i+2); cfa=present(i+3)
    m1=mat[w,i+1]
    corr=np.corrcoef(m0,m1)[0,1] if m0.std()>0 and m1.std()>0 and (m1>0).sum()>=3 else float("nan")
    print(f"{nm:<24}{rt[jm]:>9.0f}{mat[jm,i]:>11.0f}{('Y' if c1 else '-'):>6}{('Y' if ccl else '-'):>6}{('Y' if cfa else '-'):>6}{corr:>17.3f}")

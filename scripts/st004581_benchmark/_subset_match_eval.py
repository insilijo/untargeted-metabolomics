"""Tiered RT/RI -> MS1 matching of detected centWave features against the
Metabolon library (lc/ms neg subset), scored per-feature TP/FP/FN vs the
Metabolon MAF ground truth.

Calibration (RI -> seconds) is a per-platform PchipInterpolator fit ONLY on
mass-unique library compounds (no other neg library entry within the mass
tolerance), so the RT/RI tier is evaluated on the isobaric cases it is meant
to resolve rather than the easy unique-mass ones it was trained on.

Outcomes per GT compound (the 'feature' = its expected LC peak):
  TP  detected near (mz, RI->sec) AND tiered matcher assigns the correct ik14
  FP  detected but matcher assigns a DIFFERENT compound (isobaric confusion)
  FN  not detected / not retrieved within mz+RT window
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd

PLATFORM = "lc/ms neg"
MASS_PPM = 15.0          # mz match tolerance (retrieval + MS1 tier)
RT_WIN   = 30.0          # seconds, RT/RI tier window (matches build_anchor_rt_observations)
MIN_REP  = 3             # feature must appear in >= this many sample files
MIN_ANCHORS = 10
FEATS = "data/st004581_work/features_all.parquet"
MAF   = "/home/jgardner/squid_results/st004581/annotations_repaired.csv"

ik14 = lambda s: (s or "")[:14]

# ---- load library/GT (same Metabolon panel; mz=observed ion, rt=RI) ----
lib = []
for r in csv.DictReader(open(MAF)):
    if r["platform"] != PLATFORM: continue
    try: mz=float(r["mz"]); ri=float(r["rt"])
    except (ValueError, KeyError): continue
    if not (mz>0 and ri>0): continue
    lib.append({"name":r["name"],"ik14":ik14(r["inchikey"]),"mz":mz,"ri":ri,
                "unann":(r.get("unannotatable","")=="true")})
print(f"library/GT neg entries with mz+RI: {len(lib)}")

# ---- detected features (sample = COLU only) ----
df = pd.read_parquet(FEATS)
s = df[df.sample_type=="sample"].copy()
print(f"detected sample features: {len(s)} from {s.source_file.nunique()} files")
fmz = s.mz.to_numpy(); frt = s.rt.to_numpy(); fint = s.intensity.to_numpy()
fsrc = s.source_file.to_numpy()
order = np.argsort(fmz); fmz=fmz[order]; frt=frt[order]; fint=fint[order]; fsrc=fsrc[order]

def mz_window(mz):
    tol = mz*MASS_PPM*1e-6
    lo = np.searchsorted(fmz, mz-tol); hi = np.searchsorted(fmz, mz+tol)
    return lo, hi

def consensus_peak(mz, rt_center=None, rt_win=None):
    """Return (rt_med, n_files, total_int) for sample features near mz
    (and optionally near rt_center). None if < MIN_REP files."""
    lo,hi = mz_window(mz)
    if hi<=lo: return None
    rt=frt[lo:hi]; it=fint[lo:hi]; src=fsrc[lo:hi]
    if rt_center is not None:
        m = np.abs(rt-rt_center)<=rt_win
        if not m.any(): return None
        rt,it,src = rt[m],it[m],src[m]
    else:
        # collapse to the dominant RT cluster around the most-intense feature
        c = rt[np.argmax(it)]
        m = np.abs(rt-c)<=20.0
        rt,it,src = rt[m],it[m],src[m]
    nf = len(set(src))
    if nf < MIN_REP: return None
    return float(np.median(rt)), nf, float(it.sum())

# ---- mass-unique flag (within neg library) ----
lib_mz = np.array(sorted(c["mz"] for c in lib))
def n_lib_within(mz):
    tol=mz*MASS_PPM*1e-6
    return np.searchsorted(lib_mz,mz+tol)-np.searchsorted(lib_mz,mz-tol)
for c in lib: c["unique_mass"] = (n_lib_within(c["mz"])==1)
print(f"mass-unique neg compounds: {sum(c['unique_mass'] for c in lib)}")

# ---- calibration: RI -> sec from mass-unique detected anchors ----
anchors=[]
for c in lib:
    if not c["unique_mass"] or c["unann"]: continue
    cp = consensus_peak(c["mz"])
    if cp is None: continue
    anchors.append((c["ri"], cp[0]))
anchors.sort()
print(f"calibration anchors (mass-unique, detected): {len(anchors)}")
if len(anchors)<MIN_ANCHORS:
    sys.exit("Not enough calibration anchors")
# dedup duplicate RI -> median sec ; ensure strictly increasing for PCHIP
ri2rt=defaultdict(list)
for ri,rt in anchors: ri2rt[round(ri,1)].append(rt)
ris=sorted(ri2rt); rts=[float(np.median(ri2rt[r])) for r in ris]
pchip=PchipInterpolator(np.array(ris,float), np.array(rts,float), extrapolate=True)
# fit residuals
res=np.array([rt-float(pchip(ri)) for ri,rt in anchors])
print(f"  RI range [{ris[0]:.0f},{ris[-1]:.0f}]  sec range [{min(rts):.0f},{max(rts):.0f}]"
      f"  resid std {res.std():.1f}s  |resid|<30s: {(np.abs(res)<30).mean():.1%}")

lib_rtsec=np.array([float(pchip(c["ri"])) for c in lib])  # predicted sec per library entry

def tiered_match(mz, rt, use_rt=True):
    """Return best library index by Tier1(RT/RI) then Tier2(MS1), or None."""
    cand=[]
    tol=mz*MASS_PPM*1e-6
    for i,c in enumerate(lib):
        if abs(c["mz"]-mz)>tol: continue              # Tier2 MS1
        if use_rt and abs(lib_rtsec[i]-rt)>RT_WIN: continue  # Tier1 RT/RI
        # score: prefer mass-closeness then rt-closeness
        score = abs(c["mz"]-mz)/tol + (abs(lib_rtsec[i]-rt)/RT_WIN if use_rt else 0)
        cand.append((score,i))
    if not cand: return None
    cand.sort()
    return cand[0][1]

# ---- score every GT compound (exclude unannotatable) ----
def evaluate(use_rt):
    tp=fp=fn=0; rows=[]; conf_pairs=defaultdict(int)
    for c in lib:
        if c["unann"]: continue
        exp_rt=float(pchip(c["ri"]))
        cp = consensus_peak(c["mz"], rt_center=exp_rt if use_rt else None,
                            rt_win=RT_WIN if use_rt else None)
        if cp is None:
            fn+=1; rows.append((c["name"],c["ik14"],"FN_not_detected","")); continue
        rt_obs=cp[0]
        j = tiered_match(c["mz"], rt_obs, use_rt=use_rt)
        if j is None:
            fn+=1; rows.append((c["name"],c["ik14"],"FN_no_candidate","")); continue
        if lib[j]["ik14"]==c["ik14"]:
            tp+=1; rows.append((c["name"],c["ik14"],"TP",lib[j]["name"]))
        else:
            fp+=1; rows.append((c["name"],c["ik14"],"FP_misassigned",lib[j]["name"]))
            conf_pairs[(c["name"],lib[j]["name"])]+=1
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    F1=2*P*R/(P+R) if P+R else 0
    return dict(tp=tp,fp=fp,fn=fn,P=P,R=R,F1=F1,rows=rows,conf=conf_pairs)

ms1 = evaluate(use_rt=False)
both= evaluate(use_rt=True)

n_eval=sum(1 for c in lib if not c["unann"])
print(f"\n=== Eval set: {n_eval} annotatable neg Metabolon compounds ===")
hdr=f"{'method':<16}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}"
print(hdr); print("-"*len(hdr))
for name,m in [("MS1 only",ms1),("RT/RI + MS1",both)]:
    print(f"{name:<16}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['P']:>8.3f}{m['R']:>8.3f}{m['F1']:>8.3f}")

print(f"\n=== Confusion (RT/RI+MS1): per-compound 3-way ===")
from collections import Counter
cnt=Counter(r[2].split('_')[0] for r in both['rows'])
print(f"  correct(TP) {both['tp']}   misassigned(FP) {both['fp']}   not-found(FN) {both['fn']}")
print(f"\n  Top isobaric confusions (true -> assigned) [RT/RI+MS1]:")
for (a,b),n in sorted(both['conf'].items(), key=lambda x:-x[1])[:12]:
    print(f"    {n}x  {a[:34]:<34} -> {b[:34]}")
print(f"\n  MS1-only misassigned {ms1['fp']} -> RT/RI cuts to {both['fp']} "
      f"(RT/RI resolved {ms1['fp']-both['fp']} isobaric confusions)")

# save artifacts
out=pd.DataFrame(both['rows'], columns=["true_name","true_ik14","outcome","assigned_name"])
out.to_csv("data/st004581_work/per_compound_outcomes_rtri_ms1.csv", index=False)
pd.DataFrame(ms1['rows'], columns=["true_name","true_ik14","outcome","assigned_name"]).to_csv(
    "data/st004581_work/per_compound_outcomes_ms1only.csv", index=False)
print("\nwrote per_compound_outcomes_{rtri_ms1,ms1only}.csv")

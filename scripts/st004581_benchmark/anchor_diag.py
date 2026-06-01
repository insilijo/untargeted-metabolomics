"""Diagnose the RI ladder BEFORE optimizing the anchor panel (understand-before-tweak).

Three questions, per platform:
  1. PRECISION: leave-one-anchor-out CV on the pooled (detected-sec -> DD-RI) ladder.
     For each anchor, refit PCHIP on the others, predict its RI from its detected RT,
     report |error| in RI units AND in seconds (err/slope). This is the ladder's
     intrinsic placement error — the noise floor every match inherits.
  2. COVERAGE: where do anchors sit vs where the library/MAF compounds sit? Largest
     RI gaps = regions matched by extrapolation/long interpolation (least trustworthy).
  3. REPRODUCIBILITY: per-anchor detected-RT spread (P90-P10 across injections).
     Wobbly anchors poison the local ladder.

Then ties it together: is LOO error WORSE in the RI regions where the MAF compounds
we FAIL to recover live? If yes, anchors are a real lever there; if not, they aren't.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_match_rtri as M
import pandas as pd

DD   = "/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT   = "/root/SQuID-INC/data/st004581/annotations_repaired.csv"
FEAT = "/tmp/feat_colu.parquet"
ANC  = "/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
MZ_PPM = 20.0


def detected_anchor_pairs(df, anchors, mz_ppm):
    """Pooled (detected_sec, DD_RI, anchor_mz) per platform, dominant-cluster RT."""
    out = defaultdict(list)
    for plat, g in df.groupby("platform"):
        g = g.sort_values("mz")
        fmz = g.mz.to_numpy(); frt = g.rt.to_numpy(); fint = g.intensity.to_numpy()
        for mz, ri in anchors.get(plat, []):
            t = mz * mz_ppm * 1e-6
            lo = np.searchsorted(fmz, mz - t); hi = np.searchsorted(fmz, mz + t)
            if hi <= lo:
                continue
            rt, it = frt[lo:hi], fint[lo:hi]
            c = rt[np.argmax(it)]; m = np.abs(rt - c) <= 20
            out[plat].append((float(np.median(rt[m])), float(ri), float(mz)))
    return out


def main():
    df = pd.read_parquet(FEAT)
    df["platform"] = df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
    df = df.dropna(subset=["platform"])
    anchors = M.load_anchor_points(Path(ANC))
    lib = M.expand_library_adducts(M.load_library(Path(DD)))
    pairs = detected_anchor_pairs(df, anchors, MZ_PPM)

    # MAF compounds + their RI, and which are retrievable (in-library)
    gt = defaultdict(dict)
    with open(GT) as f:
        for r in csv.DictReader(f):
            if r.get("unannotatable", "") == "true":
                continue
            plat = (r.get("platform") or "").strip().lower()
            try:
                gt[plat].setdefault(M.ik14(r["inchikey"]), float(r["rt"]))
            except (ValueError, KeyError):
                continue

    for plat in sorted(pairs):
        P = sorted(pairs[plat])
        secs = np.array([p[0] for p in P]); ris = np.array([p[1] for p in P])
        # collapse duplicate secs (PCHIP needs strictly increasing x)
        uniq = {}
        for s, r in zip(secs, ris):
            uniq.setdefault(round(s, 2), []).append(r)
        xs = np.array(sorted(uniq)); ys = np.array([np.median(uniq[x]) for x in xs])
        if len(xs) < 4:
            print(f"{plat}: only {len(xs)} usable anchors — skip"); continue
        slope = np.median(np.diff(ys) / np.diff(xs))   # RI per sec, local median
        # 1. leave-one-out CV
        loo_ri = []
        for i in range(len(xs)):
            xtr = np.delete(xs, i); ytr = np.delete(ys, i)
            if len(xtr) < 3:
                continue
            f = PchipInterpolator(xtr, ytr, extrapolate=True)
            loo_ri.append((xs[i], abs(float(f(xs[i])) - ys[i])))
        errs_ri = np.array([e for _, e in loo_ri])
        errs_sec = errs_ri / abs(slope) if slope else errs_ri
        print(f"\n=== {plat} ===")
        print(f"anchors usable: {len(xs)}   RI span [{ys.min():.0f},{ys.max():.0f}]   "
              f"slope ~{slope:.1f} RI/s")
        print(f"LOO ladder error: median {np.median(errs_ri):.0f} RI "
              f"({np.median(errs_sec):.1f}s)   P90 {np.percentile(errs_ri,90):.0f} RI "
              f"({np.percentile(errs_sec,90):.1f}s)   max {errs_ri.max():.0f} RI "
              f"({errs_sec.max():.1f}s)")
        # 2. coverage gaps (in RI)
        gaps = np.diff(np.sort(ys))
        order = np.argsort(gaps)[::-1][:3]
        srt = np.sort(ys)
        print("largest RI gaps (under-anchored): " +
              ", ".join(f"[{srt[o]:.0f}-{srt[o+1]:.0f}] (Δ{gaps[o]:.0f})" for o in order))
        # 3. tie to recall: LOO error in regions of recovered vs missed MAF compounds.
        # build the real per-batch ladder + match to get TP/FN ik14, then bin by RI.
        # (cheap proxy here: compare anchor-poor vs anchor-rich halves)
        gt_ris = np.array(list(gt.get(plat, {}).values()))
        if len(gt_ris):
            # local anchor density per MAF compound = # anchors within ±300 RI
            dens = np.array([np.sum(np.abs(ys - r) <= 300) for r in gt_ris])
            print(f"MAF compounds: {len(gt_ris)}   median local anchor density (±300 RI): "
                  f"{np.median(dens):.0f}   under-anchored (<2 nearby): "
                  f"{int((dens < 2).sum())} ({(dens<2).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()

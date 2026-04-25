"""Auto-calibrate centwave_py params per-platform from SST anchors.

Reads anchor compounds (kit), opens one representative mzML per platform,
extracts XICs at each anchor's expected mz, fits Gaussian-ish peak FWHM
and intensity distribution. Output: ``run_params.json`` with per-platform
overrides for chrom_fwhm, noise_threshold_int, expected RT range.

Idea: different gradients / instruments produce peaks of different
widths and intensity distributions. Hardcoding chrom_fwhm=5 across all
methods misses peaks on slower gradients (e.g. polar) and over-detects
on sharper ones. Anchors give us the ground truth shape per platform.

Usage:
    python calibrate_ff_params.py \
        --anchors /root/results/anchors_v1.csv \
        --mzml-dir /mnt/volume-hel1-1/data/raw/ST004581 \
        --output run_params.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PLATFORM_FROM_METHOD = {
    "Method1": "lc/ms pos early",
    "Method2": "lc/ms pos late",
    "Method3": "lc/ms neg",
    "Method4": "lc/ms polar",
}


def load_anchors(path: Path):
    """Load anchor compound list. Returns dicts with mz/rt/inchikey/name."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        def col(*ns):
            for n in ns:
                if n in cols: return cols[n]
            return None
        mz_col = col("predicted_ms1_mz", "mz", "predicted_mz")
        rt_col = col("predicted_rt", "rt")
        ik_col = col("inchikey")
        name_col = col("name", "compound_name")
        for r in reader:
            try: mz = float(r.get(mz_col) or 0) if mz_col else 0
            except: mz = 0
            try: rt = float(r.get(rt_col) or 0) if rt_col else 0
            except: rt = 0
            if mz <= 0: continue
            rows.append({
                "name": (r.get(name_col) or "").strip() if name_col else "",
                "inchikey": (r.get(ik_col) or "")[:14] if ik_col else "",
                "mz": mz,
                "rt": rt,
            })
    return rows


def find_mzml(mzml_dir: Path, method: str, prefer="COLU"):
    """Pick a representative sample mzML for the method."""
    candidates = sorted(mzml_dir.glob(f"{method}_*.mzML"))
    for c in candidates:
        if prefer in c.name and "PRCS" not in c.name and "CMTRX" not in c.name:
            return c
    return candidates[0] if candidates else None


def extract_xic(mzml_path: Path, mz_target: float, mz_ppm: float = 5.0):
    """Return [(rt_s, intensity)] sorted by RT for the XIC at mz_target."""
    from pyteomics import mzml
    xic = []
    tol = mz_target * mz_ppm / 1e6
    import bisect
    with mzml.read(str(mzml_path)) as reader:
        for spec in reader:
            if spec.get("ms level", 1) != 1:
                continue
            rt = spec.get("scanList", {}).get("scan", [{}])[0].get(
                "scan start time", 0)
            rt_s = float(rt) * 60.0 if rt < 100 else float(rt)
            mzs = spec.get("m/z array")
            ints = spec.get("intensity array")
            if mzs is None or len(mzs) == 0:
                continue
            lo = bisect.bisect_left(mzs, mz_target - tol)
            hi = bisect.bisect_right(mzs, mz_target + tol)
            if lo < hi:
                xic.append((rt_s, float(max(ints[lo:hi]))))
            else:
                xic.append((rt_s, 0.0))
    return xic


def measure_peak_fwhm(xic):
    """Find apex, compute FWHM in seconds. Returns (apex_rt, apex_int, fwhm_s)
    or None if no clear peak."""
    if not xic:
        return None
    apex_idx = max(range(len(xic)), key=lambda i: xic[i][1])
    apex_rt, apex_int = xic[apex_idx]
    if apex_int <= 0:
        return None
    half = apex_int / 2.0
    # Walk left
    left_rt = xic[0][0]
    for i in range(apex_idx, -1, -1):
        if xic[i][1] < half:
            left_rt = xic[i][0]
            break
    # Walk right
    right_rt = xic[-1][0]
    for i in range(apex_idx, len(xic)):
        if xic[i][1] < half:
            right_rt = xic[i][0]
            break
    fwhm = right_rt - left_rt
    return (apex_rt, apex_int, fwhm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", required=True)
    p.add_argument("--mzml-dir", required=True)
    p.add_argument("--per-platform", type=int, default=20,
                   help="Anchors to probe per platform (cap)")
    p.add_argument("--mz-ppm", type=float, default=5.0)
    p.add_argument("--output", default="run_params.json")
    args = p.parse_args()

    anchors = load_anchors(Path(args.anchors))
    print(f"Loaded {len(anchors)} anchors")

    # Probe one mzML per method
    mzml_dir = Path(args.mzml_dir)
    per_platform_params: dict[str, dict] = {}

    for method, platform in PLATFORM_FROM_METHOD.items():
        mp = find_mzml(mzml_dir, method)
        if mp is None:
            print(f"  no mzML for {method}")
            continue
        print(f"\n{platform}  ({mp.name})")

        # Probe up to N anchors
        probe = anchors[:args.per_platform]
        fwhms, apex_intensities, apex_rts = [], [], []
        n_found = 0
        for a in probe:
            try:
                xic = extract_xic(mp, a["mz"], mz_ppm=args.mz_ppm)
            except Exception as e:
                continue
            fit = measure_peak_fwhm(xic)
            if fit is None:
                continue
            apex_rt, apex_int, fwhm = fit
            if fwhm <= 0 or fwhm > 600:  # sanity
                continue
            n_found += 1
            fwhms.append(fwhm)
            apex_intensities.append(apex_int)
            apex_rts.append(apex_rt)

        if not fwhms:
            print(f"  no anchor peaks found, using defaults")
            continue

        med_fwhm = statistics.median(fwhms)
        p5_int = sorted(apex_intensities)[max(0, int(0.05 * len(apex_intensities)))]
        med_int = statistics.median(apex_intensities)
        rt_min, rt_max = min(apex_rts), max(apex_rts)

        params = {
            "n_anchors_probed":   len(probe),
            "n_anchors_with_peak": n_found,
            "anchor_locate_frac": round(n_found / len(probe), 3),
            "chrom_fwhm":         round(med_fwhm, 1),
            "noise_threshold_int": round(max(50, p5_int / 10), 0),
            "anchor_intensity_p5":  round(p5_int, 0),
            "anchor_intensity_med": round(med_int, 0),
            "rt_range_seconds":   [round(rt_min, 1), round(rt_max, 1)],
        }
        per_platform_params[platform] = params
        print(f"  n_anchors_with_peak: {n_found}/{len(probe)}")
        print(f"  measured_fwhm_median: {med_fwhm:.1f}s   p5/median intensity: {p5_int:.0f}/{med_int:.0f}")
        print(f"  → suggested chrom_fwhm={params['chrom_fwhm']}  noise_threshold_int={params['noise_threshold_int']}")

    # Write run_params.json
    out = Path(args.output)
    out.write_text(json.dumps({
        "platforms": per_platform_params,
        "anchors_path": str(args.anchors),
        "mz_ppm_tol": args.mz_ppm,
    }, indent=2))
    print(f"\nWrote calibration → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

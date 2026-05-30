"""Tiered RT/RI -> MS1 library matcher — the open-source core of Metabolon-style
targeted annotation.

Annotate LC-MS features against a reference standard library (a purchaseable
spike-in panel + public dictionary) using a calibrated RI->seconds bridge built
from RT anchors. Implements the choices validated on ST004581
(scripts/st004581_benchmark/): per-platform PCHIP RI->sec calibration from
evenly-spaced anchors, structured-candidate preference (a structureless library
entry can only win if it is the sole option), and configurable detection
reproducibility.

CLI:
  poetry run python scripts/library_match_rtri.py \
      --features data/interim/features.tsv \
      --library  data/raw/metabolon_data_dictionary.csv \
      --anchors  data/anchor_panels/anchors_all_platforms.csv \
      --out      data/processed/rtri_annotations.csv

Library columns (case-insensitive, aliases accepted):
  name | inchikey | smiles | platform | mz(=mass) | ri(=rt)
Anchor columns: platform, ri, observed_rt_sec   (e.g. data/anchor_panels/)
Feature columns: source_file, mz, rt, intensity  (rt in seconds)
Platform is taken from the mzML method prefix (Method1..4) unless a `platform`
column is present; override the prefix->platform map with --platform-map.
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DEFAULT_PREFIX_MAP = {
    "Method1": "lc/ms pos early", "Method2": "lc/ms pos late",
    "Method3": "lc/ms neg",       "Method4": "lc/ms polar",
}
ik14 = lambda s: (s or "")[:14]


def _col(row, *names):
    for n in names:
        for k in row:
            if k.lower().lstrip("﻿") == n: return row[k]
    return ""


def load_library(path: Path) -> dict[str, list[dict]]:
    """Reference library grouped by platform. Each entry: name, ik14, smiles, mz, ri."""
    lib: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                mz = float(_col(r, "mz", "mass", "precursor_mz", "pepmass"))
                ri = float(_col(r, "ri", "rt", "retention_index", "retention_time"))
            except (ValueError, TypeError):
                continue
            if mz <= 0 or ri <= 0:
                continue
            plat = (_col(r, "platform", "method") or "").strip().lower()
            lib[plat].append({
                "name": _col(r, "name", "biochemical", "compound"),
                "ik14": ik14(_col(r, "inchikey")),
                "smiles": _col(r, "smiles"),
                "mz": mz, "ri": ri,
            })
    for p in lib:
        lib[p].sort(key=lambda c: c["mz"])
    return lib


def build_calibrators(path: Path) -> dict[str, PchipInterpolator]:
    """Per-platform monotone RI->seconds calibrators from an anchor panel."""
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            plat = (_col(r, "platform", "method") or "").strip().lower()
            try:
                ri = float(_col(r, "ri", "rt"))
                sec = float(_col(r, "observed_rt_sec", "rt_sec", "observed_rt"))
            except (ValueError, TypeError):
                continue
            if ri > 0 and sec > 0:
                pairs[plat].append((ri, sec))
    cals = {}
    for plat, pr in pairs.items():
        if len(pr) < 3:
            continue
        agg = defaultdict(list)
        for ri, sec in pr:
            agg[round(ri, 1)].append(sec)
        ris = sorted(agg)
        secs = [float(np.median(agg[r])) for r in ris]
        cals[plat] = PchipInterpolator(np.array(ris, float), np.array(secs, float),
                                       extrapolate=True)
    return cals


def consensus_features(df: pd.DataFrame, mz_ppm: float, rt_tol: float,
                       min_rep: int) -> pd.DataFrame:
    """Collapse per-injection features into consensus peaks per platform.

    Two-pass greedy: split on m/z gaps > ppm tol, then on rt gaps > rt_tol;
    keep clusters seen in >= min_rep distinct source files.
    """
    out = []
    for plat, g in df.groupby("platform"):
        g = g.sort_values("mz")
        mz = g.mz.to_numpy(); rt = g.rt.to_numpy()
        inten = g.intensity.to_numpy(); src = g.source_file.to_numpy()
        i = 0; n = len(g)
        while i < n:
            j = i + 1
            while j < n and (mz[j] - mz[j-1]) <= mz[j-1] * mz_ppm * 1e-6:
                j += 1
            # m/z block [i,j): split by rt
            order = np.argsort(rt[i:j])
            bmz, brt, bin_, bsrc = mz[i:j][order], rt[i:j][order], inten[i:j][order], src[i:j][order]
            k = 0; m = len(bmz)
            while k < m:
                l = k + 1
                while l < m and (brt[l] - brt[l-1]) <= rt_tol:
                    l += 1
                nfiles = len(set(bsrc[k:l]))
                if nfiles >= min_rep:
                    out.append({"platform": plat,
                                "mz": float(np.median(bmz[k:l])),
                                "rt": float(np.median(brt[k:l])),
                                "intensity": float(bin_[k:l].sum()),
                                "n_files": nfiles})
                k = l
            i = j
    return pd.DataFrame(out)


def match(features: pd.DataFrame, lib: dict, cals: dict, mz_ppm: float,
          rt_win: float, prefer_structured: bool = True) -> pd.DataFrame:
    """Feature-centric tiered RT/RI -> MS1 match. One row per feature."""
    rows = []
    for plat, g in features.groupby("platform"):
        cand = lib.get(plat, [])
        cal = cals.get(plat)
        if not cand or cal is None:
            for _, f in g.iterrows():
                rows.append({**f, "match_ik14": "", "match_name": "", "n_cand": 0,
                             "score": np.nan, "score_gap": np.nan})
            continue
        cmz = np.array([c["mz"] for c in cand])
        csec = np.array([float(cal(c["ri"])) for c in cand])
        for _, f in g.iterrows():
            mz, rt = f["mz"], f["rt"]
            tol = mz * mz_ppm * 1e-6
            lo = np.searchsorted(cmz, mz - tol); hi = np.searchsorted(cmz, mz + tol)
            scored = []
            for i in range(lo, hi):
                if abs(csec[i] - rt) > rt_win:
                    continue
                sc = abs(cmz[i] - mz) / tol + abs(csec[i] - rt) / rt_win
                scored.append((sc, i))
            if not scored:
                rows.append({**f, "match_ik14": "", "match_name": "", "n_cand": 0,
                             "score": np.nan, "score_gap": np.nan})
                continue
            scored.sort()
            pool = scored
            if prefer_structured:
                struct = [s for s in scored if cand[s[1]]["ik14"]]
                if struct:
                    pool = struct
            best = pool[0]; bc = cand[best[1]]
            gap = (pool[1][0] - pool[0][0]) if len(pool) > 1 else np.nan
            rows.append({**f, "match_ik14": bc["ik14"], "match_name": bc["name"],
                         "n_cand": len(scored), "score": round(best[0], 4),
                         "score_gap": round(gap, 4) if gap == gap else np.nan})
    return pd.DataFrame(rows)


def score_against_gt(ann: pd.DataFrame, gt_path: Path, cals: dict,
                     mz_ppm: float, rt_win: float) -> None:
    """Quick per-GT-compound TP/FP/FN (answer-in-library) for sanity/eval."""
    gt = defaultdict(list)
    libik = set(ann["match_ik14"].dropna())
    with open(gt_path) as f:
        for r in csv.DictReader(f):
            if r.get("unannotatable", "") == "true":
                continue
            plat = (r.get("platform") or "").strip().lower()
            try:
                mz = float(r["mz"]); ri = float(r["rt"])
            except (ValueError, KeyError):
                continue
            gt[plat].append((mz, ri, ik14(r["inchikey"])))
    tp = fp = fn = 0
    for plat, comps in gt.items():
        cal = cals.get(plat)
        sub = ann[ann.platform == plat]
        if sub.empty:
            continue   # platform not analysed — don't penalise
        if cal is None:
            fn += len(comps); continue
        amz = sub.mz.to_numpy(); art = sub.rt.to_numpy(); aik = sub.match_ik14.to_numpy()
        for mz, ri, gik in comps:
            sec = float(cal(ri)); tol = mz * mz_ppm * 1e-6
            sel = (np.abs(amz - mz) <= tol) & (np.abs(art - sec) <= rt_win)
            if not sel.any():
                fn += 1; continue
            hit_iks = set(aik[sel])
            if gik in hit_iks: tp += 1
            else: fp += 1
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    print(f"[eval] TP {tp}  FP {fp}  FN {fn}  P {P:.3f}  R {R:.3f}  F1 {F1:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mz-ppm", type=float, default=20.0)
    ap.add_argument("--rt-win", type=float, default=30.0)
    ap.add_argument("--min-rep", type=int, default=2)
    ap.add_argument("--rt-tol", type=float, default=10.0, help="consensus rt grouping (s)")
    ap.add_argument("--no-prefer-structured", action="store_true")
    ap.add_argument("--gt", default="", help="MAF csv for an optional sanity F1")
    a = ap.parse_args()

    df = (pd.read_parquet(a.features) if a.features.endswith(".parquet")
          else pd.read_csv(a.features, sep="\t"))
    if "platform" not in df.columns:
        df["platform"] = (df.source_file.str.split("_").str[0]
                          .map(DEFAULT_PREFIX_MAP))
    df = df.dropna(subset=["platform"])
    print(f"features: {len(df)} on {df.platform.nunique()} platforms", flush=True)

    lib = load_library(Path(a.library)); cals = build_calibrators(Path(a.anchors))
    print(f"library platforms: { {p: len(v) for p, v in lib.items()} }")
    print(f"calibrators: {sorted(cals)}", flush=True)

    cons = consensus_features(df, a.mz_ppm, a.rt_tol, a.min_rep)
    print(f"consensus features: {len(cons)}", flush=True)
    ann = match(cons, lib, cals, a.mz_ppm, a.rt_win,
                prefer_structured=not a.no_prefer_structured)
    n_ann = int(ann.match_ik14.astype(bool).sum())
    print(f"annotated features: {n_ann}/{len(ann)}", flush=True)
    ann.to_csv(a.out, index=False)
    print(f"-> {a.out}", flush=True)
    if a.gt:
        score_against_gt(ann, Path(a.gt), cals, a.mz_ppm, a.rt_win)


if __name__ == "__main__":
    main()

"""Tiered RI -> MS1 library matcher — the open-source core of Metabolon-style
targeted annotation.

Metabolon spikes a retention-index ladder into every batch and reports each
compound's RETENTION INDEX normalised to those standards, so RI is
batch/instrument-invariant while seconds are not. This matcher does the same:

  1. Per (platform, batch), detect the spiked ANCHOR compounds and fit a monotone
     seconds->RI calibrator (the RI ladder).
  2. Normalise every feature's observed RT (seconds) to RI via its batch ladder.
  3. Match features to the reference library in batch-invariant RI space
     (m/z tier + RI tier), preferring structured candidates.

A purchaseable anchor kit spiked into each run is exactly what makes step 1
possible per-batch — the thing that lets an open-source pipeline reproduce
Metabolon's cross-batch alignment.

CLI:
  poetry run python scripts/library_match_rtri.py \
      --features data/interim/features.tsv \
      --library  data/raw/metabolon_data_dictionary.csv \
      --anchors  data/anchor_panels/anchors_all_platforms.csv \
      --out      data/processed/rtri_annotations.csv [--gt MAF.csv]
"""
from __future__ import annotations
import argparse, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DEFAULT_PREFIX_MAP = {"Method1": "lc/ms pos early", "Method2": "lc/ms pos late",
                      "Method3": "lc/ms neg", "Method4": "lc/ms polar"}
ik14 = lambda s: (s or "")[:14]


def _col(row, *names):
    for n in names:
        for k in row:
            if k.lower().lstrip("﻿") == n:
                return row[k]
    return ""


def load_library(path: Path) -> dict[str, list[dict]]:
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
            lib[plat].append({"name": _col(r, "name", "biochemical", "compound"),
                              "ik14": ik14(_col(r, "inchikey")), "mz": mz, "ri": ri})
    for p in lib:
        lib[p].sort(key=lambda c: c["mz"])
    return lib


def load_anchor_points(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Kit anchors as per-platform (mz, ri) — RT is detected per batch, not read."""
    pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            plat = (_col(r, "platform", "method") or "").strip().lower()
            try:
                mz = float(_col(r, "mz", "mass")); ri = float(_col(r, "ri", "rt"))
            except (ValueError, TypeError):
                continue
            if mz > 0 and ri > 0:
                pts[plat].append((mz, ri))
    return pts


def _detect_rt(fmz, frt, fint, fsrc, mz, mz_ppm, min_rep):
    """Dominant-cluster median RT (s) for a target m/z, or None."""
    t = mz * mz_ppm * 1e-6
    lo = np.searchsorted(fmz, mz - t); hi = np.searchsorted(fmz, mz + t)
    if hi <= lo:
        return None
    rt, it, sr = frt[lo:hi], fint[lo:hi], fsrc[lo:hi]
    c = rt[np.argmax(it)]; m = np.abs(rt - c) <= 20
    rt, sr = rt[m], sr[m]
    if len(set(sr)) < min_rep:
        return None
    return float(np.median(rt))


def build_batch_ladders(features: pd.DataFrame, anchors: dict, mz_ppm: float,
                        anchor_min_rep: int):
    """Per (platform, batch) monotone seconds->RI calibrator from detected anchors.
    Falls back to a platform-global ladder when a batch has too few anchors."""
    ladders, cov = {}, {}
    # platform-global pooled pairs as fallback
    pooled = defaultdict(list)
    for (plat, batch), g in features.groupby(["platform", "batch"]):
        pts = anchors.get(plat, [])
        if not pts:
            continue
        g = g.sort_values("mz")
        fmz = g.mz.to_numpy(); frt = g.rt.to_numpy(); fint = g.intensity.to_numpy(); fsrc = g.source_file.to_numpy()
        pairs = []
        for mz, ri in pts:
            sec = _detect_rt(fmz, frt, fint, fsrc, mz, mz_ppm, anchor_min_rep)
            if sec is not None:
                pairs.append((sec, ri)); pooled[plat].append((sec, ri))
        cov[(plat, batch)] = len(pairs)
        lad = _fit_sec_to_ri(pairs)
        if lad is not None:
            ladders[(plat, batch)] = lad
    pooled_lad = {p: _fit_sec_to_ri(v) for p, v in pooled.items()}
    return ladders, pooled_lad, cov, pooled


def ri_per_sec(pooled_pairs: dict) -> dict[str, float]:
    """Median RI-per-second slope per platform (to express s-windows in RI)."""
    slope = {}
    for plat, pr in pooled_pairs.items():
        if len(pr) < 2:
            continue
        secs = [s for s, _ in pr]; ris = [r for _, r in pr]
        dsec = max(secs) - min(secs)
        if dsec > 0:
            slope[plat] = (max(ris) - min(ris)) / dsec
    return slope


def _fit_sec_to_ri(pairs):
    if len(pairs) < 3:
        return None
    agg = defaultdict(list)
    for sec, ri in pairs:
        agg[round(sec, 1)].append(ri)
    secs = sorted(agg); ris = [float(np.median(agg[s])) for s in secs]
    if len(secs) < 3:
        return None
    return PchipInterpolator(np.array(secs, float), np.array(ris, float), extrapolate=True)


def normalise_ri(features, ladders, pooled_lad):
    """Add ri_norm = batch_ladder(seconds), using pooled platform ladder as fallback."""
    ri = np.full(len(features), np.nan)
    for i, (plat, batch, sec) in enumerate(zip(features.platform, features.batch, features.rt)):
        lad = ladders.get((plat, batch)) or pooled_lad.get(plat)
        if lad is not None:
            ri[i] = float(lad(sec))
    features = features.copy(); features["ri_norm"] = ri
    return features.dropna(subset=["ri_norm"])


def consensus_features(df, mz_ppm, ri_tol_by_plat, min_rep):
    """Collapse per-injection features into consensus peaks in (m/z, RI) space."""
    out = []
    for plat, g in df.groupby("platform"):
        ri_tol = ri_tol_by_plat.get(plat, 150.0)
        g = g.sort_values("mz")
        mz = g.mz.to_numpy(); ri = g.ri_norm.to_numpy()
        inten = g.intensity.to_numpy(); src = g.source_file.to_numpy()
        i = 0; n = len(g)
        while i < n:
            j = i + 1
            while j < n and (mz[j] - mz[j-1]) <= mz[j-1] * mz_ppm * 1e-6:
                j += 1
            order = np.argsort(ri[i:j])
            bmz, bri, bin_, bsrc = mz[i:j][order], ri[i:j][order], inten[i:j][order], src[i:j][order]
            k = 0; m = len(bmz)
            while k < m:
                l = k + 1
                while l < m and (bri[l] - bri[l-1]) <= ri_tol:
                    l += 1
                if len(set(bsrc[k:l])) >= min_rep:
                    out.append({"platform": plat, "mz": float(np.median(bmz[k:l])),
                                "ri_norm": float(np.median(bri[k:l])),
                                "intensity": float(bin_[k:l].sum()),
                                "n_files": len(set(bsrc[k:l]))})
                k = l
            i = j
    return pd.DataFrame(out)


def match(cons, lib, mz_ppm, ri_win_by_plat, prefer_structured=True):
    rows = []
    for plat, g in cons.groupby("platform"):
        cand = lib.get(plat, [])
        ri_win = ri_win_by_plat.get(plat, 500.0)
        if not cand:
            for _, f in g.iterrows():
                rows.append({**f, "match_ik14": "", "match_name": "", "n_cand": 0, "score": np.nan})
            continue
        cmz = np.array([c["mz"] for c in cand]); cri = np.array([c["ri"] for c in cand])
        for _, f in g.iterrows():
            mz, ri = f["mz"], f["ri_norm"]; tol = mz * mz_ppm * 1e-6
            lo = np.searchsorted(cmz, mz - tol); hi = np.searchsorted(cmz, mz + tol)
            scored = [(abs(cmz[i]-mz)/tol + abs(cri[i]-ri)/ri_win, i)
                      for i in range(lo, hi) if abs(cri[i]-ri) <= ri_win]
            if not scored:
                rows.append({**f, "match_ik14": "", "match_name": "", "n_cand": 0, "score": np.nan})
                continue
            scored.sort(); pool = scored
            if prefer_structured:
                s = [x for x in scored if cand[x[1]]["ik14"]]
                if s: pool = s
            bc = cand[pool[0][1]]
            rows.append({**f, "match_ik14": bc["ik14"], "match_name": bc["name"],
                         "n_cand": len(scored), "score": round(pool[0][0], 4)})
    return pd.DataFrame(rows)


def score_against_gt(ann, gt_path, mz_ppm, ri_win_by_plat):
    gt = defaultdict(list)
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
        sub = ann[ann.platform == plat]
        if sub.empty:
            continue
        ri_win = ri_win_by_plat.get(plat, 500.0)
        amz = sub.mz.to_numpy(); ari = sub.ri_norm.to_numpy(); aik = sub.match_ik14.to_numpy()
        for mz, ri, gik in comps:
            tol = mz * mz_ppm * 1e-6
            sel = (np.abs(amz - mz) <= tol) & (np.abs(ari - ri) <= ri_win)
            if not sel.any():
                fn += 1
            elif gik in set(aik[sel]):
                tp += 1
            else:
                fp += 1
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    print(f"[eval] TP {tp} FP {fp} FN {fn}  P {P:.3f} R {R:.3f} F1 {2*P*R/(P+R) if P+R else 0:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--anchors", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mz-ppm", type=float, default=20.0)
    ap.add_argument("--rt-win-sec", type=float, default=30.0,
                    help="RI match window, expressed in seconds (auto-scaled to RI per platform)")
    ap.add_argument("--rt-tol-sec", type=float, default=10.0,
                    help="consensus RI grouping, in seconds (auto-scaled)")
    ap.add_argument("--min-rep", type=int, default=2)
    ap.add_argument("--anchor-min-rep", type=int, default=2)
    ap.add_argument("--batch-regex", default=r"(Set\d+)")
    ap.add_argument("--no-prefer-structured", action="store_true")
    ap.add_argument("--gt", default="")
    a = ap.parse_args()

    df = (pd.read_parquet(a.features) if a.features.endswith(".parquet")
          else pd.read_csv(a.features, sep="\t"))
    if "platform" not in df.columns:
        df["platform"] = df.source_file.str.split("_").str[0].map(DEFAULT_PREFIX_MAP)
    df = df.dropna(subset=["platform"])
    df["batch"] = df.source_file.str.extract(a.batch_regex, expand=False).fillna("all")
    print(f"features: {len(df)}  platforms {df.platform.nunique()}  batches {df.batch.nunique()}", flush=True)

    lib = load_library(Path(a.library)); anchors = load_anchor_points(Path(a.anchors))
    ladders, pooled, cov, pooled_pairs = build_batch_ladders(df, anchors, a.mz_ppm, a.anchor_min_rep)
    slope = ri_per_sec(pooled_pairs)
    ri_win = {p: a.rt_win_sec * s for p, s in slope.items()}
    ri_tol = {p: a.rt_tol_sec * s for p, s in slope.items()}
    print(f"per-batch RI ladders: {len(ladders)} (anchor coverage med "
          f"{int(np.median(list(cov.values()))) if cov else 0}); RI/s slope: "
          f"{ {p: round(s,1) for p,s in slope.items()} }", flush=True)
    df = normalise_ri(df, ladders, pooled)
    print(f"features normalised to RI: {len(df)}", flush=True)
    cons = consensus_features(df, a.mz_ppm, ri_tol, a.min_rep)
    ann = match(cons, lib, a.mz_ppm, ri_win, prefer_structured=not a.no_prefer_structured)
    print(f"consensus {len(cons)}  annotated {int(ann.match_ik14.astype(bool).sum())}", flush=True)
    ann.to_csv(a.out, index=False); print(f"-> {a.out}", flush=True)
    if a.gt:
        score_against_gt(ann, Path(a.gt), a.mz_ppm, ri_win)


if __name__ == "__main__":
    main()

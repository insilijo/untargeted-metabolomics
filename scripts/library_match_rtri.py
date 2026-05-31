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


# observed-ion offset from neutral M, per mode; primary (library ion) first.
_ADDUCTS_NEG = [("[M-H]-", -1.007276), ("[M+FA-H]-", 44.998201), ("[M+Cl]-", 34.969402),
                ("[M-H2O-H]-", -19.017841)]
_ADDUCTS_POS = [("[M+H]+", 1.007276), ("[M+Na]+", 22.989218), ("[M+NH4]+", 18.033823),
                ("[M+K]+", 38.963158), ("[M+H-H2O]+", -17.002740)]
_PLAT_ADDUCTS = {"lc/ms neg": _ADDUCTS_NEG, "lc/ms pos early": _ADDUCTS_POS,
                 "lc/ms pos late": _ADDUCTS_POS, "lc/ms polar": _ADDUCTS_POS}


def expand_library_adducts(lib: dict) -> dict:
    """Expand each compound into its expected adduct ions (sharing ik/ri/name).
    The library m/z is the primary ion; neutral = mz - primary_offset; other ions
    at neutral + offset. is_primary flags the original. Recovers compounds that
    ionize as a non-primary adduct (a co-elution-confirmed recall lever)."""
    out: dict[str, list[dict]] = {}
    for plat, entries in lib.items():
        adducts = _PLAT_ADDUCTS.get(plat)
        rows = []
        for c in entries:
            if not adducts:
                rows.append({**c, "adduct": "primary", "is_primary": True}); continue
            neutral = c["mz"] - adducts[0][1]
            for j, (nm, off) in enumerate(adducts):
                rows.append({**c, "mz": neutral + off, "adduct": nm, "is_primary": j == 0})
        rows.sort(key=lambda c: c["mz"])
        out[plat] = rows
    return out


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


def anchor_rt_spread(features, anchors, mz_ppm):
    """Per-platform peak-RT reproducibility (median P90-P10, seconds) from anchors —
    the right scale for a flexible window: tight where peaks are reproducible."""
    spread = {}
    for plat, g in features.groupby("platform"):
        pts = anchors.get(plat, [])
        if not pts:
            continue
        g = g.sort_values("mz")
        fmz = g.mz.to_numpy(); frt = g.rt.to_numpy(); fint = g.intensity.to_numpy(); fsrc = g.source_file.to_numpy()
        sps = []
        for mz, _ri in pts:
            t = mz * mz_ppm * 1e-6
            lo = np.searchsorted(fmz, mz - t); hi = np.searchsorted(fmz, mz + t)
            if hi <= lo: continue
            byf = {}
            for k in range(lo, hi):
                s = fsrc[k]
                if s not in byf or fint[k] > byf[s][1]: byf[s] = (frt[k], fint[k])
            rts = np.array([v[0] for v in byf.values()])
            if len(rts) >= 5: sps.append(np.percentile(rts, 90) - np.percentile(rts, 10))
        if sps:
            spread[plat] = float(np.median(sps))
    return spread


def make_ri_win_fn(slope, anchor_ris, mode, fixed_sec, floor_sec, alpha, cap_sec, spread=None):
    """Return win(plat, ri) -> RI-window (in RI units).
      fixed     : fixed_sec * slope
      spacing   : clamp(alpha*local_anchor_gap, [floor,cap]s)   — local interpolation uncertainty
      precision : clamp(floor + alpha*measured_anchor_RT_spread, [floor,cap]s) per platform"""
    spread = spread or {}
    def win(plat, ri):
        s = slope.get(plat, 17.0)
        floor_ri, cap_ri = floor_sec * s, cap_sec * s
        if mode == "precision":
            w = (floor_sec + alpha * spread.get(plat, fixed_sec)) * s
            return float(min(cap_ri, max(floor_ri, w)))
        if mode == "spacing":
            a = anchor_ris.get(plat)
            if a is None or len(a) < 2:
                return fixed_sec * s
            i = np.searchsorted(a, ri)
            gap = (a[1]-a[0] if i == 0 else a[-1]-a[-2] if i >= len(a) else a[i]-a[i-1])
            return float(min(cap_ri, max(floor_ri, alpha * gap)))
        return fixed_sec * s
    return win


def match(cons, lib, mz_ppm, ri_win_fn, prefer_structured=True, score_mode="gated", composite_cap=3.0, adduct_penalty=0.5):
    rows = []
    for plat, g in cons.groupby("platform"):
        cand = lib.get(plat, [])
        if not cand:
            for _, f in g.iterrows():
                rows.append({**f, "match_ik14": "", "match_name": "", "n_cand": 0, "score": np.nan})
            continue
        cmz = np.array([c["mz"] for c in cand]); cri = np.array([c["ri"] for c in cand])
        cwin = np.array([ri_win_fn(plat, r) for r in cri])
        for _, f in g.iterrows():
            mz, ri = f["mz"], f["ri_norm"]; tol = mz * mz_ppm * 1e-6
            lo = np.searchsorted(cmz, mz - tol); hi = np.searchsorted(cmz, mz + tol)
            if score_mode == "composite":
                # soft: no hard RT gate (loose cap only); Euclidean distance across dims
                scored = []
                for i in range(lo, hi):
                    drt = abs(cri[i]-ri)/cwin[i]
                    if drt > composite_cap: continue
                    pen = 0.0 if cand[i].get("is_primary", True) else adduct_penalty
                    scored.append((((abs(cmz[i]-mz)/tol)**2 + drt**2)**0.5 + pen, i))
            else:
                scored = [(abs(cmz[i]-mz)/tol + abs(cri[i]-ri)/cwin[i]
                           + (0.0 if cand[i].get("is_primary", True) else adduct_penalty), i)
                          for i in range(lo, hi) if abs(cri[i]-ri) <= cwin[i]]
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


def score_against_gt(ann, gt_path, mz_ppm, ri_win_fn):
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
    # Adduct-aware, by InChIKey within the RI window (feature may sit at an adduct m/z).
    # recall: GT compounds recovered. precision: of calls TO a study compound, fraction
    # placed at the right RT (catches adduct/composite-induced mis-placement).
    tp = fn = 0; pc_ok = pc_tot = 0
    for plat, comps in gt.items():
        gt_ri = {}
        for mz, ri, gik in comps: gt_ri.setdefault(gik, ri)
        sub = ann[(ann.platform == plat) & (ann.match_ik14.astype(bool))]
        ari = sub.ri_norm.to_numpy(); aik = sub.match_ik14.to_numpy()
        for gik, ri in gt_ri.items():
            ri_win = ri_win_fn(plat, ri)
            if ((aik == gik) & (np.abs(ari - ri) <= ri_win)).any(): tp += 1
            else: fn += 1
        for k, rn in zip(aik, ari):           # precision: study-compound calls placed right?
            if k in gt_ri:
                pc_tot += 1
                if abs(rn - gt_ri[k]) <= ri_win_fn(plat, gt_ri[k]): pc_ok += 1
    R = tp/(tp+fn) if tp+fn else 0; P = pc_ok/pc_tot if pc_tot else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    print(f"[eval] recall {R:.3f} ({tp}/{tp+fn})  precision {P:.3f} ({pc_ok}/{pc_tot})  F1 {F1:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--anchors", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mz-ppm", type=float, default=20.0)
    ap.add_argument("--rt-win-sec", type=float, default=30.0,
                    help="RI match window in seconds (auto-scaled to RI per platform). "
                         "Optimal value depends on peak-RT PRECISION. With sharp/aligned RT "
                         "(idealised GT-centric query) ~15s is best — it separates ~30s-apart "
                         "isomers (beta-alanine/alanine) and lifts precision 0.84->0.88. But the "
                         "blind feature-centric pipeline has noisier CONSENSUS RI, so 15s drops "
                         "recall hard (feature-centric F1 0.51@15s vs 0.57@30s on ST004581; 67%% "
                         "of FNs are RI-shift). Default 30s is safe for consensus matching; "
                         "tighten toward 15s only once consensus-RT is sharpened (the real lever "
                         "for BOTH recall and isomer precision).")
    ap.add_argument("--rt-tol-sec", type=float, default=10.0,
                    help="consensus RI grouping, in seconds (auto-scaled)")
    ap.add_argument("--min-rep", type=int, default=2,
                    help="min injections a consensus peak must appear in. 1 recovers minor "
                         "real clusters (ST004581 full set: F1 0.565->0.596, recall +0.045, "
                         "precision held) at 3x consensus count — recommended for targeted "
                         "library matching; keep >=2 for open untargeted to suppress singletons.")
    ap.add_argument("--anchor-min-rep", type=int, default=0,
                    help="0 = auto (1 for injection-level alignment, 2 otherwise)")
    ap.add_argument("--align-level", choices=["injection", "batch", "platform"],
                    default="batch",
                    help="RT-alignment grain: detect the anchor ladder per injection "
                         "(finest; removes injection-to-injection drift before consensus -> "
                         "tighter consensus RI), per batch, or one per platform.")
    ap.add_argument("--batch-regex", default=r"(Set\d+)")
    ap.add_argument("--window-mode", choices=["fixed", "spacing", "precision"], default="fixed",
                    help="fixed: rt-win-sec everywhere. spacing: scale by local anchor gap. "
                         "precision: per-platform window from measured anchor RT reproducibility "
                         "(tight where peaks are reproducible e.g. pos-early ~1s; wide where noisy "
                         "e.g. neg). 'precision' is the recommended flexible mode.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="spacing: fraction of local anchor gap; precision: multiple of RT spread")
    ap.add_argument("--rt-floor-sec", type=float, default=10.0)
    ap.add_argument("--rt-cap-sec", type=float, default=60.0)
    ap.add_argument("--score-mode", choices=["gated", "composite"], default="gated",
                    help="gated: hard m/z AND RT windows. composite: no hard RT gate (loose "
                         "cap only), rank by Euclidean distance across normalized dims so a "
                         "strong m/z offsets a marginal RT. composite recovers just-outside-"
                         "window true compounds (ST004581: F1 0.687->0.728, recall +0.09).")
    ap.add_argument("--composite-cap", type=float, default=3.0,
                    help="composite: max RT deviation (multiples of the RI window) to consider")
    ap.add_argument("--no-densify-calibration", dest="densify_calibration", action="store_false",
                    help="disable bootstrap densification of the RI->sec ladder with mass-unique "
                         "library compounds (default on; lifts detectable GT 84%%->89%% on ST004581 "
                         "by translating MAF RI->RT accurately, vs the sparse kit-panel ladder).")
    ap.add_argument("--no-adducts", dest="adducts", action="store_false",
                    help="expand each library compound to its expected adduct ions ([M+Na]+, "
                         "[M+NH4]+, [M+FA-H]-, ...) so compounds that ionize as a non-primary "
                         "adduct still match. ST004581: ~47%% of primary-undetected compounds "
                         "co-elute as an adduct. Primary ion preferred via --adduct-penalty.")
    ap.add_argument("--adduct-penalty", type=float, default=0.5,
                    help="distance penalty for matching a non-primary adduct ion")
    ap.add_argument("--no-prefer-structured", action="store_true")
    ap.add_argument("--gt", default="")
    a = ap.parse_args()

    df = (pd.read_parquet(a.features) if a.features.endswith(".parquet")
          else pd.read_csv(a.features, sep="\t"))
    if "platform" not in df.columns:
        df["platform"] = df.source_file.str.split("_").str[0].map(DEFAULT_PREFIX_MAP)
    df = df.dropna(subset=["platform"])
    # alignment grain -> the 'batch' key the per-group ladder is fit on
    if a.align_level == "injection":
        df["batch"] = df.source_file
    elif a.align_level == "platform":
        df["batch"] = df.platform
    else:
        df["batch"] = df.source_file.str.extract(a.batch_regex, expand=False).fillna("all")
    amr = a.anchor_min_rep or (2 if a.align_level != "injection" else 1)
    print(f"features: {len(df)}  platforms {df.platform.nunique()}  "
          f"align={a.align_level} ({df.batch.nunique()} groups, anchor_min_rep={amr})", flush=True)

    lib = load_library(Path(a.library)); anchors = load_anchor_points(Path(a.anchors))
    # Bootstrap-densify the RI->sec ladder: union the kit-anchor seed with mass-unique
    # LIBRARY compounds (unambiguous by mass, so detecting their peak gives a reliable
    # (RI,sec) point — no GT). Kit seeds, library densifies. (build_rt_calibration logic.)
    if a.densify_calibration:
        cal_anchors = {p: list(pts) for p, pts in anchors.items()}
        for plat, entries in lib.items():
            mzs = np.array(sorted(c["mz"] for c in entries))
            for c in entries:
                tol = c["mz"] * a.mz_ppm * 1e-6
                if (np.searchsorted(mzs, c["mz"]+tol) - np.searchsorted(mzs, c["mz"]-tol)) == 1:
                    cal_anchors.setdefault(plat, []).append((c["mz"], c["ri"]))
        print(f"densified calibration anchors: { {p: len(v) for p, v in cal_anchors.items()} }", flush=True)
    else:
        cal_anchors = anchors
    if a.adducts:
        lib = expand_library_adducts(lib)
        print(f"adduct-expanded library: { {p: len(v) for p, v in lib.items()} }", flush=True)
    ladders, pooled, cov, pooled_pairs = build_batch_ladders(df, cal_anchors, a.mz_ppm, amr)
    slope = ri_per_sec(pooled_pairs)
    ri_tol = {p: a.rt_tol_sec * s for p, s in slope.items()}
    anchor_ris = {p: np.array(sorted(ri for _, ri in pts)) for p, pts in anchors.items()}
    spread = anchor_rt_spread(df, anchors, a.mz_ppm) if a.window_mode == "precision" else {}
    ri_win_fn = make_ri_win_fn(slope, anchor_ris, a.window_mode,
                               a.rt_win_sec, a.rt_floor_sec, a.alpha, a.rt_cap_sec, spread)
    wdesc = (f"precision(floor{a.rt_floor_sec}+{a.alpha}*spread, spread_s={ {p:round(v,1) for p,v in spread.items()} })"
             if a.window_mode == "precision" else
             f"spacing(alpha={a.alpha},[{a.rt_floor_sec},{a.rt_cap_sec}]s)" if a.window_mode == "spacing"
             else f"{a.rt_win_sec}s fixed")
    print(f"per-batch RI ladders: {len(ladders)} (anchor coverage med "
          f"{int(np.median(list(cov.values()))) if cov else 0}); RI/s slope: "
          f"{ {p: round(s,1) for p,s in slope.items()} }; window={wdesc}", flush=True)
    df = normalise_ri(df, ladders, pooled)
    print(f"features normalised to RI: {len(df)}", flush=True)
    cons = consensus_features(df, a.mz_ppm, ri_tol, a.min_rep)
    ann = match(cons, lib, a.mz_ppm, ri_win_fn, prefer_structured=not a.no_prefer_structured,
                score_mode=a.score_mode, composite_cap=a.composite_cap, adduct_penalty=a.adduct_penalty)
    print(f"consensus {len(cons)}  annotated {int(ann.match_ik14.astype(bool).sum())}", flush=True)
    ann.to_csv(a.out, index=False); print(f"-> {a.out}", flush=True)
    if a.gt:
        score_against_gt(ann, Path(a.gt), a.mz_ppm, ri_win_fn)


if __name__ == "__main__":
    main()

"""EIC-targeted annotation with per-compound permutation FDR — the precision-oriented
companion to library_match_rtri (the consensus matcher).

Why this exists: the consensus matcher inherits pure-python centWave's ~7s RT apex jitter,
which makes the RI dimension non-discriminating (an RI-shuffled decoy library matches as
well as the real one). Re-deriving each compound's RT from the RAW EIC apex recovers ~2s
precision (Metabolon-grade), at which a tight RT window makes RI discriminating again.

Method, per library compound:
  1. expected_sec = inverse_ladder(DD RI)               (the per-batch sec<->RI ladder)
  2. across N injections, is there an EIC peak > floor within +-window of expected_sec?
     -> presence count (0..N)
  3. local null: peak-presence rate at n_rand RANDOM RTs for the same m/z (the m/z-density
     floor). The DD RI is PREDICTED, not a true standard recording, so a call only counts
     if its peak is at the expected RT *more than chance for that mass region*.
  4. binomial p = P(presence >= observed | local null); BH q-value across compounds.
  5. annotate at --fdr.

Pure Python (pymzML). Reuses library_match_rtri for library load + sec->RI ladders.

  poetry run python scripts/st004581_benchmark/eic_annotate.py \
      --library DD.csv --features feat.parquet --anchors anchors.csv \
      --mzml-dir /path/to/mzml --out eic_annotations.csv [--gt MAF.csv] \
      --platforms "lc/ms neg" --n-inj 6 --window 5 --mz-ppm 7 --fdr 0.05
"""
import argparse, csv, glob, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_match_rtri as M

PLAT_METHOD = {"lc/ms pos early": "Method1", "lc/ms pos late": "Method2",
               "lc/ms neg": "Method3", "lc/ms polar": "Method4"}


def inverse_ladder(pooled_pairs_plat):
    """RI -> sec PCHIP from the pooled (sec, RI) anchor pairs of a platform."""
    agg = defaultdict(list)
    for sec, ri in pooled_pairs_plat:
        agg[round(ri, 1)].append(sec)
    xs = np.array(sorted(agg))
    ys = np.array([np.median(agg[x]) for x in xs])
    if len(xs) < 3:
        return None
    return PchipInterpolator(xs, ys, extrapolate=True)


def eic_pass(mzml, mzs, mz_ppm):
    """One pass over an mzML: return rt[T] and mat[T, n_mz] of summed intensity per m/z."""
    tol = mzs * mz_ppm * 1e-6
    rts, rows = [], []
    for spec in pymzml.run.Reader(mzml):
        if spec.ms_level != 1:
            continue
        smz = np.asarray(spec.mz); si = np.asarray(spec.i)
        rts.append(spec.scan_time_in_minutes() * 60)
        if not len(smz):
            rows.append(np.zeros(len(mzs))); continue
        lo = np.searchsorted(smz, mzs - tol); hi = np.searchsorted(smz, mzs + tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k] > lo[k] else 0.0
                              for k in range(len(mzs))]))
    return np.array(rts), np.array(rows)


def bh_qvalues(pval):
    """Benjamini-Hochberg q-values."""
    m = len(pval); order = np.argsort(pval); q = np.empty(m); cur = 1.0
    for r, i in enumerate(order[::-1]):
        rank = m - r
        cur = min(cur, pval[i] * m / rank); q[i] = cur
    return q


def annotate_platform(plat, lib0, inv, mzml_files, window, mz_ppm, floor, min_rep, n_rand, seed):
    comp = {}
    for c in lib0.get(plat, []):
        nm = M._norm(c["name"])
        if nm and nm not in comp:
            comp[nm] = (c["mz"], c["ri"], c["name"], c.get("ik14", ""))
    names = list(comp)
    if not names or inv is None or not mzml_files:
        return []
    mzc = np.array([comp[n][0] for n in names])
    esec = np.array([float(inv(comp[n][1])) for n in names])
    rng = np.random.RandomState(seed)
    N = len(mzml_files)
    present = np.zeros(len(names)); bg_hits = np.zeros(len(names)); bg_tot = np.zeros(len(names))
    for mp in mzml_files:
        rt, mat = eic_pass(mp, mzc, mz_ppm)
        if not len(rt):
            continue
        centers = rng.uniform(rt.min() + window, rt.max() - window, n_rand)
        for k in range(len(names)):
            col = mat[:, k]
            if col[np.abs(rt - esec[k]) <= window].max(initial=0) > floor:
                present[k] += 1
            for c0 in centers:
                bg_tot[k] += 1
                if col[np.abs(rt - c0) <= window].max(initial=0) > floor:
                    bg_hits[k] += 1
        print(f"    {Path(mp).name}", flush=True)
    b = np.clip(bg_hits / np.maximum(bg_tot, 1), 1e-6, 0.999)
    pval = binom.sf(present - 1, N, b)
    q = bh_qvalues(pval)
    out = []
    for k, nm in enumerate(names):
        out.append({"platform": plat, "name": comp[nm][2], "ik14": comp[nm][3],
                    "mz": round(comp[nm][0], 5), "ri": comp[nm][1],
                    "present": int(present[k]), "n_inj": N,
                    "local_null": round(float(b[k]), 3), "qvalue": round(float(q[k]), 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", required=True); ap.add_argument("--features", required=True)
    ap.add_argument("--anchors", required=True); ap.add_argument("--mzml-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gt", default="")
    ap.add_argument("--platforms", default="lc/ms neg")
    ap.add_argument("--n-inj", type=int, default=6)
    ap.add_argument("--sample-tag", default="COLU", help="mzML filename tag to use as samples")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--mz-ppm", type=float, default=7.0)
    ap.add_argument("--floor", type=float, default=5e4)
    ap.add_argument("--min-rep", type=int, default=2)
    ap.add_argument("--n-rand", type=int, default=40)
    ap.add_argument("--fdr", type=float, default=0.05)
    a = ap.parse_args()

    df = (pd.read_parquet(a.features) if a.features.endswith(".parquet")
          else pd.read_csv(a.features, sep="\t"))
    if "platform" not in df.columns:
        df["platform"] = df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
    df = df.dropna(subset=["platform"]); df["batch"] = df.source_file
    lib0 = M.load_library(Path(a.library)); anchors = M.load_anchor_points(Path(a.anchors))
    # densify the ladder anchors with mass-unique library compounds (same as the matcher)
    cal = {p: list(v) for p, v in anchors.items()}
    for plat, ent in lib0.items():
        mzs = np.array(sorted(c["mz"] for c in ent))
        for c in ent:
            tol = c["mz"] * a.mz_ppm * 1e-6
            if (np.searchsorted(mzs, c["mz"] + tol) - np.searchsorted(mzs, c["mz"] - tol)) == 1:
                cal.setdefault(plat, []).append((c["mz"], c["ri"]))
    _, _, _, pp = M.build_batch_ladders(df, cal, a.mz_ppm, 1, True)

    rows = []
    for plat in [p.strip() for p in a.platforms.split(",") if p.strip()]:
        meth = PLAT_METHOD.get(plat)
        files = sorted(glob.glob(f"{a.mzml_dir}/{meth}_*{a.sample_tag}*.mzML"))[:a.n_inj] if meth else []
        print(f"[{plat}] {len(files)} injections", flush=True)
        rows += annotate_platform(plat, lib0, inverse_ladder(pp.get(plat, [])), files,
                                  a.window, a.mz_ppm, a.floor, a.min_rep, a.n_rand, seed=0)
    rows.sort(key=lambda r: r["qvalue"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    called = [r for r in rows if r["qvalue"] < a.fdr and r["present"] >= a.min_rep]
    print(f"\nwrote {len(rows)} compounds -> {a.out}; called at FDR<{a.fdr} (>= {a.min_rep} inj): {len(called)}", flush=True)

    if a.gt:
        maf = defaultdict(set)
        for r in csv.DictReader(open(a.gt)):
            if r.get("unannotatable", "") == "true":
                continue
            p = (r.get("platform") or "").strip().lower(); nm = M._norm(r.get("name") or "")
            if nm:
                maf[p].add(nm)
        n_maf = sum(len(maf[p]) for p in {r["platform"] for r in rows})
        print(f"\n{'FDR<':>6}{'called':>8}{'TP':>6}{'precision':>11}{'recall':>9}")
        for lvl in [0.01, 0.05, 0.10, 0.20]:
            keep = [r for r in rows if r["qvalue"] < lvl and r["present"] >= a.min_rep]
            tp = sum(1 for r in keep if M._norm(r["name"]) in maf.get(r["platform"], set()))
            c = len(keep)
            print(f"{lvl:>6.2f}{c:>8}{tp:>6}{(tp/c if c else 0):>11.3f}{(tp/n_maf if n_maf else 0):>9.3f}")


if __name__ == "__main__":
    main()

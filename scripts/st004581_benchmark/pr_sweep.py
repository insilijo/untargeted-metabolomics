"""Full precision/recall sweep for the RI->MS1 matcher on the ST004581 neg subset.

Exposes the operating curve the single-point [eval] line hides:
  * recall over ALL GT compounds (denom = #GT)
  * identity-restricted precision (pc_ok/pc_tot) — the matcher's own number:
    of calls whose ik14 IS a GT compound, fraction placed at the right RT.
    HIGH BY CONSTRUCTION — it ignores calls to non-GT DD compounds.
  * the hidden counts: total calls, calls-to-GT (the precision denom), and
    calls-to-non-GT (open-world: a DD compound not in this study's curated GT;
    we can't label these right/wrong, but their VOLUME is the honesty check).
  * retrievability ceiling: #GT compounds that even have a candidate ion in the
    (adduct-expanded) library — recall can never exceed this.

Sweeps the RI window (fixed sec) and reports baseline vs dual-RT side by side,
reusing library_match_rtri's exact pipeline (ladders, consensus, match, eval).
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_match_rtri as M
import pandas as pd

DD   = "/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT   = "/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SMI  = "/tmp/dd_pubchem_smiles.csv"
FEAT = "/tmp/feat_colu.parquet"
ANC  = "/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"
MZ_PPM = 20.0; MIN_REP = 1; WINDOWS = [10, 15, 20, 30, 45, 60, 90, 120]


def load_gt():
    gt = defaultdict(dict)
    with open(GT) as f:
        for r in csv.DictReader(f):
            if r.get("unannotatable", "") == "true":
                continue
            plat = (r.get("platform") or "").strip().lower()
            try:
                ri = float(r["rt"])
            except (ValueError, KeyError):
                continue
            gt[plat].setdefault(M.ik14(r["inchikey"]), ri)
    return gt


def full_score(ann, gt, ri_win_fn, qsrr_by_ik):
    """Return dict of the full count breakdown for one annotation set."""
    tp = fn = pc_ok = pc_tot = 0
    n_calls = n_call_gt = n_call_nongt = 0
    for plat, gt_ri in gt.items():
        sub = ann[(ann.platform == plat) & (ann.match_ik14.astype(bool))]
        ari = sub.ri_norm.to_numpy(); aik = sub.match_ik14.to_numpy()
        n_calls += len(aik)
        def near(rn, gik, ri):
            w = ri_win_fn(plat, ri)
            if abs(rn - ri) <= w:
                return True
            q = qsrr_by_ik.get(gik)
            return q is not None and abs(rn - q) <= w
        for gik, ri in gt_ri.items():
            sel = np.where(aik == gik)[0]
            if any(near(ari[i], gik, ri) for i in sel):
                tp += 1
            else:
                fn += 1
        for i in range(len(aik)):
            k = aik[i]
            if k in gt_ri:
                n_call_gt += 1; pc_tot += 1
                if near(ari[i], k, gt_ri[k]):
                    pc_ok += 1
            else:
                n_call_nongt += 1
    R = tp/(tp+fn) if tp+fn else 0.0
    P = pc_ok/pc_tot if pc_tot else 0.0
    F1 = 2*P*R/(P+R) if P+R else 0.0
    return dict(R=R, P=P, F1=F1, tp=tp, ngt=tp+fn, pc_ok=pc_ok, pc_tot=pc_tot,
                n_calls=n_calls, n_call_gt=n_call_gt, n_call_nongt=n_call_nongt)


def main():
    gt = load_gt()
    ngt = sum(len(v) for v in gt.values())
    # --- pipeline (mirrors library_match_rtri.main) ---
    df = pd.read_parquet(FEAT)
    if "platform" not in df.columns:
        df["platform"] = df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
    df = df.dropna(subset=["platform"])
    df["batch"] = df.source_file.str.extract(r"(Set\d+)", expand=False).fillna("all")
    lib = M.load_library(Path(DD))
    anchors = M.load_anchor_points(Path(ANC))
    lib = M.expand_library_adducts(lib)
    ladders, pooled, cov, pooled_pairs = M.build_batch_ladders(df, anchors, MZ_PPM, 2)
    slope = M.ri_per_sec(pooled_pairs)
    ri_tol = {p: 10.0 * s for p, s in slope.items()}
    anchor_ris = {p: np.array(sorted(ri for _, ri in pts)) for p, pts in anchors.items()}
    df = M.normalise_ri(df, ladders, pooled)
    cons = M.consensus_features(df, MZ_PPM, ri_tol, MIN_REP)

    # retrievability ceiling: GT compounds with >=1 adduct ion within m/z tol of some feature
    ceiling = 0
    for plat, gt_ri in gt.items():
        cand = lib.get(plat, [])
        cmz = np.array(sorted(c["mz"] for c in cand)) if cand else np.array([])
        cik = set(c["ik14"] for c in cand)
        for gik in gt_ri:
            if gik in cik:
                ceiling += 1
    print(f"GT compounds: {ngt}  |  in-library (any adduct ion): {ceiling}  "
          f"(retrievability ceiling recall <= {ceiling/ngt:.3f})\n", flush=True)

    # dual-RT: attach QSRR ri once (window-independent), park it in '_q' so base
    # runs see no ri_qsrr and dual runs restore it.
    smi = M.load_smiles_map(Path(SMI))
    qmodels = M.fit_qsrr_ri(cons, lib, smi, MZ_PPM)
    M.attach_qsrr_ri(lib, qmodels, smi)
    qsrr_by_ik = {}
    for cand in lib.values():
        for c in cand:
            c["_q"] = c.pop("ri_qsrr", None)
            if c["_q"] is not None:
                qsrr_by_ik.setdefault(c["ik14"], c["_q"])
    print(f"dual-RT QSRR ik14 coverage: {len(qsrr_by_ik)}\n", flush=True)

    hdr = (f"{'win_s':>5} {'mode':>8} | {'recall':>6} {'prec*':>6} {'F1':>6} | "
           f"{'tp':>4}/{ngt:<4} | {'calls':>5} {'->GT':>5} {'->nonGT':>7} {'%nonGT':>6}")
    print(hdr); print("-"*len(hdr), flush=True)
    for sm in ["gated", "composite"]:
        print(f"\n### score-mode = {sm}", flush=True)
        for win in WINDOWS:
            ri_win_fn = M.make_ri_win_fn(slope, anchor_ris, "fixed", float(win), 10.0, 0.5, 60.0, {})
            for mode, qmap in [("base", {}), ("dual-rt", qsrr_by_ik)]:
                for cand in lib.values():
                    for c in cand:
                        if mode == "dual-rt":
                            c["ri_qsrr"] = c["_q"]
                        else:
                            c.pop("ri_qsrr", None)
                ann = M.match(cons, lib, MZ_PPM, ri_win_fn, prefer_structured=True, score_mode=sm)
                r = full_score(ann, gt, ri_win_fn, qmap)
                pct = 100*r["n_call_nongt"]/r["n_calls"] if r["n_calls"] else 0
                print(f"{win:>5} {mode:>8} | {r['R']:>6.3f} {r['P']:>6.3f} {r['F1']:>6.3f} | "
                      f"{r['tp']:>4}/{r['ngt']:<4} | {r['n_calls']:>5} {r['n_call_gt']:>5} "
                      f"{r['n_call_nongt']:>7} {pct:>5.0f}%", flush=True)
    print("\n* prec = identity-restricted (pc_ok/pc_tot): of calls to a GT compound, "
          "fraction at right RT. Calls to non-GT DD compounds are NOT counted as FP "
          "(open-world: can't verify). '%nonGT' = share of all calls that are unverifiable.")


if __name__ == "__main__":
    main()

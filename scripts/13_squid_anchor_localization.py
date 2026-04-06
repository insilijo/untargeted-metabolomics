"""Step 13: SQuID-InC anchor-guided localization, embedding, and clustering.

Uses the selected anchor compounds to:
  1. Localize ALL features (annotated + unannotated) via Leiden community
     message passing — annotation confidence is a continuous axis, not a
     hard split.  A library match with cosine=0.65 is uncertain; it gets
     embedded alongside unknowns with its confidence score intact.
  2. Build a joint embedding (m/z + RT + MS2 spectral fingerprint + chemical
     fingerprint of best candidate + graph message prior) and project to 2D
     via UMAP for visualisation and nD for clustering.
  3. Write localization + embedding results and benchmark vs. baselines.

Outputs
-------
  processed/squid_features.parquet
      All features (annotated + unannotated) with UMAP coords, community ID,
      message prior, best candidate, and annotation confidence.

  processed/squid_community_summary.tsv
      Per-Leiden-community: size, n_anchors, coverage fraction, blind flag.

  processed/squid_benchmark.csv
      GT coverage and annotation lift vs. random/frequency/diversity baselines.

Config (config.yaml squid section)
-----------------------------------
  squid:
    anchor_set: results/anchors_balanced.csv  # relative to SQUID_INC_ROOT
    graph_dir:  data/processed/graph/ready    # relative to SQUID_INC_ROOT
    min_cosine: 0.7
    mz_tolerance_da: 0.01
    leiden_resolution: 1.0
    cross_community_damping: 0.5
    umap_n_neighbors: 15
    umap_min_dist: 0.1
    umap_n_components_cluster: 10
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config

# ---------------------------------------------------------------------------
# Locate SQuID-InC root
# ---------------------------------------------------------------------------

def _squid_root() -> Path:
    env = os.environ.get("SQUID_INC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[2]
    for candidate in [here / "SQuID-INC", here.parent / "SQuID-INC"]:
        if (candidate / "squid_inc").is_dir():
            return candidate
    raise EnvironmentError(
        "SQuID-InC root not found. Set SQUID_INC_ROOT environment variable."
    )


squid_root = _squid_root()
if str(squid_root) not in sys.path:
    sys.path.insert(0, str(squid_root))

from squid_inc.benchmark.calibrate import CoordinateCalibrator, build_calibration_points
from squid_inc.benchmark.coverage import (
    compute_coverage,
    diversity_only,
    frequency_top_n,
    random_anchor_sample,
)
from squid_inc.benchmark.embed import (
    EmbeddingConfig,
    FeatureRecord,
    embedding_to_dataframe,
    ms2_fingerprint,
    project_umap,
)
from squid_inc.benchmark.ground_truth import estimate_rt_scale_factor
from squid_inc.benchmark.ingest import load_pipeline_outputs
from squid_inc.features.fingerprints import fingerprint_smiles
from squid_inc.features.predict import predict_coordinates
from squid_inc.graph.communities import (
    apply_observation_vector,
    assign_anchors_to_communities,
    detect_communities,
)
from squid_inc.inference.localize import build_localization_state, localize_feature
from squid_inc.io import read_csv_rows, write_csv_rows


def _load_ms2_lookup(pipeline_dir: Path) -> dict[str, dict]:
    """Build a (source_file, precursor_mz_rounded) → MS2 spectrum lookup."""
    ms2_path = pipeline_dir / "data" / "interim" / "ms2_spectra.parquet"
    if not ms2_path.exists():
        return {}
    df = pd.read_parquet(ms2_path)
    lookup: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        key = (row.get("source_file", ""), round(float(row.get("precursor_mz") or 0), 3))
        if key not in lookup:
            lookup[key] = {
                "mz_array": list(row.get("mz_array") or []),
                "intensity_array": list(row.get("intensity_array") or []),
            }
    return lookup


def _best_ms2(mz: float, ms2_lookup: dict, mz_tol: float = 0.02) -> tuple[list, list]:
    """Find the closest MS2 spectrum for a given precursor m/z."""
    best_key = None
    best_dist = float("inf")
    for (_, pmz), _ in ms2_lookup.items():
        d = abs(pmz - mz)
        if d < best_dist and d <= mz_tol:
            best_dist = d
            best_key = (_, pmz)
    if best_key:
        s = ms2_lookup[best_key]
        return s["mz_array"], s["intensity_array"]
    return [], []


def main() -> None:
    cfg = load_config()
    pipeline_dir = Path(__file__).resolve().parents[1]
    processed_dir = pipeline_dir / Path(cfg["paths"]["processed_dir"])
    ensure_dirs([processed_dir])

    squid_cfg = cfg.get("squid", {})
    mz_tol = float(squid_cfg.get("mz_tolerance_da", cfg["library_search"]["mz_tolerance_da"]))
    min_cosine = float(squid_cfg.get("min_cosine", cfg["library_search"]["min_cosine"]))
    leiden_res = float(squid_cfg.get("leiden_resolution", 1.0))
    damping = float(squid_cfg.get("cross_community_damping", 0.5))

    # ── Anchor set + graph ────────────────────────────────────────────────────
    anchor_path = squid_root / squid_cfg.get("anchor_set", "results/anchors_balanced.csv")
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor set not found: {anchor_path}")
    anchor_rows = read_csv_rows(anchor_path)
    anchor_ids = {r.get("compound_id", "") for r in anchor_rows}
    print(f"Anchors: {len(anchor_rows)} from {anchor_path.name}")

    graph_dir = squid_root / squid_cfg.get("graph_dir", "data/processed/graph/ready")
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph not found: {graph_dir}")

    _CHEM_FP_BITS = 512   # must match EmbeddingConfig.chem_fp_bits

    universe_csv = squid_root / "data/processed/compound_universe.csv"
    universe_rows = read_csv_rows(universe_csv) if universe_csv.exists() else anchor_rows
    universe_with_coords = [
        {**r, "predicted_mz": predict_coordinates(r).get("ms1", 0.0)}
        for r in universe_rows
    ]
    # Pre-computed fingerprint column (comma-separated bits), padded to 512
    chem_fp_by_id: dict[str, list[int]] = {}
    for r in universe_rows:
        cid = r.get("compound_id", "")
        raw = [int(b) for b in (r.get("fingerprint") or "").split(",") if b]
        if raw:
            if len(raw) < _CHEM_FP_BITS:
                raw += [0] * (_CHEM_FP_BITS - len(raw))
            chem_fp_by_id[cid] = raw[:_CHEM_FP_BITS]

    # ── Pipeline outputs ──────────────────────────────────────────────────────
    print("Loading pipeline outputs …")
    annot_csv_path = squid_cfg.get("annotation_csv")
    annotation_csv = (pipeline_dir / annot_csv_path) if annot_csv_path else None
    if annotation_csv and not annotation_csv.exists():
        raise FileNotFoundError(f"annotation_csv not found: {annotation_csv}")

    ground_truth, unannotated_dicts, observed_intensities, annot_ms2 = load_pipeline_outputs(
        pipeline_dir, min_cosine=min_cosine, annotation_csv=annotation_csv
    )
    ms2_lookup = _load_ms2_lookup(pipeline_dir)
    # Merge annotation-CSV MS2 spectra (keyed by compound_id) into the lookup
    for cid, (mzs, ints) in annot_ms2.items():
        key = ("annotation_csv", round(
            next((gt.measured_mz for gt in ground_truth if gt.compound_id == cid), 0.0), 3
        ))
        if key not in ms2_lookup:
            ms2_lookup[key] = {"mz_array": mzs, "intensity_array": ints}
    print(
        f"  Annotated: {len(ground_truth)}  Unannotated: {len(unannotated_dicts)}  "
        f"MS2 spectra: {len(ms2_lookup)}"
        + (f"  (annotation CSV: {annotation_csv.name})" if annotation_csv else "")
    )

    # ── Fingerprints from annotation CSV SMILES ───────────────────────────────
    # Compounds in the annotation CSV may not be in compound_universe.csv.
    # Compute Morgan fingerprints on the fly from SMILES so the embedding
    # chemical-fingerprint dimension is populated for annotated features.
    n_computed = 0
    for gt in ground_truth:
        if gt.compound_id in chem_fp_by_id:
            continue
        smiles = (gt.extra or {}).get("smiles", "")
        if not smiles and gt.inchikey:
            # Try universe_rows by inchikey as a fallback
            for r in universe_rows:
                if r.get("inchikey", "") == gt.inchikey:
                    smiles = r.get("smiles", "")
                    break
        if smiles:
            fp = fingerprint_smiles(smiles, n_bits=_CHEM_FP_BITS)
            chem_fp_by_id[gt.compound_id] = fp
            n_computed += 1
    if n_computed:
        print(f"  Computed {n_computed} fingerprints from annotation CSV SMILES")

    # ── Leiden community detection ────────────────────────────────────────────
    print("Running Leiden community detection …")
    communities = detect_communities(graph_dir, resolution=leiden_res)
    print(f"  {len(communities)} communities  "
          f"(largest: {communities[0].size if communities else 0})")

    # ── RT calibration + message passing ─────────────────────────────────────
    predicted_coords = [
        {"inchikey": r.get("inchikey", ""), "rt": predict_coordinates(r).get("rt")}
        for r in anchor_rows
    ]
    rt_scale = estimate_rt_scale_factor(ground_truth, predicted_coords)
    cal_points = build_calibration_points(anchor_rows, ground_truth, rt_scale_factor=rt_scale)
    print(f"  RT scale: {rt_scale:.3f}  calibration points: {len(cal_points)}")

    gt_inchikeys = {gt.inchikey for gt in ground_truth if gt.inchikey}
    present_ids = {r.get("compound_id", "") for r in anchor_rows
                   if r.get("inchikey", "") in gt_inchikeys}
    absent_ids = anchor_ids - present_ids

    local_communities = copy.deepcopy(communities)
    assign_anchors_to_communities(local_communities, anchor_ids)
    apply_observation_vector(local_communities, present_ids, absent_ids)
    blind = sum(1 for c in local_communities if c.is_blind)
    print(f"  Blind communities: {blind}/{len(local_communities)}")

    print("Running message passing …")
    state = build_localization_state(
        graph_dir, local_communities, present_ids, absent_ids,
        cross_community_damping=damping,
    )

    # ── Build FeatureRecord list for ALL features ─────────────────────────────
    # Annotated features: from ground_truth with cosine score
    # Unannotated features: from unannotated_dicts with cosine=0
    all_features: list[FeatureRecord] = []

    print("Building feature records …")

    # Annotated
    for gt in ground_truth:
        ms2_mz, ms2_int = _best_ms2(gt.measured_mz, ms2_lookup, mz_tol)
        candidates = [
            c for c in universe_with_coords
            if abs(float(c.get("predicted_mz", 0.0)) - gt.measured_mz) <= mz_tol * 3
        ]
        loc = localize_feature(
            {"mz": gt.measured_mz, "rt": gt.measured_rt},
            state, candidates, mz_tolerance=mz_tol * 3,
        )
        rec = FeatureRecord(
            feature_id=gt.compound_id,
            mz=gt.measured_mz,
            rt=gt.measured_rt,
            intensity=float(observed_intensities.get(gt.inchikey, 0.5)),
            inchikey=gt.inchikey,
            annotation_name=gt.name,
            cosine_score=float(gt.extra.get("cosine", 1.0)) if gt.extra else 1.0,
            ms2_mz=ms2_mz,
            ms2_intensity=ms2_int,
        )
        if loc:
            rec.candidate_compound_id = loc.best_candidate_id
            rec.chemical_fingerprint = chem_fp_by_id.get(loc.best_candidate_id, [])
            rec.message_prior = loc.message_prior
            rec.community_id = loc.best_candidate_community
        all_features.append(rec)

    # Unannotated
    for i, feat in enumerate(unannotated_dicts):
        ms2_mz, ms2_int = _best_ms2(feat["mz"], ms2_lookup, mz_tol)
        candidates = [
            c for c in universe_with_coords
            if abs(float(c.get("predicted_mz", 0.0)) - feat["mz"]) <= mz_tol * 3
        ]
        loc = localize_feature(feat, state, candidates, mz_tolerance=mz_tol * 3)
        rec = FeatureRecord(
            feature_id=feat.get("feature_id", f"unk_{i}"),
            mz=feat["mz"],
            rt=feat.get("rt"),
            intensity=float(feat.get("intensity", 0.0)),
            cosine_score=0.0,
            ms2_mz=ms2_mz,
            ms2_intensity=ms2_int,
        )
        if loc:
            rec.candidate_compound_id = loc.best_candidate_id
            rec.chemical_fingerprint = chem_fp_by_id.get(loc.best_candidate_id, [])
            rec.message_prior = loc.message_prior
            rec.community_id = loc.best_candidate_community
        all_features.append(rec)

    print(f"  Total features for embedding: {len(all_features)} "
          f"({sum(f.is_annotated for f in all_features)} annotated, "
          f"{sum(not f.is_annotated for f in all_features)} unannotated)")

    # ── UMAP embedding ────────────────────────────────────────────────────────
    print("Building UMAP embedding …")
    embed_cfg = EmbeddingConfig()
    try:
        result = project_umap(
            all_features,
            config=embed_cfg,
            n_neighbors=int(squid_cfg.get("umap_n_neighbors", 15)),
            min_dist=float(squid_cfg.get("umap_min_dist", 0.1)),
            n_components_cluster=int(squid_cfg.get("umap_n_components_cluster", 10)),
        )
        feat_df = embedding_to_dataframe(result)
        print(f"  Embedding shape: {result.X_full.shape}  → 2D + "
              f"{result.X_nd.shape[1] if result.X_nd is not None else 0}D")
    except ImportError:
        print("  WARN: umap-learn not installed — skipping UMAP (pip install umap-learn)")
        feat_df = pd.DataFrame([
            {
                "feature_id": f.feature_id,
                "mz": f.mz,
                "rt": f.rt,
                "intensity": f.intensity,
                "umap_x": None,
                "umap_y": None,
                "is_annotated": f.is_annotated,
                "annotation_confidence": f.annotation_confidence,
                "inchikey": f.inchikey,
                "annotation_name": f.annotation_name,
                "cosine_score": f.cosine_score,
                "message_prior": f.message_prior,
                "community_id": f.community_id,
                "candidate_compound_id": f.candidate_compound_id,
            }
            for f in all_features
        ])

    out_path = processed_dir / "squid_features.parquet"
    feat_df.to_parquet(out_path, index=False)
    print(f"  → {out_path}")

    # ── Community summary ─────────────────────────────────────────────────────
    comm_df = pd.DataFrame([
        {
            "community_id": c.community_id,
            "size": c.size,
            "n_anchors": c.n_anchors,
            "n_present": len(c.anchor_present),
            "n_absent": len(c.anchor_absent),
            "coverage": round(c.coverage, 4),
            "is_blind": c.is_blind,
            "anchor_ids": ";".join(c.anchor_ids),
        }
        for c in local_communities
    ])
    comm_df.to_csv(processed_dir / "squid_community_summary.tsv", sep="\t", index=False)
    print(f"  → {processed_dir / 'squid_community_summary.tsv'}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    calibrator = CoordinateCalibrator().fit(cal_points)
    calibrated_gt = [calibrator.calibrate({"inchikey": gt.inchikey,
                                            "compound_id": gt.compound_id,
                                            "smiles": ""})
                     for gt in ground_truth]
    calibrated_unannotated = [calibrator.calibrate({"compound_id": f"feat_{i}",
                                                      "smiles": "", **feat})
                               for i, feat in enumerate(unannotated_dicts)]

    benchmark_rows = []
    for bname, banchors in [
        (anchor_path.stem,         anchor_rows),
        ("baseline_random",        random_anchor_sample(universe_rows, len(anchor_rows))),
        ("baseline_frequency",     frequency_top_n(universe_rows, len(anchor_rows))),
        ("baseline_diversity",     diversity_only(universe_rows, len(anchor_rows))),
    ]:
        bcal = build_calibration_points(banchors, ground_truth, rt_scale_factor=rt_scale)
        cov = compute_coverage(
            bname, banchors, ground_truth, bcal,
            calibrated_gt, unannotated_dicts, calibrated_unannotated,
            mz_tolerance=mz_tol,
        )
        benchmark_rows.append({
            "anchor_set": bname,
            "n_anchors": cov.n_anchors,
            "n_matched_gt": cov.n_anchors_matched_to_gt,
            "gt_coverage_mz": f"{cov.gt_coverage_mz:.4f}",
            "n_lifted_mz": cov.n_lifted_mz,
            "lift_fraction_mz": f"{cov.lift_fraction_mz:.4f}",
            "mz_rmse_raw": f"{cov.mz_rmse_raw:.6f}",
            "mz_rmse_calibrated": f"{cov.mz_rmse_calibrated:.6f}",
        })
        print(f"  {bname:<30} coverage={cov.gt_coverage_mz:.1%}  "
              f"lift={cov.n_lifted_mz}  rmse={cov.mz_rmse_raw:.5f}")

    write_csv_rows(
        processed_dir / "squid_benchmark.csv",
        benchmark_rows,
        list(benchmark_rows[0].keys()),
    )
    print(f"  → {processed_dir / 'squid_benchmark.csv'}")
    print("\nStep 13 complete.")


if __name__ == "__main__":
    main()

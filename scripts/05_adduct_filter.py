from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_config


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    # Compute Pearson correlation with guard rails.
    if a.size != b.size:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    # Remove adduct/isotope duplicates based on m/z, RT, and correlation.
    cfg = load_config()
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs([processed_dir])

    filtered_path = processed_dir / "feature_groups_filtered.tsv"
    groups_path = interim_dir / "feature_groups.tsv"
    wide_path = interim_dir / "feature_groups_wide.tsv"

    if not filtered_path.exists():
        raise SystemExit("Missing feature_groups_filtered.tsv. Run 04_blank_subtract.py first.")
    if not groups_path.exists() or not wide_path.exists():
        raise SystemExit("Missing feature_groups tables. Run 03_align_features.py first.")

    filtered = pd.read_csv(filtered_path, sep="\t")
    groups = pd.read_csv(groups_path, sep="\t")
    wide = pd.read_csv(wide_path, sep="\t")

    mix_cols = [c for c in wide.columns if c.startswith("MIX_")]
    if not mix_cols:
        raise SystemExit("No MIX columns found in feature_groups_wide.tsv")

    merged = filtered.merge(
        groups[["group_id", "mz_mean", "rt_mean", "max_mix_intensity", "mix_file_count"]],
        on="group_id",
        how="left",
        suffixes=("", "_group"),
    ).merge(wide, on="group_id", how="left")

    # Only keep features seen in all mix replicates, unless marked as known.
    if "known_match" in merged.columns:
        merged = merged[(merged["mix_file_count"] == 3) | (merged["known_match"] == True)].reset_index(drop=True)
    else:
        merged = merged[merged["mix_file_count"] == 3].reset_index(drop=True)

    if merged.empty:
        raise SystemExit("No features with mix_file_count=3 to filter.")

    arrays = merged[mix_cols].to_numpy(dtype=float)

    adduct_cfg = cfg["adduct_filter"]
    corr_thresh = float(adduct_cfg["corr_threshold"])
    rt_tol = float(adduct_cfg["rt_tolerance_sec"])
    mz_tol = float(adduct_cfg["mz_tolerance_da"])
    adducts = adduct_cfg["adduct_deltas"]

    to_remove = set()
    pairs = []
    known_ids = set()
    if "known_match" in merged.columns:
        known_ids = set(merged.loc[merged["known_match"] == True, "group_id"].astype(int))

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            rt_diff = abs(merged.loc[i, "rt_mean"] - merged.loc[j, "rt_mean"])
            if rt_diff > rt_tol:
                continue
            mz_diff = abs(merged.loc[i, "mz_mean"] - merged.loc[j, "mz_mean"])

            hit = None
            for adduct in adducts:
                if abs(mz_diff - adduct["delta_mz"]) <= mz_tol:
                    hit = adduct["label"]
                    break
            if hit is None:
                continue

            corr = pearson_corr(arrays[i], arrays[j])
            if not np.isfinite(corr) or corr < corr_thresh:
                continue

            # Remove the lower-intensity feature
            intensity_i = merged.loc[i, "max_mix_intensity"]
            intensity_j = merged.loc[j, "max_mix_intensity"]
            if intensity_i >= intensity_j:
                drop_id = int(merged.loc[j, "group_id"])
                keep_id = int(merged.loc[i, "group_id"])
            else:
                drop_id = int(merged.loc[i, "group_id"])
                keep_id = int(merged.loc[j, "group_id"])

            if drop_id in known_ids or keep_id in known_ids:
                continue

            to_remove.add(drop_id)
            pairs.append(
                {
                    "group_id_keep": keep_id,
                    "group_id_drop": drop_id,
                    "delta_mz": mz_diff,
                    "delta_rt": rt_diff,
                    "corr": corr,
                    "type": hit,
                }
            )

    filtered_out = merged[~merged["group_id"].isin(to_remove)].copy()
    out_path = processed_dir / "feature_groups_filtered_adduct.tsv"
    filtered_out.to_csv(out_path, sep="\t", index=False)

    pairs_path = processed_dir / "adduct_pairs.tsv"
    pd.DataFrame(pairs).to_csv(pairs_path, sep="\t", index=False)

    print(f"Removed {len(to_remove)} features as adduct/isotope candidates.")
    print(f"Wrote {out_path}")
    print(f"Wrote {pairs_path}")


if __name__ == "__main__":
    main()

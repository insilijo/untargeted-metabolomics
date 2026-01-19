from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_config


def ppm_to_da(mz: float, ppm: float) -> float:
    return mz * ppm * 1e-6


def main() -> None:
    cfg = load_config()
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    ms2_path = interim_dir / "ms2_spectra.parquet"
    features_path = interim_dir / "features_grouped.tsv"
    filtered_path = Path(cfg["paths"]["processed_dir"]) / "feature_groups_filtered_adduct.tsv"
    if not ms2_path.exists():
        raise SystemExit("Missing ms2_spectra.parquet. Run 06_extract_ms2.py first.")
    if not features_path.exists():
        raise SystemExit("Missing features_grouped.tsv. Run 03_align_features.py first.")

    ms2 = pd.read_parquet(ms2_path)
    features = pd.read_csv(features_path, sep="\t")
    if filtered_path.exists():
        filtered = pd.read_csv(filtered_path, sep="\t")
        features = features[features["group_id"].isin(filtered["group_id"])]

    link_cfg = cfg["ms2_linking"]
    mz_tol_ppm = float(link_cfg["mz_tolerance_ppm"])
    rt_tol = float(link_cfg["rt_tolerance_sec"])

    links = []
    for source_file, ms2_group in ms2.groupby("source_file"):
        feats = features[features["source_file"] == source_file]
        if feats.empty:
            continue

        feat_mz = feats["mz"].to_numpy()
        feat_rt = feats["rt"].to_numpy()

        for idx, row in ms2_group.iterrows():
            prec = row["precursor_mz"]
            if pd.isna(prec):
                continue
            mz_tol = ppm_to_da(float(prec), mz_tol_ppm)

            mz_diff = np.abs(feat_mz - prec)
            rt_diff = np.abs(feat_rt - row["rt"])
            mask = (mz_diff <= mz_tol) & (rt_diff <= rt_tol)
            if not np.any(mask):
                continue

            candidate_idx = np.where(mask)[0]
            best = candidate_idx[np.lexsort((rt_diff[candidate_idx], mz_diff[candidate_idx]))][0]
            feat_row = feats.iloc[best]

            links.append(
                {
                    "ms2_index": idx,
                    "source_file": source_file,
                    "ms2_rt": row["rt"],
                    "precursor_mz": prec,
                    "feature_id": feat_row["feature_id"],
                    "group_id": feat_row["group_id"],
                    "feature_mz": feat_row["mz"],
                    "feature_rt": feat_row["rt"],
                    "feature_intensity": feat_row["intensity"],
                    "delta_mz": float(mz_diff[best]),
                    "delta_rt": float(rt_diff[best]),
                }
            )

    links_df = pd.DataFrame(links)
    out_path = interim_dir / "ms2_feature_links.parquet"
    links_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

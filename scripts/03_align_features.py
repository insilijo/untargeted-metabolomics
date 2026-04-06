from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config, load_sample_metadata, _infer_type_from_name


def ppm_to_da(mz: float, ppm: float) -> float:
    # Convert a ppm tolerance into Da at a given m/z.
    return mz * ppm * 1e-6


def assign_groups(df: pd.DataFrame, mz_tol_ppm: float, rt_tol_sec: float) -> pd.Series:
    # Greedily group features by m/z and RT tolerances.
    df_sorted = df.sort_values(["mz", "rt"]).copy()
    group_ids = [-1] * len(df_sorted)
    groups = []

    for pos, (_, row) in enumerate(df_sorted.iterrows()):
        mz = float(row["mz"])
        rt = float(row["rt"])
        mz_tol = ppm_to_da(mz, mz_tol_ppm)

        assigned = False
        for g in reversed(groups):
            if g["mz_mean"] < mz - mz_tol:
                break
            if abs(g["mz_mean"] - mz) <= mz_tol and abs(g["rt_mean"] - rt) <= rt_tol_sec:
                group_ids[pos] = g["group_id"]
                g["count"] += 1
                g["mz_mean"] += (mz - g["mz_mean"]) / g["count"]
                g["rt_mean"] += (rt - g["rt_mean"]) / g["count"]
                assigned = True
                break

        if not assigned:
            group_id = len(groups) + 1
            groups.append({"group_id": group_id, "mz_mean": mz, "rt_mean": rt, "count": 1})
            group_ids[pos] = group_id

    df_sorted["group_id"] = group_ids
    return df_sorted["group_id"].reindex(df.index).values


def main() -> None:
    # Assign groups and write long/summary/wide alignment tables.
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    features_path = interim_dir / "features.tsv"
    if not features_path.exists():
        raise SystemExit("Missing features.tsv. Run 02_feature_finding.py first.")

    df = pd.read_csv(features_path, sep="\t")

    # Build filename → sample_type lookup from metadata (falls back to name inference)
    _, meta = load_sample_metadata(raw_dir, interim_dir, cfg)
    if meta is not None:
        type_map = dict(zip(meta["filename"], meta["sample_type"]))
        df["sample_type"] = df["source_file"].map(type_map).fillna(
            df["source_file"].apply(_infer_type_from_name)
        )
    else:
        df["sample_type"] = df["source_file"].apply(_infer_type_from_name)

    align_cfg = cfg["feature_alignment"]
    df["group_id"] = assign_groups(df, align_cfg["mz_tolerance_ppm"], align_cfg["rt_tolerance_sec"])

    long_path = interim_dir / "features_grouped.tsv"
    df.to_csv(long_path, sep="\t", index=False)

    summary = (
        df.groupby("group_id")
        .agg(
            mz_mean=("mz", "mean"),
            rt_mean=("rt", "mean"),
            n_features=("mz", "count"),
            n_files=("source_file", "nunique"),
        )
        .reset_index()
    )

    mix_stats = (
        df[df["sample_type"] == "mix"]
        .groupby("group_id")["intensity"]
        .agg(max_mix_intensity="max", mean_mix_intensity="mean")
    )
    blank_stats = (
        df[df["sample_type"] == "blank"]
        .groupby("group_id")["intensity"]
        .agg(max_blank_intensity="max", mean_blank_intensity="mean")
    )
    mix_files = df[df["sample_type"] == "mix"].groupby("group_id")["source_file"].nunique()
    blank_files = df[df["sample_type"] == "blank"].groupby("group_id")["source_file"].nunique()

    summary = summary.join(mix_stats, on="group_id").join(blank_stats, on="group_id")
    summary["mix_file_count"] = summary["group_id"].map(mix_files).fillna(0).astype(int)
    summary["blank_file_count"] = summary["group_id"].map(blank_files).fillna(0).astype(int)
    summary = summary.fillna(0)

    summary_path = interim_dir / "feature_groups.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    wide = (
        df.pivot_table(
            index="group_id",
            columns="source_file",
            values="intensity",
            aggfunc="max",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide_path = interim_dir / "feature_groups_wide.tsv"
    wide.to_csv(wide_path, sep="\t", index=False)

    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {wide_path}")


if __name__ == "__main__":
    main()

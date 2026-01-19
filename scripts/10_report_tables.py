from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator

from utils import ensure_dirs, load_config


def normalize(intensity: np.ndarray) -> np.ndarray:
    # Normalize a spectrum intensity vector.
    total = np.linalg.norm(intensity)
    if total == 0:
        return intensity
    return intensity / total


def cosine_similarity(
    # Compute cosine similarity for spectra.
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    mz_tolerance: float,
) -> float:
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    order_a = np.argsort(mz_a)
    order_b = np.argsort(mz_b)
    mz_a = mz_a[order_a]
    int_a = normalize(int_a[order_a])
    mz_b = mz_b[order_b]
    int_b = normalize(int_b[order_b])

    score = 0.0
    i = 0
    j = 0
    while i < len(mz_a) and j < len(mz_b):
        diff = mz_a[i] - mz_b[j]
        if abs(diff) <= mz_tolerance:
            score += int_a[i] * int_b[j]
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return float(score)


def knee_point_rank(intensities: np.ndarray, min_n: int = 10) -> int:
    # Find an elbow rank on the intensity curve.
    if intensities.size == 0:
        return min_n
    x = np.arange(1, len(intensities) + 1, dtype=float)
    y = np.log10(np.sort(intensities)[::-1] + 1.0)
    if len(y) < 3:
        return max(min_n, len(y))
    kl = KneeLocator(x, y, curve="convex", direction="decreasing")
    knee_rank = int(kl.knee) if kl.knee is not None else len(y)
    return max(min_n, min(knee_rank, len(y)))


def main() -> None:
    # Generate report tables and top candidates.
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    ensure_dirs([reports_dir])

    hits_path = processed_dir / "library_hits.parquet"
    if not hits_path.exists():
        raise SystemExit("Missing library hits parquet. Run 09_library_search.py first.")

    hits = pd.read_parquet(hits_path)
    if hits.empty:
        print("No hits to report.")
        return

    hits["library_compound_name"] = hits["library_compound_name"].fillna("")
    hits["library_inchikey"] = hits["library_inchikey"].fillna("")
    hits["library_title"] = hits["library_title"].fillna("")
    hits["compound_label"] = hits["library_compound_name"].replace("", pd.NA)
    hits["compound_label"] = hits["compound_label"].fillna(hits["library_title"])

    summary = (
        hits.groupby(["compound_label", "library_inchikey"])
        .agg(
            n_matches=("cosine", "count"),
            best_cosine=("cosine", "max"),
            median_cosine=("cosine", "median"),
            mean_precursor_mz=("precursor_mz", "mean"),
        )
        .reset_index()
        .sort_values(["best_cosine", "n_matches"], ascending=False)
    )

    hits_csv = reports_dir / "library_hits.csv"
    summary_csv = reports_dir / "library_summary.csv"
    hits.to_csv(hits_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # Top candidate table after adduct filtering.
    features_path = processed_dir / "feature_groups_filtered_adduct.tsv"
    if not features_path.exists():
        features_path = processed_dir / "feature_groups_filtered.tsv"
    if features_path.exists():
        features = pd.read_csv(features_path, sep="\t")
        top_n = knee_point_rank(features["max_mix_intensity"].fillna(0).to_numpy())
        mz_tol = cfg["library_search"]["mz_tolerance_da"]

        top = features.sort_values("max_mix_intensity", ascending=False).head(top_n).copy()
        best_cos = []
        best_name = []
        best_title = []
        for _, row in top.iterrows():
            mz = row["mz_mean"]
            subset = hits[(hits["precursor_mz"] >= mz - mz_tol) & (hits["precursor_mz"] <= mz + mz_tol)]
            if subset.empty:
                best_cos.append(np.nan)
                best_name.append(None)
                best_title.append(None)
                continue
            best = subset.sort_values("cosine", ascending=False).iloc[0]
            best_cos.append(best["cosine"])
            best_name.append(best.get("library_compound_name"))
            best_title.append(best.get("library_title"))

        top["best_cosine"] = best_cos
        top["best_library_name"] = [n if n not in (None, "", np.nan) else t for n, t in zip(best_name, best_title)]
        top_path = reports_dir / "top_candidates.csv"
        top.to_csv(top_path, index=False)
        print(f"Wrote {top_path}")

    # Unmatched MS2 clustering for high-intensity features.
    links_path = interim_dir / "ms2_feature_links.parquet"
    spectra_path = interim_dir / "ms2_spectra.parquet"
    if features_path.exists() and links_path.exists() and spectra_path.exists():
        features = pd.read_csv(features_path, sep="\t")
        links = pd.read_parquet(links_path)
        spectra = pd.read_parquet(spectra_path)

        mz_tol = cfg["library_search"]["mz_tolerance_da"]
        min_cos = cfg["library_search"]["min_cosine"]

        # Find best GNPS hit per group_id (if any)
        best_hit = {}
        for _, row in hits.iterrows():
            mz = row["precursor_mz"]
            best_hit.setdefault(mz, []).append(row)

        def feature_has_hit(mz: float) -> bool:
            subset = hits[(hits["precursor_mz"] >= mz - mz_tol) & (hits["precursor_mz"] <= mz + mz_tol)]
            return not subset.empty

        features["has_gnps_hit"] = features["mz_mean"].apply(feature_has_hit)
        features = features.sort_values("max_mix_intensity", ascending=False)
        unmatched = features[~features["has_gnps_hit"]]

        if not unmatched.empty:
            # Pick the strongest MS2 per group_id
            links = links[links["group_id"].isin(unmatched["group_id"])]
            if not links.empty:
                spectra_index = spectra.set_index(spectra.index)
                best_ms2 = {}
                for _, link in links.iterrows():
                    idx = int(link["ms2_index"])
                    spec = spectra_index.loc[idx]
                    intensity = float(np.sum(spec["intensity_array"]))
                    gid = int(link["group_id"])
                    if gid not in best_ms2 or intensity > best_ms2[gid]["intensity"]:
                        best_ms2[gid] = {
                            "group_id": gid,
                            "precursor_mz": float(spec["precursor_mz"]),
                            "rt": float(spec["rt"]),
                            "mz_array": np.array(spec["mz_array"], dtype=float),
                            "intensity_array": np.array(spec["intensity_array"], dtype=float),
                            "intensity": intensity,
                        }

                ms2_list = list(best_ms2.values())
                if len(ms2_list) >= 2:
                    parent = list(range(len(ms2_list)))

                    def find(x: int) -> int:
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x

                    def union(a: int, b: int) -> None:
                        ra = find(a)
                        rb = find(b)
                        if ra != rb:
                            parent[rb] = ra

                    for i in range(len(ms2_list)):
                        for j in range(i + 1, len(ms2_list)):
                            score = cosine_similarity(
                                ms2_list[i]["mz_array"],
                                ms2_list[i]["intensity_array"],
                                ms2_list[j]["mz_array"],
                                ms2_list[j]["intensity_array"],
                                mz_tol,
                            )
                            if score >= min_cos:
                                union(i, j)

                    cluster_ids = [find(i) for i in range(len(ms2_list))]
                    cluster_map = {cid: idx for idx, cid in enumerate(sorted(set(cluster_ids)), start=1)}
                    rows = []
                    for i, item in enumerate(ms2_list):
                        rows.append(
                            {
                                "group_id": item["group_id"],
                                "cluster_id": cluster_map[cluster_ids[i]],
                                "precursor_mz": item["precursor_mz"],
                                "rt": item["rt"],
                            }
                        )

                    clusters = pd.DataFrame(rows)
                    cluster_path = reports_dir / "unmatched_ms2_clusters.csv"
                    clusters.to_csv(cluster_path, index=False)
                    print(f"Wrote {cluster_path}")

                    summary = (
                        clusters.groupby("cluster_id")
                        .agg(
                            n_features=("group_id", "count"),
                            mz_min=("precursor_mz", "min"),
                            mz_max=("precursor_mz", "max"),
                            rt_min=("rt", "min"),
                            rt_max=("rt", "max"),
                        )
                        .reset_index()
                        .sort_values(["n_features"], ascending=False)
                    )
                    summary_path = reports_dir / "unmatched_ms2_cluster_summary.csv"
                    summary.to_csv(summary_path, index=False)
                    print(f"Wrote {summary_path}")

    print(f"Wrote {hits_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

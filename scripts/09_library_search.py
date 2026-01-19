from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_config


def parse_pepmass(value: str | float | None) -> float | None:
    # Parse PEPMASS values to float if present.
    if value is None:
        return None
    if isinstance(value, float):
        return value
    parts = str(value).strip().split()
    try:
        return float(parts[0])
    except ValueError:
        return None


def normalize(intensity: np.ndarray) -> np.ndarray:
    # Normalize a spectrum intensity vector.
    total = np.linalg.norm(intensity)
    if total == 0:
        return intensity
    return intensity / total


def cosine_similarity(
    # Compute cosine similarity with an m/z tolerance.
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    mz_tolerance: float,
) -> float:
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    # Ensure inputs are sorted by m/z for consistent two-pointer matching.
    order_a = np.argsort(mz_a)
    order_b = np.argsort(mz_b)
    mz_a = mz_a[order_a]
    int_a = normalize(int_a[order_a])
    mz_b = mz_b[order_b]
    int_b = normalize(int_b[order_b])

    score = 0.0
    i = 0
    j = 0
    # One-to-one peak matching within tolerance.
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


def main() -> None:
    # Search the GNPS index and write per-spectrum hits.
    cfg = load_config()
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs([processed_dir])

    spectra_path = interim_dir / "ms2_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit("Missing MS2 spectra parquet. Run 06_extract_ms2.py first.")

    ms2 = pd.read_parquet(spectra_path)
    db_path = interim_dir / cfg["inputs"]["gnps_sqlite"]
    if not db_path.exists():
        raise SystemExit("Missing GNPS library sqlite. Run 08_build_library_index.py first.")
    conn = sqlite3.connect(db_path)
    adduct_filtered_path = processed_dir / "feature_groups_filtered_adduct.tsv"
    adduct_filtered = adduct_filtered_path.exists()

    mz_tol = cfg["library_search"]["mz_tolerance_da"]
    min_cosine = cfg["library_search"]["min_cosine"]
    top_n = cfg["library_search"]["top_n"]

    hits = []
    for idx, row in ms2.iterrows():
        prec = row["precursor_mz"]
        if pd.isna(prec):
            continue
        mz_a = np.array(row["mz_array"], dtype=float)
        int_a = np.array(row["intensity_array"], dtype=float)

        cursor = conn.execute(
            """
            SELECT title, name, inchikey, pepmass, mz_array, intensity_array
            FROM spectra
            WHERE pepmass BETWEEN ? AND ?
            """,
            (prec - mz_tol, prec + mz_tol),
        )
        rows = cursor.fetchall()
        if not rows:
            continue

        for title, name, inchikey, pepmass, mz_json, int_json in rows:
            mz_b = np.array(json.loads(mz_json), dtype=float)
            int_b = np.array(json.loads(int_json), dtype=float)
            score = cosine_similarity(mz_a, int_a, mz_b, int_b, mz_tol)
            if score >= min_cosine:
                hits.append(
                    {
                        "source_file": row["source_file"],
                        "rt": row["rt"],
                        "precursor_mz": prec,
                        "library_title": title,
                        "library_compound_name": name,
                        "library_inchikey": inchikey,
                        "library_pepmass": pepmass,
                        "adduct_filtered": adduct_filtered,
                        "cosine": score,
                    }
                )

    hits_df = pd.DataFrame(hits)
    if hits_df.empty:
        print("No library matches found at current thresholds.")
    else:
        hits_df = hits_df.sort_values(["cosine"], ascending=False).groupby(
            ["source_file", "precursor_mz"]
        ).head(top_n)

    out_path = processed_dir / "library_hits.parquet"
    hits_df.to_parquet(out_path, index=False)
    conn.close()
    print(f"Wrote library hits to {out_path}")


if __name__ == "__main__":
    main()

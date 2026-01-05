from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dirs, load_config


def parse_pepmass(value: str | float | None) -> float | None:
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
    total = np.linalg.norm(intensity)
    if total == 0:
        return intensity
    return intensity / total


def cosine_similarity(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    mz_tolerance: float,
) -> float:
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    int_a = normalize(int_a)
    int_b = normalize(int_b)

    score = 0.0
    j = 0
    for i in range(len(mz_a)):
        mz_i = mz_a[i]
        while j < len(mz_b) and mz_b[j] < mz_i - mz_tolerance:
            j += 1
        if j >= len(mz_b):
            break
        if abs(mz_b[j] - mz_i) <= mz_tolerance:
            score += int_a[i] * int_b[j]
    return float(score)


def main() -> None:
    cfg = load_config()
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs([processed_dir])

    spectra_path = interim_dir / "ms2_spectra.parquet"
    library_path = interim_dir / "gnps_library.parquet"
    if not spectra_path.exists():
        raise SystemExit("Missing MS2 spectra parquet. Run 02_extract_ms2.py first.")
    if not library_path.exists():
        raise SystemExit("Missing GNPS library parquet. Run 03_build_library_index.py first.")

    ms2 = pd.read_parquet(spectra_path)
    library = pd.read_parquet(library_path)

    library["pepmass"] = library["pepmass"].apply(parse_pepmass)
    library = library.dropna(subset=["pepmass"]).reset_index(drop=True)

    mz_tol = cfg["library_search"]["mz_tolerance_da"]
    min_cosine = cfg["library_search"]["min_cosine"]
    top_n = cfg["library_search"]["top_n"]

    hits = []
    for idx, row in ms2.iterrows():
        prec = row["precursor_mz"]
        if pd.isna(prec):
            continue
        candidates = library[(library["pepmass"] - prec).abs() <= mz_tol]
        if candidates.empty:
            continue

        mz_a = np.array(row["mz_array"], dtype=float)
        int_a = np.array(row["intensity_array"], dtype=float)

        for _, lib in candidates.iterrows():
            mz_b = np.array(lib["mz_array"], dtype=float)
            int_b = np.array(lib["intensity_array"], dtype=float)
            score = cosine_similarity(mz_a, int_a, mz_b, int_b, mz_tol)
            if score >= min_cosine:
                hits.append(
                    {
                        "source_file": row["source_file"],
                        "rt": row["rt"],
                        "precursor_mz": prec,
                        "library_title": lib.get("title"),
                        "library_compound_name": lib.get("name"),
                        "library_inchikey": lib.get("inchikey"),
                        "library_pepmass": lib.get("pepmass"),
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
    print(f"Wrote library hits to {out_path}")


if __name__ == "__main__":
    main()

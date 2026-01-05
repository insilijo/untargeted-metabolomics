from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    ensure_dirs([reports_dir])

    hits_path = processed_dir / "library_hits.parquet"
    if not hits_path.exists():
        raise SystemExit("Missing library hits parquet. Run 04_library_search.py first.")

    hits = pd.read_parquet(hits_path)
    if hits.empty:
        print("No hits to report.")
        return

    summary = (
        hits.groupby(["library_compound_name", "library_inchikey"])
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

    print(f"Wrote {hits_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

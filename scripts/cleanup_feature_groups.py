from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def main() -> None:
    path = Path("data/processed/feature_groups_filtered_adduct.tsv")
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path, sep="\t")

    # Drop any merge suffix columns like *_x, *_y, *_x.1, *_y.1.1, etc.
    suffix_re = re.compile(r".*(_x|_y)(\..*)?$")
    drop_cols = [c for c in df.columns if suffix_re.match(c)]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.to_csv(path, sep="\t", index=False)
    print(f"Cleaned {path} (dropped {len(drop_cols)} columns).")


if __name__ == "__main__":
    main()

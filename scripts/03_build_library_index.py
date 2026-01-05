from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config


def parse_mgf(handle: io.TextIOBase) -> list[dict]:
    spectra = []
    current = {}
    mzs = []
    intensities = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.upper() == "BEGIN IONS":
            current = {}
            mzs = []
            intensities = []
            continue
        if line.upper() == "END IONS":
            current["mz_array"] = mzs
            current["intensity_array"] = intensities
            spectra.append(current)
            current = {}
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            current[key.lower()] = value
        else:
            parts = line.split()
            if len(parts) >= 2:
                mzs.append(float(parts[0]))
                intensities.append(float(parts[1]))
    return spectra


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    mgf_zip_path = raw_dir / cfg["inputs"]["gnps_mgf_zip"]
    if not mgf_zip_path.exists():
        raise SystemExit(f"Missing GNPS MGF zip: {mgf_zip_path}")

    spectra = []
    with zipfile.ZipFile(mgf_zip_path, "r") as zf:
        mgf_names = [n for n in zf.namelist() if n.lower().endswith(".mgf")]
        if not mgf_names:
            raise SystemExit("No MGF file found inside GNPS_SUBSET.mgf.zip")
        for name in mgf_names:
            with zf.open(name, "r") as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
                spectra.extend(parse_mgf(text))

    df = pd.DataFrame(spectra)
    out_path = interim_dir / "gnps_library.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} library spectra to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyopenms as oms

from utils import ensure_dirs, load_config, load_sample_metadata


def extract_ms2_spectra(mzml_path: Path, min_ms2_peaks: int, min_intensity: float) -> list[dict]:
    # Extract MS2 spectra from mzML with intensity/peak filters.
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)

    spectra = []
    for spec in exp:
        if spec.getMSLevel() != 2:
            continue
        peaks = spec.get_peaks()
        if len(peaks[0]) < min_ms2_peaks:
            continue
        if peaks[1].max() < min_intensity:
            continue
        precursors = spec.getPrecursors()
        precursor_mz = precursors[0].getMZ() if precursors else None
        spectra.append(
            {
                "source_file": mzml_path.name,
                "rt": spec.getRT(),
                "precursor_mz": precursor_mz,
                "mz_array": peaks[0].tolist(),
                "intensity_array": peaks[1].tolist(),
            }
        )
    return spectra


def main() -> None:
    # Run MS2 extraction for all mzML files and write parquet.
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    (sample_files, blank_files), _ = load_sample_metadata(raw_dir, interim_dir, cfg)

    ms2_cfg = cfg["ms2_extraction"]
    min_ms2_peaks = ms2_cfg["min_ms2_peaks"]
    min_intensity = ms2_cfg["min_intensity"]

    records = []
    for mzml_path in sample_files + blank_files:
        records.extend(extract_ms2_spectra(mzml_path, min_ms2_peaks, min_intensity))

    if not records:
        raise SystemExit("No MS2 spectra extracted. Check thresholds or inputs.")

    df = pd.DataFrame(records)
    out_path = interim_dir / "ms2_spectra.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} MS2 spectra to {out_path}")


if __name__ == "__main__":
    main()

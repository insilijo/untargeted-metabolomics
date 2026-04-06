from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config, load_sample_metadata


KNOWN_ADDUCTS = {
    "M": 0.0,
    "M+H": 1.007276,
    "M+Na": 22.989218,
    "M+K": 38.963158,
    "M+NH4": 18.033823,
}


def load_known_masses(cfg: dict) -> list[float]:
    # Load known monoisotopic masses from CSV.
    raw_dir = Path(cfg["paths"]["raw_dir"])
    known_name = cfg["inputs"]["known_masses_csv"]
    path = raw_dir / known_name
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "monoisotopic_mass" in df.columns:
        return df["monoisotopic_mass"].dropna().astype(float).tolist()
    # Fallback: first numeric column.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].dropna().astype(float).tolist()
    return []


def match_known(mz: float, masses: list[float], ppm: float) -> tuple[bool, float | None, str | None, float | None]:
    # Check if an m/z matches any known mass/adduct within ppm.
    if not masses:
        return False, None, None, None
    best = None
    for mass in masses:
        for label, add in KNOWN_ADDUCTS.items():
            target = mass + add
            delta_ppm = abs(mz - target) / target * 1e6
            if delta_ppm <= ppm:
                if best is None or delta_ppm < best[0]:
                    best = (delta_ppm, mass, label)
    if best is None:
        return False, None, None, None
    return True, best[1], best[2], best[0]


def main() -> None:
    # Apply blank/noise filters and keep known-mass matches.
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs([processed_dir])

    groups_path = interim_dir / "feature_groups.tsv"
    if not groups_path.exists():
        raise SystemExit("Missing feature_groups.tsv. Run 03_align_features.py first.")

    groups = pd.read_csv(groups_path, sep="\t")
    blank_cfg = cfg["blank_subtraction"]
    align_cfg = cfg["feature_alignment"]

    min_sample_intensity = float(blank_cfg["min_sample_intensity"])
    max_ratio = float(blank_cfg["max_blank_to_sample_ratio"])
    min_reps = int(align_cfg["min_replicate_count"])
    noise_floor = float(blank_cfg.get("noise_floor_intensity", 0.0))
    mz_tolerance_ppm = float(blank_cfg.get("mz_tolerance_ppm", 10.0))

    # Determine required replicate count from actual sample files
    (sample_files, _), _ = load_sample_metadata(raw_dir, interim_dir, cfg)
    required_mix = min(len(sample_files), min_reps) if sample_files else min_reps

    known_masses = load_known_masses(cfg)

    max_mix = groups["max_mix_intensity"].fillna(0)
    max_blank = groups["max_blank_intensity"].fillna(0)

    keep = (max_mix >= min_sample_intensity) & (max_blank <= max_ratio * max_mix)
    if "mix_file_count" in groups.columns:
        keep &= groups["mix_file_count"] >= required_mix
    if noise_floor > 0:
        keep &= max_mix >= noise_floor

    # Tag known-mass matches so they survive blank filtering.
    known_flags = []
    known_mass = []
    known_adduct = []
    known_delta_ppm = []
    for mz in groups["mz_mean"].fillna(0):
        flag, mass, adduct, delta = match_known(float(mz), known_masses, mz_tolerance_ppm)
        known_flags.append(flag)
        known_mass.append(mass)
        known_adduct.append(adduct)
        known_delta_ppm.append(delta)

    groups["known_match"] = known_flags
    groups["known_mass"] = known_mass
    groups["known_adduct"] = known_adduct
    groups["known_delta_ppm"] = known_delta_ppm

    if known_masses:
        keep |= groups["known_match"].fillna(False)

    filtered = groups[keep].copy()
    out_path = processed_dir / "feature_groups_filtered.tsv"
    filtered.to_csv(out_path, sep="\t", index=False)
    print(f"Kept {len(filtered)} / {len(groups)} groups after blank subtraction.")
    if noise_floor > 0:
        print(f"Applied noise floor: {noise_floor}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyopenms as oms

from utils import ensure_dirs, list_files, load_config


def run_feature_finding(mzml_path: Path, cfg: dict) -> oms.FeatureMap:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)
    exp.sortSpectra(True)

    ffm = oms.FeatureFindingMetabo()
    params = ffm.getDefaults()

    params.setValue("noise_threshold_int", float(cfg["noise_threshold_int"]))
    params.setValue("mass_error_ppm", float(cfg["mass_error_ppm"]))
    params.setValue("chrom_fwhm", float(cfg["chrom_fwhm"]))
    params.setValue("min_fwhm", float(cfg["min_fwhm"]))
    params.setValue("max_fwhm", float(cfg["max_fwhm"]))

    ffm.setParameters(params)

    fmap = oms.FeatureMap()
    # FeatureFindingMetabo performs mass trace detection internally.
    ffm.run(exp, fmap)
    fmap.setUniqueIds()
    return fmap


def feature_map_to_df(fmap: oms.FeatureMap, source_file: str) -> pd.DataFrame:
    records = []
    for feat in fmap:
        records.append(
            {
                "source_file": source_file,
                "feature_id": feat.getUniqueId(),
                "mz": feat.getMZ(),
                "rt": feat.getRT(),
                "intensity": feat.getIntensity(),
                "charge": feat.getCharge(),
                "quality": feat.getQuality(),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    mix_files = list_files(raw_dir, cfg["inputs"]["mix_glob"])
    blank_files = list_files(raw_dir, cfg["inputs"]["blank_glob"])
    files = mix_files + blank_files

    if not files:
        raise SystemExit("No mzML files found for feature finding.")

    ff_cfg = cfg["feature_finding"]
    all_features = []
    for mzml_path in files:
        fmap = run_feature_finding(mzml_path, ff_cfg)
        feature_xml = interim_dir / f"{mzml_path.stem}_features.featureXML"
        oms.FeatureXMLFile().store(str(feature_xml), fmap)

        df = feature_map_to_df(fmap, mzml_path.name)
        all_features.append(df)

    combined = pd.concat(all_features, ignore_index=True)
    out_path = interim_dir / "features.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyopenms as oms

from utils import ensure_dirs, list_files, load_config


def set_param_if_exists(params: oms.Param, key: str, value) -> None:
    if params.exists(key):
        params.setValue(key, value)


def run_feature_finding(mzml_path: Path, cfg: dict) -> tuple[oms.FeatureMap, list[oms.Kernel_MassTrace]]:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)
    exp.sortSpectra(True)

    # Ensure MS1 spectra are centroided for mass trace detection.
    picker = oms.PeakPickerHiRes()
    picked = oms.MSExperiment()
    # Not all pyopenms builds expose setExperimentalSettings on MSExperiment.
    for spec in exp:
        if spec.getMSLevel() == 1:
            picked_spec = oms.MSSpectrum()
            picker.pick(spec, picked_spec)
            picked_spec.setMSLevel(1)
            picked_spec.setRT(spec.getRT())
            picked_spec.setPrecursors(spec.getPrecursors())
            picked_spec.setNativeID(spec.getNativeID())
            picked.addSpectrum(picked_spec)
        else:
            picked.addSpectrum(spec)
    exp = picked

    mtd = oms.MassTraceDetection()
    mtd_params = mtd.getDefaults()
    set_param_if_exists(mtd_params, "noise_threshold_int", float(cfg["noise_threshold_int"]))
    set_param_if_exists(mtd_params, "mass_error_ppm", float(cfg["mass_error_ppm"]))
    set_param_if_exists(mtd_params, "chrom_peak_snr", float(cfg["chrom_peak_snr"]))
    set_param_if_exists(mtd_params, "min_trace_length", float(cfg["min_trace_length"]))
    set_param_if_exists(mtd_params, "max_trace_length", float(cfg["max_trace_length"]))
    mtd.setParameters(mtd_params)

    traces: list[oms.Kernel_MassTrace] = []
    mtd.run(exp, traces, 0)

    ffm = oms.FeatureFindingMetabo()
    ffm_params = ffm.getDefaults()
    set_param_if_exists(ffm_params, "chrom_fwhm", float(cfg["chrom_fwhm"]))
    ffm.setParameters(ffm_params)

    fmap = oms.FeatureMap()
    chromatograms: list[list[oms.MSChromatogram]] = []
    ffm.run(traces, fmap, chromatograms)
    fmap.setUniqueIds()
    return fmap, traces


def feature_map_to_df(
    fmap: oms.FeatureMap, traces: list[oms.Kernel_MassTrace], source_file: str
) -> pd.DataFrame:
    records = []
    for feat in fmap:
        try:
            quality_rt = feat.getQuality(0)
            quality_mz = feat.getQuality(1)
        except Exception:
            quality_rt = None
            quality_mz = None
        intensity = feat.getIntensity()
        keys: list[bytes] = []
        feat.getKeys(keys)
        meta = {k: feat.getMetaValue(k) for k in keys}
        if intensity == 0.0:
            max_height = meta.get(b"max_height")
            masstrace_intensity = meta.get(b"masstrace_intensity")
            if isinstance(max_height, (int, float)) and max_height > 0:
                intensity = float(max_height)
            elif isinstance(masstrace_intensity, (list, tuple)) and masstrace_intensity:
                intensity = float(sum(masstrace_intensity))
        if intensity == 0.0:
            label = meta.get(b"label")
            if label and str(label).startswith("T"):
                try:
                    idx = int(str(label)[1:])
                    if 0 <= idx < len(traces):
                        intensity = float(traces[idx].getMaxIntensity(False))
                except ValueError:
                    pass
        records.append(
            {
                "source_file": source_file,
                "feature_id": feat.getUniqueId(),
                "mz": feat.getMZ(),
                "rt": feat.getRT(),
                "intensity": intensity,
                "charge": feat.getCharge(),
                "quality_overall": feat.getOverallQuality(),
                "quality_rt": quality_rt,
                "quality_mz": quality_mz,
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
        fmap, traces = run_feature_finding(mzml_path, ff_cfg)
        feature_xml = interim_dir / f"{mzml_path.stem}_features.featureXML"
        oms.FeatureXMLFile().store(str(feature_xml), fmap)

        df = feature_map_to_df(fmap, traces, mzml_path.name)
        all_features.append(df)

    combined = pd.concat(all_features, ignore_index=True)
    out_path = interim_dir / "features.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

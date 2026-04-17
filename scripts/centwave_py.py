"""Pure-Python centWave-style peak finder.

Two-phase algorithm (Tautenhahn et al. 2008, simplified):

  1. ROI detection — build mass-trace "regions of interest": groups of
     (scan_idx, mz, intensity) triplets where consecutive scans share
     an m/z within ppm tolerance. Traces shorter than min_trace_length
     (seconds) or entirely below noise are discarded.

  2. Peak detection per ROI — 1D peak detection on the intensity-vs-RT
     vector inside each ROI via ``scipy.signal.find_peaks`` with:
       * min height = noise_threshold × chrom_peak_snr
       * min prominence = noise_threshold × chrom_peak_snr
       * min width (points) derived from chrom_fwhm and avg scan period

     Each peak yields a feature (mz = intensity-weighted mean over the
     peak boundary, rt = apex, intensity = trapezoidal area).

This module deliberately skips:
  - Isotope-pattern grouping (FFM does this and rejects lone peaks —
    the whole reason we swapped off FFM)
  - Wavelet smoothing (scipy find_peaks + minimum-prominence heuristic
    is good enough for HRMS data; add-back later if SNR drops)
  - Charge deconvolution (downstream adduct filter handles this)

Config keys read (passed as dict):
  noise_threshold_int : float — global noise floor (intensity units)
  mass_error_ppm      : float — ROI m/z tolerance (ppm)
  chrom_peak_snr      : float — required peak height / noise ratio
  chrom_fwhm          : float — expected peak FWHM in seconds
  min_trace_length    : float — min ROI duration (seconds)
  max_trace_length    : float — max ROI duration (-1 = unlimited)
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class _ROI:
    mzs: list[float]          # per-scan observed m/z
    rts: list[float]          # per-scan RT (seconds)
    intensities: list[float]  # per-scan peak intensity
    scan_idxs: list[int]      # per-scan index (sequential)


def _build_rois(
    spectra: list[tuple[float, np.ndarray, np.ndarray]],
    ppm: float,
    noise: float,
    min_trace_length_s: float,
    max_trace_length_s: float,
) -> list[_ROI]:
    """Greedy ROI builder — linear sweep across scans extending open ROIs."""
    open_rois: list[_ROI] = []
    closed_rois: list[_ROI] = []

    def _flush(roi: _ROI) -> None:
        dur = roi.rts[-1] - roi.rts[0]
        if dur < min_trace_length_s:
            return
        if max_trace_length_s > 0 and dur > max_trace_length_s:
            return
        closed_rois.append(roi)

    for scan_idx, (rt, mzs, intens) in enumerate(spectra):
        # Filter to above-noise peaks only; sort by m/z for bisect
        mask = intens > noise
        if not mask.any():
            continue
        sel_mzs = mzs[mask]
        sel_int = intens[mask]
        order = np.argsort(sel_mzs)
        sel_mzs = sel_mzs[order]
        sel_int = sel_int[order]

        next_open: list[_ROI] = []
        used = np.zeros(len(sel_mzs), dtype=bool)
        for roi in open_rois:
            ref_mz = roi.mzs[-1]
            tol = ref_mz * ppm * 1e-6
            # Find nearest peak to ref_mz within tol
            lo = bisect.bisect_left(sel_mzs, ref_mz - tol)
            hi = bisect.bisect_right(sel_mzs, ref_mz + tol)
            best_i = -1
            best_err = tol + 1e-12
            for i in range(lo, hi):
                if used[i]:
                    continue
                err = abs(sel_mzs[i] - ref_mz)
                if err < best_err:
                    best_err = err
                    best_i = i
            if best_i >= 0:
                roi.mzs.append(float(sel_mzs[best_i]))
                roi.rts.append(rt)
                roi.intensities.append(float(sel_int[best_i]))
                roi.scan_idxs.append(scan_idx)
                used[best_i] = True
                next_open.append(roi)
            else:
                # Gap: close the ROI (no extension this scan)
                _flush(roi)

        # Start new ROIs for any unused peaks
        for i in range(len(sel_mzs)):
            if used[i]:
                continue
            next_open.append(_ROI(
                mzs=[float(sel_mzs[i])], rts=[rt],
                intensities=[float(sel_int[i])], scan_idxs=[scan_idx],
            ))
        open_rois = next_open

    # Flush remaining open ROIs at end of run
    for roi in open_rois:
        _flush(roi)
    return closed_rois


def _detect_peaks_in_roi(
    roi: _ROI, noise: float, snr: float, chrom_fwhm_s: float,
) -> list[dict]:
    """Run 1D peak detection on this ROI's intensity-vs-RT profile."""
    from scipy.signal import find_peaks
    ints = np.asarray(roi.intensities, dtype=np.float64)
    rts = np.asarray(roi.rts, dtype=np.float64)
    mzs = np.asarray(roi.mzs, dtype=np.float64)
    if ints.size < 3:
        return []
    scan_dt = float(np.median(np.diff(rts))) if rts.size > 1 else 0.5
    if scan_dt <= 0:
        scan_dt = 0.5
    # Width constraint: real chromatographic peaks need at least a few scans
    # above half-max. Don't require full FWHM as the min — that rejects narrow
    # real peaks. Use 0.2 × FWHM as a floor, 3 × FWHM as a ceiling.
    min_width_scans = max(2, int(round(chrom_fwhm_s * 0.2 / scan_dt)))
    max_width_scans = max(min_width_scans * 5, int(round(chrom_fwhm_s * 3.0 / scan_dt)))
    height = noise * snr
    prominence = noise * snr * 0.5
    peak_idxs, props = find_peaks(
        ints, height=height, prominence=prominence,
        width=(min_width_scans, max_width_scans),
    )
    features = []
    left_bases = props.get("left_bases")
    right_bases = props.get("right_bases")
    for k, apex in enumerate(peak_idxs):
        lb = int(left_bases[k]) if left_bases is not None else max(0, apex - min_width_scans)
        rb = int(right_bases[k]) if right_bases is not None else min(len(ints) - 1, apex + min_width_scans)
        rb = min(rb, len(ints) - 1)
        if rb <= lb:
            continue
        weights = ints[lb:rb + 1]
        total = float(weights.sum())
        if total <= 0:
            continue
        mz_weighted = float(np.sum(mzs[lb:rb + 1] * weights) / total)
        # Trapezoidal integration for peak area (intensity units × seconds)
        area = float(np.trapz(ints[lb:rb + 1], rts[lb:rb + 1]))
        features.append({
            "mz": mz_weighted,
            "rt": float(rts[apex]),
            "intensity": float(ints[apex]),
            "area": area,
            "fwhm_s": float(rts[rb] - rts[lb]),
            "n_scans": int(rb - lb + 1),
        })
    return features


def find_features(mzml_path: Path, cfg: dict) -> pd.DataFrame:
    """Pure-Python centWave-style feature finder.

    Returns a DataFrame with the same columns as the OpenMS FFM path
    for drop-in compatibility with downstream pipeline steps.
    """
    import pyopenms as oms
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)
    exp.sortSpectra(True)

    # Ensure MS1 is centroided — re-pick if needed, matching FFM behaviour.
    picker = oms.PeakPickerHiRes()
    spectra: list[tuple[float, np.ndarray, np.ndarray]] = []
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        if spec.getType() != oms.SpectrumSettings.SpectrumType.CENTROID:
            picked = oms.MSSpectrum()
            picker.pick(spec, picked)
            mzs, ints = picked.get_peaks()
        else:
            mzs, ints = spec.get_peaks()
        spectra.append((float(spec.getRT()), np.asarray(mzs, dtype=np.float64),
                        np.asarray(ints, dtype=np.float64)))

    rois = _build_rois(
        spectra,
        ppm=float(cfg["mass_error_ppm"]),
        noise=float(cfg["noise_threshold_int"]),
        min_trace_length_s=float(cfg["min_trace_length"]),
        max_trace_length_s=float(cfg.get("max_trace_length", -1) or -1),
    )

    feature_rows: list[dict] = []
    for roi in rois:
        peaks = _detect_peaks_in_roi(
            roi,
            noise=float(cfg["noise_threshold_int"]),
            snr=float(cfg["chrom_peak_snr"]),
            chrom_fwhm_s=float(cfg["chrom_fwhm"]),
        )
        for feat in peaks:
            feature_rows.append(feat)

    df = pd.DataFrame(feature_rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "source_file", "feature_id", "mz", "rt", "intensity", "charge",
            "quality_overall", "quality_rt", "quality_mz",
        ])
        return df

    # Stable unique IDs per (mz, rt) for alignment-pass compatibility
    df = df.sort_values(["mz", "rt"]).reset_index(drop=True)
    df["source_file"] = mzml_path.name
    df["feature_id"] = [hash((mzml_path.name, i)) & ((1 << 63) - 1)
                        for i in range(len(df))]
    df["charge"] = 0
    df["quality_overall"] = df.get("area", df["intensity"]) / df["intensity"].replace(0, 1)
    df["quality_rt"] = df["fwhm_s"] / float(cfg["chrom_fwhm"])
    df["quality_mz"] = 0.0
    return df[["source_file", "feature_id", "mz", "rt", "intensity",
               "charge", "quality_overall", "quality_rt", "quality_mz"]]

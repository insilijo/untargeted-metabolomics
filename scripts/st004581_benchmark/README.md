# ST004581 Metabolon library-match benchmark

Tiered **RT/RI → MS1 (→ MS2)** annotation of ST004581 (Metabolon) mzML against
the public **Metabolon data dictionary** (`metabolon_data_dictionary_PMC_OA_subset`),
scored per-feature TP/FP/FN vs the study MAF (`annotations_repaired.csv`).

## Two script families
- `_subset_*.py` / `_build_gnps_*.py` — **local subset** runs (one batch/method).
  Feature-finding (`_subset_ff.py`, `_subset_pospolar*.py`), MS2 extraction
  (`_subset_extract_ms2.py`), GNPS neg/pos library build (`_build_gnps_*`), and
  the matchers (`_subset_match_eval{,_dd,_allplat,_ms2}.py`).
- `c_*.py` — **full-dataset** runs on the VPS, off precomputed
  `features_centwave.tsv` (3.9M features, 480 mzML) + `ms2_spectra.parquet`.
  `c_eval` (all-platform F1), `c_errors` (FP/FN decomposition), `c_crossmethod`,
  `c_fixes` (no-ik / presence / MIN_REP levers), `c_presence`, `c_isotope`,
  `c_noik`, `c_ddfix` (DD InChIKey backfill), `c_anchors` (RI-covering panels).

## Headline results (full dataset, 480 mzML)
| Setting | F1 |
|---|---|
| RT/RI+MS1, answer-in-library (pooled, the Metabolon paradigm) | **0.763** |
| + forbid structureless (no-InChIKey) candidates from winning *(leak-free)* | **0.845** |
| + MIN_REP=1 sensitivity | **~0.87** |
| End-to-end incl. coverage gap, + DD InChIKey backfill *(full-library proxy)* | 0.628 → **0.777** |

- **RT/RI is the load-bearing tier** once calibration is well-fed; +0.03–0.06 F1/platform.
- **MS2 is marginal** here — residual confusions are co-eluting *same-formula* isomers
  that co-fragment (RT separates them, MS2 can't).
- **FP decomposition:** 100% mass-coincident; 54% lost to structureless DD entries
  (47% of the public DD has no InChIKey — named regiochemical-ambiguity compounds,
  not errata), 16% to compounds not in the study, **only 30% genuine co-eluting isobars**.
- **FN:** 84% non-detection (MIN_REP/abundance), 14% RT-window/calibration tail.

## Anchor panels (`../../data/anchor_panels/`)
Per-platform RT-anchor panels selected from mass-unique, reproducibly-detected
compounds, **tiled evenly across the RI gradient** so RI→sec calibration has no
large gaps (built by `c_anchors.py`). Columns: `compound_id,name,inchikey,smiles,
platform,method,pubchem,mz,ri,observed_rt_sec,intensity`. Usable directly by
`SQuID-INC/scripts/build_anchor_rt_observations*.py` and `build_rt_calibration.py`.
`c_anchors.py` also reports **thin RI regions** where a physical spike-in kit
should add standards.

> Note: `c_*` scripts hardcode VPS paths (`/root/SQuID-INC`, `/mnt/volume-hel1-1`).
> Backfill / presence-prior / MAF-anchor results that import study-resolved
> structures are upper bounds (full-library proxy), not leak-free numbers.

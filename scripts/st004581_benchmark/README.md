# ST004581 Metabolon library-match benchmark

Tiered **RT/RI → MS1 (→ MS2)** annotation of ST004581 (Metabolon) mzML against
the public **Metabolon data dictionary** (`metabolon_data_dictionary_PMC_OA_subset`),
scored vs the study MAF (`annotations_repaired.csv`).

> **📦 Packaged result → [`RESULTS_PACKAGE.md`](RESULTS_PACKAGE.md)** (+ [`results_vs_metabolon.csv`](results_vs_metabolon.csv)).
> Self-contained writeup for paper/deck/grant: a sparse purchasable kit + the public
> dictionary reproduces **~85% of Metabolon's expert manually-curated annotations**
> (732/862 across 4 platforms) at **<2% misidentification**, no proprietary library /
> standards / MS2. Run via `precision_diagnostics/forest_sweep_h2h.py`.
>
> **🔓 Vendor-free / library-agnostic result → [`standalone/`](standalone/)** + supporting docs
> ([`fair_library_comparison_canonical.md`](fair_library_comparison_canonical.md),
> [`comprehensive_adducts_recover_observed_mz.md`](comprehensive_adducts_recover_observed_mz.md),
> [`library_coverage_via_mapper.md`](library_coverage_via_mapper.md),
> [`bootstrap_kitfree_results.md`](bootstrap_kitfree_results.md)).
> Once identities are unified via the metabolite mapper, a **fully public library** (HMDB
> metabolome) **ties or beats** Metabolon's data dictionary (canonical-matched weighted recall
> HMDB 0.72 vs DD 0.69). Every vendor "advantage" is reproducible: **m/z** = computed adducts
> (`standalone/build_library.py`, recovers + exceeds observed m/z), **measured RI** = self-built
> quasi-library (kit-free bootstrap), **identity** = public mapper, **kit** = self-bootstrapped.
> Vendor-dependency ledger: **empty**. Honest floor = stereo isomers (need measured RT) + ~3–4%
> genuinely-ambiguous lipids (need a standard).

## Headline — sparse-kit vs Metabolon (cardinality-scored, RT/RI+MS1, no MS2, stereo-isomer-aware)

| platform | distinct compounds | recall | identity precision |
|---|---|---|---|
| lc/ms pos early | 204 | **0.887** | 1.000 |
| lc/ms neg | 403 | **0.854** | 0.992 |
| lc/ms pos late | 192 | **0.833** | 0.990 |
| lc/ms polar | 93 | **0.796** | 1.000 |
| **weighted** | **892** | **0.851** | **~0.996** |

**Stereo/geometric-isomer resolution** (retention separates what MS2 can't): 18/22 isomer
skeletons fully recovered — all 8 polar pairs (fumarate+maleate, erythronate+threonate, …).
Scored on distinct compounds (full InChIKey), not merged skeletons.

Comparators (neg, correct reference impls): GNPS/matchms **0.078**, OpenMS **0.27** recall
— this method is **3–11×** at higher precision. Identity precision = misID rate on
curator-adjudicated peaks (the meaningful number); closed-world precision (0.55–0.68)
conservatively counts every out-of-scope call as our error. Engineering: streaming EIC
extractor scales to 10⁵ ions in a bounded footprint (validated 99.94% vs dense).

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

## Earlier per-feature F1 analysis (full dataset, 480 mzML — historical, summarized-feature scoring)
*Superseded as the headline by the cardinality-scored sparse-kit results above; kept for the FP/FN decomposition.*
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

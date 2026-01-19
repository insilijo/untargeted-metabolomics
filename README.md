# Untargeted Metabolomics Challenge

Starter workflow for identifying compounds from LC-MS/MS data using PyOpenMS.

## Project layout

- `data/raw/`: place raw input files here (mzML, mgf zip, known masses CSV)
- `data/interim/`: intermediate exports (feature tables, MS/MS spectra)
- `data/processed/`: processed outputs for figures and results
- `scripts/`: reproducible analysis scripts
- `notebooks/`: optional exploratory notebooks (keep minimal)
- `reports/`: slide exports or PDFs
- `figures/`: figures for reports

## Expected inputs

Put these files in `data/raw/` when ready:

- `MIX_01.mzML`, `MIX_02.mzML`, `MIX_03.mzML`
- `BLANK_01.mzML`, `BLANK_02.mzML`, `BLANK_03.mzML`
- `GNPS_SUBSET.mgf.zip`
- `KNOWN_MASSES.csv`

Alternatively, you can place a single `.zip` file in `data/raw/` that contains all of the
items above; `01_validate_inputs.py` will accept the bundle.

## Quick start

1) Create a Python env and install dependencies.

```bash
poetry install
```

2) Run the pipeline (stepwise).

```bash
poetry run python scripts/01_validate_inputs.py
poetry run python scripts/02_feature_finding.py
poetry run python scripts/03_align_features.py
poetry run python scripts/04_blank_subtract.py
poetry run python scripts/05_adduct_filter.py
poetry run python scripts/06_extract_ms2.py
poetry run python scripts/07_link_ms2_features.py
poetry run python scripts/08_build_library_index.py
poetry run python scripts/09_library_search.py
poetry run python scripts/10_report_tables.py
poetry run python scripts/11_predict_structures.py
```

## Notes

- The scripts are minimal scaffolding to make the analysis reproducible once data are available.
- Feature finding is currently run file-by-file with FeatureFinderMetabo and exported to
  `data/interim/*_features.featureXML` and `data/interim/features.tsv`.
- Feature alignment is a simple mz/rt grouping step to track features across replicates,
  followed by blank subtraction before MS/MS linking and library search.
- MS2 linking writes `data/interim/ms2_feature_links.parquet` for tying library hits
  back to grouped features.
- GNPS library indexing is streamed into `data/interim/gnps_library.sqlite` to keep
  memory usage low.
- Adduct/isotope filtering writes `data/processed/feature_groups_filtered_adduct.tsv` and
  `data/processed/adduct_pairs.tsv` for review.
- Structure prediction export writes `reports/unmatched_ms2.mgf` for downstream tools
  such as spec2mol.
- Spec2Mol execution uses `conda run -n spec2mol` by default; adjust `spec2mol.conda_env`
  in `scripts/config.yaml` if needed.

## Alternative setup

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

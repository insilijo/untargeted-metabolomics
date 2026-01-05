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

## Quick start

1) Create a Python env and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the pipeline (stepwise).

```bash
python scripts/01_validate_inputs.py
python scripts/02_feature_finding.py
python scripts/02_extract_ms2.py
python scripts/03_build_library_index.py
python scripts/04_library_search.py
python scripts/05_report_tables.py
```

## Notes

- The scripts are minimal scaffolding to make the analysis reproducible once data are available.
- Feature finding is currently run file-by-file with FeatureFinderMetabo and exported to
  `data/interim/*_features.featureXML` and `data/interim/features.tsv`.
- You can swap in MZmine feature tables later; the pipeline supports a fallback mode
  that uses raw MS/MS extraction + spectral library search.

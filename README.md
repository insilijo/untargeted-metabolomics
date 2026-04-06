# Untargeted Metabolomics Pipeline

LC-MS/MS feature finding, library search, and SQuID-InC anchor-guided localization.

## Project layout

```
data/raw/          raw inputs (mzML, spectral library, annotation CSV)
data/interim/      intermediate exports (feature tables, MS2 spectra, sqlite)
data/processed/    final outputs (library hits, embeddings, benchmark)
scripts/           reproducible pipeline scripts
notebooks/         optional exploratory notebooks
reports/           slide exports, PDFs, unmatched MGF
figures/           figures for reports
```

## Inputs

### mzML files

Place mzML files in `data/raw/` (any filenames), or drop a `.zip` of mzMLs there — step 01 extracts them automatically.

Sample vs blank assignment uses the first available source:

1. **`data/raw/sample_metadata.csv`** — explicit sheet (recommended for real data)
2. **Filename inference** — files containing `blank`, `blk`, `cmtrx` → blank; everything else → sample
3. **Legacy globs** — `MIX_*.mzML` / `BLANK_*.mzML` if no metadata sheet and no zip

#### Sample metadata sheet

```csv
filename,sample_type,batch,subject_id
COLU-00965.mzML,sample,1,P001
COLU-00966.mzML,sample,1,P002
CMTRX-94927.mzML,blank,1,
```

| Column | Values | Notes |
|---|---|---|
| `filename` | `*.mzML` | Stem or full name both accepted |
| `sample_type` | `sample`, `blank`, `qc`, `other` | **Required** |
| `batch`, `subject_id`, … | anything | Passed through to outputs |

Point to it in `scripts/config.yaml`:
```yaml
inputs:
  sample_metadata: sample_metadata.csv
```

### Spectral library (optional)

`data/raw/GNPS_SUBSET.mgf.zip` — an MGF file (zipped) used for MS2 library search (steps 8–9).
Download a subset from [GNPS](https://gnps.ucsd.edu/ProteoSAFe/libraries.jsp) or build your own
(see annotation CSV section below for when you can skip this).

### Known masses (optional)

`data/raw/KNOWN_MASSES.csv` — monoisotopic masses used to flag known features during blank
subtraction.  Two columns: `compound_id`, `monoisotopic_mass`.

### Annotation CSV (optional, replaces or supplements GNPS library search)

If you already have putative annotations, supply them directly.  Only `mz` is required;
all other columns are optional.

```csv
mz,rt,inchikey,smiles,name,ccs,confidence,ms2
234.1128,4.2,COLNVLDHVKWLRT-QMMMGPOBSA-N,CC(N)Cc1ccccc1,Phenylalanine,142.3,0.95,"[[91.1,1000],[120.1,800]]"
146.0579,1.8,WHUUTDBJXJRKMK-VKHMYHEASA-N,,Glutamate,,,
```

| Column | Aliases accepted | Notes |
|---|---|---|
| `mz` | `mass`, `precursor_mz`, `pepmass` | **Required** |
| `rt` | `retention_time`, `ri` | Seconds, minutes, or RI units |
| `inchikey` | `inchi_key` | Used for graph lookup and benchmarking |
| `smiles` | — | Used for predicted MS2 fingerprint |
| `name` | `compound_name`, `biochemical` | Display label |
| `ccs` | — | Collision cross section (Å²) |
| `confidence` | `cosine`, `score` | 0–1; defaults to 1.0 |
| `ms2` | `ms2_peaks` | `[[mz,int],…]` JSON, `mz:int;mz:int`, or flat interleaved |

Point to it in `scripts/config.yaml`:

```yaml
squid:
  annotation_csv: data/raw/my_annotations.csv
```

## Setup

```bash
poetry install
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install [SQuID-InC](https://github.com/insilijo/SQuID-INC) and set the root:

```bash
export SQUID_INC_ROOT=~/SQuID-INC
```

## Running the pipeline

### Full pipeline

```bash
poetry run python scripts/run_pipeline.py
```

### Partial run (start / end / skip)

```bash
# Start from a specific step
poetry run python scripts/run_pipeline.py --start 06

# Run only a single step
poetry run python scripts/run_pipeline.py --start 13 --end 13

# Skip steps you don't need
poetry run python scripts/run_pipeline.py --skip 08_build_library_index,09_library_search
```

Step numbers can be given as `06`, `6`, or the full name `06_extract_ms2`.

### Steps

| Step | Script | Output |
|---|---|---|
| 01 | `01_validate_inputs.py` | `data/interim/input_check.json` |
| 02 | `02_feature_finding.py` | `data/interim/features.tsv` |
| 03 | `03_align_features.py` | `data/interim/feature_groups.tsv` |
| 04 | `04_blank_subtract.py` | `data/interim/feature_groups_subtracted.tsv` |
| 05 | `05_adduct_filter.py` | `data/processed/feature_groups_filtered_adduct.tsv` |
| 06 | `06_extract_ms2.py` | `data/interim/ms2_spectra.parquet` |
| 07 | `07_link_ms2_features.py` | `data/interim/ms2_feature_links.parquet` |
| 08 | `08_build_library_index.py` | `data/interim/gnps_library.sqlite` |
| 09 | `09_library_search.py` | `data/processed/library_hits.parquet` |
| 10 | `10_report_tables.py` | `data/processed/report_*.tsv` |
| 11 | `11_predict_structures.py` | `reports/unmatched_ms2.mgf` |
| 12 | `12_package_results.py` | `reports/results.zip` |
| 13 | `13_squid_anchor_localization.py` | `data/processed/squid_features.parquet` |

### Step 13 without running steps 1–12

If you're starting from an annotation CSV and already have processed features:

```bash
SQUID_INC_ROOT=~/SQuID-INC poetry run python scripts/13_squid_anchor_localization.py
```

Step 13 requires at least one of:
- `data/processed/library_hits.parquet` (from step 09), or
- `annotation_csv` set in `config.yaml`

And from SQuID-InC:
- An anchor set at `$SQUID_INC_ROOT/results/anchors_balanced.csv`
- A prepared graph at `$SQUID_INC_ROOT/data/processed/graph/ready/`

### Step 13 outputs

| File | Contents |
|---|---|
| `data/processed/squid_features.parquet` | All features (annotated + unannotated) with UMAP coords, community ID, message prior, annotation confidence |
| `data/processed/squid_community_summary.tsv` | Per-community: size, anchor count, coverage, blind flag |
| `data/processed/squid_benchmark.csv` | GT coverage and annotation lift vs. random / frequency / diversity baselines |

## Configuration

All parameters are in `scripts/config.yaml`.  Key sections:

```yaml
library_search:
  mz_tolerance_da: 0.01
  min_cosine: 0.6

squid:
  anchor_set: results/anchors_balanced.csv   # relative to SQUID_INC_ROOT
  graph_dir:  data/processed/graph/ready     # relative to SQUID_INC_ROOT
  annotation_csv: data/raw/my_annotations.csv  # optional
  min_cosine: 0.7
  leiden_resolution: 1.0
  umap_n_neighbors: 15
```

# Sparse-anchor annotation — standalone, vendor-free

Self-contained RT/RI + MS1 metabolite annotation. No `squid_inc` / `library_match` dependency, no
hardcoded paths — point it at your own data via env vars. Built and validated on ST004581 (Metabolon),
where it reproduces ~85% of the expert-curated annotations from a sparse purchasable kit, and where a
fully **public** library (HMDB metabolome) matches or beats Metabolon's own data dictionary once
identities are unified.

## Two scripts

| script | role |
|---|---|
| `build_library.py` | Build a candidate library from a structure source (inchikey/name[/smiles]) → per-platform **computed** m/z over a comprehensive adduct set (SMILES→ExactMolWt, lipid-name→formula fallback). No vendor m/z needed. |
| `sparse_anchor_annotate.py` | Annotate raw mzML against the library: kit-calibrated RI→sec ladder + structure→RT, forest propagation, cardinality assignment. |

## Quick start

```bash
# 1. build a candidate library from any structure list (public is fine)
python build_library.py compounds.csv library.csv      # compounds.csv: inchikey,name[,smiles]

# 2. annotate
FLOOR=8000 KIT_MODE=chem ROUNDS=6 PEAK_CAP=200 \
PLATFORM="lc/ms neg" \
LIBRARY_CSV=library.csv \
KIT_CSV=anchors_lc_ms_neg.csv \
MZML_GLOB="/data/*COLU*.mzML" \
OUTPUT_TSV=annotations.tsv \
ANSWER_KEY_CSV=truth.csv            # optional; if given, also scores recall/precision \
  python sparse_anchor_annotate.py 0 0 10
```

## Annotator env vars

Required: `LIBRARY_CSV`, `KIT_CSV`, `MZML_GLOB`, `PLATFORM`.
Optional:
- `ANSWER_KEY_CSV` — score vs a truth MAF (recall / precision / isomer recovery). Omit → annotate-only.
- `OUTPUT_TSV` — per-call table (compound, inchikey, mz, rt, intensity, reps, in_answer_key).
- `SMILES_CSV` — ik14→smiles fallback if the library lacks SMILES.
- `STRUCT_RT=1` — predict RT from structure (use when the library has no RI; the public-library path).
- `HYBRID=1` — per-candidate: use library RI where present, else structure→RT. **Recommended** for mixed libraries.
- `CANON=canon_map.csv` — unify identities via a name/ik→canonical map (e.g. from a metabolite mapper) so
  libraries with different nomenclature are matched fairly. Strongly recommended for cross-library work.
- `PEAK_CAP=200` — max EIC peaks per m/z per file (memory bound; 200 saturates a normal library).
- `KIT_MODE=chem` (chem|random|spread|gap), `ROUNDS=10`, `FLOOR`, `CACHE_DIR`, `MAX_MZML=8`, `PLATFORM_TAG`.

Positional args: `KIT_SIZE(0=all) KIT_SEED K(chains)` — e.g. `0 0 10`.

## What was proven vendor-free on ST004581 (see ../*.md)

| dimension | vendor-free source | result |
|---|---|---|
| compound list / identity | public metabolome + metabolite mapper | public ≥ vendor (canonical-matched: HMDB 0.72 vs DD 0.69 weighted) |
| m/z (small molecules) | `build_library.py` comprehensive adducts | recovers + exceeds vendor's observed m/z |
| m/z (lipids) | `build_library.py` lipid-name parser | ~50–60% of structureless lipids (validated vs vendor m/z) |
| measured RI | self-built quasi-library (kit-free bootstrap) | small isobar lever (~+0.05) |
| kit calibration | bootstrap from confident MS1-unique + MS2 IDs | matches/beats curated kit |

Honest floor: stereo/geometric isomers need *measured* retention (no 2D model separates them), and a
small slice (~3–4%) of genuinely-ambiguous lipids need an authentic standard.

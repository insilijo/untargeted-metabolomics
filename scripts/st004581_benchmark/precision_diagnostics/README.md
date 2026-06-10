# Precision diagnostics (VPS-run scratch)

Scripts behind `../PRECISION_ANALYSIS.md`. Paths are hardcoded for the VPS environment
(`feat_colu.parquet`, `/root/SQuID-INC/...` DD + MAF, `/tmp/dd_pubchem_smiles.csv`).
Run with the `mmenv` interpreter; each prints its result to stdout.

- `pr_tradeoff.py`   — closed-world compound-level precision/recall, swept by match score
- `cw_eval.py`       — closed-world (presence) precision/recall from an annotation CSV
- `fp_taxonomy.py` / `fp_char.py` — classify FP calls into mechanistic groups
- `noise_classify.py`— find the pattern in the 71% noise bucket (satellite/weak)
- `pr_intensity.py` / `quality_test.py` / `adduct_envelope.py` / `corr_test.py` / `presence_gate.py`
  — per-peak presence-signal discrimination tests (all AUC <= 0.63)

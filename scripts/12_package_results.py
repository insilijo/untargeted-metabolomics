from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config


def build_feature_summary(
    features_path: Path, reports_dir: Path, interim_dir: Path
) -> Path:
    # Build a simple per-feature summary table with evidence level.
    features = pd.read_csv(features_path, sep="\t")

    summary_path = reports_dir / "library_summary.csv"
    hits_path = reports_dir / "library_hits.csv"
    spec2mol_path = reports_dir / "spec2mol_decoded_all.csv"
    ms2_links_path = interim_dir / "ms2_feature_links.parquet"

    hits = pd.read_csv(hits_path) if hits_path.exists() else None
    spec2mol = pd.read_csv(spec2mol_path) if spec2mol_path.exists() else None

    out = features.copy()

    # Attach best cosine/name per rounded precursor m/z when available.
    if hits is not None and "cosine" in hits.columns and "precursor_mz" in hits.columns:
        hits = hits.copy()
        hits["precursor_mz_round"] = hits["precursor_mz"].round(3)
        name_col = "library_compound_name" if "library_compound_name" in hits.columns else "compound_label"
        best = (
            hits.groupby("precursor_mz_round")
            .agg(best_cosine=("cosine", "max"), best_library_name=(name_col, "first"))
            .reset_index()
        )
        out["precursor_mz_round"] = out["mz_mean"].round(3)
        out = out.merge(best, on="precursor_mz_round", how="left")
        out = out.drop(columns=["precursor_mz_round"])

    # Spec2Mol predictions (top SMILES list).
    if spec2mol is not None and not spec2mol.empty:
        spec2mol = spec2mol.copy()
        spec2mol["group_id"] = spec2mol["target_group"].str.replace(
            "target_group_", "", regex=False
        ).astype(int)
        spec2mol["predicted_smiles"] = spec2mol["predicted_smiles_list"].fillna("")
        out = out.merge(spec2mol[["group_id", "predicted_smiles"]], on="group_id", how="left")
    else:
        out["predicted_smiles"] = ""

    def evidence(row: pd.Series) -> str:
        if pd.notna(row.get("best_cosine")):
            return "MS2 library"
        if isinstance(row.get("ms1_library_name"), str) and row.get("ms1_library_name"):
            return "MS1 library"
        if isinstance(row.get("predicted_smiles"), str) and row.get("predicted_smiles"):
            return "Spec2Mol"
        return "Unassigned"

    out["evidence_level"] = out.apply(evidence, axis=1)

    cols = [
        "group_id",
        "mz_mean",
        "rt_mean",
        "max_mix_intensity",
        "best_library_name",
        "best_cosine",
        "ms1_library_name",
        "ms1_library_title",
        "ms1_library_pepmass",
        "predicted_smiles",
        "evidence_level",
    ]
    cols = [c for c in cols if c in out.columns]
    summary_out = out[cols].copy().sort_values("max_mix_intensity", ascending=False)

    out_path = reports_dir / "feature_summary.csv"
    summary_out.to_csv(out_path, index=False)
    return out_path


def build_top_candidates_enriched(
    reports_dir: Path, processed_dir: Path
) -> Path | None:
    # Attach Spec2Mol SMILES + MS1/library columns to top_candidates.csv.
    top_path = reports_dir / "top_candidates.csv"
    if not top_path.exists():
        return None
    top = pd.read_csv(top_path)

    # Prefer the enriched feature table if it exists (contains MS1 matches).
    enriched_path = processed_dir / "feature_groups_filtered_adduct.tsv"
    if enriched_path.exists():
        features = pd.read_csv(enriched_path, sep="\t")
        keep_cols = [
            "group_id",
            "library_compound_name",
            "library_cosine",
            "ms1_library_name",
            "ms1_library_title",
            "ms1_library_pepmass",
            "ms1_library_delta_ppm",
            "ms1_library_smiles",
        ]
        keep_cols = [c for c in keep_cols if c in features.columns]
        if keep_cols:
            top = top.merge(features[keep_cols], on="group_id", how="left")
    else:
        # Fallback: merge MS1 annotations from spec2mol targets features if present.
        ms1_fallback = reports_dir / "spec2mol_targets_features.csv"
        if ms1_fallback.exists():
            features = pd.read_csv(ms1_fallback)
            keep_cols = [
                "group_id",
                "ms1_library_name",
                "ms1_library_title",
                "ms1_library_pepmass",
                "ms1_library_delta_ppm",
            ]
            keep_cols = [c for c in keep_cols if c in features.columns]
            if keep_cols:
                top = top.merge(features[keep_cols], on="group_id", how="left")

    spec2mol_path = reports_dir / "spec2mol_decoded_all.csv"
    if spec2mol_path.exists():
        spec = pd.read_csv(spec2mol_path)
        if "target_group" in spec.columns:
            spec = spec.copy()
            spec["group_id"] = spec["target_group"].str.replace(
                "target_group_", "", regex=False
            ).astype(int)
            spec["spec2mol_smiles"] = spec["predicted_smiles_list"].fillna("")
            top = top.merge(spec[["group_id", "spec2mol_smiles"]], on="group_id", how="left")

    out_path = reports_dir / "top_candidates_enriched.csv"
    top.to_csv(out_path, index=False)
    return out_path


def write_key_tables(files: list[Path], reports_dir: Path) -> list[Path]:
    # Return a curated list of key files to place at the bundle root.
    kept: list[Path] = []
    for file in files:
        if file is None:
            continue
        p = Path(file)
        if p.exists():
            kept.append(p)
    return kept


def collect_files(root: Path) -> list[Path]:
    # Collect outputs to include in the results bundle.
    candidates = [
        root / "reports" / "feature_summary.csv",
        root / "reports" / "top_candidates.csv",
        root / "reports" / "library_summary.csv",
        root / "reports" / "library_hits.csv",
        root / "reports" / "spec2mol_decoded_all.csv",
        root / "reports" / "spec2mol_targets.mgf",
        root / "reports" / "spec2mol_targets_features.csv",
        root / "reports" / "spec2mol_targets_spectra.csv",
        root / "reports" / "spec2mol_inputs",
        root / "reports" / "spec2mol_outputs",
        root / "reports" / "pubchem_smiles_cache.json",
        root / "figures",
        root / "notebooks",
    ]

    files: list[Path] = []
    for item in candidates:
        if item.is_dir():
            files.extend([p for p in item.rglob("*") if p.is_file()])
        elif item.exists():
            files.append(item)
    return sorted(set(files))


def write_readme(path: Path, file_list: list[Path]) -> None:
    # Write a short README describing the bundled outputs.
    lines = [
        "# Untargeted Metabolomics Results Bundle",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Included files:",
    ]
    for file in file_list:
        rel = file.as_posix()
        lines.append(f"- {rel}")
    lines.append("")
    lines.append("Key tables:")
    lines.append("- reports/feature_summary.csv: final per-feature summary with evidence level.")
    lines.append("- reports/top_candidates.csv: top features table.")
    lines.append("- reports/library_summary.csv: GNPS library summary.")
    lines.append("- reports/spec2mol_decoded_all.csv: Spec2Mol decoded SMILES list.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Build a summary table and zip all relevant outputs with a README.
    cfg = load_config()
    root = Path(".")
    reports_dir = Path(cfg["paths"]["reports_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs([reports_dir])

    features_path = processed_dir / "feature_groups_filtered_adduct.tsv"
    if features_path.exists():
        build_feature_summary(features_path, reports_dir, interim_dir)

    enriched_top = build_top_candidates_enriched(reports_dir, processed_dir)

    # Curated key files to place at the bundle root.
    pdf_path = reports_dir / "pipeline_overview.pdf"
    if not pdf_path.exists():
        pdf_path = root / "notebooks" / "pipeline_overview.pdf"
    key_files = write_key_tables(
        [
            enriched_top,
            processed_dir / "feature_groups_filtered_adduct.tsv",
            pdf_path,
        ],
        reports_dir,
    )

    files = collect_files(root)
    readme_path = reports_dir / "README_results.txt"
    write_readme(readme_path, files)

    zip_path = reports_dir / "results_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_path, "README_results.txt")
        # Add curated key files at the bundle root.
        for file in key_files:
            zf.write(file, Path(file).name)
        # Add the rest of the files under their paths.
        for file in files:
            zf.write(file, file.as_posix())

    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()

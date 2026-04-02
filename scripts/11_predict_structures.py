from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import sqlite3
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:
    from utils import ensure_dirs, load_config  # type: ignore
except ModuleNotFoundError:
    def ensure_dirs(paths):  # type: ignore
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)

    def load_config(path: str = "scripts/config.yaml") -> dict:  # type: ignore
        # Minimal fallback parser for key: value YAML with one-level nesting.
        config = {
            "paths": {
                "raw_dir": "data/raw",
                "interim_dir": "data/interim",
                "processed_dir": "data/processed",
                "reports_dir": "reports",
            },
            "library_search": {"mz_tolerance_da": 0.01},
            "spec2mol": {"conda_env": "spec2mol", "conda_exe": "conda"},
        }
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return config

        current = None
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0].strip()
                current = key
                continue
            if current is None or ":" not in line:
                continue
            subkey, value = line.strip().split(":", 1)
            value = value.strip()
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if current in config:
                if value.replace(".", "", 1).isdigit():
                    value = float(value)
                config[current][subkey] = value
        return config


def write_mgf(path: Path, spectra: list[dict]) -> None:
    # Write a list of spectra to MGF format.
    with open(path, "w", encoding="utf-8") as handle:
        for spec in spectra:
            handle.write("BEGIN IONS\n")
            handle.write(f"PEPMASS={spec['precursor_mz']}\n")
            handle.write(f"RTINSECONDS={spec['rt']}\n")
            handle.write(f"TITLE={spec['title']}\n")
            for mz, intensity in zip(spec["mz_array"], spec["intensity_array"]):
                handle.write(f"{mz} {intensity}\n")
            handle.write("END IONS\n")


def write_spec2mol_csv(path: Path, mz_array, intensity_array) -> None:
    # Write a Spec2Mol input CSV for one spectrum.
    with open(path, "w", encoding="utf-8") as handle:
        mz_list = list(mz_array)
        intensity_list = list(intensity_array)
        for mz, intensity in zip(mz_list, intensity_list):
            handle.write(f"{mz},{intensity}\n")


def fetch_pubchem_smiles(inchikey: str, cache: dict[str, str]) -> str:
    # Fetch PubChem SMILES by InChIKey with caching.
    if not inchikey:
        return ""
    if inchikey in cache:
        if cache[inchikey]:
            return cache[inchikey]
    try:
        encoded = urllib.parse.quote(inchikey)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            f"{encoded}/property/CanonicalSMILES/JSON"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                cache[inchikey] = ""
                return ""
            data = json.loads(resp.read().decode("utf-8"))
            smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            cache[inchikey] = smiles
            return smiles
    except Exception:
        cache[inchikey] = ""
        return ""


def fetch_pubchem_smiles_by_name(name: str, cache: dict[str, str]) -> str:
    # Fetch PubChem SMILES by name with caching.
    if not name:
        return ""
    candidates = [name]
    if "_" in name:
        candidates.append(name.replace("_", " "))
    if name.startswith("\"") and name.endswith("\""):
        candidates.append(name.strip("\""))
    candidates = [c.strip() for c in candidates if c.strip()]
    for cand in candidates:
        key = f"name:{cand}"
        if key in cache:
            if cache[key]:
                return cache[key]
            continue
        try:
            encoded = urllib.parse.quote(cand)
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{encoded}/property/CanonicalSMILES/JSON"
            )
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status != 200:
                    cache[key] = ""
                    continue
                data = json.loads(resp.read().decode("utf-8"))
                smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                cache[key] = smiles
                return smiles
        except Exception:
            cache[key] = ""
            continue
    return ""


def select_targets(
    # Select top unannotated features for Spec2Mol.
    features: pd.DataFrame,
    hits: pd.DataFrame,
    mz_tol: float,
    top_n: int = 25,
) -> pd.DataFrame:
    def has_hit(mz: float) -> bool:
        subset = hits[(hits["precursor_mz"] >= mz - mz_tol) & (hits["precursor_mz"] <= mz + mz_tol)]
        return not subset.empty

    features = features.copy()
    features["has_hit"] = features["mz_mean"].apply(has_hit)

    # Top N features by intensity, then filter to unannotated.
    top_features = features.sort_values("max_mix_intensity", ascending=False).head(top_n)
    candidates = top_features[~top_features["has_hit"]]

    return candidates.sort_values("max_mix_intensity", ascending=False)


def main() -> None:
    # Run Spec2Mol workflow and merge annotations.
    import argparse

    parser = argparse.ArgumentParser(description="Spec2Mol export + prediction.")
    parser.add_argument("--top-n", type=int, default=25, help="Top N unannotated features to predict.")
    args = parser.parse_args()

    cfg = load_config()
    interim_dir = Path(cfg["paths"]["interim_dir"]).resolve()
    reports_dir = Path(cfg["paths"]["reports_dir"]).resolve()
    ensure_dirs([reports_dir])

    top_candidates_path = reports_dir / "top_candidates.csv"
    features_path = Path(cfg["paths"]["processed_dir"]) / "feature_groups_filtered_adduct.tsv"
    hits_path = Path(cfg["paths"]["processed_dir"]) / "library_hits.parquet"
    links_path = interim_dir / "ms2_feature_links.parquet"
    spectra_path = interim_dir / "ms2_spectra.parquet"

    if not top_candidates_path.exists():
        raise SystemExit("Missing top_candidates.csv. Run 10_report_tables.py first.")
    if not features_path.exists() or not hits_path.exists():
        raise SystemExit("Missing adduct-filtered features or library hits.")
    if not links_path.exists() or not spectra_path.exists():
        raise SystemExit("Missing MS2 links/spectra. Run 06_extract_ms2.py and 07_link_ms2_features.py first.")

    top_candidates = pd.read_csv(top_candidates_path)
    features = pd.read_csv(features_path, sep="\t")
    hits = pd.read_parquet(hits_path)
    links = pd.read_parquet(links_path)
    spectra = pd.read_parquet(spectra_path)

    spectra_index = spectra.set_index(spectra.index)
    ms2_by_group = links.groupby("group_id")["ms2_index"].first().to_dict()

    # Attach best library hit and PubChem SMILES to the features table.
    cache_path = reports_dir / "pubchem_smiles_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        cache = {}

    ms2_available = set(links["group_id"]) if links is not None and not links.empty else set()
    features["ms2_status"] = features["group_id"].apply(
        lambda gid: "ok" if int(gid) in ms2_available else "no MS2"
    )

    # Features already carry MS2 annotations from earlier steps; do not rebuild them here.

    # If MS2 annotations are missing, fill from library_hits using MS2 links.
    if (
        hits is not None
        and not hits.empty
        and links is not None
        and not links.empty
        and "library_compound_name" in features.columns
    ):
        mz_tol = cfg["library_search"]["mz_tolerance_da"]
        hits_small = hits[
            ["source_file", "precursor_mz", "library_compound_name", "library_inchikey", "cosine"]
        ].copy()
        links_small = links[["group_id", "source_file", "precursor_mz"]].copy()
        merged_rows = []
        for _, link in links_small.iterrows():
            prec = float(link["precursor_mz"])
            subset = hits_small[
                (hits_small["source_file"] == link["source_file"])
                & (hits_small["precursor_mz"] >= prec - mz_tol)
                & (hits_small["precursor_mz"] <= prec + mz_tol)
            ]
            if subset.empty:
                continue
            best = subset.sort_values("cosine", ascending=False).iloc[0].to_dict()
            best["group_id"] = link["group_id"]
            merged_rows.append(best)
        if merged_rows:
            merged_hits = pd.DataFrame(merged_rows).sort_values("cosine", ascending=False)
            hit_summary = merged_hits.groupby("group_id", as_index=False).head(1)
            hit_summary = hit_summary.rename(columns={"cosine": "library_cosine"})
            features = features.merge(
                hit_summary[["group_id", "library_compound_name", "library_inchikey", "library_cosine"]],
                on="group_id",
                how="left",
                suffixes=("", "_ms2"),
            )
            for col in ["library_compound_name", "library_inchikey", "library_cosine"]:
                ms2_col = f"{col}_ms2"
                if ms2_col in features.columns:
                    features[col] = features[col].fillna(features[ms2_col])
                    features.drop(columns=[ms2_col], inplace=True)

    # Ensure columns exist even if there were no MS2 hits to merge.
    if "library_compound_name" not in features.columns:
        features["library_compound_name"] = pd.NA
    if "library_inchikey" not in features.columns:
        features["library_inchikey"] = pd.NA
    if "library_cosine" not in features.columns:
        features["library_cosine"] = np.nan

    if "library_inchikey" in features.columns:
        features["library_smiles"] = features["library_inchikey"].fillna("").apply(
            lambda key: fetch_pubchem_smiles(str(key), cache)
        )
        if "library_compound_name" in features.columns:
            missing = features["library_smiles"] == ""
            features.loc[missing, "library_smiles"] = features.loc[missing, "library_compound_name"].fillna("").apply(
                lambda name: fetch_pubchem_smiles_by_name(str(name), cache)
            )
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    else:
        print("Library InChIKey column missing; skipping PubChem SMILES lookup.")

    # MS1-only matching for features without MS2-based hits or cosine.
    db_path = interim_dir / cfg.get("inputs", {}).get("gnps_sqlite", "gnps_library.sqlite")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        mz_tol = cfg["library_search"]["mz_tolerance_da"]
        isotope_delta = 1.003355
        ms1_name = []
        ms1_inchikey = []
        ms1_title = []
        ms1_pepmass = []
        ms1_ppm = []
        for _, row in features.iterrows():
            if row.get("ms2_status") != "no MS2":
                ms1_name.append("")
                ms1_inchikey.append("")
                ms1_title.append("")
                ms1_pepmass.append(np.nan)
                ms1_ppm.append(np.nan)
                continue
            if pd.notna(row.get("library_cosine")):
                ms1_name.append("")
                ms1_inchikey.append("")
                ms1_title.append("")
                ms1_pepmass.append(np.nan)
                ms1_ppm.append(np.nan)
                continue
            mz = float(row["mz_mean"])
            candidates = [mz, mz - isotope_delta, mz + isotope_delta]
            candidates = [m for m in candidates if m > 0]
            best_row = None
            best_ppm = None
            for cand in candidates:
                cursor = conn.execute(
                    """
                    SELECT title, name, inchikey, pepmass
                    FROM spectra
                    WHERE pepmass BETWEEN ? AND ?
                    """,
                    (cand - mz_tol, cand + mz_tol),
                )
                rows = cursor.fetchall()
                if not rows:
                    continue
                local_best = min(rows, key=lambda r: abs(r[3] - cand))
                delta_ppm = (local_best[3] - cand) / cand * 1e6
                if best_ppm is None or abs(delta_ppm) < abs(best_ppm):
                    best_ppm = delta_ppm
                    best_row = local_best
            if best_row is None:
                ms1_name.append("")
                ms1_inchikey.append("")
                ms1_title.append("")
                ms1_pepmass.append(np.nan)
                ms1_ppm.append(np.nan)
                continue
            ms1_title.append(best_row[0])
            ms1_name.append(best_row[1] or "")
            ms1_inchikey.append(best_row[2] or "")
            ms1_pepmass.append(float(best_row[3]))
            ms1_ppm.append(float(best_ppm))
        conn.close()
        features["ms1_library_name"] = ms1_name
        features["ms1_library_inchikey"] = ms1_inchikey
        features["ms1_library_title"] = ms1_title
        features["ms1_library_pepmass"] = ms1_pepmass
        features["ms1_library_delta_ppm"] = ms1_ppm

        if "ms1_library_inchikey" in features.columns:
            features["ms1_library_smiles"] = features["ms1_library_inchikey"].fillna("").apply(
                lambda key: fetch_pubchem_smiles(str(key), cache)
            )
            if "ms1_library_name" in features.columns:
                missing = features["ms1_library_smiles"] == ""
                features.loc[missing, "ms1_library_smiles"] = features.loc[missing, "ms1_library_name"].fillna("").apply(
                    lambda name: fetch_pubchem_smiles_by_name(str(name), cache)
                )
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        # Promote MS1-only match to primary label if no MS2 hit exists.
        name_missing = features["library_compound_name"].isna() | (
            features["library_compound_name"] == ""
        )
        key_missing = features["library_inchikey"].isna() | (features["library_inchikey"] == "")
        features["library_compound_name"] = features["library_compound_name"].where(
            ~name_missing, features["ms1_library_name"]
        )
        features["library_inchikey"] = features["library_inchikey"].where(
            ~key_missing, features["ms1_library_inchikey"]
        )
        features["library_match_type"] = np.where(
            features["library_compound_name"].notna() & (features["library_compound_name"] != ""),
            np.where(features["library_cosine"].notna(), "ms2", "ms1"),
            "none",
        )
    else:
        features["library_match_type"] = "none"

    # Ensure MS1 columns exist even if MS1 matching didn't run.
    for col in [
        "ms1_library_name",
        "ms1_library_inchikey",
        "ms1_library_title",
        "ms1_library_pepmass",
        "ms1_library_delta_ppm",
        "ms1_library_smiles",
    ]:
        if col not in features.columns:
            features[col] = "" if col.endswith("name") or col.endswith("title") or col.endswith("smiles") else np.nan

    features_path.parent.mkdir(parents=True, exist_ok=True)
    # Clean any accidental merge suffix columns from prior runs.
    # Drop any lingering merge suffix columns from earlier runs.
    suffix_pattern = r"_x(\\.|$)|_y(\\.|$)"
    suffix_cols = [c for c in features.columns if pd.Series([c]).str.contains(suffix_pattern).iloc[0]]
    if suffix_cols:
        features.drop(columns=suffix_cols, inplace=True)
    features.to_csv(features_path, sep="\t", index=False)
    print(f"Wrote {features_path} with library annotations.")

    targets = select_targets(
        features=features,
        hits=hits,
        mz_tol=cfg["library_search"]["mz_tolerance_da"],
        top_n=args.top_n,
    )
    targets_path = reports_dir / "spec2mol_targets_features.csv"
    targets.to_csv(targets_path, index=False)
    print(f"Wrote {targets_path}")

    export = []
    ms2_available = set(ms2_by_group.keys())
    for _, row in targets.iterrows():
        gid = int(row["group_id"])
        targets.loc[targets["group_id"] == gid, "ms2_status"] = (
            "ok" if gid in ms2_available else "no MS2"
        )
        if gid not in ms2_by_group:
            continue
        idx = int(ms2_by_group[gid])
        spec = spectra_index.loc[idx]
        export.append(
            {
                "title": f"target_group_{gid}",
                "precursor_mz": float(spec["precursor_mz"]),
                "rt": float(spec["rt"]),
                "mz_array": spec["mz_array"],
                "intensity_array": spec["intensity_array"],
            }
        )

    if not export:
        raise SystemExit("No MS2 spectra found to export for unmatched clusters.")
    print(f"Found {len(export)} target MS2 spectra for Spec2Mol.")

    mgf_path = reports_dir / "spec2mol_targets.mgf"
    write_mgf(mgf_path, export)
    print(f"Wrote {mgf_path}")

    csv_path = reports_dir / "spec2mol_targets_spectra.csv"
    rows = []
    for spec in export:
        rows.append(
            {
                "title": spec["title"],
                "precursor_mz": spec["precursor_mz"],
                "rt": spec["rt"],
                "mz_array_json": json.dumps(list(spec["mz_array"])),
                "intensity_array_json": json.dumps(list(spec["intensity_array"])),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    spec2mol_inputs = reports_dir / "spec2mol_inputs"
    spec2mol_inputs.mkdir(parents=True, exist_ok=True)
    # Remove legacy empty placeholder files from earlier runs.
    for stale in spec2mol_inputs.glob("*_pos_high.csv"):
        stale.unlink()
    for stale in spec2mol_inputs.glob("*_neg_low.csv"):
        stale.unlink()
    for stale in spec2mol_inputs.glob("*_neg_high.csv"):
        stale.unlink()
    for spec in export:
        base = spec2mol_inputs / spec["title"]
        write_spec2mol_csv(
            base.with_name(base.name + "_pos_low.csv"),
            spec["mz_array"],
            spec["intensity_array"],
        )
    print(f"Wrote spec2mol input CSVs to {spec2mol_inputs}")

    spec2mol_dir = Path("/home/jgardner/Spec2Mol")
    if not spec2mol_dir.exists():
        print("Spec2Mol directory not found. Run Spec2Mol manually with reports/spec2mol_inputs/*_pos_low.csv.")
        return

    # Run Spec2Mol in-process (expects this script to run inside the spec2mol conda env).
    sys.path.insert(0, str(spec2mol_dir))
    sys.path.insert(0, str(spec2mol_dir / "decoder"))

    from argparse import Namespace
    from predict_embs import main as encode_main
    from decoder.scripts import decode_embeddings

    model_dir = spec2mol_dir / "decoder" / "models"
    outputs_root = reports_dir / "spec2mol_outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(spec2mol_dir)
    failures = 0
    for spec in export:
        pos_low = spec2mol_inputs / f"{spec['title']}_pos_low.csv"
        if not pos_low.exists() or pos_low.stat().st_size == 0:
            print(f"Skipping {spec['title']}: missing/empty {pos_low}.")
            failures += 1
            continue
        out_dir = outputs_root / spec["title"]
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            args = Namespace(
                pos_low_file=str(pos_low),
                pos_high_file="",
                neg_low_file="",
                neg_high_file="",
            )
            emb = encode_main(args)
            embeddings_path = out_dir / "predicted_embeddings.pt"
            import torch

            torch.save({spec["title"]: emb.squeeze(0)}, embeddings_path)

            decode_cfg = Namespace(
                output_file=str(out_dir / "decoded_output.csv"),
                predicted_embeddings=str(embeddings_path),
                model="translation",
                device="cpu",
                model_load=str(model_dir / "model.pt"),
                vocab_load=str(model_dir / "vocab.nb"),
                config_load=str(model_dir / "config.nb"),
                n_batch=65,
                num_variants=3,
            )
            decode_embeddings.main(decode_cfg.model, decode_cfg)
            print(f"Wrote Spec2Mol output for {spec['title']}.")
        except Exception as exc:
            failures += 1
            print(f"Spec2Mol failed for {spec['title']}: {exc}")

    os.chdir(cwd)
    print(f"Wrote Spec2Mol outputs to {outputs_root} with {failures} failures.")

    # Merge Spec2Mol outputs into a single table for downstream review.
    merged_rows = []
    for out_dir in outputs_root.glob("target_group_*"):
        decoded_path = out_dir / "decoded_output.csv"
        if not decoded_path.exists():
            continue
        df = pd.read_csv(decoded_path)
        df["target_group"] = out_dir.name
        merged_rows.append(df)
    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        merged_path = reports_dir / "spec2mol_decoded_all.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Wrote {merged_path}")


if __name__ == "__main__":
    main()

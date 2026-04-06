"""Prepare SQuID-InC inputs from a Metabolon Excel data table.

Outputs two files:

  metabolon_annotations.csv
      Merged Chemical Annotation + data dictionary: mz (MASS), rt (RI),
      inchikey, smiles, name, platform.  Used as annotation_csv in config.yaml.

  sample_metadata.csv
      One row per mzML file with filename, sample_type, and all metadata
      columns from the Sample Meta Data sheet.
      sample_type mapping:
        CMTRX prefix → PSS   (process system suitability)
        BLANK prefix  → BLANK
        QC prefix     → QC
        anything else → SAMPLE

Join strategy for annotations
------------------------------
1. Primary   — INCHIKEY exact match between Chemical Annotation and dict
2. Secondary — CHEMICAL_NAME vs BIOCHEMICAL (case-insensitive) for rows
               the dict lacks an INCHIKEY for

Usage
-----
    python scripts/prepare_metabolon.py \\
        --excel data/raw/MyStudy_DataTables.xlsx \\
        --mzml-zip data/raw/mzmls.zip           # optional; infers filenames
        --out   data/raw/metabolon_annotations.csv

    # config.yaml is updated automatically
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Sample type mapping
# ---------------------------------------------------------------------------

_SAMPLE_TYPE_MAP = {
    "cmtrx": "PSS",
    "blank":  "BLANK",
    "qc":     "QC",
    "prcs":   "BLANK",   # process blank
}


def _infer_sample_type(sample_name: str) -> str:
    prefix = str(sample_name).split("-")[0].lower()
    return _SAMPLE_TYPE_MAP.get(prefix, "SAMPLE")


# ---------------------------------------------------------------------------
# Platform normalisation
# ---------------------------------------------------------------------------

_PLATFORM_MAP = {
    "pos early": "lc/ms pos early",
    "pos late":  "lc/ms pos late",
    "neg":       "lc/ms neg",
    "polar":     "lc/ms polar",
}


def _norm_platform(val) -> str:
    if not val or pd.isna(val):
        return ""
    return _PLATFORM_MAP.get(str(val).strip().lower(), str(val).strip().lower())


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------

def build_sample_metadata(
    excel_path: Path,
    mzml_zip: Path | None,
    out_dir: Path,
) -> Path:
    """Read Sample Meta Data sheet and match to mzML filenames.

    mzML files are identified from mzml_zip (if supplied) or assumed to sit
    in out_dir's parent raw/ directory.  Files are matched to metadata rows
    by PARENT_SAMPLE_NAME appearing anywhere in the mzML filename stem.
    Files not matched to any metadata row are typed by name prefix.
    """
    print("Loading Sample Meta Data …")
    smd = pd.read_excel(excel_path, sheet_name="Sample Meta Data")
    smd.columns = [c.strip() for c in smd.columns]

    # Build PARENT_SAMPLE_NAME → row lookup
    smd["sample_type"] = smd["PARENT_SAMPLE_NAME"].apply(_infer_sample_type)
    meta_by_name = {
        str(row["PARENT_SAMPLE_NAME"]).strip(): row
        for _, row in smd.iterrows()
    }

    # Collect mzML filenames
    mzml_names: list[str] = []
    if mzml_zip and mzml_zip.exists():
        with zipfile.ZipFile(mzml_zip) as zf:
            mzml_names = [
                Path(n).name for n in zf.namelist()
                if Path(n).suffix.lower() == ".mzml"
                and not Path(n).name.startswith("._")
                and "__MACOSX" not in n
            ]
    else:
        # Look for already-extracted files in out_dir (raw/)
        mzml_names = [p.name for p in sorted(out_dir.glob("*.mzML"))]

    # Match each mzML to a metadata row by PARENT_SAMPLE_NAME suffix
    rows = []
    for fname in sorted(mzml_names):
        stem = Path(fname).stem  # e.g. Method3_Set181307_07_COLU-00965
        # Try longest suffix match: last _-delimited token, then progressively more
        matched_row = None
        parts = stem.split("_")
        for n in range(1, len(parts) + 1):
            candidate = "_".join(parts[-n:])
            if candidate in meta_by_name:
                matched_row = meta_by_name[candidate]
                break

        if matched_row is not None:
            rec = matched_row.to_dict()
            rec["filename"] = fname
            rec["sample_type"] = _infer_sample_type(str(matched_row["PARENT_SAMPLE_NAME"]))
        else:
            # Not in metadata sheet — type by filename
            sample_name = parts[-1] if parts else stem
            rec = {
                "filename":            fname,
                "PARENT_SAMPLE_NAME":  sample_name,
                "sample_type":         _infer_sample_type(sample_name),
            }
        rows.append(rec)

    meta_df = pd.DataFrame(rows)

    # Ensure filename and sample_type are first columns
    cols = ["filename", "sample_type"] + [
        c for c in meta_df.columns if c not in ("filename", "sample_type")
    ]
    meta_df = meta_df[[c for c in cols if c in meta_df.columns]]

    out_path = out_dir / "sample_metadata.csv"
    meta_df.to_csv(out_path, index=False)

    counts = meta_df["sample_type"].value_counts().to_dict()
    print(f"  Sample metadata: {len(meta_df)} files — {counts}")
    print(f"  → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Annotation merge
# ---------------------------------------------------------------------------

def merge_annotations(
    excel_path: Path,
    dict_path: Path,
    out_path: Path,
    update_config: bool = True,
) -> None:
    print(f"Loading Chemical Annotation from {excel_path.name} …")
    ca = pd.read_excel(excel_path, sheet_name="Chemical Annotation")
    ca.columns = [c.strip() for c in ca.columns]

    print(f"Loading data dictionary from {dict_path.name} …")
    dd = pd.read_csv(dict_path)
    dd.columns = [c.strip() for c in dd.columns]

    ca["_ik"]   = ca["INCHIKEY"].fillna("").str.strip()
    ca["_name"] = ca["CHEMICAL_NAME"].fillna("").str.strip().str.lower()
    dd["_ik"]   = dd["INCHIKEY"].fillna("").str.strip()
    dd["_name"] = dd["BIOCHEMICAL"].fillna("").str.strip().str.lower()

    # Pass 1: INCHIKEY join
    dd_keyed = dd[dd["_ik"] != ""].set_index("_ik")
    rows = []
    matched_ik: set[str] = set()

    for _, r in ca.iterrows():
        ik = r["_ik"]
        rec = _base_record(r)
        if ik and ik in dd_keyed.index:
            _fill_from_dict(rec, dd_keyed.loc[ik])
            matched_ik.add(ik)
        rows.append(rec)

    # Pass 2: name join for unmatched
    dd_named = dd[dd["_ik"] == ""].set_index("_name")
    dd_named_extra = dd[~dd["_ik"].isin(matched_ik)].set_index("_name")

    n_name_matched = 0
    for rec in rows:
        if rec["mz"] is not None:
            continue
        name = rec.pop("_name_key", "")
        if name in dd_named.index:
            _fill_from_dict(rec, dd_named.loc[name])
            n_name_matched += 1
        elif name in dd_named_extra.index:
            _fill_from_dict(rec, dd_named_extra.loc[name])
            n_name_matched += 1
        else:
            rec["_name_key"] = name  # put back if not matched (cleaned below)

    for rec in rows:
        rec.pop("_name_key", None)

    out_df = pd.DataFrame(rows)
    out_df = out_df[out_df["mz"].notna() | out_df["inchikey"].notna() | out_df["smiles"].notna()]
    out_df = out_df.sort_values("mz", na_position="last").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nAnnotation merge summary")
    print(f"  Total rows      : {len(out_df)}")
    print(f"  INCHIKEY matched: {len(matched_ik)}")
    print(f"  Name matched    : {n_name_matched}")
    print(f"  With m/z        : {out_df['mz'].notna().sum()}")
    print(f"  With RT/RI      : {out_df['rt'].notna().sum()}")
    print(f"  With SMILES     : {out_df['smiles'].notna().sum()}")
    print(f"  With InChIKey   : {out_df['inchikey'].notna().sum()}")
    print(f"\n→ {out_path}")

    if update_config:
        _write_config(out_path)


def _base_record(r: pd.Series) -> dict:
    return {
        "chem_id":   r.get("CHEM_ID"),
        "name":      r.get("CHEMICAL_NAME", ""),
        "inchikey":  r["_ik"] or None,
        "smiles":    r.get("SMILES") if pd.notna(r.get("SMILES", None)) else None,
        "platform":  _norm_platform(r.get("PLATFORM")),
        "pubchem":   r.get("PUBCHEM") if pd.notna(r.get("PUBCHEM", None)) else None,
        "mz":        None,
        "rt":        None,
        "_name_key": r["_name"],
    }


def _fill_from_dict(rec: dict, drow: pd.Series | pd.DataFrame) -> None:
    if isinstance(drow, pd.DataFrame):
        drow = drow.iloc[0]
    mass = drow.get("MASS")
    ri   = drow.get("RI")
    if pd.notna(mass):
        rec["mz"] = float(mass)
    if pd.notna(ri):
        rec["rt"] = float(ri)
    if not rec["inchikey"]:
        ik = drow.get("INCHIKEY", "")
        rec["inchikey"] = str(ik).strip() if pd.notna(ik) and str(ik).strip() else None
    if not rec["name"]:
        rec["name"] = str(drow.get("BIOCHEMICAL", "")).strip()


def _write_config(out_path: Path) -> None:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        return

    try:
        rel = out_path.relative_to(config_path.parent.parent)
    except ValueError:
        rel = out_path

    text = config_path.read_text()
    new_line = f"  annotation_csv: {rel}"

    if "annotation_csv:" in text:
        text = re.sub(r"[ \t]*#?[ \t]*annotation_csv:.*", new_line, text)
    else:
        text = re.sub(
            r"([ \t]*cross_community_damping:[^\n]*)",
            r"\1\n" + new_line,
            text,
        )

    config_path.write_text(text)
    print(f"  Updated {config_path.name}: annotation_csv = {rel}")

    # Also wire sample_metadata into inputs section
    meta_path = out_path.parent / "sample_metadata.csv"
    if meta_path.exists():
        try:
            meta_rel = meta_path.relative_to(config_path.parent.parent)
        except ValueError:
            meta_rel = meta_path
        text = config_path.read_text()
        new_meta_line = f"  sample_metadata: {meta_rel}"
        if "sample_metadata:" in text:
            text = re.sub(r"[ \t]*#?[ \t]*sample_metadata:.*", new_meta_line, text)
            config_path.write_text(text)
        else:
            text = re.sub(
                r"([ \t]*blank_glob:[^\n]*)",
                r"\1\n" + new_meta_line,
                text,
            )
            config_path.write_text(text)
        print(f"  Updated {config_path.name}: sample_metadata = {meta_rel}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dict() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parents[1], here.parents[2] / "SQuID-INC"]:
        p = base / "data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
        if p.exists():
            return p
    return Path("data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--excel",    required=True,
                   help="Metabolon Excel data tables (.xlsx)")
    p.add_argument("--mzml-zip", default=None,
                   help="Zip of mzML files (optional; used to build sample_metadata.csv)")
    p.add_argument("--dict",     default=str(_default_dict()),
                   help="Metabolon data dictionary CSV (auto-detected by default)")
    p.add_argument("--out",      default="data/raw/metabolon_annotations.csv",
                   help="Output annotation CSV (default: data/raw/metabolon_annotations.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    excel_path = Path(args.excel).expanduser().resolve()
    out_path   = Path(args.out).expanduser()
    out_dir    = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mzml_zip = Path(args.mzml_zip).expanduser().resolve() if args.mzml_zip else None

    build_sample_metadata(excel_path, mzml_zip, out_dir)
    merge_annotations(
        excel_path=excel_path,
        dict_path=Path(args.dict).expanduser().resolve(),
        out_path=out_path,
        update_config=True,
    )


if __name__ == "__main__":
    main()

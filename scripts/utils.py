from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def load_config(path: str | Path = "scripts/config.yaml") -> Dict:
    # Load YAML configuration into a dict.
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    # Create directories if they do not exist.
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict) -> None:
    # Write a JSON payload to disk.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def list_files(root: str | Path, pattern: str) -> List[Path]:
    # Return sorted files matching a glob under root (searches recursively).
    root_path = Path(root)
    return sorted(root_path.glob(pattern)) or sorted(root_path.glob(f"**/{pattern}"))


def require_files(files: Iterable[Path]) -> Tuple[bool, List[str]]:
    # Check for missing files and return status plus list.
    missing = [str(f) for f in files if not f.exists()]
    return (len(missing) == 0, missing)


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------

_SAMPLE_TYPES  = {"sample", "mix", "qc"}
_BLANK_TYPES   = {"blank", "process_blank", "solvent_blank", "method_blank"}
_IGNORE_TYPES  = {"reference", "standard", "ignore", "skip"}

# Columns recognised as sample_type (case-insensitive)
_TYPE_COLS = ["sample_type", "sampletype", "type", "class", "group"]
# Columns recognised as the filename (stem or full name)
_FILE_COLS = ["filename", "file_name", "file", "sample_name", "samplename", "mzml"]


def load_sample_metadata(
    raw_dir: Path,
    interim_dir: Path,
    cfg: Dict,
) -> Tuple[List[Path], List[Path], "pd.DataFrame | None"]:
    """Return (sample_files, blank_files, metadata_df).

    Resolution order
    ----------------
    1. ``interim/sample_metadata.tsv`` written by step 01 (preferred — already resolved)
    2. A metadata sheet named in ``cfg["inputs"]["sample_metadata"]`` (raw_dir or project root)
    3. Globs from ``cfg["inputs"]["mix_glob"]`` / ``cfg["inputs"]["blank_glob"]`` (legacy fallback)

    The metadata DataFrame (or None if globs were used) has at minimum:
        filename     — bare filename (e.g. ``Sample01.mzML``)
        filepath     — absolute Path
        sample_type  — normalised: ``sample`` | ``blank`` | ``qc`` | ``other``
    Plus any extra columns from the original sheet (batch, subject_id, etc.).
    """
    import pandas as pd

    resolved = interim_dir / "sample_metadata.tsv"
    if resolved.exists():
        meta = pd.read_csv(resolved, sep="\t")
        return _split_from_meta(meta, raw_dir), meta

    # User-supplied sheet
    sheet_name = (cfg.get("inputs") or {}).get("sample_metadata")
    if sheet_name:
        for base in [raw_dir, raw_dir.parent]:
            sheet_path = base / sheet_name
            if sheet_path.exists():
                meta = _read_metadata_sheet(sheet_path, raw_dir)
                return _split_from_meta(meta, raw_dir), meta

    # Glob fallback
    sample_files = list_files(raw_dir, cfg["inputs"]["mix_glob"])
    blank_files  = list_files(raw_dir, cfg["inputs"]["blank_glob"])
    return (sample_files, blank_files), None


def build_sample_metadata(
    raw_dir: Path,
    user_sheet: Path | None,
    zip_manifest: List[dict] | None = None,
) -> "pd.DataFrame":
    """Build a sample_metadata DataFrame from available information.

    Priority: user_sheet > zip manifest > all mzML files in raw_dir.
    Unknown filenames that exist in raw_dir are typed as 'sample' by default.
    """
    import pandas as pd

    if user_sheet is not None and user_sheet.exists():
        meta = _read_metadata_sheet(user_sheet, raw_dir)
    elif zip_manifest:
        rows = []
        for entry in zip_manifest:
            rows.append({
                "filename":    entry["filename"],
                "filepath":    str(raw_dir / entry["filename"]),
                "sample_type": _norm_type(entry.get("sample_type", "sample")),
                **{k: v for k, v in entry.items() if k not in ("filename", "sample_type")},
            })
        meta = pd.DataFrame(rows)
    else:
        # Discover all mzML files, infer type from filename
        mzml_files = sorted(raw_dir.glob("*.mzML")) + sorted(raw_dir.glob("*.mzml"))
        rows = []
        for p in mzml_files:
            rows.append({
                "filename":    p.name,
                "filepath":    str(p),
                "sample_type": _infer_type_from_name(p.name),
            })
        meta = pd.DataFrame(rows)

    return meta


def write_sample_metadata(meta: "pd.DataFrame", interim_dir: Path) -> Path:
    """Persist metadata to interim/sample_metadata.tsv and return path."""
    out = interim_dir / "sample_metadata.tsv"
    meta.to_csv(out, sep="\t", index=False)
    return out


def _read_metadata_sheet(path: Path, raw_dir: Path) -> "pd.DataFrame":
    """Read a user-supplied metadata sheet, normalise columns, resolve filepaths."""
    import pandas as pd

    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find filename column
    file_col = next((c for c in _FILE_COLS if c in df.columns), None)
    if file_col is None:
        raise ValueError(
            f"Metadata sheet {path.name} has no filename column. "
            f"Expected one of: {_FILE_COLS}"
        )
    df = df.rename(columns={file_col: "filename"})
    df["filename"] = df["filename"].astype(str).str.strip()
    # Add .mzML extension if missing
    df["filename"] = df["filename"].apply(
        lambda f: f if f.lower().endswith(".mzml") else f + ".mzML"
    )
    df["filepath"] = df["filename"].apply(lambda f: str(raw_dir / f))

    # Find and normalise sample_type column
    type_col = next((c for c in _TYPE_COLS if c in df.columns), None)
    if type_col:
        df["sample_type"] = df[type_col].apply(_norm_type)
        if type_col != "sample_type":
            df = df.drop(columns=[type_col])
    else:
        df["sample_type"] = df["filename"].apply(_infer_type_from_name)

    return df


def _split_from_meta(
    meta: "pd.DataFrame", raw_dir: Path
) -> Tuple[List[Path], List[Path]]:
    sample_files = [
        Path(r["filepath"]) for _, r in meta.iterrows()
        if r["sample_type"] in _SAMPLE_TYPES and Path(r["filepath"]).exists()
    ]
    blank_files = [
        Path(r["filepath"]) for _, r in meta.iterrows()
        if r["sample_type"] in _BLANK_TYPES and Path(r["filepath"]).exists()
    ]
    return sample_files, blank_files


def _norm_type(value: str) -> str:
    v = str(value).strip().lower()
    if v in _SAMPLE_TYPES:
        return "sample" if v == "mix" else v
    if v in _BLANK_TYPES:
        return "blank"
    if v in _IGNORE_TYPES:
        return "other"
    return "sample"  # default unknown to sample


def _infer_type_from_name(name: str) -> str:
    low = name.lower()
    if any(m in low for m in ("blank", "blk", "cmtrx", "process_blank")):
        return "blank"
    if any(m in low for m in ("qc", "quality_control")):
        return "qc"
    return "sample"

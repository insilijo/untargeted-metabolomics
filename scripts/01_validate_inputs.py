from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

from utils import (
    build_sample_metadata,
    ensure_dirs,
    load_config,
    write_json,
    write_sample_metadata,
)


def _extract_mzml_zip(zip_path: Path, raw_dir: Path) -> list[dict]:
    """Extract mzML files from a zip, keeping original filenames.

    Returns a list of dicts with keys: filename, sample_type (inferred).
    """
    from utils import _infer_type_from_name

    manifest = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [
                n for n in zf.namelist()
                if Path(n).suffix.lower() == ".mzml"
                and not Path(n).name.startswith("._")
                and "__MACOSX" not in n
            ]
            if not members:
                return []
            for member in members:
                dest = raw_dir / Path(member).name
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                manifest.append({
                    "filename":    Path(member).name,
                    "sample_type": _infer_type_from_name(Path(member).name),
                })
    except zipfile.BadZipFile:
        return []
    return manifest


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([raw_dir, interim_dir])

    # ── Locate and extract mzML files ────────────────────────────────────────
    zip_manifest: list[dict] | None = None
    mzml_files = sorted(raw_dir.glob("*.mzML")) + sorted(raw_dir.glob("*.mzml"))

    if not mzml_files:
        for zip_path in sorted(raw_dir.glob("*.zip")):
            manifest = _extract_mzml_zip(zip_path, raw_dir)
            if manifest:
                zip_manifest = manifest
                sample_count = sum(1 for m in manifest if m["sample_type"] == "sample")
                blank_count  = sum(1 for m in manifest if m["sample_type"] == "blank")
                qc_count     = sum(1 for m in manifest if m["sample_type"] == "qc")
                print(
                    f"Extracted {len(manifest)} mzML files from {zip_path.name} "
                    f"({sample_count} samples, {blank_count} blanks, {qc_count} QC)"
                )
                break
        mzml_files = sorted(raw_dir.glob("*.mzML")) + sorted(raw_dir.glob("*.mzml"))

    if not mzml_files:
        raise SystemExit("No mzML files found in data/raw/ and no zip containing mzML files.")

    # ── Build sample metadata ─────────────────────────────────────────────────
    # Priority: user-supplied sheet > zip manifest inference > filename inference
    sheet_name = (cfg.get("inputs") or {}).get("sample_metadata")
    user_sheet = None
    if sheet_name:
        for base in [raw_dir, raw_dir.parent]:
            candidate = base / sheet_name
            if candidate.exists():
                user_sheet = candidate
                print(f"Using sample metadata sheet: {candidate}")
                break
        if user_sheet is None:
            print(f"WARNING: sample_metadata '{sheet_name}' not found — inferring from filenames")

    meta = build_sample_metadata(raw_dir, user_sheet, zip_manifest)
    meta_path = write_sample_metadata(meta, interim_dir)

    n_sample = (meta["sample_type"] == "sample").sum()
    n_qc     = (meta["sample_type"] == "qc").sum()
    n_blank  = (meta["sample_type"] == "blank").sum()
    n_other  = (meta["sample_type"] == "other").sum()
    print(
        f"Sample metadata: {len(meta)} files — "
        f"{n_sample} samples, {n_blank} blanks, {n_qc} QC"
        + (f", {n_other} other" if n_other else "")
    )
    print(f"  → {meta_path}")

    if n_sample == 0:
        raise SystemExit(
            "No sample files found. Check sample_metadata or mzML filenames.\n"
            f"Extra columns in metadata: {list(meta.columns)}"
        )

    # ── Optional inputs (GNPS / known masses) ─────────────────────────────────
    has_annotation_csv = bool((cfg.get("squid") or {}).get("annotation_csv"))
    gnps_zip     = raw_dir / cfg["inputs"]["gnps_mgf"]
    known_masses = raw_dir / cfg["inputs"]["known_masses_csv"]
    gnps_ok      = gnps_zip.exists() or has_annotation_csv
    known_ok     = known_masses.exists() or has_annotation_csv

    missing = []
    if not gnps_ok:
        missing.append(str(gnps_zip))
    if not known_ok:
        missing.append(str(known_masses))

    status = {
        "n_sample_files": int(n_sample),
        "n_blank_files":  int(n_blank),
        "n_qc_files":     int(n_qc),
        "gnps_zip_ok":    gnps_ok,
        "known_masses_ok": known_ok,
        "annotation_csv":  has_annotation_csv,
        "missing":         missing,
    }
    write_json(interim_dir / "input_check.json", {"status": status})

    if missing:
        raise SystemExit(f"Missing required files:\n" + "\n".join(missing))

    print("Input validation complete.")


if __name__ == "__main__":
    main()

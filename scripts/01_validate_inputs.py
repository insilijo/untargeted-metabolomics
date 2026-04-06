from __future__ import annotations

from pathlib import Path
import fnmatch
import zipfile

from utils import ensure_dirs, list_files, load_config, require_files, write_json


def main() -> None:
    # Validate required inputs and write an input check summary.
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])
    mix_files = list_files(raw_dir, cfg["inputs"]["mix_glob"])
    blank_files = list_files(raw_dir, cfg["inputs"]["blank_glob"])
    gnps_zip = raw_dir / cfg["inputs"]["gnps_mgf_zip"]
    known_masses = raw_dir / cfg["inputs"]["known_masses_csv"]

    checks = {
        "mix_files": [str(p) for p in mix_files],
        "blank_files": [str(p) for p in blank_files],
        "gnps_zip": str(gnps_zip),
        "known_masses": str(known_masses),
    }

    ok_mix, missing_mix = require_files(mix_files)
    ok_blank, missing_blank = require_files(blank_files)
    ok_gnps, missing_gnps = require_files([gnps_zip])
    ok_known, missing_known = require_files([known_masses])

    zip_candidates = sorted(raw_dir.glob("*.zip"))
    zip_ok = False
    zip_contents = []
    mix_pattern = cfg["inputs"]["mix_glob"]
    blank_pattern = cfg["inputs"]["blank_glob"]
    gnps_name = cfg["inputs"]["gnps_mgf_zip"]
    known_name = cfg["inputs"]["known_masses_csv"]
    if zip_candidates and (missing_mix or missing_blank or missing_gnps or missing_known):
        for zip_path in zip_candidates:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = {Path(n).name for n in zf.namelist()}
            except zipfile.BadZipFile:
                continue
            has_mix = any(fnmatch.fnmatch(name, mix_pattern) for name in names)
            has_blank = any(fnmatch.fnmatch(name, blank_pattern) for name in names)
            has_gnps = gnps_name in names
            has_known = known_name in names
            if has_mix and has_blank and has_gnps and has_known:
                zip_ok = True
                zip_contents = sorted(names)
                checks["raw_zip"] = str(zip_path)
                break

    # ── mzML-only zip: extract mzML files directly into raw_dir ──────────────
    # Handles a zip that contains only mzML files (named MIX_*.mzML / BLANK_*.mzML).
    # Files are extracted in-place; existing files are not overwritten.
    if not zip_ok and (missing_mix or missing_blank):
        for zip_path in zip_candidates:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    mzml_members = [
                        n for n in zf.namelist()
                        if Path(n).suffix.lower() == ".mzml"
                        and not Path(n).name.startswith("._")
                    ]
                    if not mzml_members:
                        continue
                    extracted = []
                    for member in mzml_members:
                        dest = raw_dir / Path(member).name
                        if not dest.exists():
                            with zf.open(member) as src, open(dest, "wb") as dst:
                                dst.write(src.read())
                        extracted.append(dest.name)
                    print(f"Extracted {len(extracted)} mzML files from {zip_path.name}")
                    checks["mzml_zip"] = str(zip_path)
                    checks["mzml_zip_extracted"] = extracted
            except zipfile.BadZipFile:
                continue
        # Re-check after extraction
        mix_files = list_files(raw_dir, mix_pattern)
        blank_files = list_files(raw_dir, cfg["inputs"]["blank_glob"])
        ok_mix, missing_mix = require_files(mix_files)
        ok_blank, missing_blank = require_files(blank_files)

    status = {
        "mix_files_ok": ok_mix or zip_ok,
        "blank_files_ok": ok_blank or zip_ok,
        "gnps_zip_ok": ok_gnps or zip_ok,
        "known_masses_ok": ok_known or zip_ok,
        "zip_ok": zip_ok,
        "zip_contents": zip_contents,
        "missing": missing_mix + missing_blank + missing_gnps + missing_known,
    }

    out_path = interim_dir / "input_check.json"
    write_json(out_path, {"checks": checks, "status": status})

    if not all([status["mix_files_ok"], status["blank_files_ok"], status["gnps_zip_ok"], status["known_masses_ok"]]):
        missing_msg = "\n".join(status["missing"])
        raise SystemExit(f"Missing required files:\n{missing_msg}")

    print("Input validation complete.")


if __name__ == "__main__":
    main()

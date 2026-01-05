from __future__ import annotations

from pathlib import Path

from utils import list_files, load_config, require_files, write_json


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
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

    status = {
        "mix_files_ok": ok_mix,
        "blank_files_ok": ok_blank,
        "gnps_zip_ok": ok_gnps,
        "known_masses_ok": ok_known,
        "missing": missing_mix + missing_blank + missing_gnps + missing_known,
    }

    out_path = Path(cfg["paths"]["interim_dir"]) / "input_check.json"
    write_json(out_path, {"checks": checks, "status": status})

    if not all([ok_mix, ok_blank, ok_gnps, ok_known]):
        missing_msg = "\n".join(status["missing"])
        raise SystemExit(f"Missing required files:\n{missing_msg}")

    print("Input validation complete.")


if __name__ == "__main__":
    main()

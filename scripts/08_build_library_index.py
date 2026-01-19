from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Dict, Iterable

from utils import ensure_dirs, load_config


def parse_pepmass(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    parts = str(value).strip().split()
    try:
        return float(parts[0])
    except ValueError:
        return None


def parse_mgf(handle: io.TextIOBase) -> Iterable[Dict]:
    current: Dict[str, object] = {}
    mzs: list[float] = []
    intensities: list[float] = []
    in_ions = False

    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.upper() == "BEGIN IONS":
            current = {}
            mzs = []
            intensities = []
            in_ions = True
            continue
        if line.upper() == "END IONS":
            if in_ions:
                current["mz_array"] = mzs
                current["intensity_array"] = intensities
                yield current
            current = {}
            in_ions = False
            continue
        if not in_ions:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            current[key.lower()] = value
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mzs.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spectra (
            id INTEGER PRIMARY KEY,
            title TEXT,
            name TEXT,
            inchikey TEXT,
            pepmass REAL,
            mz_array TEXT,
            intensity_array TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spectra_pepmass ON spectra (pepmass)")
    conn.commit()
    return conn


def pick_field(spec: Dict, *keys: str) -> str | None:
    for key in keys:
        value = spec.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def insert_batch(conn: sqlite3.Connection, batch: list[Dict]) -> None:
    rows = []
    for spec in batch:
        pepmass = parse_pepmass(spec.get("pepmass"))
        name = pick_field(spec, "name", "compound_name", "compoundname")
        inchikey = pick_field(spec, "inchikey", "inchi_key", "inchi-key")
        rows.append(
            (
                pick_field(spec, "title", "spectrumid"),
                name,
                inchikey,
                pepmass,
                json.dumps(spec.get("mz_array", [])),
                json.dumps(spec.get("intensity_array", [])),
            )
        )
    conn.executemany(
        """
        INSERT INTO spectra (title, name, inchikey, pepmass, mz_array, intensity_array)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    mgf_zip_path = raw_dir / cfg["inputs"]["gnps_mgf_zip"]
    if not mgf_zip_path.exists():
        raise SystemExit(f"Missing GNPS MGF zip: {mgf_zip_path}")

    db_path = interim_dir / cfg["inputs"]["gnps_sqlite"]
    if db_path.exists():
        db_path.unlink()

    conn = init_db(db_path)
    total = 0
    batch: list[Dict] = []
    batch_size = 1000

    with zipfile.ZipFile(mgf_zip_path, "r") as zf:
        mgf_names = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".mgf")
            and not n.startswith("__MACOSX/")
            and not Path(n).name.startswith("._")
        ]
        if not mgf_names:
            raise SystemExit("No MGF file found inside GNPS_SUBSET.mgf.zip")
        for name in mgf_names:
            with zf.open(name, "r") as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
                for spec in parse_mgf(text):
                    batch.append(spec)
                    if len(batch) >= batch_size:
                        insert_batch(conn, batch)
                        total += len(batch)
                        batch = []

    if batch:
        insert_batch(conn, batch)
        total += len(batch)

    conn.commit()
    conn.close()
    print(f"Wrote {total} library spectra to {db_path}")


if __name__ == "__main__":
    main()

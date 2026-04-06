from __future__ import annotations

"""Build the GNPS library SQLite index from a raw ALL_GNPS MGF file.

Filtration applied (mirrors HOC annotation pipeline standards):
  - MS2 only (discard MS1-only records)
  - Precursor m/z present and > 0
  - At least min_peaks fragment ions (default 5)
  - Library quality tier: Gold (1) or Silver (2) preferred; Bronze (3)
    accepted only when the entry has an InChIKey and ≥ min_peaks_bronze peaks
  - Known contaminant / exogenous compound names removed
  - Duplicate (pepmass, inchikey) pairs deduplicated — best peak count kept

Supported input:
  - Plain MGF        (raw_dir / gnps_mgf, e.g. ALL_GNPS.mgf)
  - Gzip-compressed  (ALL_GNPS.mgf.gz)
  - Zip archive      (ALL_GNPS.mgf.zip  — first .mgf inside)

Progress is printed every 100 000 records.
"""

import gzip
import io
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator

from utils import ensure_dirs, load_config

# ---------------------------------------------------------------------------
# Filtration constants (mirror HOC pipeline)
# ---------------------------------------------------------------------------

# GNPS Library_Class field: 1 = Gold (reference standards), 2 = Silver,
# 3 = Bronze (community, less curated), 10 = third-party (treat as Bronze).
GOLD   = {"1"}
SILVER = {"2"}
BRONZE = {"3", "10", ""}          # empty = field absent → treat conservatively

MIN_PEAKS_GOLD_SILVER = 5         # minimum fragment ions for Gold/Silver
MIN_PEAKS_BRONZE      = 8         # stricter threshold for Bronze
REQUIRE_INCHIKEY_BRONZE = True    # Bronze entries without InChIKey are dropped

# Exogenous / contaminant keywords — compound names containing these are dropped.
# Derived from annotate.py in the HOC pipeline.
EXOGENOUS_KEYWORDS = (
    "chlorophenyl", "bromophenyl", "fluorophenyl", "iodophenyl",
    "pesticide", "herbicide", "fungicide",
    "amphetamine", "benzodiazep", "barbit", "sulfonamide",
    "anthramycin", "mitoxantrone", "tyrphostin",
)

# Ion mode strings that indicate positive or negative mode (lower-cased).
# We store them as-is; downstream search steps filter by mode when needed.
# No hard removal here — the pipeline searches both modes against the right data.

# ---------------------------------------------------------------------------
# MGF parser
# ---------------------------------------------------------------------------

def _open_mgf(path: Path) -> Iterator[io.TextIOBase]:
    """Yield an open text handle for the MGF content regardless of compression."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            yield fh
    elif suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            mgf_names = [
                n for n in zf.namelist()
                if n.lower().endswith(".mgf")
                and not n.startswith("__MACOSX/")
                and not Path(n).name.startswith("._")
            ]
            if not mgf_names:
                raise SystemExit(f"No .mgf file found inside {path}")
            for name in mgf_names:
                with zf.open(name, "r") as raw:
                    yield io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            yield fh


def parse_mgf(handle: io.TextIOBase) -> Iterable[Dict]:
    """Stream-parse an MGF handle into spec dicts with mz_array / intensity_array."""
    current: Dict = {}
    mzs: list[float] = []
    intensities: list[float] = []
    in_ions = False

    for line in handle:
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "BEGIN IONS":
            current = {}
            mzs = []
            intensities = []
            in_ions = True
            continue
        if upper == "END IONS":
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
            current[key.lower()] = value.strip()
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mzs.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _pick(spec: Dict, *keys: str) -> str:
    for k in keys:
        v = spec.get(k, "")
        if v and str(v).strip() not in ("", "N/A", "n/a", "0", "null"):
            return str(v).strip()
    return ""


def _parse_pepmass(spec: Dict) -> float | None:
    raw = _pick(spec, "pepmass")
    if not raw:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def _count_peaks(spec: Dict) -> int:
    return len(spec.get("mz_array", []))


# ---------------------------------------------------------------------------
# Filtration logic
# ---------------------------------------------------------------------------

def _is_exogenous(name: str) -> bool:
    name_lc = name.lower()
    return any(kw in name_lc for kw in EXOGENOUS_KEYWORDS)


def _library_class(spec: Dict) -> str:
    """Return GNPS Library_Class as a string ('1', '2', '3', '10', or '')."""
    return _pick(spec, "library_class", "libraryclass", "library class").split()[0] \
        if _pick(spec, "library_class", "libraryclass", "library class") else ""


def _inchikey(spec: Dict) -> str:
    return _pick(spec, "inchikey", "inchi_key", "inchi-key")


def accept(spec: Dict) -> bool:
    """Return True if this spectrum passes all quality filters."""
    pepmass = _parse_pepmass(spec)
    if pepmass is None or pepmass <= 0:
        return False

    n_peaks = _count_peaks(spec)
    lc = _library_class(spec)

    # Peak count threshold depends on tier
    if lc in GOLD | SILVER:
        if n_peaks < MIN_PEAKS_GOLD_SILVER:
            return False
    else:
        # Bronze / unknown tier — stricter
        if n_peaks < MIN_PEAKS_BRONZE:
            return False
        if REQUIRE_INCHIKEY_BRONZE and not _inchikey(spec):
            return False

    # MS level: skip anything that isn't MS2
    ms_level = _pick(spec, "mslevel", "ms_level", "mslev")
    if ms_level and ms_level not in ("2", "MS2", "ms2"):
        return False

    # Exogenous compound name rejection
    name = _pick(spec, "name", "compound_name", "compoundname")
    if name and _is_exogenous(name):
        return False

    return True


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spectra (
    id              INTEGER PRIMARY KEY,
    title           TEXT,
    name            TEXT,
    inchikey        TEXT,
    pepmass         REAL,
    ion_mode        TEXT,
    adduct          TEXT,
    library_class   TEXT,
    n_peaks         INTEGER,
    mz_array        TEXT,
    intensity_array TEXT
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pepmass    ON spectra (pepmass);",
    "CREATE INDEX IF NOT EXISTS idx_inchikey   ON spectra (inchikey);",
    "CREATE INDEX IF NOT EXISTS idx_ion_mode   ON spectra (ion_mode);",
]


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()
    return conn


def insert_batch(conn: sqlite3.Connection, batch: list[Dict]) -> None:
    rows = []
    for spec in batch:
        rows.append((
            _pick(spec, "title", "spectrumid"),
            _pick(spec, "name", "compound_name", "compoundname"),
            _inchikey(spec),
            _parse_pepmass(spec),
            _pick(spec, "ionmode", "ion_mode", "ion mode"),
            _pick(spec, "charge", "adduct"),
            _library_class(spec),
            _count_peaks(spec),
            json.dumps(spec.get("mz_array", [])),
            json.dumps(spec.get("intensity_array", [])),
        ))
    conn.executemany(
        """
        INSERT INTO spectra
            (title, name, inchikey, pepmass, ion_mode, adduct,
             library_class, n_peaks, mz_array, intensity_array)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(conn: sqlite3.Connection) -> int:
    """Remove duplicate (pepmass-rounded-3dp, inchikey) rows, keeping most peaks.

    Returns number of rows deleted.
    """
    conn.execute("""
        DELETE FROM spectra
        WHERE id NOT IN (
            SELECT MIN(id) FROM (
                SELECT id,
                       ROUND(pepmass, 3) AS mz_r,
                       COALESCE(NULLIF(inchikey,''), title) AS key_col,
                       n_peaks
                FROM spectra
            ) sub
            GROUP BY mz_r, key_col
            HAVING n_peaks = MAX(n_peaks)
            -- break ties by smallest id
        )
    """)
    deleted = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    raw_dir   = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    ensure_dirs([interim_dir])

    # Locate MGF — accept plain, .gz, or .zip
    mgf_stem = cfg["inputs"].get("gnps_mgf", "ALL_GNPS.mgf")
    candidates = [
        raw_dir / mgf_stem,
        raw_dir / (mgf_stem + ".gz"),
        raw_dir / (mgf_stem + ".zip"),
        raw_dir / (Path(mgf_stem).stem + ".mgf.zip"),
    ]
    mgf_path = next((p for p in candidates if p.exists()), None)
    if mgf_path is None:
        raise SystemExit(
            f"Cannot find GNPS library MGF under {raw_dir}. "
            f"Tried: {[str(p) for p in candidates]}"
        )
    print(f"Using library: {mgf_path}")

    db_path = interim_dir / cfg["inputs"]["gnps_sqlite"]
    if db_path.exists():
        db_path.unlink()

    conn = init_db(db_path)

    total_seen = 0
    total_kept = 0
    batch: list[Dict] = []
    batch_size = 2000

    for handle in _open_mgf(mgf_path):
        for spec in parse_mgf(handle):
            total_seen += 1
            if not accept(spec):
                continue
            batch.append(spec)
            total_kept += 1
            if len(batch) >= batch_size:
                insert_batch(conn, batch)
                conn.commit()
                batch = []
            if total_seen % 100_000 == 0:
                print(f"  Scanned {total_seen:,}  kept {total_kept:,} …")

    if batch:
        insert_batch(conn, batch)
        conn.commit()

    print(f"Ingested {total_seen:,} records → kept {total_kept:,} after quality filters.")

    deleted = deduplicate(conn)
    final = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
    print(f"Deduplicated: removed {deleted:,} duplicates → {final:,} unique entries in {db_path}")

    conn.close()


if __name__ == "__main__":
    main()

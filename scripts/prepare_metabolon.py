"""Prepare a SQuID-InC annotation CSV from a Metabolon Excel data table
and the Metabolon public data dictionary.

Merges the Excel's Chemical Annotation sheet (INCHIKEY, SMILES, CHEMICAL_NAME,
PLATFORM) with the data dictionary (BIOCHEMICAL, MASS, RI, INCHIKEY) to produce
a flat CSV with mz, rt, inchikey, smiles, name, and platform columns suitable
for use as annotation_csv in config.yaml.

Join strategy
-------------
1. Primary   — INCHIKEY exact match
2. Secondary — CHEMICAL_NAME (annot) vs BIOCHEMICAL (dict), case-insensitive,
               for compounds the dict lacks an INCHIKEY for
Compounds in Chemical Annotation with no MASS/RI in either source are still
included (they contribute INCHIKEY + SMILES to the embedding even without RT).

Usage
-----
    python scripts/prepare_metabolon.py \\
        --excel   data/raw/MyStudy_DataTables.xlsx \\
        --dict    data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv \\
        --out     data/raw/metabolon_annotations.csv

    # then in config.yaml:
    #   squid:
    #     annotation_csv: data/raw/metabolon_annotations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Platform normalisation (dict uses verbose names; annot uses short codes)
# ---------------------------------------------------------------------------

_PLATFORM_MAP = {
    "pos early": "lc/ms pos early",
    "pos late":  "lc/ms pos late",
    "neg":       "lc/ms neg",
    "polar":     "lc/ms polar",
}


def _norm_platform(val: str | None) -> str:
    if not val or pd.isna(val):
        return ""
    return _PLATFORM_MAP.get(str(val).strip().lower(), str(val).strip().lower())


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge(excel_path: Path, dict_path: Path, out_path: Path, update_config: bool = True) -> None:
    print(f"Loading Chemical Annotation from {excel_path.name} …")
    ca = pd.read_excel(excel_path, sheet_name="Chemical Annotation")
    ca.columns = [c.strip() for c in ca.columns]

    print(f"Loading data dictionary from {dict_path.name} …")
    dd = pd.read_csv(dict_path)
    dd.columns = [c.strip() for c in dd.columns]

    # Normalise key columns
    ca["_ik"]   = ca["INCHIKEY"].fillna("").str.strip()
    ca["_name"] = ca["CHEMICAL_NAME"].fillna("").str.strip().str.lower()
    dd["_ik"]   = dd["INCHIKEY"].fillna("").str.strip()
    dd["_name"] = dd["BIOCHEMICAL"].fillna("").str.strip().str.lower()

    # ── Pass 1: INCHIKEY join ─────────────────────────────────────────────────
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

    # ── Pass 2: name join for unmatched rows ─────────────────────────────────
    dd_named = dd[dd["_ik"] == ""].set_index("_name")
    # also include dict rows whose IK wasn't in annot
    dd_named_extra = dd[~dd["_ik"].isin(matched_ik)].set_index("_name")

    n_name_matched = 0
    for rec in rows:
        if rec["mz"] is not None:           # already filled by IK pass
            continue
        name = rec.get("_name_key", "")
        if name in dd_named.index:
            _fill_from_dict(rec, dd_named.loc[name])
            n_name_matched += 1
        elif name in dd_named_extra.index:
            _fill_from_dict(rec, dd_named_extra.loc[name])
            n_name_matched += 1

    # Drop internal key
    for rec in rows:
        rec.pop("_name_key", None)

    out_df = pd.DataFrame(rows)

    # Drop rows with no identifying information at all
    out_df = out_df[out_df["mz"].notna() | out_df["inchikey"].notna() | out_df["smiles"].notna()]

    # Sort by mz ascending, NaN last
    out_df = out_df.sort_values("mz", na_position="last").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_total   = len(out_df)
    n_mz      = out_df["mz"].notna().sum()
    n_rt      = out_df["rt"].notna().sum()
    n_smiles  = out_df["smiles"].notna().sum()
    n_ik      = out_df["inchikey"].notna().sum()

    print(f"\nMerge summary")
    print(f"  Total rows          : {n_total}")
    print(f"  INCHIKEY matched    : {len(matched_ik)}")
    print(f"  Name matched        : {n_name_matched}")
    print(f"  With m/z (MASS)     : {n_mz}")
    print(f"  With RT/RI          : {n_rt}")
    print(f"  With SMILES         : {n_smiles}")
    print(f"  With InChIKey       : {n_ik}")
    print(f"\n→ {out_path}")

    if update_config:
        _write_config(out_path)


def _write_config(out_path: Path) -> None:
    """Write annotation_csv into the squid section of config.yaml."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        print(f"  config.yaml not found at {config_path} — skipping auto-update")
        return

    try:
        rel = out_path.relative_to(config_path.parent.parent)
    except ValueError:
        rel = out_path

    text = config_path.read_text()
    new_line = f"  annotation_csv: {rel}"

    if "annotation_csv:" in text:
        # Replace existing (commented or uncommented)
        import re
        text = re.sub(r"[ \t]*#?[ \t]*annotation_csv:.*", new_line, text)
    else:
        # Append inside squid: block — after cross_community_damping or at end of block
        import re
        text = re.sub(
            r"([ \t]*cross_community_damping:[^\n]*)",
            r"\1\n" + new_line,
            text,
        )

    config_path.write_text(text)
    print(f"  Updated {config_path.name}: annotation_csv = {rel}")


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
    # drow may be a DataFrame when multiple dict rows share the same key
    if isinstance(drow, pd.DataFrame):
        drow = drow.iloc[0]
    mass = drow.get("MASS")
    ri   = drow.get("RI")
    if pd.notna(mass):
        rec["mz"] = float(mass)
    if pd.notna(ri):
        rec["rt"] = float(ri)
    # Fill INCHIKEY from dict if annot was missing
    if not rec["inchikey"]:
        ik = drow.get("INCHIKEY", "")
        rec["inchikey"] = str(ik).strip() if pd.notna(ik) and str(ik).strip() else None
    # Fill name from dict BIOCHEMICAL if annot name was blank
    if not rec["name"]:
        rec["name"] = str(drow.get("BIOCHEMICAL", "")).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_dict() -> Path:
    here = Path(__file__).resolve()
    # Look relative to the untargeted-metabolomics project root,
    # then fall back to SQUID_INC_ROOT
    for base in [here.parents[1], here.parents[2] / "SQuID-INC"]:
        p = base / "data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
        if p.exists():
            return p
    return Path("data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excel", required=True,
                   help="Path to Metabolon Excel data tables (.xlsx)")
    p.add_argument("--dict",  default=str(_default_dict()),
                   help="Path to Metabolon data dictionary CSV (default: auto-detected)")
    p.add_argument("--out",   default="data/raw/metabolon_annotations.csv",
                   help="Output annotation CSV path (default: data/raw/metabolon_annotations.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge(
        excel_path=Path(args.excel).expanduser().resolve(),
        dict_path=Path(args.dict).expanduser().resolve(),
        out_path=Path(args.out).expanduser(),
        update_config=True,
    )


if __name__ == "__main__":
    main()

# build_references — bootstrappable MS2 reference libraries (from scratch)

`python build_references.py all` downloads every public source and builds MS2 reference libraries for
the public Metabolon DD + HMDB, so the kit-free bootstrap can confirm anchors by reference-MS2 match.
Library-matching with self-bootstrapping calibration, **not** de novo.

Stages: `download | fast | mona | hmdb | dd | emit` (or `all`). Outputs land in `out/` (gitignored).

Sources auto-download (GNPS-LIBRARY, GNPS HMDB library, MassBank latest MSP, MoNA-LipidBlast) + PubChem
PUG REST for DD SMILES. The **HMDB compound universe is taken from the GNPS HMDB library** (`HMDB.json`),
not the licence-gated hmdb.ca XML — so the whole build is hands-off, no manual/academic download step.
The Metabolon DD PMC-OA subset is bundled in `inputs/` (no clean public URL).

Outputs: `metabolon_dd_ms2.mgf`, `hmdb_ms2.mgf`, `ms2_ref_{neg,pos}_ik14.json` (bootstrap drop-in),
`metabolon_dd_corrected.csv`.

**Private/internal** — compound identities trace to HMDB; don't surface this bundle publicly.

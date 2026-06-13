# build_references — bootstrappable MS2 reference libraries (from scratch)

`python build_references.py all` downloads every public source and builds MS2 reference libraries for
the public Metabolon DD + HMDB, so the kit-free bootstrap can confirm anchors by reference-MS2 match.
Library-matching with self-bootstrapping calibration, **not** de novo.

Stages: `download | fast | mona | hmdb | dd | emit` (or `all`). Outputs land in `out/` (gitignored).

**Private/internal only** — HMDB is academic-licence and taints the bundle. HMDB's XML is licence-gated
and may 403 automated requests; if so, download `hmdb_metabolites.xml` manually (the script prints the URL).
The Metabolon DD PMC-OA subset is bundled in `inputs/` (no clean public URL). All other sources auto-download.

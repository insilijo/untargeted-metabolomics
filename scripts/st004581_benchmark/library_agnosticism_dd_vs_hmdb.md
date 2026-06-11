# Library-agnosticism: DD vs HMDB/metabolome calls (structure->RT, no RI, all 4 platforms)

Same kit-calibrated structure->RT engine, two candidate libraries (Metabolon DD ~1.6k vs public
human-metabolome ~1.85k), equal footing (no library RI for either). How similar are the CALLS?

| platform   | DD recovers | HMDB recovers | recovered-set Jaccard | shared-peak call agreement |
|------------|-------------|---------------|-----------------------|----------------------------|
| neg        | 176         | 128           | 0.50                  | **0.77** (197/257)         |
| pos-early  | 124         | 96            | 0.52                  | **0.76** (168/220)         |
| pos-late   | 82          | 57            | 0.60                  | **0.95** (91/96)           |
| polar      | 38          | 15            | 0.10                  | 0.56 (9/16, tiny n)        |

structure-RT recall (H2H): DD neg 0.78 / pe 0.85 / pl 0.84 / pol 0.79 ; HMDB 0.48 / 0.67 / 0.34 / 0.29.

## Finding (disagreement decomposes cleanly)
- PER-PEAK CALL agreement 76-95%: the IDENTITY the method assigns a given peak is largely
  library-AGNOSTIC -- the data drives the call, not the library.
- RECOVERED-SET overlap only 0.50-0.60: driven by COVERAGE (different compound lists, different
  isobaric neighbors), not by the method assigning different identities.
- => swapping vendor DD for a public metabolome changes WHICH peaks get called (reach), not WHAT a
  given peak is called (identity). Engine stable; library sets the reach.
- Polar weak (Jaccard 0.10) on tiny n -- thin public polar-metabolite SMILES coverage.

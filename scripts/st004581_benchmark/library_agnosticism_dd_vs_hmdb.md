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

## CORRECTION (the DD>HMDB gap is a naming artifact, NOT competition)
Earlier framing attributed the recovered-set gap to "coverage/competition". Measured directly:
- Isobaric density at MAF peaks: DD median 2/mean 2.9 ; HMDB-[M-H] median 1/mean 1.4 ; HMDB-multi
  median 2/mean 2.1. => HMDB has FEWER isobars, NOT more. Competition is NOT the cause.
- Effective coverage of the 406 neg MAF (ik14 OR name -- the benchmark's own matching):
    DD   : by ik14 74% | by NAME 92% | by EITHER 93%
    HMDB : by ik14 80% | by NAME 53% | by EITHER 82%
  => By STRUCTURE (ik14) HMDB covers the MAF BETTER (80% vs 74%). The DD only wins on NAME (92% vs 53%)
  because the DD's compound names ARE the MAF's names (same Metabolon source) -- a benchmark artifact
  (answer key scored in Metabolon's nomenclature, which the vendor library trivially shares).
HMDB->DD recall gap (0.56->0.78) = ~11pt effective-coverage (mostly the shared-naming artifact) +
adduct/m-z construction efficiency (multi-adduct closes part). NOT isobaric competition.
Implication: structurally the public metabolome is the MORE complete library; the DD's real residual
edge is only pre-computed observed-adduct m/z (reproducible) + measured RI (small recall lever).

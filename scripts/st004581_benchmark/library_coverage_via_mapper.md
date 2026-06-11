# Public-library coverage of ST004581 neg MAF — via GIZMO MetaboliteMapper

The "47% not in public libraries" was an artifact of measuring MS2-SPECTRAL-library coverage
(which needs a physically deposited spectrum). For our STRUCTURE-based method the right measure is
structure/metabolome coverage, and it's near-total.

| coverage measure                                   | of 386 neg MAF | %   |
|----------------------------------------------------|----------------|-----|
| MS2 spectral libs (GNPS+MassBank+MoNA)             | 204            | 53% |
| structure in human-metabolome graph (InChIKey14)   | 310            | 80% |
| MetaboliteMapper by NAME (weird Metabolon names)   | 380/406        | 94% |
| combined (mapper-name OR structure-in-graph)       | 398/406        | 98% |

Mapper = gizmo.evidence.mappers.MetaboliteMapper on GIZMO human_full graph (2,642 ik14 metabolites,
2,819 HMDB ids). It bridges Metabolon's idiosyncratic names (S-methylcysteine, pantoate, o-cresol
sulfate, ...) to known human metabolites via fuzzy/abbreviation/salt/stereo handling.

## Implication
- Library dependency for COVERAGE is ~fully satisfiable from PUBLIC structure sources (HMDB/metabolome ~98%).
- What the DD uniquely provides vs HMDB is the RI prior (-> structure-RT NORI path, ~-2.6 pts) and
  lower isobaric competition. Coverage is NOT the bottleneck.
- Spectral matching (GNPS-style) is deposit-limited (53%); structure+RT matching is not.

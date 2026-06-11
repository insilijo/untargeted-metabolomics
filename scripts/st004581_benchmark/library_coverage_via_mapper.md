# Public-library coverage of ST004581 MAF — via GIZMO MetaboliteMapper (ALL 4 platforms)

The "compounds not in public libraries" gap was an artifact of measuring MS2-SPECTRAL-library
coverage (needs a physically deposited spectrum). For our STRUCTURE-based method the right measure
is structure/metabolome coverage, and via the GIZMO MetaboliteMapper it is near-total on every platform.

| platform        | nMAF | MS2-spectral | structure (ik14) | mapper by NAME | combined |
|-----------------|------|--------------|------------------|----------------|----------|
| lc/ms neg       | 406  | 53%          | 76%              | 94%            | **98%**  |
| lc/ms pos-early | 205  | 71%          | 93%              | 99%            | **100%** |
| lc/ms pos-late  | 201  | 56%          | 37%              | 97%            | **100%** |
| lc/ms polar     | 93   | 61%          | 86%              | 94%            | **100%** |

Mapper = gizmo.evidence.mappers.MetaboliteMapper on GIZMO data/processed/human_full/graph.json
(2,642 ik14 metabolites, 2,819 HMDB ids; node-link JSON -> nx.DiGraph -> mg.graph). Resolves
Metabolon's idiosyncratic names (S-methylcysteine, pantoate, o-cresol sulfate, lipid shorthand)
via fuzzy/abbreviation/salt/stereo handling.

## Key points
- Combined structure coverage 98-100% on ALL platforms -> no coverage ceiling for structure+RT matching.
- pos-late: structure-by-ik14 only 37% but mapper-by-name 97% -- lipid-heavy platform where raw
  InChIKey match badly undercounts; the mapper is essential ("weird names").
- MS2-spectral coverage (53-71%) is deposit-limited -- the misleading metric.
- DD's unique value vs public HMDB/metabolome = RI prior (NORI -2.6 pts) + lower isobaric competition,
  NOT coverage. Spectral matching (GNPS-style) stays deposit-limited.

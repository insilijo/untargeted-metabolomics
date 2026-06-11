# Multi-adduct MS1 + RT-prediction / isobar-cardinality findings

## DD-vs-HMDB recovery gap mechanism (DD recovers but public HMDB doesn't)
| platform  | DD-only | absent(coverage) | adduct | competition/RT |
|-----------|---------|------------------|--------|----------------|
| neg       | 74      | 0                | 13     | 61 (82%)       |
| pos-early | 49      | 0                | 3      | 46 (94%)       |
| pos-late  | 30      | 20 (lipids)      | 4      | 6 (20%)        |
| polar     | 33      | 0                | 3      | 30 (91%)       |
=> not coverage (~0 absent except pos-late lipids); ~18% adduct; ~82% competition/RT.

## Extension 1: multi-adduct MS1 (predict alternative adduct ions, not just [M-H]/[M+H])
HMDB structure-RT recall, single-adduct -> multi-adduct:
  neg 0.476->0.556 (+.080) | pos-early 0.667->0.721 (+.054) | pos-late 0.339->0.365 (+.026) | polar 0.290->0.462 (+.172)
Recovers MORE than the flagged adduct bucket -- alt adducts also dodge isobaric competition. KEEP.

## Isobar cardinality: it's a PREDICTION problem, not chromatography
Isobars resolve as distinct peaks on the column; structure-RT just can't assign them.
RT prediction error (|pred-observed|, neg recovered):
  structure-RT (kit) median 22s ; measured-RI median 8s.
Functional-group descriptors (user: "should resolve based on functional group"):
  n=30 kit:   19desc 22.2s -> +FG 26.6s (OVERFITS, worse)
  n=176:      19desc 14.5s -> +FG 13.7s (FG HELPS; error halved by data density)
=> functional groups DO encode isobar-differentiating retention, but require DATA DENSITY to learn;
   sparse 30-kit overfits. Path = quasi-library accumulation (kit-free bootstrap) + FG descriptors ->
   approaches measured-RI accuracy. Stereo/geometric isomers = hard floor (no 2D model separates them;
   measured RT only).

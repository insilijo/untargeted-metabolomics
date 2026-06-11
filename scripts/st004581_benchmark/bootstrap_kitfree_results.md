# Kit-free / RI-free bootstrap results (ST004581, cap=200)

Discover the RT-calibration seed FROM THE DATA (mass-unique MS1 + reference-MS2 match >= TAU) instead
of the spike-in kit. Compared WITHIN the ik14-based bootstrap scorer (read deltas, not vs the 0.854
full-InChIKey headline).

| platform   | kit   | bootstrap | confident seeds | delta  |
|------------|-------|-----------|-----------------|--------|
| pos-late   | 0.832 | **0.879** | 16              | +0.047 |
| lc/ms neg  | 0.824 | **0.834** | 19              | +0.010 |
| pos-early  | 0.887 | **0.892** | 23              | +0.005 |
| polar      | 0.795 | 0.771     | 4               | -0.024 |

TAU sweep (neg): TAU 0.4 -> 24 seeds / 0.839 ; 0.5 -> 19 / 0.834 ; 0.6 -> 17 / 0.829 ; 0.7 -> 15 / 0.824.
RI-FREE (NORI, neg): 0.808  (vs bootstrap+RI 0.834 -> library RI worth ~0.026, not essential).

## Takeaways
- Data-discovered anchors MATCH OR BEAT the curated kit on 3/4 platforms (pos-late +0.047).
- Polar fails (only 4 confident seeds) -> kit earns its keep where MS2/mass-uniqueness is thin.
- Kit-free AND RI-free still reaches ~0.81 on neg.
- HONEST CEILING: every number is recall AGAINST THE LIBRARY. Bootstrap removes the *calibration*
  dependency (kit, RI); the *compound list* is irreducible -- this is a library-MATCHING method, not
  de novo. Cheaper way to match a library, not a way to not need one.

Run: forest_sweep_bootstrap.py  SEED=bootstrap [TAU_SEED=..] [NORI=1] MS2MODE=gate PLATFORM=.. LIBRARY=dd

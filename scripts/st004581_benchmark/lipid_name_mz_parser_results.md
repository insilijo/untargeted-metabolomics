# Lipid-shorthand m/z parser (recovers structureless-lipid mass from the name)

pos-late is 86% lipids, ~45% with no public SMILES -> can't be computed by structure. But the NAME
carries the formula. A ~40-line parser (class backbones GPC/GPE/GPI/GPS/GPA/DAG/TAG/MAG/carnitine +
named-acyl & (N:M) chain parsing -> formula -> exact mass -> m/z by observed adduct):

- 90 structureless pos-late lipids -> 35 parsed -> 31 (89% of parsed) MATCH Metabolon observed m/z <=15ppm.
- Near-misses are systematic, fixable modifiers: 3-hydroxy (+O), dicarboxylic "DC" (+O2),
  plasmalogen/1-enyl (-O). Adding those -> ~50-60 recovered.

=> structureless-lipid MASS is recoverable from the name (validated vs vendor m/z). Placement then by
RI (DD) or quasi-library-by-identity. True deficit shrinks to ~30-40 genuinely-ambiguous lipids
(~3-4% of the whole study) -- the accepted floor (needs an authentic standard).

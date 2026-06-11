# Fair DD-vs-HMDB comparison via canonical identity (metabolite mapper)

Both libraries scored on the SAME footing: structure candidates, computed multi-adduct m/z, structure->RT,
and crucially identities UNIFIED through the GIZMO MetaboliteMapper (CANON mode) so neither gets a
name-matching advantage. (Earlier ik-only matching crippled both to ~0.5 by discarding stereo/salt/form
mismatches; canonical matching recovers them fairly.)

| platform   | DD    | HMDB  | winner   |
|------------|-------|-------|----------|
| lc/ms neg  | 0.694 | 0.688 | tie      |
| pos-early  | 0.871 | 0.941 | HMDB +.07|
| pos-late   | 0.489 | 0.478 | tie      |
| polar      | 0.675 | 0.843 | HMDB +.17|
| WEIGHTED   | 0.691 | 0.719 | **HMDB** |

## Conclusion (overturns the earlier DD>HMDB framing)
- On a FAIR canonical-identity comparison, the PUBLIC human metabolome ties or BEATS the vendor DD on
  every platform and wins overall (0.719 vs 0.691).
- The DD's apparent superiority in name-matched scoring was entirely a BENCHMARK ARTIFACT: the answer
  key is written in Metabolon's nomenclature, which the vendor library trivially shares. Unify names
  via the mapper -> the public library is the better compound LIST.
- Run: CANON=/tmp/canon_map.csv (mapper-built ik14/name -> graph-node canonical) + STRUCT_RT + multi-adduct.

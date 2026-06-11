# Hybrid cardinality: published RI where available, structure-RT otherwise

Per-candidate: use inv(library RI) if the library provides an RI (Metabolon DD), else kit-trained
structure->RT. Mixed library = DD compounds (4029 rows, with RI) UNION HMDB/metabolome-only (5864
rows, no RI). HYBRID mode in standalone (sparse_anchor_annotate.py).

| platform   | DD-RI only | HYBRID (RI + structure-RT) | delta  |
|------------|------------|----------------------------|--------|
| lc/ms neg  | 0.829      | **0.856**                  | +0.027 |
| pos-early  | 0.887      | **0.926**                  | +0.039 |
| pos-late   | 0.833      | 0.833                      | 0      |
| polar      | 0.785      | **0.860**                  | +0.075 |

## Findings
- Mechanism works: DD compounds resolve isobars via Metabolon's PUBLISHED RI (legitimate -- DD is the
  public candidate library, not the answer key); HMDB-only compounds use structure-RT.
- Recall IMPROVES on 3/4 (not just holds): the larger candidate pool feeds the iterative self-training
  more admitted anchors -> tighter RT model -> more of the DD's own MAF compounds get admitted.
  (= quasi-library effect, live.) Caveat: small neg lift may be partly better SMILES coverage.
- REACH benefit (true positives OUTSIDE the vendor library) NOT demonstrable on ST004581 (MAF subset
  of DD) -- needs a non-DD-bounded ground truth (MTBLS136 serum, or known non-DD spike-ins).

## Practical method = hybrid
Published RI wherever it exists (vendor DD / any RI library / prior runs' quasi-library) -> measured
isobar resolution, high recall. Structure-RT + quasi-library only for compounds no one has measured.

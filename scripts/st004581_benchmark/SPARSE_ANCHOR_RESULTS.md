# Sparse-anchor forest propagation — ST004581 results

Working results for the paper *"Maximizing informational space in sparse anchors enables
industry-competitive prediction of MS compounds."* All numbers are leak-free (SMILES from
PubChem, never the MAF) and scored against the **public Metabolon data dictionary (DD)** as
the candidate library, with the **MAF as the answer key**. Built/validated on ST004581
**lc/ms neg** (COLU, Method3); POS-mode generalization in progress.

## The method

A sparse spike-in **kit** (the SQuID-INC anchor panels) calibrates a structure→RT predictor;
annotation runs against the full DD via **precision-first iterative propagation**, scored as
**compound-clusters**. Components, in order:

1. **EIC-precise RT** — raw-EIC apex (~2s) from pymzML fixes pure-python centWave's ~7s apex
   jitter; this is what makes RI/RT discriminate at all.
2. **Precision-first propagation** ("linear label-propagation", the graph-message-passing
   analog): seed = certain calls → train structure→seconds on the admitted set → admit only
   peaks in a TIGHT window (±6s) of the prediction, reproducible (≥2 inj), isobar-resolved,
   FDR<0.01 → admitted EIC-apexes become new training anchors → predictions tighten → next
   structural tier crosses the gate → expand. FPs are withheld below the gate.
3. **Forest** (the xgboost-forest-in-tandem idea): K=10 chains, each = bootstrapped seed +
   random descriptor subset, with **early-clip** (stop a chain when its held-out OOB-seed RT
   error rises = drift onset) and **consensus** admission (≥M chains). M is the precision
   knob / Pareto-front slider. The forest fixes the seed-sensitivity ("starting point
   matters"): diverse seeds reach structure-space neighborhoods one seed can't.
4. **Compound-cluster scoring**: a co-eluting isobaric group is ONE annotation (isobaric
   peers listed honestly), TP if ANY member is in the MAF. This stops counting one peak as
   two FPs without deleting either compound (recall held, precision up).

## Headline P/R curve (neg, 412 MAF compounds, 8 injections)

| consensus ≥M | clusters | recall | precision |
|---|---|---|---|
| 1 | 234 | **0.532** | **0.658** |
| 2 | 191 | 0.441 | 0.665 |
| 3 | 171 | 0.397 | 0.702 |
| 4 | 149 | 0.330 | 0.685 |
| 5 | 119 | 0.259 | 0.714 |

**Pareto-dominates blanket m/z+RI matching** (0.490 / 0.435) — cluster M=1 is +0.04 recall
AND +0.22 precision, both axes at once. The forest union (M=1) also beats a single
propagation chain (0.352 / 0.552) on recall by +0.18 at equal precision.

## Recall is RT-misprediction-bound, not detection-bound

Of 412 MAF neg compounds:

| | n | |
|---|---|---|
| detectable at predicted RT | 209 | recovered |
| **present but RT-mislocated** | **148** | peak elsewhere at its m/z (median 32s, p90 193s off) |
| truly absent / sub-floor | 55 | the real wall |

**True recoverable ceiling = 357/412 = 0.867.** Sparse anchors fail by *mislocating*, not
missing. But only **11 of the 148** mislocated are mass-unique (recoverable by m/z alone);
**137 are isobaric** and need RT precise enough to *disambiguate*. The wide-window structure-
model **rescue pass recovers only 7** of them (recall 0.532→0.549) — confirming the recall
bottleneck is **anchor placement, not the algorithm**. The lever is the kit-design map:
concentrate anchors where RT↔sec is steep/sparse (early-RI 152s-spread region, late tail),
which library densification backfills for free.

## Precision is identity-bound; the residual error is tiny

Decomposition of the M=1 admitted set:
- TP 0.547, **isobar duplicates 0.208** (fixed by cluster scoring), `novel_real` 0.177
  (unconfirmed — see discipline), weak/noise 0.035, **genuine wrong-isomer (substitution)
  only 0.033**.

So the genuine identity-error rate is ~3.3%. Pushing precision past the cluster ~0.66–0.71
needs an **independent identity axis** — MS2 or the isotope/adduct envelope. The chemical-
neighbor (squid_inc `localize.py`) presence prior is *partly redundant* with our structure-
space propagation; a biochemical/pathway prior can only be a **flagged final tiebreaker**
(it inherits prior calls = circular).

## Disciplines (load-bearing, do not violate)

- **The MAF is the answer key.** Closed-world precision vs the MAF is the only precision
  number. We never overrule the key to claim a higher number.
- **A coincident peak at predicted m/z+RT NEVER establishes presence** (the identity bound).
  This kills claims in both directions: it can't confirm our annotation, and it can't convict
  the MAF of a miss. So there is **no answer-key-incompleteness credit** — the "278 MAF-FN
  pool" and `novel_real` are *unconfirmed*, treated as FPs until independent evidence says
  otherwise. `truePrec` is retired.
- **The cluster/dedup gain is legitimate** — duplicates are wrong by the MAF's own standard
  (one peak, named once). Fixing the double-count is not reclassifying the key.

## Paper figures (candidate)

1. P/R curve: forest-cluster front vs blanket (Pareto dominance).
2. Propagation expansion: recall/precision per round, forest vs single chain.
3. Recall-ceiling decomposition (at-predicted / mislocated / absent) + mislocation histogram.
4. Kit-design map: per-RI prediction spread → where to place anchors.
5. FP decomposition + the 3.3% genuine identity-error residual.

## Open / next

- **Recall lever**: empirical kit redesign (concentrate anchors per the map) → re-measure.
- **Precision lever**: MS2 / isotope-envelope identity axis for the 3.3% + the isobaric ties.
- **Generalization**: POS-early/late + polar (in progress).

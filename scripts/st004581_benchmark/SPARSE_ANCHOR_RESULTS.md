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

## Headline P/R curve (neg, 412 MAF compounds), compound-cluster scoring

The front has **two regimes** along the consensus knob M; the operating point is a choice.

**High-precision regime — propagation-only, 8 inj (SMILES-gated):**

| ≥M | recall | precision |
|---|---|---|
| 1 | 0.532 | 0.658 |
| 3 | 0.397 | 0.702 |
| 5 | 0.259 | 0.714 |

**High-recall regime — 16 inj + ladder-fallback (no-SMILES via RI→sec), MINREP=30%:**

| ≥M | recall | precision |
|---|---|---|
| 1 | **0.665** | 0.592 |
| 3 | 0.552 | 0.593 |
| 5 | 0.419 | 0.567 |
| +rescue | 0.680 | 0.578 |

**Both Pareto-dominate blanket m/z+RI matching** (0.490 / 0.435). The two regimes *cross* — at
matched recall ~0.53 propagation-only wins on precision (0.658 vs ~0.59); the ladder-fallback
*extends* recall to ~0.67–0.70 (real reachable ceiling, vs the 0.80 SMILES+detection cap) but
at lower precision, because no-SMILES compounds sit at the imprecise ladder RT with no
structure-model refinement. **NOTE: the original 0.532 "headline" was SMILES-gating-suppressed
— 84 MAF compounds (20%) had no SMILES and were auto-FN'd; PubChem name-resolution recovers
only 2/84 (Metabolon-proprietary names), so the ladder-fallback is the reachability fix.**

To move the WHOLE front out (both axes), the two levers are an **identity axis (MS2/envelope)**
for precision and **better anchor placement** (kit-design map) to tighten the ladder so the
no-SMILES/mislocated compounds stop being low-precision. MINREP/FDR only slide along the front.

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

## ABLATION WATERFALL — marginal P/R gain per lever (neg, cluster M=1)

| lever (cumulative) | recall | precision | ΔRecall | ΔPrec |
|---|---|---|---|---|
| blanket m/z+RI [ref] | 0.490 | 0.435 | — | — |
| + forest propagation | 0.552 | 0.541 | +0.062 | +0.106 |
| + cluster scoring | 0.552 | 0.648 | +0.000 | **+0.107** |
| + InChIKey matching | **0.565** | **0.680** | +0.013 | +0.032 |
| + ladder-fallback | 0.583 | 0.590 | +0.018 | −0.090 |
| + MS2 certify (as filter) | 0.479 | 0.561 | −0.104 | −0.029 |

**Best operating point = forest + cluster + InChIKey: 0.565 / 0.680.** The top three are clean
wins (cluster scoring is the biggest single precision lever, +0.107, recall-free). Ladder-fallback
is a recall-for-precision trade (off by default); **MS2-as-a-filter is counterproductive** —
the MS2-covered subset is already high-precision (0.82, DDA abundance bias), so filtering within
it shrinks the high-precision portion and drags the average down. **MS2 is a CERTIFICATION overlay
(flag a ≥0.90-precision tier, entropy≥0.7), NOT a filter.** Reference-MS2 entropy AUC TP-vs-FP
= 0.603 (in-silico 0.555), capped by DDA chimeric spectra.

## Scoring correctness + lever taxonomy (2026-06-07)

**MAF matching fix.** Name-keying undercounts: the DD carries synonym variants the normalized-
name key splits (`isovalerate` vs `isovalerate (c5)`, `azelate (nonanedioate; c9)` vs `azelaic
acid`). Admitting one synonym while the MAF lists another scores FP + FN for the *same*
compound. Fix = **InChIKey-OR-name matching, recall over distinct MAF compounds** (`forest_prop.py`).
Honest impact: raw precision +0.106 but **cluster precision only +0.022** (cluster scoring
already collapses co-eluting synonyms); recall flat. Real correctness fix, modest headline.

**Lever taxonomy — what actually moves the front:**
| lever | effect | verdict |
|---|---|---|
| compound-cluster scoring | recall held, precision +0.11 vs raw dedup | **CLEAN** |
| InChIKey-OR-name matching | cluster precision +0.02, recall ~0 | **CLEAN (correctness)** |
| ladder-fallback (no-SMILES, RI→sec) | recall +0.13, precision −0.07 | trade (recall lever) |
| more injections (8→16) | recall +0.13, precision −0.11 | trade (recall lever) |
| adduct/fragment filter | precision +0.10, recall −0.10 (cuts 41 TP) | trade |
| mass-unique ladder gate | recall −0.09, precision ~0 | net loss |
| **MS2 / isotope-envelope identity** | not built | **only thing that breaks the ~0.59 ceiling** |

FP composition at the cluster operating point: **conflation 6%, novel/noise 58%, ladder
no-SMILES 36%** — precision is genuinely identity-bound, not a scoring artifact.

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

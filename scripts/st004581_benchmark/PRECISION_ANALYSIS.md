# Closed-world precision analysis (ST004581 vs the Metabolon data dictionary)

The headline matcher number — **F1 0.800 (recall 0.672 / precision 0.988)** — is the
*identity-restricted* metric: it scores only calls whose identity is a MAF compound
(precision = of those, the fraction at the right RT) and recall over MAF compounds. That
is the right number for **matching quality**, but it is **not** how good our annotation is
versus Metabolon's actual output, because it ignores every call we make to a library
compound that *isn't* present.

This folder holds the diagnostics that compute the honest, **closed-world** picture. All
scripts run on the VPS against `feat_colu.parquet` (the ST004581 COLU features) + the
public DD + the repaired MAF; paths are hardcoded for that environment.

## The closed-world frame (mirrors how Metabolon operates)

Metabolon matches every feature against their internal library and reports the compounds
judged present (the MAF). So, treating the MAF as truth:

- **TP** = a library compound we annotate that is present (in the MAF).
- **FP** = a library compound we annotate that is **not** present.
- **FN** = a present compound we don't annotate.

| metric | value | script |
|---|---|---|
| **closed-world precision** | **0.31** | `pr_tradeoff.py` |
| closed-world recall | 0.67 | |
| compounds we call present | 1905 / 3214 DD = **59%** | `cw_eval.py` |
| compounds Metabolon calls present | 832 / 3214 = **26%** | |

We over-call presence by ~2.3×. The data dict is a ~3,200-compound catalog; only ~26% are
in any given sample, but the feature space is dense enough that ~60% of the catalog finds
a convincing m/z+RI look-alike. **Precision is bounded by presence discrimination, not by
matching** — even perfectly placing every present compound caps precision at 832/1905 = 0.44.

## Why no per-peak signal fixes it (`pr_intensity`, `quality_test`, `adduct_envelope`, `corr_test`)

Every per-peak presence signal we tested separates *strong from weak*, not *present from
absent* — because the FPs are **real compound peaks** (just not the named/present one):

| signal | AUC (TP vs FP) |
|---|---|
| log intensity | 0.628 |
| n_files (reproducibility) | 0.609 |
| match score (m/z+RI) | 0.604 |
| isotope ratio vs formula | 0.566 |
| adduct-envelope size | 0.558 |
| M0↔M+1 cross-sample correlation | 0.547 |
| centWave peak quality | 0.537 |

None beat ~0.63. "Is this a real peak?" is ~yes for everything; the question is "is this
the *named* compound?", which these don't answer.

## FP taxonomy (`fp_taxonomy.py`, `fp_char.py`)

Of ~1300 FP calls: ~0% substitution (the one-to-one Hungarian assignment already fixed
present-compound mislabels), ~9% isotope/adduct **duplicates** of present compounds, ~7%
close-m/z-far-RT, ~13% reproducible **novel/real-other** (genuine compounds not in the
MAF), and **~71% noise** (no M+1, sparse). ~91% of FP feature-calls sit at an m/z where no
present compound exists — we annotate the non-library metabolome with library names.

## The noise pattern + the deconvolution attempt (`noise_classify.py`)

The 71% noise *does* have structure: **~62% are isotope/adduct/in-source-fragment debris
of a stronger co-eluting peak**, and they're ~47× weaker than TPs. The discriminating
signal is **relational** ("already explained by a bigger peak nearby"), not per-peak.

But the implemented deconvolution (`--deconvolve`, library_match_rtri) is a **net loss**:
the wide consensus RI window over-flags true compounds in the dense space, killing ~120
TPs (recall 0.672→0.537) for only +0.018 precision. It needs a tight (~1–2 s) same-peak
co-elution window or a **learned peak curator** to be recall-safe. Left off by default.

## Where this points

The correct output is **probabilistic, FDR-controlled annotation + a learned curator**,
not hard calls + hand rules — independently the design of **MassID / PeakDetective /
DecoID2** (Stancliffe et al., Panome Bio, 2026), whose human-plasma result (>4,000
annotated, ~1,200 at FDR<5% ≈ 30%) lands on the same presence-discipline fraction as
Metabolon's 26%. For SQuID-INC the presence prior is meant to come from the
anchor + biochemistry graph rather than MS2.

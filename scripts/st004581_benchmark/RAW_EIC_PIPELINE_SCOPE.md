# Raw-EIC Python pipeline — scope

## Why
Every precision and recall lever we tried tops out at ~+0.04 with a recall cost, because
we work from **summarized consensus features** (m/z, RI, intensity) — we threw away the
chromatographic peak shape. Consequences, all measured on ST004581:

- **Precision (closed-world 0.31).** ~36% of FPs come from blind 5× adduct expansion;
  we can't validate adducts because per-peak signals fail (AUC ≤ 0.63 — the FPs are real
  peaks) and relational grouping (deconvolution, CAMERA-lite) **over-connects in the dense
  feature space** (CAMERA-lite: precision +0.05 but recall −0.40). Real CAMERA confirms a
  family with **EIC peak-shape correlation within each sample**, which the consensus table
  doesn't carry.
- **Recall (0.67).** ~20% of misses are true centWave detection failures (pos-late: 48
  compounds absent from the feature table); the pure-python centWave is weaker than XCMS.

Both are the **same root gap**: we need to work at the **raw extracted-ion-chromatogram
(EIC)** level, not the summarized table. This is library-guided/targeted (a Metabolon-style
benchmark), which makes it tractable — we only probe each library compound's ions, not the
whole feature space.

## Payoffs
1. **Adduct grouping → precision.** Extract each compound's candidate ion EICs in a sample;
   require ≥2 ions whose EIC peak shapes correlate (real family) to annotate. Kills the
   adduct double-count FPs without the over-connection of consensus-level grouping.
2. **Targeted detection → recall.** Pull the EIC at each library ion's m/z + expected RT
   (RI→sec via the per-injection ladder) and detect the peak directly — recovers compounds
   the global centWave missed (the detection-loss bucket).

## Architecture (targeted, library-guided)
For each (library compound, adduct ion, injection):
1. **EIC extractor** — `eic(mzml, mz, ppm) -> (rt[], intensity[])` via pymzML/pyOpenMS.
2. **EIC peak detector** — apex + integration in a window around the expected RT
   (expected_rt = inverse-ladder(DD RI) per injection).
3. **Family peak-shape correlation** — Pearson between a compound's ion EICs in the same
   sample; presence requires the principal ion + ≥1 shape-correlated partner (isotope or
   adduct). This is the real-CAMERA discriminator we currently lack.
4. **Call** — annotate the compound once (its neutral) if the family is confirmed; carry
   per-sample integrated intensities for quant.

## Phases
- **Phase 1 — EIC extractor + single-ion detection.** Install pymzML in `mmenv`; build
  `eic()` + peak detection; validate by reproducing ~20 known features' RT/intensity from
  the consensus table. Deliverable: `raw_eic.py` + a validation print.
- **Phase 2 — adduct-family shape correlation → precision.** For the current FP set, test
  whether requiring an EIC-shape-correlated family separates TP from the adduct-FP
  double-counts (the thing the *cross-sample* correlation couldn't, because shape is
  within-sample and far sharper). Measure closed-world precision/recall.
- **Phase 3 — targeted detection → recall.** Probe the detection-loss compounds' EICs at
  expected RT; measure recall recovery vs the global centWave table.

## Data / infra
- Raw mzML: `/mnt/volume-hel1-1/data/raw/ST004581` (+ `/root/SQuID-INC/data/st004581/mzml`).
- Install: `pymzml` (light) into `/root/mmenv`; pyOpenMS only if we need its peak picker.
- Cost: targeted extraction is ~(library ions × injections) EIC pulls per platform; cache
  per (mzml, m/z) and restrict to library ions to keep it tractable. Random-access mzML
  indexing (pymzML) avoids full-file scans.

## Honest risk
Phase 2 is the bet: EIC *shape* correlation is much sharper than the cross-sample intensity
correlation that failed (AUC 0.55) — but if our weak ion detection means the family members
still aren't *there*, shape can't rescue them. Phase 1 validation will tell us whether the
faint family ions are recoverable from raw before we commit to Phase 2/3.

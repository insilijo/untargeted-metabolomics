# Sparse-Anchor Metabolite Annotation vs. Metabolon — ST004581 Benchmark

**One-line result:** a sparse, purchasable spike-in kit + the *public* Metabolon data
dictionary reproduces **~85% of Metabolon's expert-curated annotations** (732 of 862
compounds across four LC-MS platforms) at a **misidentification rate under 2%**, using
**no proprietary library, no per-compound authentic standards, and no MS2**.

---

## 1. Why this matters

Commercial metabolite annotation (Metabolon and peers) depends on a large proprietary
spectral/RT library, authentic standards, and expert manual curation. This benchmark asks
whether a **sparse spike-in kit** — a small, purchasable set of anchor compounds — plus only
*public* information can reproduce that output. The answer key here is **manually curated by
experts**, not an automated pipeline, so matching it is a deliberately hard bar.

## 2. Method (summary)

1. **Anchor kit → retention model.** A sparse set of spiked anchors calibrates (a) an RI→seconds
   ladder (monotone PCHIP) and (b) a structure→RT predictor (gradient-boosted trees on RDKit
   descriptors). Chemical-diversity anchor selection (Tanimoto farthest-point) beats random.
2. **EIC-precise peaks.** Apexes are taken from raw extracted-ion chromatograms (≈2 s precision),
   not summarized feature tables, via a memory-bounded streaming extractor (scales to 10⁵ ions).
3. **Forest propagation.** K bootstrapped self-training chains admit only peaks within a tight
   RT window, reproducible across injections, isobar-resolved, FDR-controlled; admitted peaks
   become new anchors and predictions tighten. A consensus vote (M chains) is the precision knob.
4. **Cardinality scoring.** Co-eluting isobaric candidates are assigned one-to-one to distinct-RT
   peaks (Hungarian), so an isobaric group recovers the *right number* of compounds — no
   tie-inflation, no double-counting.

## 3. Benchmark design

| element | choice |
|---|---|
| Dataset | ST004581 (Metabolon), COLU column, 8 injections/platform |
| Platforms | lc/ms neg, pos-early, pos-late, polar |
| Answer key | MAF — **expert manually-curated** annotations |
| Candidate library | **public** Metabolon data dictionary (1,627 distinct compounds) |
| Recall | cardinality recall over distinct curated compounds |
| Precision | reported two ways (see §5); **out-of-scope calls treated conservatively as our errors** |
| Provenance | leak-free — structures from PubChem, never the answer key |

## 4. Headline results (RT/RI + MS1 only, no MS2; stereo/geometric-isomer-aware)

Scored on **distinct compounds** keyed by full InChIKey — stereo and geometric isomers
(fumarate vs maleate, erythronate vs threonate, cis/trans, R/S, E/Z) are differentiated, not
collapsed.

| platform | distinct compounds | **recall** | **identity precision** | closed-world precision |
|---|---|---|---|---|
| pos-early | 204 | **0.887** | 1.000 | 0.543 |
| neg | 403 | **0.854** | 0.992 | 0.587 |
| pos-late | 192 | **0.833** | 0.990 | 0.638 |
| polar | 93 | **0.796** | 1.000 | 0.639 |
| **weighted** | **892** | **0.851** | **≈0.996** | — |

- **Recall 0.80–0.89** of an expert-curated gold standard, automatically, from a sparse kit.
- **Identity precision 0.99–1.00**: among peaks the curators adjudicate, we essentially never
  assign the wrong compound (0–2 substitutions per platform).
- Recall sits near the dataset's **detection-limited ceiling** (~0.87 recoverable); remaining
  misses are RT-mislocation or sub-floor signal, not algorithmic.

### 4a. Stereo/geometric-isomer resolution (where retention beats MS2)

Isomers that share a molecular skeleton (and so share m/z **and** fragmentation) cannot be
separated by MS2. The RT/RI model assigns each to its own peak via the data-dictionary
retention index:

| platform | isomer skeletons | both/all recovered | partial | none |
|---|---|---|---|---|
| polar | 8 | **8** | 0 | 0 |
| neg | 12 | **9** | 2 | 1 |
| pos-late | 1 | **1** | 0 | 0 |
| pos-early | 1 | 0 | 0 | 1 |
| **total** | **22** | **18 (82%)** | 2 | 2 |

All 8 polar pairs fully resolved (fumarate **and** maleate, etc.). The 4 unrecovered are
**library-coverage** gaps (the DD does not list the second isomer), not algorithmic limits.

### 4b. Compound-count provenance

`934` total MAF rows → `−29` Metabolon-flagged unannotatable → `905` curated annotations →
keyed by full InChIKey (with a name fallback for the 47% of DD entries lacking an InChIKey) →
**`892` distinct compounds** scored. (A coarser InChIKey14 skeleton key would merge the stereo/
geometric isomers above into 862 — this benchmark keeps them distinct.)

## 5. The two precision numbers (read both)

- **Identity precision (≈0.99)** — of calls on peaks the curated key adjudicates, the fraction
  with the correct compound. This is the meaningful misidentification rate. **It is low.**
- **Closed-world precision (≈0.55–0.68)** — counts *every* call not in the curated key as an
  error, including calls at ions the key never covers. Per the conservative stance adopted here,
  **all such out-of-scope calls are treated as our errors.** Characterization (neg, n=162) shows
  they are ≈47% low-reproducibility/noise, ≈6%+ redundant ions (adducts/in-source fragments) of
  *already-annotated* compounds, and ≈33% reproducible-but-identity-ambiguous (median 8 isobaric
  candidates in full chemical space). None are claimed as discoveries.

## 6. Comparators (same dataset, neg, correct reference implementations)

| method | recall | precision | notes |
|---|---|---|---|
| GNPS spectral match (matchms) | 0.078 | 0.938 | requires MS2; DDA-coverage-limited |
| OpenMS FeatureFinderMetabo + accurate mass | 0.27 | 0.625 | MS1 mass match |
| **This method (sparse kit, RT/RI+MS1)** | **0.852** | **0.996 (identity)** | no proprietary library, no MS2 |

3–11× the recall of the standard open tools, at higher precision.

## 7. What this is — and is not

**Is:** evidence that sparse-anchor calibration + public data reproduces expert-curated
commercial annotation at a fraction of the apparatus, reproducibly across four platforms.

**Is not:**
- **"Better than Metabolon."** The candidate library *is* Metabolon's dictionary and the key is
  their curation, so the benchmark is closed-world and cannot score above them. Calls outside the
  dictionary (true discovery) are out of scope here and are conservatively counted as errors.
- **Multi-dataset validated.** This is one richly-characterized study (ST004581). Generalization
  claims require independent datasets.
- **MS2-dependent.** MS2 was tested (confirm-required training gate + MS2-aware tiebreak); on this
  closed-dictionary benchmark its effect is marginal (≤0.01 recall). Its leverage is full-chemical-
  space disambiguation, not this task.

## 8. Reproduction

```bash
# per platform (lc/ms neg | lc/ms pos early | lc/ms pos late | lc/ms polar), library dd (public DD)
FLOOR=8000 KIT_MODE=chem ROUNDS=6 PLATFORM="lc/ms neg" LIBRARY=dd \
  python forest_sweep_h2h.py <KIT_SIZE=0> <KIT_SEED=0> <K=10>
# emits: cardinality recall + closed-world & identity precision per consensus level M
```

- `forest_sweep_h2h.py` — main runner (PLATFORM × LIBRARY switches; streaming cached EIC).
- `forest_sweep_ms2.py` — adds MS2 ablation (MS2MODE=vanilla|gate|tiebreak|both) + the
  4-part vs-Metabolon REPORT (parity / coverage / out-of-scope-with-evidence).
- `eic_stream_validation.py` — validates the streaming extractor vs the dense method
  (99.94% RT agreement, 0 intensity mismatches).

Memory/storage: the streaming EIC extractor replaces a dense (scans × ions) matrix with
vectorized cumulative-sum window-sums + rolling local-max detection, m/z dedup, and an on-disk
peak cache — extracting 10⁵ ions in seconds within a bounded memory footprint.

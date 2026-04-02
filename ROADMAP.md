# untargeted-metabolomics Roadmap

## Current State (2026-03-23)

### Working well
- The repo has a coherent stepwise script pipeline from input validation through feature finding, blank subtraction, library search, reporting, and structure-prediction export.
- The workflow is documented more clearly than many early-stage analysis repos.
- There is already an explicit bridge to Spec2Mol-style downstream prediction.

### Quality assessment
- Overall quality: well-structured analysis scaffold, but still scaffold-heavy.
- Strengths are the numbered script pipeline and clear expected inputs/outputs.
- Weaknesses are the lack of tests, likely dependence on operator-managed environments, and the fact that much of the pipeline has not yet been hardened around real challenge data.
- The repo currently looks close to useful, but not yet trustworthy enough for unattended reruns.

### Highest risks
- The pipeline spans many failure-prone file transformations without automated regression coverage.
- Integration with external tools such as Spec2Mol is still environment-sensitive.
- Script churn in late-stage steps suggests packaging/output conventions are still moving.

## Priority Roadmap

### Phase 1 — Make the Pipeline Re-runnable
- [ ] Add a single command that runs the full pipeline reproducibly on a known sample dataset.
- [ ] Add smoke tests for each stage's expected outputs.
- [ ] Lock down config handling and environment assumptions.

### Phase 2 — Harden Real-Data Behavior
- [ ] Validate feature alignment, blank subtraction, and library-linking steps on real challenge inputs.
- [ ] Add failure diagnostics for missing or empty intermediate files.
- [ ] Finalize the structure-prediction packaging step and document downstream handoff clearly.

### Phase 3 — Reporting
- [ ] Produce one canonical report bundle from the pipeline.
- [ ] Make provenance of matches and unmatched spectra explicit in the final outputs.

## Recommendation

This repo is close to being a dependable workflow, but it needs regression checks and one canonical sample run before it should be treated as a robust analysis pipeline.

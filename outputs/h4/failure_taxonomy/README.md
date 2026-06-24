# Failure-mode taxonomy: why do generated ADC candidates fail?

Wave-0 task 0.3. Pure tabulation of existing per-molecule gate results (no GPU / RDKit). Re-run with `python3 analyze_failures.py`.

## The funnel (strict, first-failed-stage)

Each candidate is classified by the **first** gate it fails:

1. `not_parsed`    -- the SDF could not be parsed at all.
2. `disconnected`  -- parsed, but the linker is **not natively bonded** to payload+antibody handle (the *connectivity wall*).
3. `mmff_fail`     -- connected, but MMFF relaxation / valence fails.
4. `no_adc_motif`  -- MMFF-clean, but lacks a Val-Cit or self-immolative urea cleavage motif.
5. `usable`        -- survives every gate.

These five are mutually exclusive and exhaustive (verified: each file's five buckets sum to its row count).

## Cohorts and files included

- **rebaseline_n500** (authoritative n=500 for the 4 main configs): `outputs/h4/eval/rebaseline_n500/{curriculum851,multisite884_base,h4_lambda0.01,h4_lambda0_control}_per_mol.csv`
- **per_n200** (earlier n=200 runs, *only configs not in rebaseline* to avoid double-counting): `outputs/h4/eval/per/{lambda0.1,lambda1.0,lambda10,lambda10_strict}/<run>_per_mol.csv`
- **deliverable** (n=500 best model): `outputs/h4/deliverable/final_best_per_mol.csv` (byte-identical to `outputs/h4/eval/per/final_best/final_best_per_mol.csv`; counted once).

The four rebaseline configs' n=200 counterparts (curriculum851, multisite884_lam0, lambda0.01, lambda0_control) are **excluded** from per_n200 so no config is double-counted across cohorts.

## Breakdown table (counts, with % in parentheses)

| config | cohort | n | not_parsed | disconnected | mmff_fail | no_adc_motif | usable |
|---|---|---:|---:|---:|---:|---:|---:|
| curriculum851 | rebaseline_n500 | 500 | 0 (0.0%) | 396 (79.2%) | 3 (0.6%) | 58 (11.6%) | 43 (8.6%) |
| h4_lambda0.01 | rebaseline_n500 | 500 | 0 (0.0%) | 334 (66.8%) | 2 (0.4%) | 113 (22.6%) | 51 (10.2%) |
| h4_lambda0_control | rebaseline_n500 | 500 | 0 (0.0%) | 423 (84.6%) | 1 (0.2%) | 53 (10.6%) | 23 (4.6%) |
| multisite884_base | rebaseline_n500 | 500 | 0 (0.0%) | 300 (60.0%) | 3 (0.6%) | 173 (34.6%) | 24 (4.8%) |
| lambda0.1 | per_n200 | 200 | 0 (0.0%) | 126 (63.0%) | 1 (0.5%) | 55 (27.5%) | 18 (9.0%) |
| lambda1.0 | per_n200 | 200 | 0 (0.0%) | 136 (68.0%) | 1 (0.5%) | 38 (19.0%) | 25 (12.5%) |
| lambda10 | per_n200 | 200 | 0 (0.0%) | 157 (78.5%) | 1 (0.5%) | 28 (14.0%) | 14 (7.0%) |
| lambda10_strict | per_n200 | 200 | 0 (0.0%) | 154 (77.0%) | 1 (0.5%) | 34 (17.0%) | 11 (5.5%) |
| final_best | deliverable | 500 | 0 (0.0%) | 309 (61.8%) | 2 (0.4%) | 141 (28.2%) | 48 (9.6%) |
| **ALL POOLED** | all | 3300 | 0 (0.0%) | 2335 (70.8%) | 15 (0.5%) | 693 (21.0%) | 257 (7.8%) |

## Headline finding

**Disconnection is the wall.** Across all 3300 pooled candidates, **70.8%** are lost to `disconnected` -- the single largest failure mode by a wide margin. Parsing essentially never fails (0.0%); the model reliably emits a valid SDF but fails to *bond the linker into one covalent ADC graph*.

### Conditional drop-off down the funnel (pooled)

- Of all candidates, **29.2%** reach connectivity (965 / 3300 parsed-and-connected).
- Of the **connected** ones, **1.6%** then fail MMFF / valence (15 / 965).
- Of the **MMFF-clean** ones, **72.9%** lack an ADC cleavage motif (693 / 950) and are dropped at `no_adc_motif`.
- Net **usable yield (pooled): 7.8%** (257 / 3300).

So the loss chain is overwhelmingly front-loaded: the connectivity gate destroys the majority of candidates, MMFF prunes a small slice of survivors, and the motif filter is the *second* big gate -- it removes a large fraction of otherwise-valid 3D molecules.

## Per-config contrasts

Using the authoritative n=500 rebaseline configs:

| config | connectivity (1 - disc) | MMFF-clean | usable | motif yield of MMFF-clean |
|---|---:|---:|---:|---:|
| curriculum851 | 20.8% | 20.2% | 8.6% | 42.6% |
| multisite884_base | 40.0% | 39.4% | 4.8% | 12.2% |
| h4_lambda0.01 | 33.2% | 32.8% | 10.2% | 31.1% |
| h4_lambda0_control | 15.4% | 15.2% | 4.6% | 30.3% |

**Multi-site has higher connectivity but lower motif yield -- CONFIRMED.** `multisite884_base` reaches the *highest* connectivity of any config (40.0% vs 20.8% for curriculum), yet only **12.2%** of its MMFF-clean molecules carry an ADC motif (vs 42.6% for curriculum). The motif collapse is severe enough that multi-site's final usable yield (4.8%) is *below* curriculum's (8.6%) despite nearly double the connectivity -- it generates well-bonded scaffolds that mostly lack a cleavable Val-Cit/urea linker.

**lambda0.01 is the best 4-config rebaseline:** highest usable yield at **10.2%**, driven by good connectivity (33.2%) *and* the best motif retention.

**lambda0_control (no connectivity reward) is the worst:** connectivity only 15.4% and usable only 4.6% -- direct evidence the connectivity objective is doing real work.

**final_best (deliverable, n=500):** usable **9.6%** (connectivity 38.2%) -- matches the known ~9-10/100 target for the best model.

## Files in this directory

- `analyze_failures.py` -- deterministic, re-runnable generator.
- `failure_breakdown.csv` -- per (config, cohort) counts + percentages, plus an ALL_POOLED row.
- `failure_funnel.png` / `.pdf` -- stacked-bar funnel per config.
- `README.md` -- this report.

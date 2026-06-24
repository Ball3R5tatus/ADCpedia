# Wave 1 — generalization & the Pareto frontier

Two GPU experiments that turn the generation side from "we tried checkpoints"
into a reproducible map of model behaviour. Both reuse the frozen
`data/processed/dataset_manifest.csv` (Wave 0) and the unified
`src/diffusion/gate` (Wave 0). The split builder is
`src/diffusion/data/build_split.py`.

**Rigor (Master Finding 9):** single-checkpoint comparisons at n<500 cannot
separate signal from GPU run-to-run variance. So: **n≥500 generations per
config, ≥3 training seeds, Wilson 95% CIs on every yield.** `eval_sweep.py`
computes the CIs; the plot averages replicates per fraction.

---

## 1.2 Pareto frontier (cysteine sampling fraction) — *do this first*

**Question:** can we steer the model along the connectivity ↔ target-chemistry
tradeoff with a single knob? The prior project saw 3 points (66 % natural,
85 % curriculum, 100 % cys-only); this densifies it to a curve.

Instead of the `WeightedRandomSampler` (which caused the VRAM death-spiral,
§19.13), we **pre-balance** the multi-site training trio at each target cysteine
fraction, at fixed total size (1240 rows) so training compute is constant.

```bash
# Step 0 — build splits (no GPU). Multi-seed for real CIs:
for s in 0 1 2; do
  PYTHONPATH=src python -m diffusion.data.build_split pareto \
      --fractions 50,66,75,85,100 --seed $s
done
# -> data/processed/splits/pareto_fXXX_sY/ each with a trio + a cloned .yml

# Step 1 — train + generate + gate-eval (GPU; from the difflinker_gpu env):
conda activate difflinker_gpu
bash experiments/wave1/run_wave1.sh           # SPLIT_GLOB="pareto_*"

# Step 2 — figure (also run automatically at the end of run_wave1.sh):
PYTHONPATH=src python experiments/wave1/eval_sweep.py plot \
    --results outputs/wave1/results.csv --out outputs/wave1/pareto_frontier.png
```

**Sanity anchor:** the 66 % point is the natural multi-site mix and should
reproduce the prior anchor (native-connected ≈ 40 %, usable ≈ 4–5 %); the
100 % point ≈ the cysteine-only fine-tune (conn ≈ 28 %, Val-Cit ≈ 23 %). If the
66 % point lands far off those, something in the data/checkpoint path drifted.

**Deliverable:** `outputs/wave1/pareto_frontier.png` — "cysteine sampling
fraction controls ADC chemistry vs 3D validity", the strong positive result.

---

## 1.1 Leave-one-payload-class-out (LOPO) — generalization vs interpolation

**Question (no current answer):** does the fine-tune *generalize* to a payload
class it never saw, or only *interpolate* within trained classes?

Feasible classes (multi-site train rows; conformer-augmented, ×20 per L/P):
auristatin 240, camptothecin 220, maytansinoid 220, PBD_anthracycline 200,
duocarmycin 140, eribulin 100 (≥100 rows → safe to hold out). taxane 20 and
calicheamicin 40 are thin → report as "untestable at current data scale" (which
is itself the data-wall finding).

```bash
# Step 0 — build LOPO splits (no GPU): one training set per held-out class
PYTHONPATH=src python -m diffusion.data.build_split lopo \
    --classes auristatin,camptothecin,maytansinoid,PBD_anthracycline,duocarmycin,eribulin --seed 0

# Step 1 — for EACH held-out class:
#   (a) export a representative held-out payload as a generation input:
PYTHONPATH=src python -m diffusion.data.difflinker_export --smiles "<held-out payload SMILES>" \
    --out data/processed/difflinker_inputs/<class>_input.sdf
#   (b) train on the LOPO trio, generate on that held-out input, gate-score.
#       Use run_wave1.sh with SPLIT_GLOB="lopo_*" and GEN_INPUT pointed at the
#       held-out input (edit per class), OR drive eval_sweep.py directly.
```

**Read-out:** if usable-3D% and motif% on the *held-out* class are comparable to
in-distribution, the model generalizes; if they collapse, it interpolates. Either
way it is a publishable answer to the reviewer's first question.

---

## Files

| file | role |
|---|---|
| `src/diffusion/data/build_split.py` | builds Pareto (resample) + LOPO (filter) trios + cloned configs |
| `experiments/wave1/run_wave1.sh` | GPU driver: seed → train → generate → gate-eval → plot |
| `experiments/wave1/eval_sweep.py` | gate aggregation, Wilson CI, novelty Tanimoto, Pareto figure |
| `data/processed/splits/SPLITS_MANIFEST.csv` | log of every split built |
| `outputs/wave1/results.csv` | per-config aggregated metrics (gitignored; regenerable) |

## Caveats baked in
- Checkpoint pick = latest epoch (≈20 past val min under early stopping). Refine
  to best-val if your ModelCheckpoint filenames encode the val loss.
- `outputs/` is gitignored by design; the figure + results.csv regenerate from
  the trios + checkpoints.
- Multi-seed is **not optional** for the headline numbers — a single seed can
  swing usable-3D ~2× (Finding 9).

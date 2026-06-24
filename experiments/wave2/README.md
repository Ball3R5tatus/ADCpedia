# Wave 2 — lock the numbers, and the one physics lever

Two parts: (B) a multi-seed, size-swept **benchmark** that pins the generation
numbers with confidence intervals, and (C) a **per-linker physics descriptor**
pilot — the only evaluation lever that adds a signal varying *per linker*.

---

## B. Multi-seed 3D-strict benchmark

**Purpose:** a *confirmation* engine, not discovery. Master Finding 9 showed a
single checkpoint can swing usable-3D ~2× from GPU variance; this re-runs the
surviving configs at **≥3 seeds, n≥500, 4 linker sizes, Wilson 95% CIs** so the
Pareto/connectivity ceiling is stated with error bars, not anecdote.

Configs (reuse Pareto splits as proxies, so no new training data):
`cysonly` = pareto_f100 · `multisite` = pareto_f066 · `curriculum` = pareto_f085
· plus **zero-shot** (GEOM checkpoint, no fine-tune). Size dimension 15/20/25/30.

```bash
# Step 0 — build the splits for every seed (no GPU):
for s in 0 1 2; do PYTHONPATH=src python -m diffusion.data.build_split \
    pareto --fractions 66,85,100 --seed $s ; done

# Step 1 — train + size-sweep generate + gate-eval + table (GPU):
conda activate difflinker_gpu
bash experiments/wave2/run_benchmark.sh

# (table can be regenerated any time from the accumulated results:)
PYTHONPATH=src python experiments/wave1/eval_sweep.py table \
    --results outputs/wave2/results.csv --out outputs/wave2/benchmark.md
```

**Deliverable:** `outputs/wave2/benchmark.md` — config × size × seed, conn% /
Val-Cit% / usable% with Wilson CIs. Expected shape: connectivity falls and ADC
chemistry rises with size; zero-shot fails outright at large ADC sizes (§12);
usable-3D ceiling ~10/100 with overlapping CIs across the top configs (i.e. no
config dominates beyond noise — that IS the finding).

---

## C. Per-linker physics descriptor (the only per-linker eval lever)

**Why it matters:** §20.13 found the Cys214 site microenvironment is identical
for every candidate, so a *site* descriptor returns the same value for all — it
ranks sites, not linkers. The signal that could rank *linkers* is how each linker
perturbs the protonation/pKa of residues around the conjugation sulfur (the
Sebastián-Pérez / Lyon-2014 ring-opening-vs-retro-Michael axis).

### C1 — cheap static surrogate: per-linker PROPKA  *(runnable, with deps)*
`per_linker_pka.py` runs PROPKA on each **conjugated complex** (protein + that
linker + payload) and reports the pKa of titratable residues near the
conjugation SG, then ranks residues by ΔpKa across linkers (the descriptor).

```bash
python experiments/wave2/per_linker_pka.py \
    --pdbs md/runs/cand_4/<conjugated_complex>.pdb \
           md/runs/cand_5/<conjugated_complex>.pdb \
           md/runs/cand_vedotin/<conjugated_complex>.pdb \
    --site-segid A --site-resid 214 --cutoff 10 \
    --out outputs/wave2/per_linker_pka.csv
```

**Honest dependencies (the script self-checks and degrades, it does not fake):**
1. **PROPKA3 binary** — the env that produced `/tmp/rec.pka` in §20.13 (not on
   the default PATH; `pip install propka` in that env, or put `propka3` on PATH).
2. **A conjugated-complex PDB per linker** containing the linker+payload (the apo
   `receptor.pdb` is identical across candidates → returns the §20.13 null; you
   must feed the conjugate). If your complexes are merged to a single chain,
   pass `--site-segid` accordingly (the script falls back to resid-only).
3. MDAnalysis (already used by `md/analysis/site_microenv.py`).

This is a *hypothesis generator*, not a validated predictor (no stability labels
exist — Master §21). Present any ΔpKa ranking as a hypothesis, never as
"prediction."

### C2 — the rigorous version: constant-pH MD  *(blocked on toolchain)*
The real lever is CpHMD (λ-dynamics) so the maleimide-microenvironment
protonation samples dynamically per linker. **Blocker:** the project's
`gmx_mpi` (/home/galeito/gromacs-mpi) shows no constant-pH / λ-dynamics support;
CpHMD needs a CpHMD-capable GROMACS build (GROMACS λ-dynamics constant-pH, or the
phbuilder/CpHMD toolkit). Protocol when that build exists:
- titratable groups: the succinimide/maleimide-adjacent ionisable group + the
  proximal basic residue(s) surfaced by C1;
- reuse the §20.7 covalent topology + the §20.11 validity gate;
- ≥100–500 ns per linker (the proven ADC protocol is 500 ns + CpHMD + QM,
  Sebastián-Pérez 2022); read out local pKa / fraction-protonated per linker.
Even CpHMD gives *proxies*, not the bond-breaking reaction (that is QM/MM) — so
scope the claim to "local protonation environment differs per linker," not
"predicted stability."

---

## Files
| file | role | runnable now? |
|---|---|---|
| `experiments/wave2/run_benchmark.sh` | multi-seed size-swept benchmark driver | GPU |
| `experiments/wave1/eval_sweep.py table` | markdown benchmark table from results | yes |
| `experiments/wave2/per_linker_pka.py` | per-linker PROPKA descriptor pilot | needs PROPKA3 + conjugate PDBs |
| (CpHMD) | rigorous per-linker protonation | needs CpHMD-capable GROMACS build |

# Paper skeleton — "Fixed-bond MD does not rank ADC linkers"

> **Status: SKELETON (2026-06-22).** Argument-per-section + figure plan + key numbers pulled
> from `docs/DIFFUSION_PIPELINE_MASTER_FINAL.md`. `[WRITE]` = prose to draft; `[FIG]`/`[TAB]` =
> artifact to produce/assemble (most already exist under `md/analysis/` and `outputs/`).
> Target: ChemRxiv preprint → methods/ML-for-science venue (NeurIPS MLSB, J. Chem. Inf. Model.).
> Author: Gaël Coulombe. The paper is a **clean negative result anchored by an FDA positive control** —
> limitations are a feature, not a weakness.

---

## Working title (pick one)

1. **Fixed-bond molecular dynamics does not rank antibody–drug conjugate linkers: a negative result validated against an FDA gold standard**
2. *When the proxy passes but the object is wrong: surface metrics mask structural errors in generative ADC linker design*

(#1 leads with the headline result; #2 leads with the cross-cutting methodological lesson. Recommend #1 as title, #2 as the framing of the Discussion.)

---

## Abstract (draft — real numbers)

Generative 3D models can now propose antibody–drug conjugate (ADC) linkers, shifting the open
problem from *generation* to *evaluation*. We fine-tune DiffLinker on 73 ADC linker-payloads and
show it learns ADC linker chemistry without memorization — cleavable motifs (Val-Cit, urea,
Gly-Gly-Phe-Gly) emerge at 10–69% vs 0–4% in the zero-shot baseline, with median Tanimoto 0.44–0.59
to the training set and **0 near-copies across 1000 generations**. The dominant practical failure is
not the model but **chemical validity**: outputs routinely lose connectivity, valence, and
stereochemistry, yielding only **~9–10 fully 3D-valid candidates per 100** after a fail-closed
validity gate. We then ask whether classical covalent molecular dynamics (MD) can *rank* the
survivors. Building covalent trastuzumab-Fab/HER2 complexes at Cys214 and running 50 ns each, all
candidates are **grossly stable** (covalent bond 1.81 Å, secondary structure preserved, per-domain
Cα RMSD 1.2–1.8 Å). An apparent payload-stability ranking is traced to a **topology artifact** (a
pentavalent conjugation carbon); after correction it vanishes and partially inverts. A pre-registered
**N=3 replicate** study finds no separation (Welch p=0.157), with inter-replicate spread as large as
the inter-candidate gap. Finally, grafting the **FDA-approved brentuximab-vedotin linker**
(mc-Val-Cit-PAB-MMAE) onto the identical scaffold yields a payload metric **statistically
indistinguishable** from the generated candidates (p=0.41, 0.20). We conclude that fixed-bond MD
payload RMSD is non-discriminative for ADC linkers because the chemistry that would discriminate them
— succinimide hydrolysis, retro-Michael deconjugation, local pKa — is absent from the model by
construction. The bottleneck has moved from generation to evaluation, and evaluation is currently
**data-bound, not method-bound**.

---

## 1. Introduction  `[WRITE]`

- ADC primer: antibody + cleavable linker + cytotoxic payload; the linker controls plasma stability /
  deconjugation, which gates efficacy and therapeutic index. (1–2 sentences, cite a review.)
- The generate-then-evaluate paradigm; 3D diffusion (DiffLinker) for linker design.
- **The gap this paper attacks:** generation methods are improving fast, but the *evaluation/ranking*
  step that selects "the best" linker is usually assumed reliable and rarely stress-tested. We
  stress-test it and it fails — cleanly, and against a clinical control.
- **Contributions (bullet list):**
  1. A reproducible DiffLinker→validity-gate→covalent-MD pipeline for ADC linkers (Cys214 thiosuccinimide).
  2. Evidence that chemical validity — not generation — is the dominant practical bottleneck; a
     fail-closed valence/connectivity/stereo gate as the operative screen.
  3. A documented **ranking artifact**: a covalent-topology valence error produced a confident but
     spurious payload-stability ranking that disappears under correction.
  4. A **pre-registered N=3 + FDA positive-control** test showing fixed-bond MD payload RMSD does not
     discriminate generated linkers from each other or from an approved clinical linker.
  5. A cross-cutting methodological lesson: *surface metrics (bond length, AUC, score) mask structural
     errors (valence, loaded weights, stereo)*; adversarial validation is required.

## 2. Background / related work  `[WRITE]`

- 3D diffusion linker generation (DiffLinker; EGNN; GEOM pretraining). Atomistic output → bond
  perception → the validity problem.
- ADC linker stability determinants: maleimide–thiol Michael addition; **succinimide ring-opening
  hydrolysis** (self-stabilization, Lyon 2014); **retro-Michael / thiol exchange** with plasma thiols
  (albumin, glutathione); local pKa / proximal-base catalysis (Sebastian-Perez axis). → these are the
  quantities that *would* rank linkers.
- Existing ADC scoring: AMM (potency classifier, 2D); MD-based stability proxies. Both implicitly
  assume the discriminating signal is captured — the assumption we test.

## 3. Methods  `[WRITE]`

- **3.1 Generation.** DiffLinker fine-tune (73 multi-site L/Ps; recipe: fix-1 isfinite guard, LR 1e-5,
  batch 4; checkpoint H4 λ=0.01 / curriculum). *Brief — supporting, not the contribution.* (master §3-§19)
- **3.2 Chemical-validity gate.** Level A (ligand RDKit: sanitize, single component, undefined
  stereocenters, open-maleimide vs thiosuccinimide); Level B (per-atom valence vs element max). Run
  fail-closed post-merge. (master §20.11)
- **3.3 Covalent complex construction.** 1N8Z (trastuzumab Fab + HER2 ECD); Cys214 selection by SG-SASA
  (43.75 Å², most exposed free Cys); **canonical MMAE stereo imposition** via CIP-driven iterative
  chiral-tag assignment (10/10 CIP); Michael-addition conjugation; geometric pose + clash min. (§19.19, §19.24)
- **3.4 MD protocol.** amber99sb-ildn / GAFF2-AM1BCC / TIP3P / 0.15 M NaCl; dodecahedron d=1.2;
  EM→NVT 100 ps→NPT 100 ps→50 ns production, 2 fs; **fixed (harmonic) covalent C–SG bond** with
  proper exclusions (the modeling choice under test). Note the staged NPT(CPU)/prod(GPU) barostat fix. (§19.25-§20.0)
- **3.5 Evaluation metrics.** MMAE-core (51 heavy atoms, order-verified) self-fitted RMSD; per-domain
  Cα RMSD (self-fitted, correct for multi-domain Fab–HER2); DSSP; C–SG trace; lig–protein contacts; Rg.
  State explicitly which are *integrity checks* (C–SG, mindist — bonded-topology-enforced) vs *behavioral* (self-RMSD). (§20.9)
- **3.6 Positive control.** Authentic mc-Val-Cit-PAB-MMAE (brentuximab vedotin, CAS 646502-53-6,
  C68H105N11O15, MW 1316.65); same scaffold, same Cys214, same thiosuccinimide chemistry, same FF and
  protocol — everything constant except the linker. MMAE-core disambiguation rule (exclude PABC
  carbamate carbon). (§20.12)
- **3.7 Statistical design (pre-registered).** N=3 independent replicates (rep1 = NPT-velocity
  continuation; rep2/3 = gen_vel with distinct seeds). Metric = MMAE-core self-RMSD, mean over 25–50 ns.
  Pre-registered separation criterion: disjoint mean±SD ranges AND Welch p<0.05. (§20.8)

## 4. Results

### 4.1 DiffLinker learns ADC chemistry without memorizing  `[WRITE]` `[TAB]`
- Motif emergence (Val-Cit/urea/GGFG) ft vs zs at matched size; 0 near-copies; Tanimoto medians 0.44–0.59.
- `[TAB 1]` motif % and Tanimoto across configs (from master §19.12 / §12.4).
- One-line honest caveat: chemistry↔connectivity and diversity↔specificity tradeoffs (cite, don't re-run).

### 4.2 Chemical validity, not generation, is the practical bottleneck  `[WRITE]` `[FIG]`
- Connectivity/valence/stereo loss in raw 3D outputs; the gate; ~9–10/100 usable-3D (Wilson CI).
- `[FIG 2]` yield funnel: generated → connected → valence-OK → stereo-correct → motif-bearing.
- This reframes the project: the validity gate is an operative deliverable.

### 4.3 Gross MD stability: all candidates pass  `[WRITE]` `[TAB]` `[FIG]`
- 4 candidates: C–SG ~1.81 Å, DSSP ~43% (std <1.1%), per-domain Cα RMSD light 1.2–1.85 / heavy 1.5–1.6 /
  HER2 dom II–III rigid, I–IV flex; no dissociation. (§20.1–§20.3, §19.31)
- `[FIG 3]` per-domain RMSD + DSSP panel (cand_4); `[TAB 2]` batch gross-stability summary.
- Message: at the **coarse** level the model is fine — set up the discrimination question.

### 4.4 The ranking artifact (the pivot)  `[WRITE]` `[FIG]` ← **key figure**
- Pre-fix apparent ranking cand_4 > cand_1 > cand_5 (cand_4 self-RMSD 1.21 Å looked rigid).
- The **pentavalent conjugation carbon** bug (added C–S without removing one H → 5 bonds). (§20.10)
- After valence fix: cand_4 self-RMSD 1.21 → **3.51 Å**; secondary metrics (Rg, contacts) invert. (§20.5–§20.6)
- `[FIG 4]` buggy-vs-fixed payload metrics — the artifact made visible.
- This is the "surface metric (distance) passed a chemically-invalid topology" instance.

### 4.5 N=3 replicates: no separation  `[WRITE]` `[FIG]`
- cand_4 3.15±0.96 Å, cand_5 4.33±0.30 Å; gap 1.18 Å; Welch t=-2.04, df=2.40, **p=0.157**; ranges overlap.
- Intra-candidate SD (cand_4, 0.96 Å) ≈ inter-candidate gap → payload not configurationally converged at 50 ns.
- `[FIG 5]` = `md/analysis/selfrmsd_n3_summary.png`. Honest caveat: N=3 is low power (catches only large effects).

### 4.6 FDA positive control: clinical linker indistinguishable  `[WRITE]` `[FIG]` ← **the closer**
- Vedotin 3.75±0.54 Å sits *between* cand_4 and cand_5; Welch p=0.41 (vs cand_4), 0.20 (vs cand_5).
- Two reads: (good) generated candidates occupy the **same dynamic regime as an approved ADC** → physically realistic;
  (decisive) the metric **does not even separate an FDA linker from generated ones** → non-discriminative.
- `[FIG 6]` = `md/analysis/selfrmsd_3systems.png`. **This figure carries the thesis** (independent of the N=3 power issue).

### 4.7 Why the metric fails — by construction  `[WRITE]`
- Fixed harmonic bond, no bond-breaking, no constant-pH, no plasma thiols → the model omits exactly the
  chemistry (ring-opening, retro-Michael, local pKa) that sets real stability.
- Site analysis (§20.13): Cys214 proxy is acidic, no persistent proximal base → predicted weak
  self-stabilization, but **site is shared across candidates** → cannot rank linkers anyway.
- *"The quantity that would rank candidates is precisely the quantity the cheap model omits."*

## 5. Discussion  `[WRITE]`

- **The bottleneck moved from generation to evaluation.** Generation is solved-enough (4.1); ranking is open.
- **Surface-metric-masks-structural-error** as the cross-cutting lesson, with the full chain of instances:
  AMM checkpoint broken under `strict=False`; covalent topology distance-OK / valence-wrong; stereo
  silently lost; small-FT yield differences within trajectory variance. Adversarial validation (negative
  controls, valence assertions, multi-seed + Wilson CI) is what caught each.
- **Site vs linker ranking are different problems** with different data needs: site (needs site diversity;
  descriptors + structures suffice) vs linker-at-fixed-site (needs deconjugation chemistry or experimental labels).
- **What a valid evaluation would require:** experimental deconjugation/plasma-stability labels (currently
  ~5 fragmented, non-poolable literature sources — a hard data wall), or reactive/constant-pH MD. Frame
  constant-pH local pKa as the cheapest physics-only next lever; reactive MD as research-grade.

## 6. Limitations  `[WRITE]` (blunt — this is the honest paper)
- N=3, low power (df≈2.4): proves "no separation survives noise at this cost," not "molecules identical."
- Cys214 is a **proxy** site (1N8Z lacks the IgG hinge Cys) → characterizes the modeling site, not clinical Adcetris.
- Single arbitrary thiosuccinimide diastereomer; cand_1/4 basic-amine modeled neutral; 300 K (not 310 K); GAFF2/AM1-BCC payload.
- Fixed-bond covalent model with bond-only junction (no explicit angle/dihedral at sulfur).
- Configurational (not just thermodynamic) equilibration not reached at 50 ns for the slow payload mode.

## 7. Conclusion  `[WRITE]`
- DiffLinker generates plausible, novel ADC linkers; chemical validity is the practical gate; but neither
  AMM (label-saturated) nor fixed-bond MD payload RMSD (chemistry-omitting) can reliably rank them — shown
  against an FDA control. The next advance is an evaluation model grounded in conjugation-site chemistry
  and/or experimental deconjugation data, not more sampling of an incomplete physics.

---

## Boxed: methodological findings (from the campaign — candidate for a highlighted box / SI table)

| # | Finding | Section |
|---|---|---|
| 1 | Sanity controls (known-junk + known-active SMILES) are mandatory before trusting any ML filter | §19.21 |
| 2 | Silent `strict=False` checkpoint loading can mask a fully broken classifier | §19.22–23 |
| 3 | Checkpoints may ship vestigial untrained heads (check `num_batches_tracked` / BN stats) | §19.23 |
| 4 | Diffusion XYZ→SDF loses stereochemistry irrecoverably; canonical-stereo imposition mandatory | §19.24 |
| 5 | For covalent MD, validate **valence**, not just bond **length** (distance holds despite wrong chemistry) | §19.28, §20.10 |
| 6 | Multi-chain PBC: `-pbc whole`→`-pbc nojump`, not `-pbc mol -center` | §19.30 |
| 7 | Small-scale FT yield differences can be within GPU trajectory variance → multi-seed + Wilson CI required | §19.18 |
| 8 | A fail-closed 3D chemical-validity gate is the scalable screen MD cannot be | §20.11 |

---

## Figure / artifact plan (most already exist)

| Fig/Tab | Content | Source |
|---|---|---|
| Fig 1 | Pipeline schematic (generate → gate → covalent build → MD → evaluate) | new (draw) |
| Tab 1 | Motif emergence + Tanimoto, ft vs zs | master §12.4 / §19.12 |
| Fig 2 | Usable-3D yield funnel | `outputs/h4/eval/` (Wilson CI numbers §19.18) |
| Fig 3 | Per-domain Cα RMSD + DSSP (cand_4) | `md/analysis/cand_4/` (dom*_rmsd.xvg, dssp.dat, compare_clean.png) |
| Tab 2 | Batch gross-stability summary (C–SG, DSSP, per-domain) | master §20.1–§20.4 |
| Fig 4 | Buggy-vs-fixed payload metrics (the artifact) | `md/analysis/` (cand_4/cand_5 buggy vs fixed, §20.5–§20.6) |
| Fig 5 | N=3 self-RMSD, cand_4 vs cand_5 | `md/analysis/selfrmsd_n3_summary.png` |
| Fig 6 | 3-system self-RMSD incl. vedotin (**the closer**) | `md/analysis/selfrmsd_3systems.png` |
| Fig 7 (opt) | Cys214 site microenvironment (static pKa + dynamic basic contacts) | `md/analysis/` (§20.13) |

---

## Pre-submission checklist  `[TODO]`

- [ ] Confirm/recompute the §4.1 generation table at a single reported config (avoid mixing §12/§19 numbers).
- [ ] Re-baseline cand_1 / cand_8 self-RMSD with the valence fix? (master ran only cand_4/cand_5 fixed; decide if needed or scope to the 2-system diagnostic + vedotin.)
- [ ] Wilson CIs on every reported yield (§19.18 convention) — state n.
- [ ] Methods reproducibility box: mdp params, FF versions, gate code, conjugation/stereo scripts (paths in master §19.25–§20.0).
- [ ] Decide title (#1 vs #2) and venue; ChemRxiv preprint first.
- [ ] Author/affiliation; acknowledge Hazem Mslati (ADCpedia/AMM) appropriately.
- [ ] Data/code availability statement (which scripts + trajectories released).

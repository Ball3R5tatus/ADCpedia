# Validated AMBER covalent-conjugate build (cand_4) — what actually worked

AmberTools24 installed via micromamba (conda libmamba was broken: libarchive.so.19).
Build/setup tools (tleap/cpinutil.py/cphstats/sander/parmed) work CPU-side. The
covalent conjugate BUILDS, VALENCE-VALIDATES, solvates, neutralizes, and the cpin
generates — proving the §25.4 AMBER engine decision in practice (zero
reparametrization: the acpype GAFF2 ligand + a small junction frcmod suffice).

## Corrections vs the original scaffold (all needed):
1. `pdb4amber -i receptor.pdb -o rec_clean.pdb --nohyd` (strip H, standardize;
   tleap re-adds H per ff14SB — the PDBFixer H names broke typing otherwise).
2. Conjugation Cys → **CYX** (the bonded no-HG variant) — NOT `remove HG`
   (which left an incomplete CYS).
3. `source leaprc.constph` — required so AS4/GL4/HIP titratable residues have types.
4. Proximal carboxylates renamed to AS4/GL4. **CORRECTION (2026-06-25):** the
   built `complex_gb.prmtop` actually contains only **AS4:1 (ASP122) + GL4:1**
   (one GLU of 213/219) — 2 renamed, not the 3 the original note claimed. This is
   mostly moot because (see §cpin below) `cpinutil.py` with no `-resnums` titrates
   a BROAD default set regardless of pre-renaming.
5. Ligand: `remove LIG.1.H76` (the attach-C's 2nd H) + fold its +0.0907 charge into
   C48 (`set LIG.1.C48 charge -0.0722`) to keep integer charge.
6. `loadamberparams junction.frcmod` — the cross-FF S(amber)–c5(gaff) bond/angle/
   dihedral terms (none exist in ff14SB or GAFF). See junction.frcmod.
7. bond: `complex.214.SG complex.1042.C48` (Cys214 = residue 214; LIG = 1042 after
   combine, = total receptor residues 1041 + 1).

## Validated outputs (outputs/wave2/cphmd/cand_4/, gitignored):
- complex_dry.prmtop: **SG = 2 bonds (CB, C48); C48 = 4 bonds (SG,C47,C49,H75)** —
  no pentavalent C (§20.10), no hypervalent S (§19.28); C-SG = 1.81 Å; charge -10 (integer).
- complex_solv.prmtop: solvated TIP3P oct, 280,111 atoms, net charge 0.003 (neutral).
- cpin (§cpin): **CORRECTION (2026-06-25)** — `cpin_gb` titrates **TRESCNT=79
  residues** (verified: 79 `FIRST_ATOM` records), NOT "3 proximal carboxylates".
  `cpinutil.py -p complex_gb.prmtop -o cpin_gb -igb 2` with no `-resnums` applies
  its broad default (all eligible ASP/GLU). Consequence for the pilot: the proximal
  observable (ASP122 = resid 122 = AS4; the proximal GLU as GL4) is read out of the
  full 79-residue titration via `cphstats`. The comparison across candidates stays
  controlled because all 3 conjugates use the identical recipe → identical 79-set.
  **If targeted (proximal-only, fixed background, cleaner convergence) is wanted,
  regenerate at launch with `cpinutil.py ... -resnums <122> <GL4-resnum>`** (a
  5-min step once the system is in front of you).
- minimization runs (system is simulatable).

## GPU production — RESOLVED (2026-06-25, §25.9):
`pmemd.cuda` is **BUILT + TESTED** on the RTX 4080 (Amber26 source: pmemd26 +
ambertools26 from `/home/galeito/Amber26/`, CUDA 12.8, gcc13, sm_89; 53 ns/day on
the 15.8k GB system). The old Amber22 path was the WRONG version (doesn't pair with
AmberTools26). GB CpHMD GPU run needs `cut=9999` (infinite cutoff, matches cpin
reference energies). **Caveat (WSL): pmemd.cuda HANGS when a 2nd CUDA context shares
the GPU** (the user's `auto_trading` python). auto_trading is never to be touched →
GPU-resident CpHMD is effectively blocked while it runs; CPU path = the truncated
active-site GB model (see DIFFUSION_PIPELINE_MASTER §25.9 / titration notes).

## Pilot set complete (2026-06-25): cand_5 + cand_vedotin built
`build_pilot_conjugates.sh` replicates this recipe for the other two §20.12 pilot
candidates (acpype params already existed → no antechamber). Both attach carbons
are GAFF **c5** (same junction.frcmod applies verbatim). Valence-validated:
- **cand_5**: attach **C55**, remove H81 → C55 charge −0.0727; SG=2 bonds (CB,C55),
  C55=4 bonds (SG,C56,C54,H80); total charge −10 (integer), LIG 0.001 (neutral).
- **cand_vedotin**: attach **C67**, remove H107 → C67 charge −0.07395; SG=2 bonds
  (CB,C67), C67=4 bonds (SG,C68,C66,H106); total −10 (integer), LIG 0.000 (neutral).
All three (cand_4/cand_5/cand_vedotin) now have complex_gb.prmtop + cpin_gb (79) +
gb_min.in + gb_cph[_gpu].tmpl staged → per-linker pKa titration is turnkey the
moment a clean GPU window or the CPU truncated-model path is chosen.

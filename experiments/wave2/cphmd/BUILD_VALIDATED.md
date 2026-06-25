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
4. Rename ONLY the proximal titratable carboxylates to AS4/GL4 (targeted, cheap
   titration): GLU213/A→GL4, ASP122/A→AS4, GLU219/B→GL4.
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
- cpin: 3 proximal carboxylates titratable.
- minimization runs (system is simulatable).

## REMAINING for GPU production (the blocker):
`pmemd.cuda` needs the **AmberTools source tree** layered with the licensed Amber22
pmemd into one `amber22_src/`, then `cmake -DCUDA=TRUE`. The user has only the
Amber22 pmemd source (Amber22.tar.bz2, md5 593ebf62…, genuine) — NOT the AmberTools
source. Get AmberTools22/23 source from ambermd.org/GetAmber.php (free, with the
license) → extract into the same amber22_src/ → build. CPU sander CpHMD is
infeasible at 280k atoms (months); GPU makes the pH-ladder titration ~<1 day.

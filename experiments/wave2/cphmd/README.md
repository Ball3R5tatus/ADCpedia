# AMBER constant-pH MD — scaffold (the rigorous per-linker physics lever)

Dynamic, per-linker version of the §25.6 static-PROPKA pilot: a pH ladder of
explicit-solvent CpHMD yields each candidate's local pKa for the titratable
residues around Cys214 (GLU213, ASP122, GLU219, LYS136, TYR186 — the
§20.13/§25.6 observable). Comparing those pKa **across candidates** is the
per-linker signal that a static structure could not provide (§25.6 returned a
NULL; dynamics may widen it — or confirm the null).

## Why AMBER (Master §25.4)
The system is **AMBER-native** (protein amber99sb-ildn≈ff14SB; ligand
GAFF2/AM1-BCC; `acpype` already wrote `md/lig_params/<cand>/LIG.acpype/LIG_AC.frcmod`
+ `LIG_bcc_gaff2.mol2`). AMBER constant-pH needs **zero reparametrization** and is
the reference implementation for residue pKa. The GROMACS-constantph fork is
**CHARMM36m-only**, and CHARMM CpHMD needs CGenFF — both force a full
protein+ligand reparametrization. So AMBER wins decisively.

## Status: SCAFFOLD — needs AmberTools (not installed here = the blocker)
Nothing is execution-validated. Install, resolve 2 per-candidate values, run:
```bash
conda create -n ambertools -c conda-forge ambertools=24 && conda activate ambertools
# (free CPU `sander` does constant-pH; GPU `pmemd.cuda` needs an Amber license)
```

## Workflow
1. **Which residues to read** (runnable now, no AmberTools):
   ```bash
   python experiments/wave2/cphmd/select_titratable.py \
       --pdb structures/1N8Z_clean.pdb --site-resid 214 --site-chain A --cutoff 12
   ```
2. **Resolve the 2 per-candidate covalent values** (the only manual step):
   - `CYS_RESNUM` — the conjugation Cys's residue number **as tleap renumbers it**
     (sequential over the combined system, not necessarily PDB 214). Find it after
     a first `loadpdb` via `parmed -p complex.prmtop` → `printResidues :CYS`.
   - `LIG_ATTACH` — the ligand attach-carbon **atom name** in `LIG_bcc_gaff2.mol2`
     (the C bonded to SG; its conjugation H was already removed when the
     maleimide-Cys adduct was built — see `md/runs/<cand>/restraint.json`
     `C3_H_removed_lig`). `grep` the mol2 atom block for the C3 analogue.
3. **Build + titrate + simulate** (needs AmberTools):
   ```bash
   CAND=cand_4 CYS_RESNUM=<...> LIG_ATTACH=<...> PH_LADDER="4 5 6 7 8" \
     bash experiments/wave2/cphmd/run_cphmd.sh
   ```
   It: renames ASP→AS4/GLU→GL4/HIS→HIP (constant-pH residues) → `tleap` covalent
   build (`bond Cys.SG – LIG.attachC`) → `cpinutil.py` (cpin) → min/heat/equil →
   `prod_cph` at each pH (`icnstph=2, ntcnstph=100, solvph=<pH>`) → `cphstats`.
4. **Read out:** fit fraction-protonated vs pH (Henderson–Hasselbalch) per residue
   → pKa; compare the proximal-residue pKa across cand_4 / cand_5 / vedotin.

## Pilot scope & caveats
- Pilot = the §20.12 set (cand_4, cand_5, vedotin) so it ties to the MD paper.
- pH ladder {4,5,6,7,8} per candidate gives a titration; ≥5–10 ns/pH for a usable
  curve (the Sebastián-Pérez protocol is 500 ns + CpHMD — budget accordingly).
- **Still a proxy:** CpHMD titrates standard residues (protonation), NOT the
  succinimide ring-opening / retro-Michael bond chemistry (that is QM/MM). It
  upgrades §25.6 from static→dynamic per-linker pKa, not to deconjugation kinetics.
- The covalent build is the fragile step: verify integer total charge and that
  the attach carbon stays tetravalent (reuse the §20.11 valence gate idea on the
  built `complex_built.pdb`).

## Files
| file | role | runnable now? |
|---|---|---|
| `select_titratable.py` | proximal titratable residues near Cys214 | **yes** |
| `run_cphmd.sh` | tleap covalent build + cpin + min/heat/equil + pH-ladder CpHMD + cphstats | needs AmberTools |

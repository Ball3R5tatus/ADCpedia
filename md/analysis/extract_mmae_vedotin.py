#!/usr/bin/env python3
"""Build the [ MMAE ] index (51 heavy atoms) for the vedotin system, in the
Complex-subset numbering (matches complex.tpr + the compressed prod*.xtc), using
the SAME canonical MMAE skeleton as cand_4/cand_5 PLUS the carbamate-exclusion
disambiguation (VcMMAE's Val-Cit linker aliases the MMAE peptide terminus; the
correct match keeps the MMAE N-methyl C and excludes the PABC carbamate C).

Mapping: merged-ligand order == posed-SDF order for indices 0..199 (the one removed
atom is the C3 hydrogen = last SDF atom 201, a hydrogen NOT in the MMAE core), so
SDF heavy-atom index i -> lig_abs[i] where lig_abs = resname MOL in complex.tpr.
"""
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda

RUN='/home/galeito/ADCpedia/md/runs/cand_vedotin'
SDF='/home/galeito/ADCpedia/md/lig_params/cand_vedotin_posed.sdf'
CANON=('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)'
       '[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C')

mol=Chem.MolFromMolFile(SDF, removeHs=False)
flat=Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(CANON), isomericSmiles=False))
matches=[list(x) for x in mol.GetSubstructMatches(flat, uniquify=True)]
carb={a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol()=='C'
      and sorted(n.GetSymbol() for n in a.GetNeighbors())==['N','O','O']}   # PABC carbamate carbon
good=[m for m in matches if not (set(m) & carb)]
assert len(good)==1 and len(good[0])==51, f'MMAE disambiguation failed: {len(matches)} matches, {len(good)} carbamate-free'
mmae_sdf=good[0]
assert max(mmae_sdf) <= 199, f'MMAE atom idx {max(mmae_sdf)} > 199 (would be affected by removed C3-H)'

# LIG group (resname MOL) in Complex-subset numbering from complex.tpr
u=mda.Universe(f'{RUN}/complex.tpr')
mol_atoms=u.select_atoms('resname MOL')
lig_abs=[int(a.index)+1 for a in mol_atoms]            # 1-based, Complex-subset
assert len(lig_abs)==200, f'MOL group has {len(lig_abs)} atoms, expected 200'

mmae_abs=sorted(lig_abs[i] for i in mmae_sdf)
# sanity: the 51 selected atoms should all be carbon/nitrogen/oxygen (MMAE has no S/P/H here)
els=[u.atoms[a-1].element if hasattr(u.atoms[a-1],'element') else '' for a in mmae_abs]
out=f'{RUN}/ana_mmae.ndx'
# write ana_mmae.ndx (just the MMAE group; selfrmsd_reps.sh only needs "MMAE")
with open(out,'w') as fh:
    fh.write('[ MMAE ]\n')
    for k in range(0, len(mmae_abs), 15):
        fh.write(' '.join(f'{x:6d}' for x in mmae_abs[k:k+15])+'\n')
print(f'flat MMAE matches={len(matches)}  carbamate-free={len(good)}  -> 51-atom MMAE unique')
print(f'MOL ligand atoms in complex.tpr: {len(lig_abs)} (range {min(lig_abs)}-{max(lig_abs)})')
print(f'MMAE abs atom range: {min(mmae_abs)}-{max(mmae_abs)}  (n={len(mmae_abs)})')
print(f'wrote {out}')

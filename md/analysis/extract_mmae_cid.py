#!/usr/bin/env python3
"""Generic [ MMAE ] index (51 heavy atoms) for any conjugated system, in Complex-subset
numbering. Matches the canonical MMAE skeleton on the posed SDF; if >1 match, disambiguate
by excluding any match that hits a PABC carbamate carbon (C bonded to N,O,O) -- vedotin case.
Maps SDF heavy-atom index i -> the i-th atom of resname MOL in complex.tpr (merged-ligand
order == SDF order for indices below the single removed C3-H).
Usage: extract_mmae_cid.py <cid> [posed_sdf_basename]"""
import sys, warnings; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda

cid=sys.argv[1]
sdfbase=sys.argv[2] if len(sys.argv)>2 else f'cand_{cid}'
RUN=f'/home/galeito/ADCpedia/md/runs/cand_{cid}'
SDF=f'/home/galeito/ADCpedia/md/lig_params/{sdfbase}_posed.sdf'
CANON=('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)'
       '[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C')

mol=Chem.MolFromMolFile(SDF, removeHs=False)
flat=Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(CANON), isomericSmiles=False))
matches=[list(x) for x in mol.GetSubstructMatches(flat, uniquify=True)]
carb={a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol()=='C'
      and sorted(n.GetSymbol() for n in a.GetNeighbors())==['N','O','O']}
good=[m for m in matches if not (set(m) & carb)] if len(matches)>1 else matches
assert good and all(len(g)==51 for g in good), f'{cid}: MMAE skeleton not 51 atoms ({len(matches)} matches)'
if len(good)>1:
    # >1 equivalent matches (e.g. N-CH3 vs valine-Calpha both bonded to the basic amine N):
    # they differ by a single terminal C -> negligible RMSD effect. Deterministic min-sorted
    # pick so prot & neut variants of the same candidate select the IDENTICAL 51 atoms.
    print(f'WARNING {cid}: {len(good)} equivalent MMAE matches '
          f'(differ by {sorted(set(good[0])^set(good[1]))}); min-sorted deterministic pick', file=sys.stderr)
    good=[min(good, key=lambda m: sorted(m))]
mmae_sdf=good[0]

u=mda.Universe(f'{RUN}/complex.tpr')
lig_abs=[int(a.index)+1 for a in u.select_atoms('resname MOL')]
assert max(mmae_sdf) < len(lig_abs), f'{cid}: MMAE idx {max(mmae_sdf)} >= ligand size {len(lig_abs)}'
mmae_abs=sorted(lig_abs[i] for i in mmae_sdf)
out=f'{RUN}/ana_mmae.ndx'
with open(out,'w') as fh:
    fh.write('[ MMAE ]\n')
    for k in range(0,len(mmae_abs),15): fh.write(' '.join(f'{x:6d}' for x in mmae_abs[k:k+15])+'\n')
print(f'{cid}: flat matches={len(matches)} -> unique 51; MOL={len(lig_abs)} atoms; MMAE abs {min(mmae_abs)}-{max(mmae_abs)} -> {out}')

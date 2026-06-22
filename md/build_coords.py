#!/usr/bin/env python
"""Build the coordinate file that matches the HG-removed, ligand-merged chain-A topology.

gmx_assemble.py writes complex.gro in order [chainA_prot, chainB, chainC, ligand] with the
conjugated-Cys HG still present. merge_chainA_ligand.py rewrites the topology to
[chainA_prot(HG deleted), ligand, chainB, chainC]. This script reorders the coordinates to
match AND removes the same HG, so grompp sees a consistent system. Deterministic & verifiable
(no ParmEd index-by-residue guessing).

Usage: python build_coords.py <cid>   ->  writes <run>/merged_noHG.gro
"""
import sys, json, re
cid = sys.argv[1]; run = f'/home/galeito/ADCpedia/md/runs/cand_{cid}'
r = json.load(open(f'{run}/restraint.json'))
npro = r['npro']; nlig = r['nlig']; hg = r['HG_global_removed']   # 1-based global, in chainA-protein block
ligH = r.get('C3_H_removed_lig')                                  # 1-based index WITHIN the ligand block (C3 hydrogen)

def itp_natoms(f):
    itp = open(f).read(); m = re.search(r'\[ *atoms *\]\s*\n(.*?)(?=^\[ )', itp, re.M | re.S)
    return sum(1 for l in m.group(1).splitlines() if l.split() and l.split()[0].isdigit())

nB = itp_natoms(f'{run}/topol_Protein_chain_B.itp')
nC = itp_natoms(f'{run}/topol_Protein_chain_C.itp')
nAfull = npro - nB - nC                      # chain A protein, HG still present
assert 1 <= hg <= nAfull, f'HG index {hg} not within chain A protein [1,{nAfull}]'

L = open(f'{run}/complex.gro').read().splitlines()
title, n = L[0], int(L[1]); atoms = L[2:2 + n]; box = L[2 + n]
assert n == npro + nlig, f'complex.gro has {n} atoms, expected {npro + nlig}'

A   = atoms[0:nAfull]
B   = atoms[nAfull:nAfull + nB]
C   = atoms[nAfull + nB:npro]
Lig = atoms[npro:npro + nlig]

hgline = A[hg - 1]                            # safety: must be the Cys thiol proton
assert hgline[10:15].strip() == 'HG' and hgline[5:10].strip() in ('CYS', 'CYX', 'CYS2'), \
    f'atom {hg} is {hgline[5:15].strip()!r}, not a Cys HG — refusing'
A = A[:hg - 1] + A[hg:]                       # delete HG (protein side)

if ligH:                                      # delete the C3 hydrogen (ligand side, valence fix)
    lhline = Lig[ligH - 1]
    assert lhline[10:15].strip().startswith('H'), \
        f'ligand atom {ligH} is {lhline[10:15].strip()!r}, not a hydrogen — refusing'
    Lig = Lig[:ligH - 1] + Lig[ligH:]

new = A + Lig + B + C                         # reorder to match merged topology
out = [a[:15] + f'{(i + 1) % 100000:>5d}' + a[20:] for i, a in enumerate(new)]   # renumber serial col
open(f'{run}/merged_noHG.gro', 'w').write(
    title + '\n' + f'{len(new)}\n' + '\n'.join(out) + '\n' + box + '\n')
print(f'merged_noHG.gro: {len(new)} atoms = chainA_prot {nAfull-1} (HG removed) '
      f'+ ligand {len(Lig)} (C3-H removed) + chainB {nB} + chainC {nC}')

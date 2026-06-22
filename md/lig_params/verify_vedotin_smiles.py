#!/usr/bin/env python3
"""Verify the authentic VcMMAE (mc-vc-PAB-MMAE) SMILES before building anything.
Checks: (1) formula/MW match authoritative C68H105N11O15 / 1316.63;
(2) maleimide present; (3) the project's canonical MMAE core (51 heavy atoms,
the SAME atoms used for cand_4/cand_5 self-RMSD) is a substructure -- flat AND
stereo-aware -- so the payload is provably identical to the candidates."""
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

# Authoritative VcMMAE, maleimide (reactive) form -- Selleck + chemicalbook/biosynth concur
AUTH = ('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@H](C)'
        '[C@H](C2=CC=CC=C2)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)N(C)C(=O)'
        'OCC3=CC=C(C=C3)NC(=O)[C@H](CCCNC(=O)N)NC(=O)[C@H](C(C)C)NC(=O)CCCCCN4C(=O)C=CC4=O')

# Project canonical MMAE (used by build_complexes_stereo / extract_mmae for the 51-atom core)
CANON_MMAE = ('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)'
              '[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C')

m = Chem.MolFromSmiles(AUTH)
assert m is not None, 'AUTH SMILES failed to parse'
formula = rdMolDescriptors.CalcMolFormula(m)
mw = Descriptors.MolWt(m)
print(f'[1] AUTH VcMMAE: formula={formula}  MW={mw:.2f}  (expect C68H105N11O15 / 1316.63)')
print(f'    formula OK: {formula == "C68H105N11O15"}    MW OK: {abs(mw-1316.63) < 0.5}')

male = Chem.MolFromSmarts('O=C1C=CC(=O)N1')
print(f'[2] maleimide present: {m.HasSubstructMatch(male)}')

mmae = Chem.MolFromSmiles(CANON_MMAE)
flat = Chem.MolFromSmiles(Chem.MolToSmiles(mmae, isomericSmiles=False))
print(f'[3] canonical MMAE: {mmae.GetNumAtoms()} heavy atoms; flat-skeleton query {flat.GetNumAtoms()} atoms')

flat_matches = m.GetSubstructMatches(flat, uniquify=True)
print(f'    flat MMAE substructure in VcMMAE: {len(flat_matches)} match(es), '
      f'size {len(flat_matches[0]) if flat_matches else 0}  (expect 1 x 51)')

stereo_matches = m.GetSubstructMatches(mmae, uniquify=True, useChirality=True)
print(f'    STEREO-aware MMAE match in VcMMAE: {len(stereo_matches)} match(es)  '
      f'-> {"SAME stereo as candidates" if stereo_matches else "STEREO MISMATCH (will graft canonical MMAE)"}')

# Also: does AUTH MMAE core have any undefined stereocenters?
und = Chem.FindMolChiralCenters(m, useLegacyImplementation=False, includeUnassigned=True, force=True)
n_unassigned = sum(1 for _, c in und if c == '?')
print(f'[4] stereocenters: {len(und)} total, {n_unassigned} UNDEFINED  (gate needs 0 undefined)')

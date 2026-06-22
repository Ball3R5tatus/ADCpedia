"""Build an [ MMAE ] index group (51 heavy atoms of the canonical payload core)
for each candidate, by substructure-matching the flat MMAE skeleton onto the
posed conjugate SDF, then mapping to absolute system atom numbers via [LIG].
Verifies SDF atom order == system ligand order before trusting the mapping."""
import warnings, re, sys; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda

RUNS = '/home/galeito/ADCpedia/md/runs'
LIGP = '/home/galeito/ADCpedia/md/lig_params'
MMAE_SMILES = ('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)'
               'C(=O)N[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H]'
               '(NC(=O)[C@@H](NC)C(C)C)C(C)C')
mmae = Chem.MolFromSmiles(MMAE_SMILES)
QRY = Chem.MolFromSmiles(Chem.MolToSmiles(mmae, isomericSmiles=False))  # flat skeleton
print(f'MMAE query heavy atoms: {QRY.GetNumAtoms()}')

MASS2EL = {12: 'C', 1: 'H', 14: 'N', 16: 'O', 32: 'S', 31: 'P', 19: 'F', 35: 'Cl'}

def read_ndx_group(path, name):
    txt = open(path).read()
    blocks = re.split(r'\[\s*(.*?)\s*\]', txt)[1:]
    groups = {blocks[i].strip(): blocks[i+1].split() for i in range(0, len(blocks), 2)}
    return [int(x) for x in groups[name]]            # 1-based absolute indices

def run(cand):
    sdf = f'{LIGP}/{cand}_posed.sdf'
    ndx = f'{RUNS}/{cand}/ana.ndx'
    tpr = f'{RUNS}/{cand}/prod.tpr'
    mol = Chem.MolFromMolFile(sdf, removeHs=False)
    lig_abs = read_ndx_group(ndx, 'LIG')             # absolute atom numbers (1-based), ligand order
    # --- verify SDF order == system ligand order (element-by-element) ---
    u = mda.Universe(tpr)
    sys_el = [MASS2EL.get(round(float(m)), '?') for m in u.atoms.masses]
    sdf_el = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    assert mol.GetNumAtoms() == len(lig_abs), \
        f'{cand}: SDF {mol.GetNumAtoms()} atoms != LIG group {len(lig_abs)}'
    mism = [(i, sdf_el[i], sys_el[a-1]) for i, a in enumerate(lig_abs) if sdf_el[i] != sys_el[a-1]]
    order_ok = (len(mism) == 0)
    # --- substructure match: MMAE skeleton -> SDF atom indices (== ligand positions) ---
    match = mol.GetSubstructMatch(QRY)
    assert len(match) == QRY.GetNumAtoms(), f'{cand}: MMAE match {len(match)}/{QRY.GetNumAtoms()}'
    mmae_abs = sorted(lig_abs[i] for i in match)     # absolute system atom numbers
    # --- write ana_mmae.ndx = ana.ndx + [ MMAE ] ---
    out = f'{RUNS}/{cand}/ana_mmae.ndx'
    with open(out, 'w') as fh:
        fh.write(open(ndx).read().rstrip() + '\n')
        fh.write('[ MMAE ]\n')
        for k in range(0, len(mmae_abs), 15):
            fh.write(' '.join(f'{x:6d}' for x in mmae_abs[k:k+15]) + '\n')
    print(f'\n[{cand}]  LIG={len(lig_abs)}  order_preserved={order_ok} (mismatches={len(mism)})'
          f'  MMAE_matched={len(match)}  -> {out}')
    if mism[:3]:
        print(f'   first mismatches (pos, sdf_el, sys_el): {mism[:3]}')
    return order_ok and len(match) == 51

ok = all(run(c) for c in ['cand_1', 'cand_4'])
print('\nALL OK' if ok else '\nPROBLEM - check output above')
sys.exit(0 if ok else 1)

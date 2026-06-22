"""Build an [ MMAE ] index group (51 heavy atoms of the payload core) for a candidate,
by substructure-matching the flat MMAE skeleton onto the posed conjugate SDF and mapping
to absolute atom numbers via the [LIG] group. Verifies SDF order == system ligand order.

Usage: python extract_mmae.py <cid>            (e.g. 5  -> cand_5)
Topology: prefers runs/cand_<cid>/complex.tpr (Complex-compressed runs); falls back to prod.tpr.
Reads the [LIG] group from runs/cand_<cid>/ana.ndx (built by analyze_traj.sh) and writes
ana_mmae.ndx alongside it.
"""
import warnings, re, sys, os; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda

cid = sys.argv[1]
cand = f'cand_{cid}'
RUNS = '/home/galeito/ADCpedia/md/runs'
LIGP = '/home/galeito/ADCpedia/md/lig_params'
MMAE_SMILES = ('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)'
               'C(=O)N[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H]'
               '(NC(=O)[C@@H](NC)C(C)C)C(C)C')
QRY = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(MMAE_SMILES), isomericSmiles=False))
MASS2EL = {12: 'C', 1: 'H', 14: 'N', 16: 'O', 32: 'S', 31: 'P', 19: 'F', 35: 'Cl'}

def read_ndx_group(path, name):
    txt = open(path).read()
    blocks = re.split(r'\[\s*(.*?)\s*\]', txt)[1:]
    groups = {blocks[i].strip(): blocks[i+1].split() for i in range(0, len(blocks), 2)}
    return [int(x) for x in groups[name]]

rundir = f'{RUNS}/{cand}'
tpr = f'{rundir}/complex.tpr' if os.path.exists(f'{rundir}/complex.tpr') else f'{rundir}/prod.tpr'
ndx = f'{rundir}/ana.ndx'
sdf = f'{LIGP}/{cand}_posed.sdf'
for p in (tpr, ndx, sdf):
    assert os.path.exists(p), f'missing prerequisite: {p}'

mol = Chem.MolFromMolFile(sdf, removeHs=False)
lig_abs = read_ndx_group(ndx, 'LIG')
u = mda.Universe(tpr)
sys_el = [MASS2EL.get(round(float(m)), '?') for m in u.atoms.masses]
sdf_el = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
assert mol.GetNumAtoms() == len(lig_abs), f'{cand}: SDF {mol.GetNumAtoms()} != LIG {len(lig_abs)}'
mism = [(i, sdf_el[i], sys_el[a-1]) for i, a in enumerate(lig_abs) if sdf_el[i] != sys_el[a-1]]
match = mol.GetSubstructMatch(QRY)
assert len(match) == QRY.GetNumAtoms(), f'{cand}: MMAE match {len(match)}/{QRY.GetNumAtoms()}'
mmae_abs = sorted(lig_abs[i] for i in match)

out = f'{rundir}/ana_mmae.ndx'
with open(out, 'w') as fh:
    fh.write(open(ndx).read().rstrip() + '\n[ MMAE ]\n')
    for k in range(0, len(mmae_abs), 15):
        fh.write(' '.join(f'{x:6d}' for x in mmae_abs[k:k+15]) + '\n')
ok = (len(mism) == 0 and len(match) == 51)
print(f'[{cand}] topo={os.path.basename(tpr)} LIG={len(lig_abs)} order_ok={len(mism)==0} '
      f'MMAE={len(match)}/51 -> {out}')
if mism[:3]:
    print(f'   mismatches: {mism[:3]}')
sys.exit(0 if ok else 1)

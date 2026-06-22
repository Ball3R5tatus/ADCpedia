#!/usr/bin/env python
"""3D chemical-validity gate for ADC linker-payloads — the pre-screen that should
sit at the DiffLinker 3D output, BEFORE any MD or synthesis.

Two levels:
  A. LIGAND level (RDKit on SMILES/SDF) — the broad screen:
       - sanitizable (valence sane)?
       - single connected component? (DiffLinker's known defect: linker terminus
         placed ~2 bond-lengths short -> disconnected fragments, Master §19.16)
       - undefined stereocenters? (linker AND payload must be fully specified)
       - conjugation state: open maleimide (unreacted) vs thiosuccinimide
       - drug-likeness sanity (MW / logP / rot-bonds)
  B. CONJUGATE/TOPOLOGY level (GROMACS merged chain-A itp) — the build-time guard:
       - per-atom valence: no carbon with >4 bonds, etc.
       This is the check that would have auto-caught the pentavalent C3 (§20.10)
       and the over-coordinated SG (HG bug, §20.0) on ALL candidates at build time.

Usage:
  python chem_validity_gate.py ligand <sdf_or_smiles> [name]
  python chem_validity_gate.py topology <merged_chainA_itp> [name]
  python chem_validity_gate.py batch        # runs both levels on cand_1/4/5/8
"""
import sys, re

MAX_VAL = {'C': 4, 'N': 4, 'O': 2, 'H': 1, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1}
# NB: rdkit is imported lazily inside check_ligand() so that check_topology_valence()
# (pure-text, rdkit-free) can be imported by the merge pipeline's system python.

def check_ligand(src, name=''):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, FindMolChiralCenters
    from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    OPEN_MAL = Chem.MolFromSmarts('O=C1C=CC(=O)N1')   # unreacted maleimide (C=C intact)
    m = Chem.MolFromSmiles(src) if not src.endswith(('.sdf', '.mol')) else Chem.MolFromMolFile(src, sanitize=False)
    flags = []
    if m is None:
        return name, ['UNPARSEABLE'], {}
    try:
        Chem.SanitizeMol(m); sane = True
    except Exception as e:
        sane = False; flags.append(f'NOT-SANITIZABLE ({str(e)[:40]})')
    frags = Chem.GetMolFrags(m)
    if len(frags) > 1:
        flags.append(f'DISCONNECTED ({len(frags)} fragments)')   # DiffLinker short-bond defect
    try:
        undef = FindMolChiralCenters(m, useLegacyImplementation=False, includeUnassigned=True)
        n_undef = sum(1 for _, c in undef if c == '?')
        if n_undef: flags.append(f'{n_undef} undefined stereocenter(s)')
    except Exception:
        pass
    if m.HasSubstructMatch(OPEN_MAL):
        flags.append('OPEN-MALEIMIDE (unreacted C=C — not a conjugated thiosuccinimide)')
    props = {'MW': round(Descriptors.MolWt(m), 0), 'logP': round(Descriptors.MolLogP(m), 1),
             'rotB': rdMolDescriptors.CalcNumRotatableBonds(m), 'TPSA': round(Descriptors.TPSA(m), 0)}
    if props['MW'] > 1600: flags.append(f"MW {props['MW']} > 1600 (very large)")
    return name, flags or ['OK'], props

def check_topology_valence(itp, name=''):
    txt = open(itp).read()
    atoms = {int(c[0]): c[1] for c in (l.split() for l in
             re.search(r'\[ *atoms *\]\s*\n(.*?)(?=^\[ )', txt, re.M | re.S).group(1).splitlines())
             if c and c[0].isdigit()}
    deg = {}
    for blk in re.findall(r'^\[ *bonds *\]\s*\n(.*?)(?=^\[ |\Z)', txt, re.M | re.S):
        for l in blk.splitlines():
            c = l.split()
            if len(c) >= 2 and c[0].isdigit() and c[1].isdigit():
                deg[int(c[0])] = deg.get(int(c[0]), 0) + 1
                deg[int(c[1])] = deg.get(int(c[1]), 0) + 1
    bad = []
    for idx, gafftype in atoms.items():
        el = {'c': 'C', 'n': 'N', 'o': 'O', 'h': 'H', 's': 'S', 'p': 'P', 'f': 'F'}.get(gafftype[0].lower())
        if el and idx in deg and deg[idx] > MAX_VAL.get(el, 4):
            bad.append((idx, gafftype, deg[idx], el))
    return name, bad

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'batch'
    if mode == 'batch':
        import json
        OURS = {'1': 'cand_1', '4': 'cand_4', '5': 'cand_5(FIXED)', '8': 'cand_8'}
        print('=== LEVEL A — ligand validity (posed SDF / DiffLinker 3D output) ===')
        for cid, nm in OURS.items():
            sdf = f'/home/galeito/ADCpedia/md/lig_params/cand_{cid}_posed.sdf'
            nm2, fl, pr = check_ligand(sdf, nm)
            print(f'  {nm2:16s}: {", ".join(fl)}   {pr}')
        print('\n=== LEVEL B — conjugate topology valence (merged chain-A itp) ===')
        for cid, nm in OURS.items():
            itp = f'/home/galeito/ADCpedia/md/runs/cand_{cid}/topol_Protein_chain_A.itp'
            try:
                nm2, bad = check_topology_valence(itp, nm)
                if bad:
                    print(f'  {nm2:16s}: ❌ {len(bad)} OVER-VALENT atom(s): ' +
                          '; '.join(f'atom {i}({t}) has {d} bonds (max {MAX_VAL[e]})' for i, t, d, e in bad))
                else:
                    print(f'  {nm2:16s}: ✅ all valences OK')
            except FileNotFoundError:
                print(f'  {nm:16s}: (no topology)')
    elif mode == 'ligand':
        print(check_ligand(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ''))
    elif mode == 'topology':
        nm, bad = check_topology_valence(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
        print(nm, '❌', bad if bad else '✅ OK')

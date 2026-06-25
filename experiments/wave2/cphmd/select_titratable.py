#!/usr/bin/env python
"""List the titratable residues near the Cys214 conjugation site — the residues
whose per-linker local pKa is the constant-pH MD observable (the dynamic version
of the §25.6 static-PROPKA NULL).

AMBER constant-pH titrates ASP(AS4)/GLU(GL4)/HIS(HIP) and (newer) LYS/TYR/CYS.
The conjugation Cys214 itself is NOT titratable (it is covalently bonded to the
ligand). This script reads the receptor PDB and reports the proximal titratable
residues + their distance to the Cys214 SG, so you know which residues to read
out of `cphstats` (and, optionally, which to restrict the cpin to).

Pure-Python PDB parse (no MDAnalysis/AmberTools needed). Runnable now.

  python experiments/wave2/cphmd/select_titratable.py \
      --pdb structures/1N8Z_clean.pdb --site-resid 214 --site-chain A --cutoff 12
"""
from __future__ import annotations

import argparse

# residues AMBER constant-pH can titrate (CYS214 excluded: it's the bonded site)
TITRATABLE = {'ASP', 'GLU', 'HIS', 'HIP', 'HID', 'HIE', 'LYS', 'TYR', 'CYS'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', default='structures/1N8Z_clean.pdb')
    ap.add_argument('--site-resid', type=int, default=214)
    ap.add_argument('--site-chain', default='A')
    ap.add_argument('--cutoff', type=float, default=12.0)
    args = ap.parse_args()

    atoms, sg = [], None
    with open(args.pdb) as fh:
        for ln in fh:
            if ln[:6] not in ('ATOM  ', 'HETATM'):
                continue
            try:
                name, rn, ch = ln[12:16].strip(), ln[17:20].strip(), ln[21].strip() or '?'
                ri = int(ln[22:26]); x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            atoms.append((rn, ri, ch, x, y, z))
            if ri == args.site_resid and name == 'SG' and (sg is None or ch == args.site_chain):
                sg = (x, y, z)
    if sg is None:
        raise SystemExit(f'no SG on resid {args.site_resid} (chain {args.site_chain}) in {args.pdb}')

    c2 = args.cutoff ** 2
    near = {}
    for rn, ri, ch, x, y, z in atoms:
        d2 = (x - sg[0]) ** 2 + (y - sg[1]) ** 2 + (z - sg[2]) ** 2
        if d2 <= c2:
            k = (rn, ri, ch)
            near[k] = min(near.get(k, 1e9), d2 ** 0.5)

    titr = sorted(((d, rn, ri, ch) for (rn, ri, ch), d in near.items()
                   if rn in TITRATABLE and not (rn == 'CYS' and ri == args.site_resid)),
                  key=lambda t: t[0])
    print(f'# titratable residues within {args.cutoff} A of {args.site_chain}:CYS{args.site_resid}:SG')
    print(f'#   (constant-pH observables; Cys{args.site_resid} excluded = bonded)')
    print(f'{"resname":8s}{"resid":>6}{"chain":>6}{"dist_SG_A":>11}')
    pdb_resids = []
    for d, rn, ri, ch in titr:
        print(f'{rn:8s}{ri:>6}{ch:>6}{d:>11.2f}')
        pdb_resids.append(ri)
    print(f'\n# PDB resids (map to prmtop numbering after tleap for cpinutil -resnum):')
    print('#   ' + ' '.join(str(r) for r in pdb_resids))


if __name__ == '__main__':
    main()

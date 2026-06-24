#!/usr/bin/env python
"""Wave-2 physics-lever PILOT: a PER-LINKER local-pKa descriptor.

Why: §20.13 showed the Cys214 site microenvironment is the SAME for every
candidate (they all sit on Cys214) — a site descriptor cannot rank linkers.
The per-linker signal can only come from how EACH linker perturbs the titratable
residues around the conjugation site. This pilot runs PROPKA on each *conjugated
complex* (protein + that linker + payload) and reports the pKa of residues near
the conjugation sulfur, so the shift across linkers becomes the descriptor.

This is the cheap, static surrogate. The rigorous version is constant-pH MD
(λ-dynamics) — see experiments/wave2/README.md; it needs a CpHMD-capable GROMACS
build, which the project's gmx_mpi does not currently have.

Dependencies (honest): PROPKA3 binary (the env that produced /tmp/rec.pka in
§20.13) + MDAnalysis + per-chain conjugated-complex PDBs. The script auto-detects
PROPKA and degrades with a clear message if absent — it does NOT fake output.

Usage:
  python experiments/wave2/per_linker_pka.py \
      --pdbs md/runs/cand_4/complex.pdb md/runs/cand_5/complex.pdb md/runs/cand_vedotin/complex.pdb \
      --site-segid A --site-resid 214 --cutoff 10 --out outputs/wave2/per_linker_pka.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

BASIC = {'LYS', 'ARG', 'HIS'}


def find_propka():
    """Locate a propka3 executable: PATH, then common conda envs."""
    p = shutil.which('propka3') or shutil.which('propka')
    if p:
        return [p]
    # try module form in known python envs
    for py in ('/home/galeito/miniconda3/envs/amm_adc_10nm/bin/python',
               sys.executable):
        if py and os.path.exists(py):
            try:
                subprocess.run([py, '-c', 'import propka'], check=True,
                               capture_output=True)
                return [py, '-m', 'propka']
            except Exception:
                pass
    return None


def run_propka(propka_cmd, pdb, workdir):
    """Run PROPKA on a PDB; return path to the .pka output (in workdir)."""
    base = os.path.splitext(os.path.basename(pdb))[0]
    # propka writes <base>.pka in the cwd
    subprocess.run(propka_cmd + [os.path.abspath(pdb)], cwd=workdir,
                   check=True, capture_output=True)
    pka = os.path.join(workdir, base + '.pka')
    if not os.path.exists(pka):
        raise FileNotFoundError(f'PROPKA produced no {pka}')
    return pka


def parse_pka(pka_path):
    """(resname, resid, chain) -> pKa, parsed ONLY from the definitive
    'SUMMARY OF THIS PREDICTION' table of a PROPKA .pka file (the detailed
    interaction sections above it would otherwise double-match and pick up the
    100.00 sentinel)."""
    out = {}
    in_summary = False
    for ln in open(pka_path):
        if 'SUMMARY OF THIS PREDICTION' in ln:
            in_summary = True
            continue
        if not in_summary:
            continue
        if ln.strip().startswith('-') or 'Free energy' in ln:
            break  # end of summary table
        # summary rows have TWO numeric columns: pKa and model-pKa
        m = re.match(r'^\s*([A-Z]{2,3})\s+(\d+)\s+([A-Za-z*])\s+([\d.]+)\s+([\d.]+)', ln)
        if m:
            val = float(m.group(4))
            if val < 50.0:  # PROPKA writes ~99.99 for non-titratable (buried/disulfide) groups
                out[(m.group(1), int(m.group(2)), m.group(3))] = val
    return out


def near_residues(pdb, segid, resid, cutoff):
    """Residues with any atom within cutoff of the conjugation Cys SG.

    Pure-Python fixed-width PDB parse (no MDAnalysis dependency): finds the SG of
    the target resid (preferring chain==segid, else any chain), then returns
    {(resname, resid, chain): min_dist_to_SG} for residues within cutoff."""
    atoms = []   # (resname, resid, chain, x, y, z, name)
    sg = None
    with open(pdb) as fh:
        for ln in fh:
            if ln[:6] not in ('ATOM  ', 'HETATM'):
                continue
            try:
                name = ln[12:16].strip()
                rn = ln[17:20].strip()
                ch = ln[21].strip() or '?'
                ri = int(ln[22:26])
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            atoms.append((rn, ri, ch, x, y, z, name))
            if ri == resid and name == 'SG':
                # prefer the SG on the requested chain, but accept any if needed
                if sg is None or ch == segid:
                    sg = (x, y, z)
    if sg is None:
        return {}
    c2 = cutoff * cutoff
    res = {}
    for rn, ri, ch, x, y, z, _ in atoms:
        d2 = (x - sg[0]) ** 2 + (y - sg[1]) ** 2 + (z - sg[2]) ** 2
        if d2 <= c2:
            key = (rn, ri, ch)
            d = round(d2 ** 0.5, 2)
            if key not in res or d < res[key]:
                res[key] = d
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description='Per-linker local-pKa descriptor (PROPKA).')
    ap.add_argument('--pdbs', nargs='+', required=True,
                    help='conjugated-complex PDBs, one per linker/candidate')
    ap.add_argument('--site-segid', default='A')
    ap.add_argument('--site-resid', type=int, default=214)
    ap.add_argument('--cutoff', type=float, default=10.0)
    ap.add_argument('--out', default='outputs/wave2/per_linker_pka.csv')
    args = ap.parse_args(argv)

    propka = find_propka()
    if propka is None:
        print('PROPKA3 not found. This pilot needs the propka3 binary (the env that '
              'produced /tmp/rec.pka in §20.13). Install with `pip install propka` in '
              'that env, or put propka3 on PATH, then re-run. No output written.',
              file=sys.stderr)
        return 2
    print(f'using PROPKA: {" ".join(propka)}', file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    workdir = os.path.dirname(os.path.abspath(args.out))
    rows = []
    GENERIC = {'complex', 'system', 'md', 'prod', 'em', 'npt', 'nvt', 'receptor'}
    for pdb in args.pdbs:
        stem = os.path.splitext(os.path.basename(pdb))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(pdb)))
        # use the file stem when it's distinctive (complex_0, complex_1, ...);
        # fall back to the parent dir for generic per-candidate filenames
        # (md/runs/cand_4/complex.pdb -> cand_4)
        label = parent if stem.lower() in GENERIC else stem
        try:
            pka_path = run_propka(propka, pdb, workdir)
            pka = parse_pka(pka_path)
        except Exception as e:  # noqa: BLE001
            print(f'  !! {label}: PROPKA failed ({str(e)[:70]})', file=sys.stderr)
            continue
        near = near_residues(pdb, args.site_segid, args.site_resid, args.cutoff)
        if near is None:
            print('  !! MDAnalysis unavailable — reporting all titratable pKa instead',
                  file=sys.stderr)
            near = {(rn, ri, ch): None for (rn, ri, ch) in pka}
        for (rn, ri, ch), dist in sorted(near.items(), key=lambda kv: (kv[1] is None, kv[1])):
            val = pka.get((rn, ri, ch)) or pka.get((rn, ri, args.site_segid))
            if val is None:
                continue
            rows.append(dict(linker=label, resname=rn, resid=ri, chain=ch,
                             dist_to_SG=dist, pKa=val, is_basic=rn in BASIC))
        print(f'  {label}: {sum(1 for r in rows if r["linker"]==label)} near-site titratable residues')

    if not rows:
        print('no pKa rows produced', file=sys.stderr)
        return 1
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # the discriminating signal: residues whose pKa varies most across linkers
    by_res = {}
    for r in rows:
        by_res.setdefault((r['resname'], r['resid'], r['chain']), []).append(r['pKa'])
    spread = sorted(((max(v) - min(v), k, v) for k, v in by_res.items() if len(v) > 1),
                    reverse=True)
    print(f'\nwrote {args.out}')
    if spread:
        print('Residues whose local pKa varies MOST across linkers (the per-linker signal):')
        for d, (rn, ri, ch), vals in spread[:8]:
            print(f'  {rn}{ri}/{ch}: ΔpKa={d:.2f}  values={[round(x,1) for x in vals]}')
    else:
        print('(only one structure given — pass multiple conjugate PDBs to see per-linker ΔpKa)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

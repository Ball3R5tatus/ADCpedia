#!/usr/bin/env python
"""Wave-1 evaluation: score a DiffLinker generation directory with the unified
chemical-validity gate and aggregate the Pareto / LOPO metrics.

Reuses ``src/diffusion/gate`` (gate_usable3d) — the SAME gate as the rest of the
pipeline, so the usable-3D definition is identical (native-connected, NO MST
closure; MMFF-relaxable; ADC motif).

Two subcommands:

  eval  --gen-dir DIR --label NAME --results CSV [--train-smiles-csv F --smiles-col C]
        Scores every output_*.sdf in DIR, appends one aggregated row to CSV:
        label, n, native_conn%, mmff_ok%, valcit%, urea%, usable%, wilson_lo, wilson_hi, median_novelty_tanimoto

  plot  --results CSV --out PNG
        Reads the results CSV and renders the Pareto frontier figure
        (cysteine fraction vs connectivity / Val-Cit / usable-3D with Wilson CI).

Run with PYTHONPATH=<repo>/src so ``import diffusion.gate`` resolves.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys


# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
def _load_train_fps(train_smiles_csv, smiles_col):
    """Morgan fingerprints of the training-set linker-payloads, for novelty."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        print('  [novelty] rdkit unavailable — skipping Tanimoto novelty')
        return None
    fps = []
    with open(train_smiles_csv) as fh:
        for r in csv.DictReader(fh):
            smi = (r.get(smiles_col) or '').strip()
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    return fps or None


def _median_novelty(usable_smiles, train_fps):
    """Median over generated molecules of (1 - max Tanimoto to any training mol).
    Returns the median *Tanimoto-to-nearest-train* (the paper's reported stat)."""
    if not train_fps or not usable_smiles:
        return ''
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    nn = []
    for smi in usable_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        if sims:
            nn.append(max(sims))
    if not nn:
        return ''
    nn.sort()
    n = len(nn)
    med = nn[n // 2] if n % 2 else 0.5 * (nn[n // 2 - 1] + nn[n // 2])
    return round(med, 3)


# --------------------------------------------------------------------------
def cmd_eval(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from diffusion.gate import gate_usable3d

    sdfs = sorted(glob.glob(os.path.join(args.gen_dir, '*.sdf')))
    if not sdfs:
        print(f'!! no *.sdf in {args.gen_dir}', file=sys.stderr)
        return 1
    recs = [gate_usable3d(s) for s in sdfs]
    n = len(recs)
    cnt = {k: sum(1 for r in recs if r[k]) for k in
           ('parsed', 'native_connected', 'mmff_ok', 'has_valcit', 'has_urea', 'usable_3d')}
    lo, hi = wilson(cnt['usable_3d'], n)

    novelty = ''
    if args.train_smiles_csv:
        fps = _load_train_fps(args.train_smiles_csv, args.smiles_col)
        usable_smiles = [r['smiles'] for r in recs if r['usable_3d'] and r['smiles']]
        novelty = _median_novelty(usable_smiles, fps)

    row = dict(label=args.label, n=n,
               native_conn_pct=round(100 * cnt['native_connected'] / n, 1),
               mmff_ok_pct=round(100 * cnt['mmff_ok'] / n, 1),
               valcit_pct=round(100 * cnt['has_valcit'] / n, 1),
               urea_pct=round(100 * cnt['has_urea'] / n, 1),
               usable_pct=round(100 * cnt['usable_3d'] / n, 1),
               wilson_lo=round(100 * lo, 1), wilson_hi=round(100 * hi, 1),
               median_novelty_tanimoto=novelty)

    header = list(row.keys())
    new = not os.path.exists(args.results)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f'{args.label}: n={n} conn={row["native_conn_pct"]}% valcit={row["valcit_pct"]}% '
          f'usable={row["usable_pct"]}% [{row["wilson_lo"]},{row["wilson_hi"]}] '
          f'novelty={novelty}')
    return 0


# --------------------------------------------------------------------------
def _frac_from_label(label):
    m = re.search(r'pareto_f(\d+)', label)
    return int(m.group(1)) if m else None


def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.results)))
    # average replicates per cysteine fraction
    by_frac = {}
    for r in rows:
        f = _frac_from_label(r['label'])
        if f is None:
            continue
        by_frac.setdefault(f, []).append(r)
    fracs = sorted(by_frac)
    if not fracs:
        print('!! no pareto_fXXX rows in results — nothing to plot', file=sys.stderr)
        return 1

    def avg(rs, k):
        return sum(float(x[k]) for x in rs) / len(rs)

    conn = [avg(by_frac[f], 'native_conn_pct') for f in fracs]
    valcit = [avg(by_frac[f], 'valcit_pct') for f in fracs]
    usable = [avg(by_frac[f], 'usable_pct') for f in fracs]
    lo = [avg(by_frac[f], 'wilson_lo') for f in fracs]
    hi = [avg(by_frac[f], 'wilson_hi') for f in fracs]
    # clamp to >=0: averaging Wilson bounds across replicates independently of
    # the mean can leave a tiny negative arm (the CI is nonlinear in p).
    yerr = [[max(0.0, u - l) for u, l in zip(usable, lo)],
            [max(0.0, h - u) for h, u in zip(hi, usable)]]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fracs, conn, 'o-', color='#0072B2', label='native-connected %')
    ax[0].plot(fracs, valcit, 's-', color='#C2570C', label='Val-Cit %')
    ax[0].set_xlabel('cysteine sampling fraction (%)')
    ax[0].set_ylabel('% of generations')
    ax[0].set_title('Connectivity vs target chemistry (the tradeoff)')
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].errorbar(fracs, usable, yerr=yerr, fmt='D-', color='#117733',
                   capsize=4, label='usable-3D % (Wilson 95% CI)')
    ax[1].set_xlabel('cysteine sampling fraction (%)')
    ax[1].set_ylabel('usable-3D % per 100')
    ax[1].set_title('Operational yield along the frontier')
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle('Wave-1 Pareto frontier: cysteine sampling fraction controls '
                 'ADC chemistry vs 3D validity', fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.splitext(args.out)[0] + '.' + ext, dpi=150, bbox_inches='tight')
    print(f'wrote {os.path.splitext(args.out)[0]}.png/.pdf  ({len(fracs)} fractions)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='eval_sweep')
    sub = ap.add_subparsers(dest='cmd', required=True)
    pe = sub.add_parser('eval', help='score a generation dir, append to results CSV')
    pe.add_argument('--gen-dir', required=True)
    pe.add_argument('--label', required=True)
    pe.add_argument('--results', required=True)
    pe.add_argument('--train-smiles-csv', default=None,
                    help='optional: training table CSV for Tanimoto novelty')
    pe.add_argument('--smiles-col', default='molecule',
                    help='SMILES column in the training table (default: molecule)')
    pe.set_defaults(func=cmd_eval)
    pp = sub.add_parser('plot', help='render the Pareto frontier figure')
    pp.add_argument('--results', required=True)
    pp.add_argument('--out', required=True)
    pp.set_defaults(func=cmd_plot)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

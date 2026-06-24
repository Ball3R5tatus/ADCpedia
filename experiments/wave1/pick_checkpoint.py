#!/usr/bin/env python
"""Select the operationally-best DiffLinker checkpoint by GENERATED CONNECTIVITY,
not by epoch or validation loss.

Motivation (Master Finding 9 + the Wave-2 §25.7 diagnostic): connectivity peaks
EARLY in fine-tuning and then degrades, so the last/`ls -t` checkpoint is
over-trained (f100: ft-ep12 conn 9.4% vs ft-ep32 conn 4.4% at size 20). And
val_loss is a denoising loss, not a generation-quality metric. So we probe a
handful of checkpoints with a small generation, gate-score native connectivity
(src/diffusion/gate), and return the best one.

Prints the chosen checkpoint PATH on the last stdout line (for `best=$(...)`);
the probe table goes to stderr.

Run in the difflinker_gpu env (needs torch+cuda for generate.py AND rdkit for the
gate; difflinker_gpu has both).

  python experiments/wave1/pick_checkpoint.py --ckpt-dir <dir> \
      --fragments <input.sdf> --linker-size 20 --n 64 --k 6 \
      --difflinker-root ~/tools/DiffLinker --device cuda
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def epoch_of(path):
    m = re.search(r'epoch=(\d+)', path)
    return int(m.group(1)) if m else -1


def candidate_checkpoints(ckpt_dir, k):
    """All trained checkpoints (exclude the epoch=00 GEOM seed), evenly subsampled
    to ~k probes spanning the fine-tune range (always include first + last)."""
    cks = [c for c in glob.glob(os.path.join(ckpt_dir, '*.ckpt')) if epoch_of(c) > 0]
    cks = sorted(set(cks), key=epoch_of)
    cks = [c for c in cks if epoch_of(c) >= 1]
    # drop the GEOM seed (lowest epoch if it is the 00 seed already excluded)
    if len(cks) <= k:
        return cks
    idx = sorted(set([round(i * (len(cks) - 1) / (k - 1)) for i in range(k)]))
    return [cks[i] for i in idx]


def probe_connectivity(ckpt, fragments, size, n, dl_root, device):
    """Generate n linkers from ckpt and return native-connected %."""
    from diffusion.gate import gate_usable3d
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, 'generate.py', '--fragments', os.path.abspath(fragments),
               '--model', os.path.abspath(ckpt), '--linker_size', str(size),
               '--output', td, '--n_samples', str(n), '--device', device]
        env = dict(os.environ, WANDB_MODE='disabled')
        r = subprocess.run(cmd, cwd=os.path.expanduser(dl_root), env=env,
                           capture_output=True, text=True)
        sdfs = sorted(glob.glob(os.path.join(td, '*.sdf')))
        if not sdfs:
            sys.stderr.write(f'  [probe] {os.path.basename(ckpt)}: NO output '
                             f'({r.stderr.strip()[-160:]})\n')
            return None
        recs = [gate_usable3d(s) for s in sdfs]
        nconn = sum(1 for x in recs if x['native_connected'])
        nusable = sum(1 for x in recs if x['usable_3d'])
        return dict(n=len(recs), conn_pct=round(100 * nconn / len(recs), 1),
                    usable_pct=round(100 * nusable / len(recs), 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--fragments', required=True)
    ap.add_argument('--linker-size', type=int, default=20)
    ap.add_argument('--n', type=int, default=64, help='probe samples per checkpoint')
    ap.add_argument('--k', type=int, default=6, help='number of checkpoints to probe')
    ap.add_argument('--difflinker-root', default='~/tools/DiffLinker')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)

    cands = candidate_checkpoints(args.ckpt_dir, args.k)
    if not cands:
        sys.stderr.write(f'no trained checkpoints in {args.ckpt_dir}\n')
        return 1
    sys.stderr.write(f'[pick_checkpoint] probing {len(cands)} checkpoints '
                     f'(n={args.n}, size={args.linker_size}) by native connectivity:\n')
    best, best_conn = None, -1.0
    for ck in cands:
        res = probe_connectivity(ck, args.fragments, args.linker_size, args.n,
                                 args.difflinker_root, args.device)
        if res is None:
            continue
        sys.stderr.write(f'  ep{epoch_of(ck):>4}: conn {res["conn_pct"]:>5}%  '
                         f'usable {res["usable_pct"]:>5}%  (n={res["n"]})\n')
        # tie-break toward HIGHER connectivity (the operational bottleneck)
        if res['conn_pct'] > best_conn:
            best_conn, best = res['conn_pct'], ck
    if best is None:
        sys.stderr.write('all probes failed\n')
        return 1
    sys.stderr.write(f'[pick_checkpoint] BEST: ep{epoch_of(best)} (conn {best_conn}%)\n')
    print(best)  # the only stdout line -> captured by the runner
    return 0


if __name__ == '__main__':
    sys.exit(main())

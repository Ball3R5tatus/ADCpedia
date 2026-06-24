#!/usr/bin/env python
"""Wave-1 split builder: Pareto (cysteine-fraction sweep) and LOPO
(leave-one-payload-class-out) training sets for the DiffLinker fine-tune.

Both experiments are produced by *resampling/filtering the existing
conformer-augmented multi-site trio* (table.csv + frag.sdf + link.sdf), NOT by
retraining the data pipeline.  This is deterministic (seeded) and — for the
Pareto sweep — avoids the ``WeightedRandomSampler`` VRAM death-spiral documented
in Master §19.13 by materialising a pre-balanced dataset at fixed total size.

DiffLinker data contract (from ~/tools/DiffLinker/src/datasets.py ZincDataset):
  * a "split" is three files sharing a prefix in one directory:
        {prefix}_table.csv, {prefix}_frag.sdf, {prefix}_link.sdf
    with the SDF molecule blocks index-aligned 1:1 to the table rows.
  * the prefix MUST contain the substring ``geom`` so ``is_geom=True`` selects
    the GEOM atom vocabulary (required to resume from geom_difflinker.ckpt).
  * ZincDataset auto-builds + caches ``{prefix}.pt`` on first access.

So each split we emit is named ``geom_<split>_train`` / ``geom_<split>_val`` and
dropped in its own directory, with a cloned fine-tune YAML pointing at it.

Source (default): data/processed/difflinker_trainset_v3_multi_site_aug/
  adc_cys_train_*  (1240 rows: 820 cysteine / 380 lysine / 40 click = 66% cys)
  adc_cys_val_*    (202 rows)  -- shared, fixed across all splits.

Usage
-----
  PYTHONPATH=src python -m diffusion.data.build_split pareto \
      --fractions 50,66,75,85,100 --seed 0
  PYTHONPATH=src python -m diffusion.data.build_split lopo \
      --classes auristatin,camptothecin,maytansinoid,duocarmycin,calicheamicin,taxane --seed 0

Outputs (per split) under data/processed/splits/<split>/:
  geom_<split>_train_{table.csv,frag.sdf,link.sdf}
  geom_<split>_val_{table.csv,frag.sdf,link.sdf}   (symlinks to the shared val)
  <split>.yml                                       (cloned fine-tune config)
and a top-level data/processed/splits/SPLITS_MANIFEST.csv recording each build.
"""
from __future__ import annotations

import argparse
import csv
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC_DIR = os.path.join(ROOT, 'data/processed/difflinker_trainset_v3_multi_site_aug')
OUT_ROOT = os.path.join(ROOT, 'data/processed/splits')
MANIFEST_PATH = os.path.join(ROOT, 'data/processed/dataset_manifest.csv')
FINETUNE_YML = os.path.expanduser('~/tools/DiffLinker/configs/adc_cys_finetune.yml')

# Conjugation-site normalisation (mirror of build_manifest.py).
def norm_site(s):
    s = (s or '').strip().lower()
    if 'cys' in s or 'cyst' in s or 'custe' in s:
        return 'cysteine'
    if 'lys' in s:
        return 'lysine'
    if 'glut' in s or 'llqg' in s:
        return 'glutamine'
    if any(k in s for k in ('click', 'azid', 'triazol', 'dbco', 'bcn', 'galnac', 'glycan', 'alkyne')):
        return 'click'
    return 'other'


# ---------------------------------------------------------------------------
# trio IO  (table rows + index-aligned SDF molecule blocks)
# ---------------------------------------------------------------------------
def read_sdf_blocks(path):
    """Return a list of molecule blocks (each ending with the '$$$$' line),
    byte-faithful (line-based, never splits inside a block)."""
    blocks, cur = [], []
    with open(path) as fh:
        for line in fh:
            cur.append(line)
            if line.strip() == '$$$$':
                blocks.append(''.join(cur))
                cur = []
    if cur and ''.join(cur).strip():
        blocks.append(''.join(cur))  # tolerate a missing final terminator
    return blocks


def read_trio(data_dir, prefix):
    with open(os.path.join(data_dir, f'{prefix}_table.csv')) as fh:
        rdr = csv.DictReader(fh)
        header = rdr.fieldnames
        rows = list(rdr)
    frag = read_sdf_blocks(os.path.join(data_dir, f'{prefix}_frag.sdf'))
    link = read_sdf_blocks(os.path.join(data_dir, f'{prefix}_link.sdf'))
    assert len(rows) == len(frag) == len(link), \
        f'{prefix}: misaligned trio rows={len(rows)} frag={len(frag)} link={len(link)}'
    return header, rows, frag, link


def write_trio(out_dir, prefix, header, rows, frag, link, order):
    """Write a trio selecting source indices in ``order`` (duplicates allowed)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{prefix}_table.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for j, i in enumerate(order):
            r = dict(rows[i])
            if 'uuid' in r and r.get('uuid'):
                r['uuid'] = f"{r['uuid']}__r{j}"   # keep ids unique under duplication
            w.writerow(r)
    with open(os.path.join(out_dir, f'{prefix}_frag.sdf'), 'w') as fh:
        fh.write(''.join(frag[i] for i in order))
    with open(os.path.join(out_dir, f'{prefix}_link.sdf'), 'w') as fh:
        fh.write(''.join(link[i] for i in order))


def link_shared_val(out_dir, split, src_dir, src_val_prefix):
    """Symlink the shared (fixed) val trio into the split dir under geom_<split>_val_*."""
    dst_prefix = f'geom_{split}_val'
    for kind in ('table.csv', 'frag.sdf', 'link.sdf'):
        src = os.path.join(src_dir, f'{src_val_prefix}_{kind}')
        dst = os.path.join(out_dir, f'{dst_prefix}_{kind}')
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(src, out_dir), dst)
    return dst_prefix


# ---------------------------------------------------------------------------
# config emission
# ---------------------------------------------------------------------------
def emit_config(out_dir, split, train_prefix, val_prefix):
    """Clone the fine-tune YAML, repoint data/prefixes/exp_name/resume."""
    if not os.path.exists(FINETUNE_YML):
        return None
    with open(FINETUNE_YML) as fh:
        txt = fh.read()
    repl = {
        'exp_name': f"exp_name: '{split}'",
        'data:': f'data: {out_dir}',
        'train_data_prefix:': f'train_data_prefix: {train_prefix}',
        'val_data_prefix:': f'val_data_prefix: {val_prefix}',
        'resume:': f"resume: '{split}'",
    }
    out_lines = []
    for line in txt.splitlines():
        key = line.split('#', 1)[0].strip()
        done = False
        for k, newline in repl.items():
            if key.startswith(k):
                out_lines.append(newline)
                done = True
                break
        if not done:
            out_lines.append(line)
    cfg_path = os.path.join(out_dir, f'{split}.yml')
    with open(cfg_path, 'w') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    return cfg_path


def record(split, kind, train_prefix, n_train, detail, cfg):
    os.makedirs(OUT_ROOT, exist_ok=True)
    splits_manifest = os.path.join(OUT_ROOT, 'SPLITS_MANIFEST.csv')
    new = not os.path.exists(splits_manifest)
    with open(splits_manifest, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['split', 'kind', 'train_prefix', 'n_train', 'detail', 'config'])
        w.writerow([split, kind, train_prefix, n_train, detail, cfg or ''])


# ---------------------------------------------------------------------------
# PARETO: cysteine-fraction sweep at fixed total size
# ---------------------------------------------------------------------------
def _sample(rng, pool, k):
    """Sample k indices from pool: without replacement up to len(pool), then
    fill the remainder with replacement (deterministic given rng)."""
    if not pool:
        return []
    if k <= len(pool):
        return rng.sample(pool, k)
    return rng.sample(pool, len(pool)) + rng.choices(pool, k=k - len(pool))


def build_pareto(fractions, seed, src_dir, train_prefix, val_prefix):
    header, rows, frag, link = read_trio(src_dir, train_prefix)
    n = len(rows)
    cys = [i for i, r in enumerate(rows) if norm_site(r.get('site_final')) == 'cysteine']
    non = [i for i, r in enumerate(rows) if norm_site(r.get('site_final')) != 'cysteine']
    print(f'[pareto] source {train_prefix}: N={n}  cys={len(cys)} ({100*len(cys)/n:.1f}%)  non-cys={len(non)}')
    for f in fractions:
        rng = random.Random(seed * 1000 + f)
        n_cys = round(f / 100.0 * n)
        sel = _sample(rng, cys, n_cys) + _sample(rng, non, n - n_cys)
        rng.shuffle(sel)
        split = f'pareto_f{f:03d}_s{seed}'
        out_dir = os.path.join(OUT_ROOT, split)
        tp = f'geom_{split}_train'
        write_trio(out_dir, tp, header, rows, frag, link, sel)
        vp = link_shared_val(out_dir, split, src_dir, val_prefix)
        cfg = emit_config(out_dir, split, tp, vp)
        got = sum(1 for i in sel if norm_site(rows[i].get('site_final')) == 'cysteine')
        detail = f'target_cys={f}% actual_cys={100*got/len(sel):.1f}% n_cys={n_cys}'
        print(f'  -> {split}: {detail}  dir={out_dir}')
        record(split, 'pareto', tp, len(sel), detail, cfg)


# ---------------------------------------------------------------------------
# LOPO: leave-one-payload-class-out
# ---------------------------------------------------------------------------
def _payload_class_map():
    """canonical_payload_name -> payload_class, read from the frozen manifest."""
    m = {}
    if not os.path.exists(MANIFEST_PATH):
        return m
    for r in csv.DictReader(open(MANIFEST_PATH)):
        name = (r.get('payload_name') or '').strip()
        if name:
            m[name] = r.get('payload_class', 'unknown')
    return m


def build_lopo(classes, seed, src_dir, train_prefix, val_prefix):
    header, rows, frag, link = read_trio(src_dir, train_prefix)
    pcmap = _payload_class_map()
    if not pcmap:
        print('[lopo] WARNING: no manifest found; run build_manifest.py first. Aborting.')
        return
    def pclass(r):
        return pcmap.get((r.get('canonical_payload_name') or '').strip(), 'unknown')
    from collections import Counter
    dist = Counter(pclass(r) for r in rows)
    print(f'[lopo] source {train_prefix}: payload_class row-counts = {dict(dist)}')
    for cls in classes:
        keep = [i for i, r in enumerate(rows) if pclass(r) != cls]
        dropped = len(rows) - len(keep)
        if dropped == 0:
            print(f'  !! class "{cls}" not present in source (0 rows) — skipping')
            continue
        rng = random.Random(seed)
        rng.shuffle(keep)
        split = f'lopo_{cls}_s{seed}'
        out_dir = os.path.join(OUT_ROOT, split)
        tp = f'geom_{split}_train'
        write_trio(out_dir, tp, header, rows, frag, link, keep)
        vp = link_shared_val(out_dir, split, src_dir, val_prefix)
        cfg = emit_config(out_dir, split, tp, vp)
        detail = f'held_out={cls} dropped_rows={dropped} kept_rows={len(keep)}'
        print(f'  -> {split}: {detail}  dir={out_dir}')
        record(split, 'lopo', tp, len(keep), detail, cfg)
    print('\n[lopo] NOTE: to evaluate generalisation, generate on the HELD-OUT class '
          'payloads (export a representative payload+stub via diffusion.data.difflinker_export) '
          'and score with diffusion.gate. The held-out class is absent from training by construction.')


def build_full(seed, src_dir, train_prefix, val_prefix):
    """Interpolation baseline: the unfiltered source trio as geom_full_s<seed>.
    Used as the in-distribution reference the LOPO (held-out) arm is compared
    against — same recipe, all classes present."""
    header, rows, frag, link = read_trio(src_dir, train_prefix)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    split = f'full_s{seed}'
    out_dir = os.path.join(OUT_ROOT, split)
    tp = f'geom_{split}_train'
    write_trio(out_dir, tp, header, rows, frag, link, order)
    vp = link_shared_val(out_dir, split, src_dir, val_prefix)
    cfg = emit_config(out_dir, split, tp, vp)
    detail = f'all_classes n={len(rows)}'
    print(f'[full] -> {split}: {detail}  dir={out_dir}')
    record(split, 'full', tp, len(rows), detail, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='diffusion.data.build_split',
                                 description='Build Pareto / LOPO / full DiffLinker training splits.')
    ap.add_argument('--src-dir', default=SRC_DIR, help='source augmented trio dir')
    ap.add_argument('--train-prefix', default='adc_cys_train')
    ap.add_argument('--val-prefix', default='adc_cys_val')
    sub = ap.add_subparsers(dest='cmd', required=True)

    pp = sub.add_parser('pareto', help='cysteine-fraction sweep (fixed total size)')
    pp.add_argument('--fractions', default='50,66,75,85,100',
                    help='comma-sep cysteine %% targets')
    pp.add_argument('--seed', type=int, default=0)

    pl = sub.add_parser('lopo', help='leave-one-payload-class-out')
    pl.add_argument('--classes', required=True, help='comma-sep payload_class names to hold out')
    pl.add_argument('--seed', type=int, default=0)

    pf = sub.add_parser('full', help='unfiltered baseline (in-distribution reference for LOPO)')
    pf.add_argument('--seed', type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == 'pareto':
        fr = [int(x) for x in args.fractions.split(',') if x.strip()]
        build_pareto(fr, args.seed, args.src_dir, args.train_prefix, args.val_prefix)
    elif args.cmd == 'lopo':
        cls = [x.strip() for x in args.classes.split(',') if x.strip()]
        build_lopo(cls, args.seed, args.src_dir, args.train_prefix, args.val_prefix)
    elif args.cmd == 'full':
        build_full(args.seed, args.src_dir, args.train_prefix, args.val_prefix)


if __name__ == '__main__':
    main()

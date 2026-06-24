#!/usr/bin/env python
"""Export one representative held-out-class payload as a DiffLinker generation
input, for the LOPO (leave-one-payload-class-out) generalization test.

For each viable payload_class (>=100 augmented training rows) we pick the
manifest's best representative (prefer fully-defined stereo, then largest) and
write payload+maleimide-stub input SDF via difflinker_export.export_adc, the
same exporter used for the cysteine MMAE input. Also emits a mapping CSV so the
LOPO runner knows which input goes with which held-out split.

Run with PYTHONPATH=<repo>/src.

  PYTHONPATH=src python experiments/wave1/export_lopo_inputs.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from pathlib import Path
from diffusion.data.difflinker_export import export_adc

MANIFEST = os.path.join(ROOT, 'data/processed/dataset_manifest.csv')
OUT_DIR = os.path.join(ROOT, 'data/processed/difflinker_inputs')
MAP_CSV = os.path.join(OUT_DIR, 'lopo_inputs_map.csv')

# Classes with >=100 augmented training rows (safe to hold out); see
# experiments/wave1/README.md. taxane/calicheamicin excluded (too thin).
VIABLE = {'auristatin', 'camptothecin', 'maytansinoid',
          'PBD_anthracycline', 'duocarmycin', 'eribulin'}


def pick_reps():
    """class -> (payload_name, payload_smiles), best = stereo-defined then longest."""
    best = {}
    with open(MANIFEST) as fh:
        for r in csv.DictReader(fh):
            cls = r['payload_class']
            if cls not in VIABLE:
                continue
            smi = (r.get('payload_smiles') or '').strip()
            if not smi or smi.lower() == 'nan':
                continue
            score = (1 if r.get('stereo_defined') == 'yes' else 0, len(smi))
            if cls not in best or score > best[cls][0]:
                best[cls] = (score, r.get('payload_name'), smi)
    return {c: (n, s) for c, (sc, n, s) in best.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reps = pick_reps()
    rows = []
    for cls in sorted(reps):
        name, smi = reps[cls]
        try:
            res = export_adc(smi, f'lopo_{cls}', Path(OUT_DIR))
        except Exception as e:  # noqa: BLE001
            print(f'  !! {cls}: export failed ({str(e)[:60]}) — skipping')
            continue
        anchors = f'{res.payload_anchor_1based},{res.maleimide_anchor_1based}'
        rows.append(dict(payload_class=cls, payload_name=name,
                         input_sdf=os.path.relpath(res.sdf_path, ROOT),
                         anchors=anchors, n_payload_atoms=res.n_payload_atoms,
                         anchor_reason=res.payload_anchor_reason))
        print(f'  {cls:18s} <- {name:24s}  -> {os.path.basename(res.sdf_path)}  anchors={anchors}')
    if rows:
        with open(MAP_CSV, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {len(rows)} inputs + map: {os.path.relpath(MAP_CSV, ROOT)}')
    else:
        print('no inputs exported', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

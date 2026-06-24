"""Export an ADC payload + a standard maleimide stub as a DiffLinker input.

DiffLinker expects a single multi-fragment SDF with 3D coords and a CLI flag
``--anchors <i,j>`` (1-indexed) pointing at the two heavy atoms where the
generated spacer will attach. No dummy ``*`` atoms in the file — the placeholder
is communicated solely via the anchor indices (see docs/DIFFLINKER_ANALYSIS.md
section Q1).

This module:
  1. Reads a payload SMILES.
  2. Builds a minimal maleimide stub (cysteine-side terminal).
  3. Picks an anchor on each fragment using SMARTS priority lists.
  4. Embeds 3D conformers (ETKDGv3 + MMFF94, UFF fallback).
  5. Combines them into one SDF with the two fragments translated apart so
     their bounding boxes do not overlap.
  6. Prints the generate.py command with the correct ``--anchors``.

The output is reproducible: same SMILES → same SDF (RDKit `randomSeed=0`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Maleimide stub — minimal N-methyl maleimide. The methyl C (atom 0) is the
# anchor: DiffLinker grows the spacer outward from that carbon. The rest of
# the ring stays intact in the generated molecule.
# ---------------------------------------------------------------------------
MALEIMIDE_STUB_SMILES = "CN1C(=O)C=CC1=O"
MALEIMIDE_ANCHOR_IDX = 0  # the methyl carbon


# Reactive-group SMARTS for picking the PAYLOAD anchor, in priority order.
# For most cytotoxic payloads conjugated through a self-immolative PABC
# linker, the attachment point is an amine or hydroxyl that forms a carbamate
# / carbonate with the linker. The N-methyl secondary amine of MMAE's
# N-Me-valine head group is the canonical example.
#
# Each entry: (name, smarts, atom_index_within_match).
PAYLOAD_REACTIVE_SMARTS: List[Tuple[str, str, int]] = [
    # Secondary aliphatic amine (-NH-R, not amide / aniline). MMAE's NMe-Val.
    ("secondary_aliphatic_amine",
     "[NX3;H1;D2;!$(N-[C,c]=[O,S]);!$(N-[c])]", 0),
    # Primary aliphatic amine (-NH2). Common in DXd, MMAF analogs.
    ("primary_aliphatic_amine",
     "[NX3;H2;D1;!$(N-[C,c]=[O,S])]", 0),
    # Aniline-style aromatic NH (still amine-reactive in some conjugations).
    ("aromatic_secondary_amine",
     "[NX3;H1;D2;$(N-[c])]", 0),
    # Aliphatic hydroxyl (-OH on sp3 C, not part of -COOH).
    ("aliphatic_hydroxyl",
     "[OX2;H1;D1;$([OH][CX4])]", 0),
    # Phenol.
    ("phenol",
     "[OX2;H1;D1;$([OH][c])]", 0),
    # Thiol (last resort — DM1/DM4 style).
    ("thiol",
     "[SX2;H1;D1]", 0),
]


@dataclass
class ExportResult:
    sdf_path: Path
    payload_anchor_1based: int
    maleimide_anchor_1based: int
    payload_anchor_reason: str
    payload_smiles: str
    n_payload_atoms: int
    n_maleimide_atoms: int

    def generate_command(self, repo_root: Path, ckpt: str, size_gnn: str,
                         output_dir: str, n_samples: int = 20) -> str:
        return (
            f"cd {repo_root} && python -W ignore generate.py "
            f"--fragments {self.sdf_path} "
            f"--model {ckpt} --linker_size {size_gnn} "
            f"--output {output_dir} --n_samples {n_samples} --device cpu"
        )


# ---------------------------------------------------------------------------
# Anchor finders
# ---------------------------------------------------------------------------

def find_payload_anchor(mol: Chem.Mol) -> Tuple[int, str]:
    """Return (atom_idx, reason) for the payload's reactive-group atom.

    Walks ``PAYLOAD_REACTIVE_SMARTS`` in priority order; first match wins.
    Raises ``ValueError`` if no reactive group is found.
    """
    for name, smarts, sub_idx in PAYLOAD_REACTIVE_SMARTS:
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            continue
        match = mol.GetSubstructMatch(q)
        if match:
            return match[sub_idx], name
    raise ValueError("No reactive functional group found on the payload")


def find_maleimide_anchor(stub_mol: Chem.Mol) -> int:
    """Locate the exocyclic sp3 carbon adjacent to the maleimide N. The
    spacer will be grown from this atom, so it must be exocyclic.
    """
    # RDKit SMARTS note: 'R5' = 'in 5 SSSR rings' (count). 'r5' would be
    # 'in a ring of size 5'. The N-CH_x check below uses neither — it
    # constrains the ring carbons by membership in any ring (capital R) and
    # the carbonyl pattern, which is enough to disambiguate maleimide.
    q = Chem.MolFromSmarts("[CX4][N;R]([C;R]=O)[C;R]=O")
    match = stub_mol.GetSubstructMatch(q)
    if not match:
        raise ValueError(
            "Maleimide stub does not contain expected exocyclic-C–N(C=O)(C=O) pattern"
        )
    return match[0]


# ---------------------------------------------------------------------------
# 3D embedding
# ---------------------------------------------------------------------------

def embed_3d(mol: Chem.Mol, seed: int = 0) -> Chem.Mol:
    """Generate a single 3D conformer with ETKDGv3 + MMFF94 (UFF fallback)."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        # Retry with looser parameters before giving up.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            raise RuntimeError("RDKit ETKDGv3 failed to embed the molecule")
    if AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500) != 0:
        # Sometimes MMFF94 lacks parameters for unusual atoms — fall back to UFF.
        if AllChem.UFFOptimizeMolecule(mol_h, maxIters=500) != 0:
            # Even an unoptimised geometry is usable for DiffLinker input.
            pass
    mol_no_h = Chem.RemoveHs(mol_h)
    return mol_no_h


# ---------------------------------------------------------------------------
# Fragment assembly
# ---------------------------------------------------------------------------

def _translate_conformer(mol: Chem.Mol, offset: np.ndarray):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + offset[0], p.y + offset[1], p.z + offset[2]))


def _atom_position(mol: Chem.Mol, idx: int) -> np.ndarray:
    p = mol.GetConformer().GetAtomPosition(idx)
    return np.array([p.x, p.y, p.z])


def _coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z]
                     for i in range(mol.GetNumAtoms())])


def assemble_fragments(payload: Chem.Mol, payload_anchor: int,
                       stub: Chem.Mol, stub_anchor: int,
                       target_anchor_distance: float = 8.0,
                       min_atom_clearance: float = 2.5) -> Chem.Mol:
    """Combine the two 3D fragments into a single mol with no inter-fragment
    bonds. Position the stub so that:
      - the two anchor atoms are exactly ``target_anchor_distance`` apart, and
      - no atom of the stub is closer than ``min_atom_clearance`` Å to any
        atom of the payload.

    DiffLinker's training distribution (GEOM) sits with inter-fragment
    centroid distances of ~10-15 Å. Putting the anchors at ~8 Å keeps us in
    that regime; ``min_atom_clearance`` then pushes the stub further out
    along the anchor-anchor axis if a clash would occur.
    """
    p_anchor_pos = _atom_position(payload, payload_anchor)
    payload_centroid = _coords(payload).mean(axis=0)

    # Direction = from payload centroid → payload anchor (pointing outward).
    direction = p_anchor_pos - payload_centroid
    nrm = np.linalg.norm(direction)
    if nrm < 1e-6:
        # Degenerate: payload anchor is at the centroid. Fall back to +x.
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / nrm

    s_anchor_pos = _atom_position(stub, stub_anchor)

    # Initial placement: stub anchor sits along the outward direction at the
    # target anchor distance from the payload anchor.
    distance = float(target_anchor_distance)
    payload_coords = _coords(payload)
    for _ in range(10):
        target = p_anchor_pos + distance * direction
        shift = target - s_anchor_pos
        stub_translated = Chem.Mol(stub)
        _translate_conformer(stub_translated, shift)
        # Check pairwise atom distance to detect clashes.
        stub_coords = _coords(stub_translated)
        diffs = stub_coords[None, :, :] - payload_coords[:, None, :]
        min_d = np.sqrt((diffs ** 2).sum(axis=-1)).min()
        if min_d >= min_atom_clearance:
            stub = stub_translated
            break
        distance += 1.0
    else:
        stub = stub_translated  # use last attempt regardless

    return Chem.CombineMols(payload, stub)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def export_adc(payload_smiles: str, adc_name: str,
               output_dir: Path,
               stub_smiles: str = MALEIMIDE_STUB_SMILES) -> ExportResult:
    payload = Chem.MolFromSmiles(payload_smiles)
    if payload is None:
        raise ValueError(f"Failed to parse payload SMILES: {payload_smiles}")
    stub = Chem.MolFromSmiles(stub_smiles)
    if stub is None:
        raise ValueError(f"Failed to parse stub SMILES: {stub_smiles}")

    payload_anchor_0, reason = find_payload_anchor(payload)
    stub_anchor_0 = find_maleimide_anchor(stub)

    payload_3d = embed_3d(payload)
    stub_3d = embed_3d(stub)

    combined = assemble_fragments(payload_3d, payload_anchor_0,
                                  stub_3d, stub_anchor_0)

    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"{adc_name}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(combined)
    writer.close()

    n_pay = payload_3d.GetNumAtoms()
    n_stub = stub_3d.GetNumAtoms()
    # CombineMols preserves order: payload [0..n_pay-1], stub [n_pay..n_pay+n_stub-1].
    payload_anchor_1based = payload_anchor_0 + 1
    stub_anchor_1based = n_pay + stub_anchor_0 + 1

    return ExportResult(
        sdf_path=sdf_path,
        payload_anchor_1based=payload_anchor_1based,
        maleimide_anchor_1based=stub_anchor_1based,
        payload_anchor_reason=reason,
        payload_smiles=payload_smiles,
        n_payload_atoms=n_pay,
        n_maleimide_atoms=n_stub,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/processed/linkers_qc_passed.csv")
    ap.add_argument("--row", type=int, default=0,
                    help="Row index in the (filtered) cysteine CSV to export")
    ap.add_argument("--site", default="cysteine",
                    help="Filter by site_final (cysteine, lysine, ...)")
    ap.add_argument("--payload-name", default=None,
                    help="Optional canonical_payload_name filter (substring match)")
    ap.add_argument("--name", default=None,
                    help="Override the ADC export name (defaults to "
                         "<payload>_<linker>_<row>)")
    ap.add_argument("--output-dir", default="data/processed/difflinker_inputs")
    ap.add_argument("--difflinker-root", default="~/tools/DiffLinker")
    ap.add_argument("--ckpt", default="models/geom_difflinker.ckpt")
    ap.add_argument("--size-gnn", default="models/geom_size_gnn.ckpt")
    ap.add_argument("--gen-output", default="./out_adc_test")
    ap.add_argument("--n-samples", type=int, default=20)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    sub = df[df["site_final"] == args.site]
    if args.payload_name:
        sub = sub[sub["canonical_payload_name"].astype(str)
                  .str.contains(args.payload_name, na=False)]
    if len(sub) == 0:
        print(f"No rows match site={args.site} payload~={args.payload_name}", file=sys.stderr)
        sys.exit(1)
    row = sub.iloc[args.row]
    payload_smiles = row["payload_smiles"]
    if args.name:
        name = args.name
    else:
        name = (f"{row['canonical_payload_name']}_{row['canonical_linker_name']}"
                .replace(" ", "_").replace("/", "_") + f"_{args.row}")

    res = export_adc(payload_smiles, name, Path(args.output_dir))
    print(f"\n=== Export complete ===")
    print(f"ADC name           : {name}")
    print(f"Payload SMILES     : {payload_smiles}")
    print(f"Linker (annotated) : {row['canonical_linker_name']}")
    print(f"Payload atoms      : {res.n_payload_atoms}")
    print(f"Stub atoms         : {res.n_maleimide_atoms}")
    print(f"Payload anchor idx : {res.payload_anchor_1based} (1-based)  "
          f"[reason={res.payload_anchor_reason}]")
    print(f"Stub anchor idx    : {res.maleimide_anchor_1based} (1-based)")
    print(f"SDF written to     : {res.sdf_path}")

    cmd = (
        f"cd {args.difflinker_root} && python -W ignore generate.py \\\n"
        f"  --fragments $(realpath '{res.sdf_path}') \\\n"
        f"  --model {args.ckpt} \\\n"
        f"  --linker_size {args.size_gnn} \\\n"
        f"  --output {args.gen_output} \\\n"
        f"  --n_samples {args.n_samples} \\\n"
        f"  --anchors {res.payload_anchor_1based},{res.maleimide_anchor_1based} \\\n"
        f"  --device cpu"
    )
    print(f"\n=== DiffLinker generation command ===\n{cmd}")


if __name__ == "__main__":
    main()

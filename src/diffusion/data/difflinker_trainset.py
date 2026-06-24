"""Build a DiffLinker fine-tuning dataset from ADC linkers.

Pipeline (refactored to use the dataset-provided ``ADC_SMILES`` directly
instead of re-building the LP from linker + payload via SMILES surgery):

  1. Load the cysteine ADC rows + join with the source splits for the
     ground-truth ``ADC_SMILES`` (LP).
  2. Derive the payload/linker boundary via
     :func:`diffusion.data.payload_anchor.derive_payload_boundary` — anchored
     on the dataset payload SMILES, no SMARTS guessing.
  3. Identify the maleimide / succinimide stub by running SMARTS over the
     LP and accepting only matches whose atoms lie ENTIRELY inside the
     ``linker_atom_idxs`` returned in step 2. Add the alpha-C exocyclic
     neighbour of the ring N to the stub set (the future spacer attachment).
  4. The spacer is then ``linker_atoms - stub_atoms``. Verify connectedness.
  5. Embed a 3D conformer of the full LP (ETKDGv2 + MMFF94 / UFF fallback).
  6. Slice (frag, link) from the LP conformer keeping coordinates.
  7. Emit ``<prefix>_frag.sdf`` (payload + stub, 2 components),
     ``<prefix>_link.sdf`` (spacer, 1 component), ``<prefix>_table.csv``.

The legacy SMARTS-surgery helpers (``find_pl_terminal``,
``find_payload_reactive``, ``build_full_adc``, ``process_row``) are kept in
this module because ``difflinker_augment.py`` still imports them. They are
no longer on the main path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Linker payload-side terminal — known patterns, with the SMARTS atom indices
# of (leaving_group_atom, bonding_carbon).
# ---------------------------------------------------------------------------
PAYLOAD_TERMINAL_PATTERNS: List[Tuple[str, str, int, int]] = [
    # ("name",                   "smarts",                        leaving_idx, bonding_idx)
    ("pabc_carbonate_oh",       "c[CH2][#8][#6](=O)[#8;H1;D1]",   5, 3),
    ("terminal_carboxylic_acid", "[#6](=O)[#8;H1;D1]",            2, 0),
]
# terminal_alcohol_sp3 is intentionally not supported: chemistry depends on
# whether the payload exports an acid (ester) or an amine (no attachment), and
# only 2 linkers in our cysteine set use it.


# ---------------------------------------------------------------------------
# Payload reactive atom — same priority list as difflinker_export.py so the
# train-time and inference-time conventions stay in sync.
# ---------------------------------------------------------------------------
PAYLOAD_REACTIVE_SMARTS: List[Tuple[str, str, int]] = [
    ("secondary_aliphatic_amine",
     "[NX3;H1;D2;!$(N-[C,c]=[O,S]);!$(N-[c])]", 0),
    ("primary_aliphatic_amine",
     "[NX3;H2;D1;!$(N-[C,c]=[O,S])]", 0),
    ("aromatic_secondary_amine",
     "[NX3;H1;D2;$(N-[c])]", 0),
    ("aliphatic_hydroxyl",
     "[OX2;H1;D1;$([OH][CX4])]", 0),
    ("phenol",
     "[OX2;H1;D1;$([OH][c])]", 0),
    ("thiol",
     "[SX2;H1;D1]", 0),
]


# ---------------------------------------------------------------------------
# Maleimide stub patterns (ring + the alpha-C bonded to the ring N, outside
# the ring). The SMARTS only finds the ring; the alpha-C is then added by
# topology inspection.
# ---------------------------------------------------------------------------
MALEIMIDE_RING_PATTERNS: List[Tuple[str, str]] = [
    # Succinimide-thioether (post-Michael adduct), 5-membered ring with S.
    ("succinimide_thioether_4c", "[#7]1[#6](=O)[#6]~[#6]([#16;D1,D2])[#6]1=O"),
    # 6-membered glutarimide-thioether variant.
    ("succinimide_thioether_5c", "[#7]1[#6](=O)[#6][#6]~[#6]([#16;D1,D2])[#6]1=O"),
    # Free maleimide ring (intact C=C, no thiol yet).
    ("free_maleimide", "[#7;X3]1[#6](=O)[#6]=,:[#6][#6]1=O"),
]


# ---------------------------------------------------------------------------
# Result / failure types
# ---------------------------------------------------------------------------
@dataclass
class RowResult:
    canonical_linker_name: str
    canonical_payload_name: str
    payload_smiles: str
    linker_smiles_original: str

    # Filled on success
    full_adc_smiles: Optional[str] = None
    stub_anchor_0based: Optional[int] = None     # in frag combined mol
    payload_anchor_0based: Optional[int] = None  # in frag combined mol
    n_stub_atoms: int = 0
    n_payload_atoms: int = 0
    n_spacer_atoms: int = 0
    pattern_pl: Optional[str] = None
    pattern_payload_reactive: Optional[str] = None
    pattern_stub: Optional[str] = None

    # Failure reason (None on success)
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_pl_terminal(linker: Chem.Mol) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Return (pattern_name, leaving_atom_idx, bonding_atom_idx) or (None, None, None)."""
    for name, smarts, leaving, bonding in PAYLOAD_TERMINAL_PATTERNS:
        q = Chem.MolFromSmarts(smarts)
        match = linker.GetSubstructMatch(q)
        if match:
            return name, match[leaving], match[bonding]
    return None, None, None


def find_payload_reactive(payload: Chem.Mol) -> Tuple[Optional[str], Optional[int]]:
    for name, smarts, sub_idx in PAYLOAD_REACTIVE_SMARTS:
        q = Chem.MolFromSmarts(smarts)
        match = payload.GetSubstructMatch(q)
        if match:
            return name, match[sub_idx]
    return None, None


def find_maleimide_stub_atoms(linker: Chem.Mol
                               ) -> Tuple[Optional[str], Optional[List[int]]]:
    """Return (pattern_name, [stub_atom_indices including ring + alpha-C])."""
    for name, smarts in MALEIMIDE_RING_PATTERNS:
        q = Chem.MolFromSmarts(smarts)
        match = linker.GetSubstructMatch(q)
        if not match:
            continue
        ring_atoms = list(match)
        # The first atom of every ring SMARTS above is the N.
        ring_n = ring_atoms[0]
        # alpha-C = the heavy neighbour of ring_n that is NOT in the ring.
        alpha = None
        for nbr in linker.GetAtomWithIdx(ring_n).GetNeighbors():
            if nbr.GetIdx() not in match:
                alpha = nbr.GetIdx()
                break
        if alpha is None:
            continue
        return name, ring_atoms + [alpha]
    return None, None


def build_full_adc(linker: Chem.Mol, payload: Chem.Mol,
                   linker_bonding_idx: int, linker_leaving_idx: int,
                   payload_reactive_idx: int
                   ) -> Tuple[Chem.Mol, Dict[int, int], Dict[int, int]]:
    """Combine the two mols and form the inter-fragment bond.

    Returns (full_mol, linker_old2new, payload_old2new) where the dicts map
    pre-merge atom indices to the indices in the resulting combined mol.
    Note: ``linker_leaving_idx`` is removed from the final mol, so it is
    absent from ``linker_old2new``.
    """
    n_linker = linker.GetNumAtoms()
    n_payload = payload.GetNumAtoms()

    combined = Chem.RWMol(Chem.CombineMols(linker, payload))
    payload_offset = n_linker
    payload_new = payload_reactive_idx + payload_offset

    # Add the linker-payload bond before deleting the leaving group.
    combined.AddBond(linker_bonding_idx, payload_new, order=Chem.BondType.SINGLE)

    # Remove the leaving-group atom. After RemoveAtom, atoms with index > the
    # removed one shift down by 1.
    combined.RemoveAtom(linker_leaving_idx)

    def shift(old_idx, offset):
        new = old_idx + offset
        if new > linker_leaving_idx:
            new -= 1
        elif new == linker_leaving_idx:
            return None
        return new

    linker_old2new = {}
    for i in range(n_linker):
        new = shift(i, 0)
        if new is not None:
            linker_old2new[i] = new
    payload_old2new = {}
    for i in range(n_payload):
        new = shift(i, payload_offset)
        if new is not None:
            payload_old2new[i] = new

    final = combined.GetMol()
    Chem.SanitizeMol(final)
    return final, linker_old2new, payload_old2new


def embed_3d(mol: Chem.Mol, seed: int = 0
             ) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Return (mol_with_3d_coords, failure_reason_or_None). Always strips Hs
    before returning so that atom indices match the input."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None, "etkdg_failed"
    try:
        rc = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        rc = -1
    if rc != 0:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            pass  # unoptimised geometry is still usable
    return Chem.RemoveHs(mol_h), None


def extract_subgraph(full: Chem.Mol, atom_indices: List[int],
                     conf_id: int = -1) -> Chem.Mol:
    """Carve out a subset of atoms from a full mol, preserving the 3D coords
    of the requested conformer (``conf_id=-1`` → the mol's default/first
    conformer, matching the RDKit convention)."""
    sub = Chem.RWMol()
    old2new = {}
    for new_idx, old_idx in enumerate(atom_indices):
        old_atom = full.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(old_atom.GetAtomicNum())
        new_atom.SetFormalCharge(old_atom.GetFormalCharge())
        new_atom.SetIsAromatic(old_atom.GetIsAromatic())
        new_atom.SetNoImplicit(False)
        sub.AddAtom(new_atom)
        old2new[old_idx] = new_idx
    for bond in full.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in old2new and b in old2new:
            sub.AddBond(old2new[a], old2new[b], bond.GetBondType())
    full_conf = full.GetConformer(conf_id)
    new_conf = Chem.Conformer(sub.GetNumAtoms())
    for old_idx, new_idx in old2new.items():
        p = full_conf.GetAtomPosition(old_idx)
        new_conf.SetAtomPosition(new_idx, (p.x, p.y, p.z))
    sub.AddConformer(new_conf, assignId=True)
    return sub.GetMol(), old2new


# ---------------------------------------------------------------------------
# LEGACY per-row processing (SMARTS-surgery pipeline)
#
# Kept because ``difflinker_augment.py`` imports the helpers above and uses
# this function as a reference. The new main path is ``process_adc_row``,
# which uses ``payload_anchor.derive_payload_boundary`` and operates on the
# ground-truth ``ADC_SMILES`` directly.
# ---------------------------------------------------------------------------

def process_row(row, seed: int = 0) -> Tuple[RowResult,
                                              Optional[Chem.Mol],
                                              Optional[Chem.Mol]]:
    """Legacy: process one CSV row via SMARTS surgery on linker+payload.

    Returns (RowResult, frag_mol, link_mol) where the mols are None on
    failure. The new main path is :func:`process_adc_row`."""
    res = RowResult(
        canonical_linker_name=str(row.get("canonical_linker_name", "")),
        canonical_payload_name=str(row.get("canonical_payload_name", "")),
        payload_smiles=str(row.get("payload_smiles", "")),
        linker_smiles_original=str(row.get("linker_smiles_original", "")),
    )

    payload = Chem.MolFromSmiles(res.payload_smiles)
    if payload is None:
        res.failure_reason = "payload_parse_fail"
        return res, None, None
    linker = Chem.MolFromSmiles(res.linker_smiles_original)
    if linker is None:
        res.failure_reason = "linker_parse_fail"
        return res, None, None

    pl_name, leaving_idx, bonding_idx = find_pl_terminal(linker)
    if pl_name is None:
        res.failure_reason = "no_pl_terminal_matched"
        return res, None, None
    res.pattern_pl = pl_name

    stub_name, stub_atom_idxs = find_maleimide_stub_atoms(linker)
    if stub_name is None:
        res.failure_reason = "no_maleimide_stub_matched"
        return res, None, None
    res.pattern_stub = stub_name

    payload_reactive_name, payload_reactive_idx = find_payload_reactive(payload)
    if payload_reactive_idx is None:
        res.failure_reason = "no_payload_reactive_group"
        return res, None, None
    res.pattern_payload_reactive = payload_reactive_name

    # Build the full ADC
    try:
        full, linker_map, payload_map = build_full_adc(
            linker, payload,
            linker_bonding_idx=bonding_idx,
            linker_leaving_idx=leaving_idx,
            payload_reactive_idx=payload_reactive_idx,
        )
    except Exception as e:
        res.failure_reason = f"adc_build_fail:{e.__class__.__name__}"
        return res, None, None
    res.full_adc_smiles = Chem.MolToSmiles(full)

    # Embed
    embedded, fail_reason = embed_3d(full, seed=seed)
    if embedded is None:
        res.failure_reason = fail_reason
        return res, None, None

    # Re-sanity check that atom indices survived AddHs/RemoveHs (they should
    # because RemoveHs preserves heavy-atom order). Verify atom count matches.
    if embedded.GetNumAtoms() != full.GetNumAtoms():
        res.failure_reason = "atom_count_mismatch_after_embed"
        return res, None, None

    # Map stub / payload / spacer indices on the embedded mol.
    # The last entry of stub_atom_idxs is, by construction in
    # find_maleimide_stub_atoms, the alpha-C (the only stub atom that the
    # spacer attaches to). Track it explicitly: when the ring carries a Cys-S
    # adduct drawn with extra heavy atoms (e.g. the CL2A variant where S is
    # D2), several stub atoms would otherwise look like "anchors", and the
    # first-match loop below would pick the wrong one.
    stub_atoms_full = [linker_map[i] for i in stub_atom_idxs if i in linker_map]
    stub_alpha_c_full = linker_map.get(stub_atom_idxs[-1])
    payload_atoms_full = [payload_map[i] for i in range(payload.GetNumAtoms())
                          if i in payload_map]
    all_linker_atoms_full = set(linker_map.values())
    spacer_atoms_full = sorted(all_linker_atoms_full - set(stub_atoms_full))

    if not stub_atoms_full or not payload_atoms_full or not spacer_atoms_full:
        res.failure_reason = "partition_empty"
        return res, None, None

    # Guard: the spacer subgraph must be connected. If it is not, the maleimide
    # SMARTS likely matched the wrong ring (e.g. a heterocycle in the body of
    # the linker rather than the actual maleimide) — refuse silently.
    spacer_sub_mol, _ = extract_subgraph(
        Chem.MolFromSmiles(Chem.MolToSmiles(full)), spacer_atoms_full
    ) if False else (None, None)  # avoid the round-trip; check on `embedded`
    spacer_only_mol = Chem.RWMol(embedded)
    keep = set(spacer_atoms_full)
    for idx in sorted(set(range(embedded.GetNumAtoms())) - keep, reverse=True):
        spacer_only_mol.RemoveAtom(idx)
    spacer_only_mol = spacer_only_mol.GetMol()
    if len(Chem.GetMolFrags(spacer_only_mol)) != 1:
        res.failure_reason = "spacer_disconnected"
        return res, None, None

    res.n_stub_atoms = len(stub_atoms_full)
    res.n_payload_atoms = len(payload_atoms_full)
    res.n_spacer_atoms = len(spacer_atoms_full)

    # Build the _frag mol (payload first, then stub) and the _link mol
    frag_indices = payload_atoms_full + stub_atoms_full
    frag_mol, frag_old2new = extract_subgraph(embedded, frag_indices)
    link_mol, link_old2new = extract_subgraph(embedded, spacer_atoms_full)

    # Compute anchor indices in the frag mol (0-based, in GEOM multifrag style)
    # Anchor = the atom of each fragment that connects to the linker. We
    # identify it as the fragment atom that is bonded to a spacer atom in the
    # original embedded mol.
    spacer_set = set(spacer_atoms_full)
    payload_anchor_old = None
    for old_idx in payload_atoms_full:
        for nbr in embedded.GetAtomWithIdx(old_idx).GetNeighbors():
            if nbr.GetIdx() in spacer_set:
                payload_anchor_old = old_idx
                break
        if payload_anchor_old is not None:
            break
    # Stub anchor MUST be the alpha-C (the atom from which the spacer was
    # grown). Picking any other stub atom that happens to border the spacer
    # would be a bug — sanity check it.
    stub_anchor_old = stub_alpha_c_full
    if stub_anchor_old is None or stub_anchor_old not in stub_atoms_full:
        res.failure_reason = "stub_anchor_missing"
        return res, None, None
    has_spacer_nbr = any(nbr.GetIdx() in spacer_set
                         for nbr in embedded.GetAtomWithIdx(stub_anchor_old).GetNeighbors())
    if not has_spacer_nbr:
        res.failure_reason = "stub_anchor_not_bordering_spacer"
        return res, None, None
    if payload_anchor_old is None:
        res.failure_reason = "anchor_lookup_failed"
        return res, None, None
    res.payload_anchor_0based = frag_old2new[payload_anchor_old]
    res.stub_anchor_0based = frag_old2new[stub_anchor_old]

    return res, frag_mol, link_mol


# ---------------------------------------------------------------------------
# NEW main path: payload-anchored decomposition
# ---------------------------------------------------------------------------

@dataclass
class AdcRowResult:
    """Per-row result from the payload-anchored pipeline."""
    canonical_linker_name: str
    canonical_payload_name: str
    payload_smiles: str
    adc_smiles: str

    # Filled on success
    full_adc_smiles_canonical: Optional[str] = None
    payload_anchor_0based: Optional[int] = None  # in frag combined mol
    stub_anchor_0based: Optional[int] = None     # in frag combined mol
    n_stub_atoms: int = 0
    n_payload_atoms: int = 0
    n_spacer_atoms: int = 0
    pattern_stub: Optional[str] = None
    boundary_method: Optional[str] = None         # exact_unique | exact_min_junctions | mcs
    site_final: Optional[str] = None              # cysteine | click_chemistry | lysine

    failure_reason: Optional[str] = None


def find_maleimide_stub_in_linker_subset(
        lp: Chem.Mol, linker_atom_set: set
        ) -> Tuple[Optional[str], Optional[List[int]]]:
    """Locate a maleimide / succinimide stub among the ``linker_atom_set``
    atoms of the LP.

    Unlike :func:`find_maleimide_stub_atoms` which is run on a standalone
    linker mol, this variant filters substructure matches to those lying
    entirely within the known linker atom set. This is the robust replacement
    for the SMARTS-on-isolated-linker approach: we know which atoms are the
    linker (from :func:`payload_anchor.derive_payload_boundary`), so any
    pattern hit outside that set is by definition a wrong match.

    Returns ``(pattern_name, [ring_atoms..., alpha_C])`` or ``(None, None)``.
    """
    for name, smarts in MALEIMIDE_RING_PATTERNS:
        q = Chem.MolFromSmarts(smarts)
        for match in lp.GetSubstructMatches(q, uniquify=True):
            ring_atoms = list(match)
            if not all(a in linker_atom_set for a in ring_atoms):
                continue  # the ring strays into payload territory — wrong hit
            ring_n = ring_atoms[0]  # all our ring SMARTS start with the N
            alpha = None
            for nbr in lp.GetAtomWithIdx(ring_n).GetNeighbors():
                idx = nbr.GetIdx()
                if idx in ring_atoms:
                    continue
                if idx not in linker_atom_set:
                    continue  # the N's exocyclic neighbour is payload — wrong hit
                alpha = idx
                break
            if alpha is None:
                continue
            return name, ring_atoms + [alpha]
    return None, None


# ---------------------------------------------------------------------------
# Click-chemistry and lysine stub detection.
#
# Unlike the maleimide (which sits at the antibody terminus), the click handle
# can be INTERNAL (e.g. a DBCO triazole with a cyclooctane on the antibody
# side) and lysine linkers carry several amides. So we cannot just take "the
# matched ring" as the stub — we must split the linker at the reactive group
# into (antibody-side stub, payload-side spacer) using the graph and the known
# payload junction. ``_split_stub_spacer`` does exactly that and returns the
# stub atom list with the spacer-bordering anchor atom LAST (matching the
# maleimide convention used by ``process_adc_row``).
# ---------------------------------------------------------------------------
CLICK_TRIAZOLE_PATTERNS: List[Tuple[str, str]] = [
    ("triazole", "[#6]1~[#6]~[#7]~[#7]~[#7]1"),   # 1,2,3-triazole (aromatic or not)
]
CLICK_ALKYNE_PATTERN = ("alkyne_handle", "[#6]#[#6]")   # unreacted propargyl/cyclooctyne
LYSINE_AMIDE_PATTERNS: List[Tuple[str, str]] = [
    ("amide", "[CX3](=O)[NX3]"),
]


def _linker_reach(lp: Chem.Mol, start: int, blocked: set, linker_atom_set: set) -> set:
    """BFS over linker atoms from ``start``, never entering ``blocked``."""
    import collections
    seen, dq = set(), collections.deque([start])
    while dq:
        x = dq.popleft()
        if x in seen:
            continue
        seen.add(x)
        for nb in lp.GetAtomWithIdx(x).GetNeighbors():
            j = nb.GetIdx()
            if j in linker_atom_set and j not in blocked and j not in seen:
                dq.append(j)
    return seen


def _split_stub_spacer(lp: Chem.Mol, core: List[int], linker_atom_set: set,
                       payload_atom_set: set) -> Optional[List[int]]:
    """Split the linker at the reactive ``core`` into (stub = core + antibody
    side) and (spacer = payload side). The spacer side is the one whose linker
    component is bonded to the payload. Returns the stub atom list with the
    spacer-bordering anchor LAST, or None if the split is ill-defined."""
    core_set = set(core)
    rest = set(linker_atom_set) - core_set
    # Linker atoms bonded to the payload (the payload-side border the spacer
    # must reach). payload_junction is a *payload* atom, so we work via this.
    border = {a for a in linker_atom_set
              if any(nb.GetIdx() in payload_atom_set
                     for nb in lp.GetAtomWithIdx(a).GetNeighbors())}
    if not border:
        return None
    # External (non-core) linker neighbours of the core, with their core atom.
    ext = [(a, nb.GetIdx())
           for a in core_set
           for nb in lp.GetAtomWithIdx(a).GetNeighbors()
           if nb.GetIdx() in rest]
    if not ext:
        return None
    # Spacer side = external neighbour whose component (blocked by core) reaches
    # a payload-bonded border atom.
    spacer_anchor = spacer_nb = spacer_comp = None
    for a, j in ext:
        comp = _linker_reach(lp, j, core_set, linker_atom_set)
        if comp & border:
            spacer_anchor, spacer_nb, spacer_comp = a, j, comp
            break
    if spacer_anchor is None:
        return None
    antibody = rest - spacer_comp           # everything not on the payload side
    stub = (core_set | antibody) - {spacer_anchor}
    if not stub:                            # need at least the anchor in stub
        stub = {spacer_anchor}
        return [spacer_anchor]
    return list(stub) + [spacer_anchor]     # anchor LAST (borders the spacer)


def find_click_stub_in_linker_subset(
        lp: Chem.Mol, linker_atom_set: set, payload_atom_set: set
        ) -> Tuple[Optional[str], Optional[List[int]]]:
    """Click stub: a 1,2,3-triazole (DBCO/BCN+azide product) if present, else an
    unreacted terminal alkyne (propargyl handle). Split into antibody-side stub
    + payload-side spacer via :func:`_split_stub_spacer`."""
    for name, sm in CLICK_TRIAZOLE_PATTERNS + [CLICK_ALKYNE_PATTERN]:
        q = Chem.MolFromSmarts(sm)
        for match in lp.GetSubstructMatches(q, uniquify=True):
            if not all(a in linker_atom_set for a in match):
                continue
            stub = _split_stub_spacer(lp, list(match), linker_atom_set, payload_atom_set)
            if stub:
                return name, stub
    return None, None


def find_lysine_stub_in_linker_subset(
        lp: Chem.Mol, linker_atom_set: set, payload_atom_set: set
        ) -> Tuple[Optional[str], Optional[List[int]]]:
    """Lysine stub: the antibody-terminal amide (NHS-ester aminolysis product).
    The linker carries several amides, so we pick the amide match that lies on
    the ANTIBODY side (its split leaves a non-trivial spacer reaching the
    payload). Among valid candidates, prefer the one whose carbonyl is farthest
    from the payload junction (closest to the antibody terminus)."""
    candidates = []
    for name, sm in LYSINE_AMIDE_PATTERNS:
        q = Chem.MolFromSmarts(sm)
        for match in lp.GetSubstructMatches(q, uniquify=True):
            if not all(a in linker_atom_set for a in match):
                continue
            stub = _split_stub_spacer(lp, list(match), linker_atom_set, payload_atom_set)
            if not stub:
                continue
            # antibody-terminal preference: larger stub (more antibody-side atoms)
            candidates.append((len(stub), name, stub))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)            # largest antibody-side stub first
    _, name, stub = candidates[0]
    return name, stub


def _is_connected_subgraph(mol: Chem.Mol, atom_indices: set) -> bool:
    if not atom_indices:
        return False
    rw = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - atom_indices, reverse=True):
        rw.RemoveAtom(idx)
    return len(Chem.GetMolFrags(rw.GetMol())) == 1


def process_adc_row(row, seed: int = 0
                     ) -> Tuple[AdcRowResult,
                                 Optional[Chem.Mol],
                                 Optional[Chem.Mol]]:
    """Process a single ADC row via the payload-anchored pipeline.

    Expected row fields:
      * ``ADC_SMILES``          — the LP (linker + payload, ground truth)
      * ``payload_smiles``      — the payload alone
      * ``canonical_linker_name``, ``canonical_payload_name``
    """
    # Import here to keep the module's top-level dependency surface lean.
    from .payload_anchor import derive_payload_boundary

    res = AdcRowResult(
        canonical_linker_name=str(row.get("canonical_linker_name", "")),
        canonical_payload_name=str(row.get("canonical_payload_name", "")),
        payload_smiles=str(row.get("payload_smiles", "")),
        adc_smiles=str(row.get("ADC_SMILES", "")),
    )
    if not res.adc_smiles or res.adc_smiles == "nan":
        res.failure_reason = "no_adc_smiles"
        return res, None, None

    # Step 1: ground-truth payload/linker boundary
    boundary = derive_payload_boundary(res.adc_smiles, res.payload_smiles)
    if not boundary.ok:
        res.failure_reason = f"boundary:{boundary.failure_reason}"
        return res, None, None
    res.boundary_method = boundary.method

    lp = Chem.MolFromSmiles(res.adc_smiles)
    linker_atom_set = set(boundary.linker_atom_idxs)
    payload_atom_set = set(boundary.payload_atom_idxs)

    # Step 2: stub detection, dispatched by conjugation site, restricted to
    # linker atoms only.
    site = str(row.get("site_final", "cysteine"))
    res.site_final = site
    if site == "cysteine":
        stub_name, stub_atom_idxs = find_maleimide_stub_in_linker_subset(
            lp, linker_atom_set)
    elif site == "click_chemistry":
        stub_name, stub_atom_idxs = find_click_stub_in_linker_subset(
            lp, linker_atom_set, payload_atom_set)
    elif site == "lysine":
        stub_name, stub_atom_idxs = find_lysine_stub_in_linker_subset(
            lp, linker_atom_set, payload_atom_set)
    else:
        res.failure_reason = f"unsupported_site:{site}"
        return res, None, None
    if stub_name is None:
        res.failure_reason = f"no_stub_in_linker:{site}"
        return res, None, None
    res.pattern_stub = stub_name
    stub_atoms = set(stub_atom_idxs)
    stub_alpha_c = stub_atom_idxs[-1]  # by construction

    # Step 3: spacer = linker atoms minus stub atoms
    spacer_atoms = linker_atom_set - stub_atoms
    if not spacer_atoms:
        res.failure_reason = "empty_spacer"
        return res, None, None
    if not _is_connected_subgraph(lp, spacer_atoms):
        res.failure_reason = "spacer_disconnected"
        return res, None, None

    # Sanity: payload | stub | spacer must exactly cover the LP
    if (payload_atom_set | stub_atoms | spacer_atoms
            != set(range(lp.GetNumAtoms()))):
        res.failure_reason = "partition_incomplete"
        return res, None, None
    if (payload_atom_set & stub_atoms) or (payload_atom_set & spacer_atoms) \
            or (stub_atoms & spacer_atoms):
        res.failure_reason = "partition_overlap"
        return res, None, None

    res.n_stub_atoms = len(stub_atoms)
    res.n_payload_atoms = len(payload_atom_set)
    res.n_spacer_atoms = len(spacer_atoms)

    # Step 4: 3D conformer of the FULL LP (single embedding here; the multi-
    # conformer augmentation is done downstream in difflinker_augment.py).
    embedded, fail_reason = embed_3d(lp, seed=seed)
    if embedded is None:
        res.failure_reason = fail_reason
        return res, None, None
    if embedded.GetNumAtoms() != lp.GetNumAtoms():
        res.failure_reason = "atom_count_mismatch_after_embed"
        return res, None, None
    res.full_adc_smiles_canonical = Chem.MolToSmiles(embedded)

    # Step 5: slice into (frag = payload + stub, link = spacer), keeping coords
    frag_atom_order = sorted(payload_atom_set) + sorted(stub_atoms)
    frag_mol, frag_old2new = extract_subgraph(embedded, frag_atom_order)
    link_mol, _ = extract_subgraph(embedded, sorted(spacer_atoms))

    if len(Chem.GetMolFrags(frag_mol)) != 2:
        res.failure_reason = "frag_components_not_2"
        return res, None, None
    if len(Chem.GetMolFrags(link_mol)) != 1:
        res.failure_reason = "link_not_connected"
        return res, None, None

    # Step 6: anchor indices in the frag mol (GEOM multifrag style, 0-based)
    # Payload anchor = the payload atom of the boundary's payload-side junction
    # Stub anchor    = the alpha-C (deterministic by construction)
    pa_old = boundary.payload_junction_atom
    sa_old = stub_alpha_c
    if pa_old not in frag_old2new or sa_old not in frag_old2new:
        res.failure_reason = "anchor_mapping_failed"
        return res, None, None
    res.payload_anchor_0based = frag_old2new[pa_old]
    res.stub_anchor_0based = frag_old2new[sa_old]

    return res, frag_mol, link_mol


# ---------------------------------------------------------------------------
# Data loading: join cys QC rows with ADC_SMILES from the source splits
# ---------------------------------------------------------------------------

def load_cys_adcs_with_lp(qc_csv: Path,
                          site: str = "cysteine",
                          include_flagged: bool = False
                          ) -> pd.DataFrame:
    """Return cys rows joined with their ground-truth ``ADC_SMILES``.

    If ``include_flagged=True``, also pull rows from
    ``linkers_qc_flagged.csv`` (same dir as ``qc_csv``). Useful to identify
    linkers the OLD SMARTS pipeline rejected as ``fragment_too_small`` /
    ``site_terminal_mismatch`` that the new boundary recovers.
    """
    qc = pd.read_csv(qc_csv)
    qc = qc[qc["site_final"] == site].copy()
    qc["_source_qc"] = "passed"

    base = qc_csv.parent
    if include_flagged:
        flagged_csv = base / "linkers_qc_flagged.csv"
        if flagged_csv.exists():
            fl = pd.read_csv(flagged_csv)
            # site_normalized or site_final, depending on the QC version.
            site_col = "site_final" if "site_final" in fl.columns else "site_normalized"
            fl = fl[fl[site_col] == site].copy()
            fl["_source_qc"] = "flagged"
            qc = pd.concat([qc, fl], ignore_index=True)

    # Guarantee a site_final column for the dispatch (flagged rows may only
    # carry site_normalized). We filtered to this site, so force it.
    qc["site_final"] = site

    parts = []
    for s in ("train", "val", "test"):
        p = base / f"{s}_split.csv"
        if p.exists():
            parts.append(pd.read_csv(p, usecols=[
                "ADC_SMILES", "ADC_Linker_SMILES", "ADC_Payload_SMILES"
            ], low_memory=False))
    ts = pd.concat(parts).drop_duplicates().reset_index(drop=True)

    merged = qc.merge(
        ts.rename(columns={"ADC_Linker_SMILES": "linker_smiles_original",
                            "ADC_Payload_SMILES": "payload_smiles"}),
        on=["linker_smiles_original", "payload_smiles"], how="left",
    )
    return merged


# ---------------------------------------------------------------------------
# Top-level dataset builder (NEW)
# ---------------------------------------------------------------------------

def build_dataset_adc_anchored(df: pd.DataFrame, out_dir: Path,
                                prefix: str = "adc_cys",
                                seed: int = 0):
    """New main builder. Uses :func:`process_adc_row` per row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_path = out_dir / f"{prefix}_frag.sdf"
    link_path = out_dir / f"{prefix}_link.sdf"
    table_path = out_dir / f"{prefix}_table.csv"

    frag_writer = Chem.SDWriter(str(frag_path))
    link_writer = Chem.SDWriter(str(link_path))

    rows_out, failures = [], []
    for _, row in df.iterrows():
        res, frag_mol, link_mol = process_adc_row(row, seed=seed)
        if res.failure_reason is not None:
            failures.append((res, row))
            continue
        uuid = f"{prefix}_{len(rows_out):03d}"
        frag_mol.SetProp("_Name", uuid)
        link_mol.SetProp("_Name", uuid)
        # SDWriter kekulizes before writing. A fragment cut may leave aromatic
        # rings half-formed → KekulizeException. Try once with sanitize, then
        # bail out for that row instead of crashing the whole build.
        try:
            for mol in (frag_mol, link_mol):
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass  # use as-is; SDWriter may still succeed
            frag_writer.write(frag_mol)
            link_writer.write(link_mol)
        except Exception as e:
            res.failure_reason = f"sdf_write_fail:{e.__class__.__name__}"
            failures.append((res, row))
            continue
        rows_out.append({
            "uuid": uuid,
            "molecule": uuid,
            "canonical_linker_name": res.canonical_linker_name,
            "canonical_payload_name": res.canonical_payload_name,
            "anchors": f"{res.payload_anchor_0based}-{res.stub_anchor_0based}",
            "anchor_1": res.payload_anchor_0based,
            "anchor_2": res.stub_anchor_0based,
            "n_stub_atoms": res.n_stub_atoms,
            "n_payload_atoms": res.n_payload_atoms,
            "n_spacer_atoms": res.n_spacer_atoms,
            "pattern_stub": res.pattern_stub,
            "boundary_method": res.boundary_method,
            "site_final": res.site_final,
            "adc_smiles": res.adc_smiles,
            "full_adc_smiles_canonical": res.full_adc_smiles_canonical,
            "_source_qc": str(row.get("_source_qc", "")),
        })

    frag_writer.close()
    link_writer.close()
    pd.DataFrame(rows_out).to_csv(table_path, index=False)
    return rows_out, failures


def print_report_adc(rows_out, failures, stream=sys.stdout):
    total = len(rows_out) + len(failures)
    print(f"\n=== DiffLinker trainset build (payload-anchored) ===", file=stream)
    print(f"Cys ADC rows processed : {total}", file=stream)
    print(f"Successful trios       : {len(rows_out)}  ({len(rows_out)/total:.1%})",
          file=stream)
    print(f"Failed                 : {len(failures)}", file=stream)

    if rows_out:
        from collections import Counter
        methods = Counter(r["boundary_method"] for r in rows_out)
        sources = Counter(r["_source_qc"] for r in rows_out)
        print(f"\nBoundary method distribution:", file=stream)
        for k, v in methods.most_common():
            print(f"  {v:4d}  {k}", file=stream)
        print(f"\nSource QC bucket distribution:", file=stream)
        for k, v in sources.most_common():
            print(f"  {v:4d}  {k}", file=stream)

    if failures:
        from collections import Counter
        reasons = Counter(r.failure_reason for r, _ in failures)
        print(f"\nFailures by reason:", file=stream)
        for k, v in reasons.most_common():
            print(f"  {v:4d}  {k}", file=stream)
        print(f"\nFailed rows:", file=stream)
        for r, row in failures:
            src = str(row.get("_source_qc", "?"))
            print(f"  [{r.failure_reason:30s}] [{src}] "
                  f"{r.canonical_linker_name[:32]:32s} + {r.canonical_payload_name[:24]}",
                  file=stream)


# ---------------------------------------------------------------------------
# Legacy top-level dataset builder (kept for the test fixtures)
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame, out_dir: Path, prefix: str = "adc_cys",
                  seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_path = out_dir / f"{prefix}_frag.sdf"
    link_path = out_dir / f"{prefix}_link.sdf"
    table_path = out_dir / f"{prefix}_table.csv"

    frag_writer = Chem.SDWriter(str(frag_path))
    link_writer = Chem.SDWriter(str(link_path))

    rows_out = []
    failures = []
    for idx, row in df.iterrows():
        res, frag_mol, link_mol = process_row(row, seed=seed)
        if res.failure_reason is not None:
            failures.append(res)
            continue
        uuid = f"{prefix}_{len(rows_out):03d}"
        frag_mol.SetProp("_Name", uuid)
        link_mol.SetProp("_Name", uuid)
        frag_writer.write(frag_mol)
        link_writer.write(link_mol)
        rows_out.append({
            "uuid": uuid,
            "canonical_linker_name": res.canonical_linker_name,
            "canonical_payload_name": res.canonical_payload_name,
            "anchors": f"{res.payload_anchor_0based}-{res.stub_anchor_0based}",
            "anchor_1": res.payload_anchor_0based,
            "anchor_2": res.stub_anchor_0based,
            "n_stub_atoms": res.n_stub_atoms,
            "n_payload_atoms": res.n_payload_atoms,
            "n_spacer_atoms": res.n_spacer_atoms,
            "pattern_stub": res.pattern_stub,
            "pattern_pl_terminal": res.pattern_pl,
            "pattern_payload_reactive": res.pattern_payload_reactive,
            "full_adc_smiles": res.full_adc_smiles,
        })

    frag_writer.close()
    link_writer.close()
    pd.DataFrame(rows_out).to_csv(table_path, index=False)
    return rows_out, failures


def print_report(rows_out, failures, stream=sys.stdout):
    total = len(rows_out) + len(failures)
    print(f"\n=== DiffLinker trainset build report ===", file=stream)
    print(f"Total cysteine rows : {total}", file=stream)
    print(f"Successful          : {len(rows_out)} ({len(rows_out)/total:.1%})",
          file=stream)
    print(f"Failed              : {len(failures)}", file=stream)
    if failures:
        from collections import Counter
        reasons = Counter(f.failure_reason for f in failures)
        print(f"Failures by reason:", file=stream)
        for reason, n in reasons.most_common():
            print(f"  {n:4d}  {reason}", file=stream)
        print(f"\nFailed rows:", file=stream)
        for f in failures:
            print(f"  [{f.failure_reason:30s}] {f.canonical_linker_name[:32]:32s} "
                  f"+ {f.canonical_payload_name[:24]}", file=stream)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-csv", default="data/processed/linkers_qc_passed.csv")
    ap.add_argument("--out-dir", default="data/processed/difflinker_trainset_v2")
    ap.add_argument("--prefix", default="adc_cys")
    ap.add_argument("--site", default="cysteine",
                    help="Single site (legacy). Ignored if --sites is given.")
    ap.add_argument("--sites", default=None,
                    help="Comma-separated sites, or 'all'. "
                         "Choices: cysteine, click, lysine. "
                         "Overrides --site. Default keeps --site (cysteine).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-flagged", action="store_true",
                    help="Also include rows from linkers_qc_flagged.csv. The "
                         "new boundary may recover linkers the old SMARTS "
                         "pipeline rejected as fragment_too_small.")
    args = ap.parse_args()

    # Resolve the site list. 'click' is shorthand for the CSV's
    # 'click_chemistry' value.
    SITE_ALIAS = {"click": "click_chemistry", "click_chemistry": "click_chemistry",
                  "cysteine": "cysteine", "lysine": "lysine"}
    if args.sites:
        if args.sites.strip() == "all":
            site_list = ["cysteine", "click_chemistry", "lysine"]
        else:
            site_list = [SITE_ALIAS.get(s.strip(), s.strip())
                         for s in args.sites.split(",")]
    else:
        site_list = [args.site]

    df = pd.concat(
        [load_cys_adcs_with_lp(Path(args.qc_csv), site=s,
                               include_flagged=args.include_flagged)
         for s in site_list],
        ignore_index=True,
    )
    log.info("Loaded %d rows for sites=%s (include_flagged=%s)",
             len(df), site_list, args.include_flagged)

    rows_out, failures = build_dataset_adc_anchored(
        df, Path(args.out_dir), prefix=args.prefix, seed=args.seed,
    )
    print_report_adc(rows_out, failures)
    print(f"\nWrote {args.out_dir}/{args.prefix}_frag.sdf  ({len(rows_out)} mols)")
    print(f"Wrote {args.out_dir}/{args.prefix}_link.sdf  ({len(rows_out)} mols)")
    print(f"Wrote {args.out_dir}/{args.prefix}_table.csv ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()

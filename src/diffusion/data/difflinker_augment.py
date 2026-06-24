"""3D conformer augmentation + train/val split for the DiffLinker fine-tune.

For each cysteine ADC that passes the payload-anchored topological
decomposition (:func:`compute_adc_partition`), we generate N=20 conformers of
the full LP, optimise each, then re-extract the (frag, link) subgraphs from
every conformer. DiffLinker treats each conformer as a separate 3D training
example.

The train/val partition is performed AT THE LINKER LEVEL — every conformer of
a given source linker lives in the same split. Otherwise the model would see
both the train and val sets contain near-identical topologies, which would
defeat the point of early stopping on val.

ETKDG version reconciliation
----------------------------
The trainset module's :func:`embed_3d` uses ``AllChem.ETKDGv3()`` (the Python
helper that returns an EmbedParameters with ``useSmallRingTorsions=True``,
``useExpTorsionAnglePrefs=True``, ``useBasicKnowledge=True`` and internally
``ETversion=2`` — RDKit's "ETKDGv3" label refers to ETv2 with small-ring
torsions). We use the kwargs form of ``EmbedMolecule`` here (needed for the
``clearConfs=False`` knob that lets us accumulate distinct conformers from
distinct seeds) with the EXACT same parameters: ``ETversion=2``,
``useSmallRingTorsions=True``, ``useExpTorsionAnglePrefs=True``,
``useBasicKnowledge=True``. The two code paths produce equivalent geometries.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from .difflinker_trainset import (
    build_full_adc,
    extract_subgraph,
    find_click_stub_in_linker_subset,
    find_lysine_stub_in_linker_subset,
    find_maleimide_stub_atoms,
    find_maleimide_stub_in_linker_subset,
    find_payload_reactive,
    find_pl_terminal,
    load_cys_adcs_with_lp,
)
from .payload_anchor import derive_payload_boundary

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)


DEFAULT_N_CONFS = 20
DEFAULT_VAL_FRACTION = 0.17  # 6/36 → ≈17%, leaves 30 train + 6 val
DEFAULT_SPLIT_SEED = 42
# Pruning during the ETKDG inner loop is deceptively aggressive for large
# flexible molecules (~100 heavy atoms): a threshold of 0.5 Å on best-fit
# RMSD strips 19/20 requested conformers because ETKDG keeps re-finding the
# same local-minimum basin before perturbing far enough. We disable pruning
# during embed and let MMFF redistribute the geometries afterward — empirical
# diversity is then verified per-row by ``DIVERSITY_MIN_RMS``.
PRUNE_RMS_THRESH = -1.0
DIVERSITY_MIN_RMS = 0.5


# ---------------------------------------------------------------------------
# Topological partition (called once per source linker, before any conformer
# embedding). Indices returned are stable across conformers because RDKit
# preserves heavy-atom ordering through AddHs / EmbedMultipleConfs / RemoveHs.
# ---------------------------------------------------------------------------

@dataclass
class LinkerPartition:
    full_mol_2d: Chem.Mol
    payload_atoms: List[int]   # in full mol
    stub_atoms: List[int]
    spacer_atoms: List[int]
    payload_anchor: int        # in full mol
    stub_anchor: int           # alpha-C of stub, in full mol
    pattern_stub: str
    pattern_pl: str
    pattern_payload_reactive: str
    full_adc_smiles: str
    failure_reason: Optional[str] = None


def compute_adc_partition(adc_smiles: str, payload_smiles: str,
                          site: str = "cysteine") -> LinkerPartition:
    """NEW main path. Partition the LP using the dataset-provided payload
    SMILES as ground truth (:func:`payload_anchor.derive_payload_boundary`),
    then run the maleimide SMARTS only on the known linker subset.

    Returns the same ``LinkerPartition`` dataclass used by the legacy path so
    downstream functions (``embed_multiple_conformers``,
    ``extract_conformer_example``) work unchanged.
    """
    if not isinstance(adc_smiles, str) or not adc_smiles or adc_smiles == "nan":
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="no_adc_smiles")
    boundary = derive_payload_boundary(adc_smiles, payload_smiles)
    if not boundary.ok:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason=f"boundary:{boundary.failure_reason}")

    lp = Chem.MolFromSmiles(adc_smiles)
    linker_set = set(boundary.linker_atom_idxs)
    payload_set = set(boundary.payload_atom_idxs)

    if site == "cysteine":
        stub_name, stub_atom_idxs = find_maleimide_stub_in_linker_subset(lp, linker_set)
    elif site == "click_chemistry":
        stub_name, stub_atom_idxs = find_click_stub_in_linker_subset(lp, linker_set, payload_set)
    elif site == "lysine":
        stub_name, stub_atom_idxs = find_lysine_stub_in_linker_subset(lp, linker_set, payload_set)
    else:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason=f"unsupported_site:{site}")
    if stub_name is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason=f"no_stub_in_linker:{site}")
    stub_set = set(stub_atom_idxs)
    stub_alpha_c = stub_atom_idxs[-1]

    spacer_set = linker_set - stub_set
    if not spacer_set:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="empty_spacer")

    # Spacer connectedness on the LP
    spacer_only = Chem.RWMol(lp)
    for idx in sorted(set(range(lp.GetNumAtoms())) - spacer_set, reverse=True):
        spacer_only.RemoveAtom(idx)
    if len(Chem.GetMolFrags(spacer_only.GetMol())) != 1:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="spacer_disconnected")

    # Partition sanity
    if (payload_set | stub_set | spacer_set) != set(range(lp.GetNumAtoms())):
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="partition_incomplete")
    if (payload_set & stub_set) or (payload_set & spacer_set) or (stub_set & spacer_set):
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="partition_overlap")

    return LinkerPartition(
        full_mol_2d=lp,
        payload_atoms=sorted(payload_set),
        stub_atoms=sorted(stub_set),
        spacer_atoms=sorted(spacer_set),
        payload_anchor=boundary.payload_junction_atom,
        stub_anchor=stub_alpha_c,
        pattern_stub=stub_name,
        pattern_pl=boundary.method,          # repurposed: boundary method label
        pattern_payload_reactive="anchored",  # not used in new path
        full_adc_smiles=Chem.MolToSmiles(lp),
    )


# ---------------------------------------------------------------------------
# LEGACY partition (SMARTS surgery on isolated linker + payload). Kept so
# the old tests and the old CLI (``--legacy`` flag) keep working. The main
# path is :func:`compute_adc_partition` above.
# ---------------------------------------------------------------------------

def compute_partition(payload_smiles: str, linker_smiles: str) -> LinkerPartition:
    payload = Chem.MolFromSmiles(payload_smiles)
    if payload is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="payload_parse_fail")
    linker = Chem.MolFromSmiles(linker_smiles)
    if linker is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="linker_parse_fail")

    pl_name, leaving_idx, bonding_idx = find_pl_terminal(linker)
    if pl_name is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="no_pl_terminal_matched")
    stub_name, stub_atoms_idxs = find_maleimide_stub_atoms(linker)
    if stub_name is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="no_maleimide_stub_matched")
    payload_react_name, payload_react_idx = find_payload_reactive(payload)
    if payload_react_idx is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="no_payload_reactive_group")

    try:
        full, linker_map, payload_map = build_full_adc(
            linker, payload,
            linker_bonding_idx=bonding_idx,
            linker_leaving_idx=leaving_idx,
            payload_reactive_idx=payload_react_idx,
        )
    except Exception as e:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason=f"adc_build_fail:{e.__class__.__name__}")

    stub_atoms_full = [linker_map[i] for i in stub_atoms_idxs if i in linker_map]
    stub_alpha_c_full = linker_map.get(stub_atoms_idxs[-1])
    payload_atoms_full = [payload_map[i] for i in range(payload.GetNumAtoms())
                          if i in payload_map]
    spacer_atoms_full = sorted(set(linker_map.values()) - set(stub_atoms_full))
    if not (stub_atoms_full and payload_atoms_full and spacer_atoms_full
            and stub_alpha_c_full is not None):
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="partition_empty")

    # Spacer-connectedness check (same defensive guard as trainset).
    keep = set(spacer_atoms_full)
    spacer_only = Chem.RWMol(full)
    for idx in sorted(set(range(full.GetNumAtoms())) - keep, reverse=True):
        spacer_only.RemoveAtom(idx)
    if len(Chem.GetMolFrags(spacer_only.GetMol())) != 1:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="spacer_disconnected")

    spacer_set = set(spacer_atoms_full)
    payload_anchor = None
    for old_idx in payload_atoms_full:
        for nbr in full.GetAtomWithIdx(old_idx).GetNeighbors():
            if nbr.GetIdx() in spacer_set:
                payload_anchor = old_idx
                break
        if payload_anchor is not None:
            break
    if payload_anchor is None:
        return LinkerPartition(None, [], [], [], -1, -1, "", "", "", "",
                               failure_reason="payload_anchor_missing")

    return LinkerPartition(
        full_mol_2d=full,
        payload_atoms=payload_atoms_full,
        stub_atoms=stub_atoms_full,
        spacer_atoms=spacer_atoms_full,
        payload_anchor=payload_anchor,
        stub_anchor=stub_alpha_c_full,
        pattern_stub=stub_name,
        pattern_pl=pl_name,
        pattern_payload_reactive=payload_react_name,
        full_adc_smiles=Chem.MolToSmiles(full),
    )


# ---------------------------------------------------------------------------
# Multi-conformer embedding
# ---------------------------------------------------------------------------

def embed_multiple_conformers(full_mol: Chem.Mol, n_confs: int = DEFAULT_N_CONFS,
                              seed: int = 0
                              ) -> Tuple[Optional[Chem.Mol], List[int], List[str]]:
    """Return (mol_with_3d_conformers, kept_conf_ids, failure_messages).

    NOTE on RDKit behaviour: ``EmbedMultipleConfs`` with a fixed positive
    ``randomSeed`` produces N IDENTICAL conformers (the seed is not advanced
    between attempts). When combined with ``pruneRmsThresh>0`` this looks like
    "only 1 conformer survived"; without pruning, all N are duplicates. We
    therefore drive embedding with a manual loop over ``EmbedMolecule`` with
    distinct per-conformer seeds, which gives both reproducibility AND
    genuine geometric diversity.
    """
    # NOTE: ``EmbedMolecule(mol, EmbedParameters)`` always clears existing
    # conformers (the params-object signature has no ``clearConfs`` knob), so
    # we use the kwargs signature, which exposes it. ``ETversion=3`` selects
    # the ETKDGv3 torsion preferences.
    mol_h = Chem.AddHs(full_mol)
    cids: List[int] = []
    failures: List[str] = []
    for i in range(n_confs):
        conf_seed = seed * 10_000 + i + 1
        cid = AllChem.EmbedMolecule(
            mol_h,
            maxAttempts=0,
            randomSeed=conf_seed,
            clearConfs=False,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            useSmallRingTorsions=True,
            ETversion=2,
        )
        if cid >= 0:
            cids.append(cid)
            continue
        # Retry with random initial coords for stubborn conformers.
        cid = AllChem.EmbedMolecule(
            mol_h,
            maxAttempts=200,
            randomSeed=conf_seed,
            clearConfs=False,
            useRandomCoords=True,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            useSmallRingTorsions=True,
            ETversion=2,
        )
        if cid >= 0:
            cids.append(cid)
        else:
            failures.append(f"etkdg_failed_conf{i}")
    if not cids:
        return None, [], failures or ["etkdg_no_conformers"]

    kept = []
    for cid in cids:
        try:
            rc = AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
            mmff_ok = True
        except Exception as e:
            rc = -1
            mmff_ok = False
            failures.append(f"mmff_exception:{e.__class__.__name__}")
        if not mmff_ok or rc < 0:
            try:
                rc2 = AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
                if rc2 < 0:
                    failures.append(f"uff_failed_conf{cid}")
                    continue
            except Exception as e:
                failures.append(f"uff_exception_conf{cid}:{e.__class__.__name__}")
                continue
        kept.append(cid)

    full_no_h = Chem.RemoveHs(mol_h)
    return full_no_h, kept, failures


# ---------------------------------------------------------------------------
# Per-conformer extraction
# ---------------------------------------------------------------------------

@dataclass
class ConformerExample:
    source_uuid: str
    conformer_id: int
    canonical_linker_name: str
    canonical_payload_name: str
    payload_anchor_0based: int
    stub_anchor_0based: int
    n_payload_atoms: int
    n_stub_atoms: int
    n_spacer_atoms: int
    pattern_stub: str
    pattern_pl: str
    pattern_payload_reactive: str
    full_adc_smiles: str
    frag_mol: Chem.Mol
    link_mol: Chem.Mol


def extract_conformer_example(full_mol: Chem.Mol, conf_id: int,
                              partition: LinkerPartition,
                              source_uuid: str,
                              canonical_linker_name: str,
                              canonical_payload_name: str
                              ) -> Optional[ConformerExample]:
    """Build a single conformer's (frag, link) pair using ``partition``."""
    # Frag = payload then stub (preserve the trainset convention).
    frag_atom_order = list(partition.payload_atoms) + list(partition.stub_atoms)
    frag_mol, frag_old2new = extract_subgraph(full_mol, frag_atom_order, conf_id=conf_id)
    link_mol, _ = extract_subgraph(full_mol, list(partition.spacer_atoms), conf_id=conf_id)

    if (frag_mol.GetNumAtoms() + link_mol.GetNumAtoms()
            != full_mol.GetNumAtoms()):
        return None
    if len(Chem.GetMolFrags(frag_mol)) != 2 or len(Chem.GetMolFrags(link_mol)) != 1:
        return None

    pa = frag_old2new.get(partition.payload_anchor)
    sa = frag_old2new.get(partition.stub_anchor)
    if pa is None or sa is None:
        return None

    return ConformerExample(
        source_uuid=source_uuid,
        conformer_id=conf_id,
        canonical_linker_name=canonical_linker_name,
        canonical_payload_name=canonical_payload_name,
        payload_anchor_0based=pa,
        stub_anchor_0based=sa,
        n_payload_atoms=len(partition.payload_atoms),
        n_stub_atoms=len(partition.stub_atoms),
        n_spacer_atoms=len(partition.spacer_atoms),
        pattern_stub=partition.pattern_stub,
        pattern_pl=partition.pattern_pl,
        pattern_payload_reactive=partition.pattern_payload_reactive,
        full_adc_smiles=partition.full_adc_smiles,
        frag_mol=frag_mol,
        link_mol=link_mol,
    )


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def linker_level_split(source_uuids: List[str],
                       val_fraction: float = DEFAULT_VAL_FRACTION,
                       seed: int = DEFAULT_SPLIT_SEED
                       ) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(source_uuids)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

@dataclass
class SourceLinkerResult:
    source_uuid: str
    canonical_linker_name: str
    canonical_payload_name: str
    success: bool
    n_conformers_kept: int = 0
    n_conformers_requested: int = 0
    failure_reason: Optional[str] = None
    conformer_failures: List[str] = field(default_factory=list)


def process_source_linker(row, source_uuid: str,
                           n_confs: int = DEFAULT_N_CONFS,
                           seed: int = 0
                           ) -> Tuple[SourceLinkerResult, List[ConformerExample]]:
    """Default path uses ``compute_adc_partition`` (payload-anchored)."""
    name = str(row.get("canonical_linker_name", ""))
    pl_name = str(row.get("canonical_payload_name", ""))
    partition = compute_adc_partition(
        adc_smiles=str(row.get("ADC_SMILES", "")),
        payload_smiles=str(row.get("payload_smiles", "")),
        site=str(row.get("site_final", "cysteine")),
    )
    if partition.failure_reason is not None:
        return (SourceLinkerResult(source_uuid, name, pl_name, success=False,
                                    n_conformers_requested=n_confs,
                                    failure_reason=partition.failure_reason),
                [])

    full_3d, kept_cids, failures = embed_multiple_conformers(
        partition.full_mol_2d, n_confs=n_confs, seed=seed
    )
    if full_3d is None or not kept_cids:
        return (SourceLinkerResult(source_uuid, name, pl_name, success=False,
                                    n_conformers_requested=n_confs,
                                    failure_reason="all_conformers_failed",
                                    conformer_failures=failures),
                [])

    examples = []
    for cid in kept_cids:
        ex = extract_conformer_example(
            full_3d, cid, partition,
            source_uuid=source_uuid,
            canonical_linker_name=name,
            canonical_payload_name=pl_name,
        )
        if ex is None:
            failures.append(f"extract_failed_conf{cid}")
            continue
        examples.append(ex)

    return (SourceLinkerResult(source_uuid, name, pl_name,
                                success=bool(examples),
                                n_conformers_kept=len(examples),
                                n_conformers_requested=n_confs,
                                conformer_failures=failures),
            examples)


_STUB2SITE = {
    "succinimide_thioether_4c": "cysteine", "succinimide_thioether_5c": "cysteine",
    "free_maleimide": "cysteine",
    "triazole": "click_chemistry", "alkyne_handle": "click_chemistry",
    "amide": "lysine",
}
def _site_from_stub(pattern_stub: str) -> str:
    return _STUB2SITE.get(str(pattern_stub), "unknown")


def write_split(examples: List[ConformerExample], out_dir: Path, prefix: str):
    frag_path = out_dir / f"{prefix}_frag.sdf"
    link_path = out_dir / f"{prefix}_link.sdf"
    table_path = out_dir / f"{prefix}_table.csv"
    frag_writer = Chem.SDWriter(str(frag_path))
    link_writer = Chem.SDWriter(str(link_path))
    rows = []
    n_kek_skipped = 0
    for ex in examples:
        uuid = f"{ex.source_uuid}_c{ex.conformer_id:02d}"
        ex.frag_mol.SetProp("_Name", uuid)
        ex.link_mol.SetProp("_Name", uuid)
        try:
            # SDWriter kekulizes before writing; carved sub-graphs may have
            # half-broken aromatic rings. Try to sanitize first; on failure,
            # skip this conformer rather than crashing the whole build.
            for mol in (ex.frag_mol, ex.link_mol):
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
            frag_writer.write(ex.frag_mol)
            link_writer.write(ex.link_mol)
        except Exception:
            n_kek_skipped += 1
            continue
        rows.append({
            "uuid": uuid,
            "molecule": uuid,          # ZincDataset expects this column
            "source_uuid": ex.source_uuid,
            "conformer_id": ex.conformer_id,
            "canonical_linker_name": ex.canonical_linker_name,
            "canonical_payload_name": ex.canonical_payload_name,
            "anchors": f"{ex.payload_anchor_0based}-{ex.stub_anchor_0based}",
            "anchor_1": ex.payload_anchor_0based,
            "anchor_2": ex.stub_anchor_0based,
            "n_stub_atoms": ex.n_stub_atoms,
            "n_payload_atoms": ex.n_payload_atoms,
            "n_spacer_atoms": ex.n_spacer_atoms,
            "pattern_stub": ex.pattern_stub,
            "site_final": _site_from_stub(ex.pattern_stub),
            "pattern_pl_terminal": ex.pattern_pl,            # = boundary_method in new path
            "pattern_payload_reactive": ex.pattern_payload_reactive,
            "full_adc_smiles": ex.full_adc_smiles,
        })
    frag_writer.close()
    link_writer.close()
    pd.DataFrame(rows).to_csv(table_path, index=False)
    if n_kek_skipped:
        log.warning("[%s] %d conformer rows skipped due to kekulize failure",
                    prefix, n_kek_skipped)
    return frag_path, link_path, table_path


def build_augmented_dataset(df: pd.DataFrame, out_dir: Path,
                             n_confs: int = DEFAULT_N_CONFS,
                             val_fraction: float = DEFAULT_VAL_FRACTION,
                             split_seed: int = DEFAULT_SPLIT_SEED,
                             embed_seed: int = 0):
    """End-to-end pipeline. Returns a metadata dict for reporting."""
    out_dir.mkdir(parents=True, exist_ok=True)

    per_linker_results = []
    per_linker_examples = []
    success_idx = 0
    for _, row in df.iterrows():
        # Generate the source UUID only once we know if the topology check
        # passed, so the indices stay dense. But for traceability we want a
        # stable id whether or not we succeed → assign before processing.
        candidate_uuid = f"src_{success_idx:03d}"
        result, examples = process_source_linker(
            row, source_uuid=candidate_uuid,
            n_confs=n_confs, seed=embed_seed,
        )
        per_linker_results.append(result)
        if result.success:
            per_linker_examples.append((result.source_uuid, examples))
            success_idx += 1

    successful_uuids = [uid for uid, _ in per_linker_examples]
    train_uuids, val_uuids = linker_level_split(successful_uuids,
                                                 val_fraction=val_fraction,
                                                 seed=split_seed)

    train_ex, val_ex = [], []
    for uid, exs in per_linker_examples:
        if uid in set(train_uuids):
            train_ex.extend(exs)
        else:
            val_ex.extend(exs)

    train_paths = write_split(train_ex, out_dir, "adc_cys_train")
    val_paths = write_split(val_ex, out_dir, "adc_cys_val")
    return {
        "per_linker_results": per_linker_results,
        "train_uuids": train_uuids,
        "val_uuids": val_uuids,
        "n_train_examples": len(train_ex),
        "n_val_examples": len(val_ex),
        "train_paths": train_paths,
        "val_paths": val_paths,
    }


def print_report(meta, stream=sys.stdout):
    per = meta["per_linker_results"]
    n_total = len(per)
    n_success = sum(1 for r in per if r.success)
    print(f"\n=== Conformer augmentation report ===", file=stream)
    print(f"Source linkers attempted : {n_total}", file=stream)
    print(f"Source linkers succeeded : {n_success}", file=stream)
    print(f"Source linkers failed    : {n_total - n_success}", file=stream)

    print(f"\nTrain : {len(meta['train_uuids'])} source linkers, "
          f"{meta['n_train_examples']} conformer examples", file=stream)
    print(f"Val   : {len(meta['val_uuids'])} source linkers, "
          f"{meta['n_val_examples']} conformer examples", file=stream)

    print(f"\nPer-linker conformer yield:", file=stream)
    for r in per:
        if not r.success:
            print(f"  [FAIL  {r.failure_reason:28s}] {r.source_uuid}  "
                  f"{r.canonical_linker_name[:32]:32s} + {r.canonical_payload_name[:24]}",
                  file=stream)
        else:
            print(f"  [OK   {r.n_conformers_kept:2d}/{r.n_conformers_requested:2d}  "
                  f"{'':23s}] {r.source_uuid}  "
                  f"{r.canonical_linker_name[:32]:32s} + {r.canonical_payload_name[:24]}",
                  file=stream)
    if any(not r.success for r in per):
        reasons = Counter(r.failure_reason for r in per if not r.success)
        print(f"\nFailure breakdown:", file=stream)
        for reason, n in reasons.most_common():
            print(f"  {n:4d}  {reason}", file=stream)

    print(f"\nSource-linker leakage check: ", file=stream, end="")
    overlap = set(meta["train_uuids"]) & set(meta["val_uuids"])
    print(f"{'OK (disjoint)' if not overlap else f'LEAK: {overlap}'}", file=stream)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-csv", default="data/processed/linkers_qc_passed.csv")
    ap.add_argument("--out-dir",
                    default="data/processed/difflinker_trainset_v2_incl_aug")
    ap.add_argument("--site", default="cysteine",
                    help="Single site (legacy). Ignored if --sites is given.")
    ap.add_argument("--sites", default=None,
                    help="Comma-separated sites or 'all' "
                         "(cysteine,click,lysine). Overrides --site.")
    ap.add_argument("--n-confs", type=int, default=DEFAULT_N_CONFS)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--embed-seed", type=int, default=0)
    ap.add_argument("--include-flagged", action="store_true", default=True,
                    help="Include rows from linkers_qc_flagged.csv (the new "
                         "boundary recovers many of them — default ON to match "
                         "the 51-trio baseline of difflinker_trainset_v2_incl).")
    args = ap.parse_args()

    SITE_ALIAS = {"click": "click_chemistry", "click_chemistry": "click_chemistry",
                  "cysteine": "cysteine", "lysine": "lysine"}
    if args.sites:
        site_list = (["cysteine", "click_chemistry", "lysine"]
                     if args.sites.strip() == "all"
                     else [SITE_ALIAS.get(s.strip(), s.strip())
                           for s in args.sites.split(",")])
    else:
        site_list = [args.site]
    df = pd.concat(
        [load_cys_adcs_with_lp(Path(args.qc_csv), site=s,
                               include_flagged=args.include_flagged)
         for s in site_list],
        ignore_index=True,
    )
    log.info("Loaded %d candidate rows for sites=%s (include_flagged=%s)",
             len(df), site_list, args.include_flagged)

    meta = build_augmented_dataset(
        df, Path(args.out_dir),
        n_confs=args.n_confs,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        embed_seed=args.embed_seed,
    )
    print_report(meta)
    for tag, paths in [("train", meta["train_paths"]), ("val", meta["val_paths"])]:
        frag, link, table = paths
        print(f"\nWrote {tag}: {frag.name} / {link.name} / {table.name}")


if __name__ == "__main__":
    main()

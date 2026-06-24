"""Tests for src/diffusion/data/difflinker_augment.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.data.difflinker_augment import (  # noqa: E402
    DEFAULT_N_CONFS,
    compute_adc_partition,
    compute_partition,
    embed_multiple_conformers,
    extract_conformer_example,
    linker_level_split,
    process_source_linker,
)


# ---------------------------------------------------------------------------
# Fixtures: the same MMAE+MC-Val-Cit-PAB pair used in the trainset tests
# ---------------------------------------------------------------------------

MMAE_SMILES = (
    "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N"
    "[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C"
)
MC_VAL_CIT_PAB_SMILES = (
    "CC(C)[C@H](NC(=O)CCCCCN1C(=O)CC(S)C1=O)C(=O)N[C@@H](CCCNC(N)=O)"
    "C(=O)Nc2ccc(COC(=O)O)cc2"
)


def _row(linker=MC_VAL_CIT_PAB_SMILES, payload=MMAE_SMILES,
         name="MC-Val-Cit-PAB", payload_name="MMAE"):
    return pd.Series({
        "canonical_linker_name": name,
        "canonical_payload_name": payload_name,
        "payload_smiles": payload,
        "linker_smiles_original": linker,
    })


def _load_mc_val_cit_pab_mmae_adc():
    """Pull the ADC_SMILES for MC-Val-Cit-PAB + MMAE from the source splits."""
    parts = []
    for s in ("train", "val", "test"):
        p = ROOT / "data" / "processed" / f"{s}_split.csv"
        if not p.exists():
            continue
        parts.append(pd.read_csv(p, usecols=[
            "ADC_SMILES", "Canonical_Linker_Name", "Canonical_Payload_Name",
        ], low_memory=False))
    if not parts:
        return None
    df = pd.concat(parts).dropna(subset=["ADC_SMILES"])
    df = df[(df["Canonical_Linker_Name"] == "MC-Val-Cit-PAB")
            & (df["Canonical_Payload_Name"] == "MMAE")]
    if df.empty:
        return None
    return df.iloc[0]["ADC_SMILES"]


def _adc_row(adc=None, payload=MMAE_SMILES,
             name="MC-Val-Cit-PAB", payload_name="MMAE"):
    if adc is None:
        adc = _load_mc_val_cit_pab_mmae_adc()
    return pd.Series({
        "canonical_linker_name": name,
        "canonical_payload_name": payload_name,
        "payload_smiles": payload,
        "ADC_SMILES": adc,
    })


# ---------------------------------------------------------------------------
# Partition: stable, repeatable, with the right counts
# ---------------------------------------------------------------------------

def test_compute_partition_succeeds_on_mc_val_cit_pab_mmae():
    """Legacy SMARTS-surgery path still works (kept for tests/back-compat)."""
    p = compute_partition(MMAE_SMILES, MC_VAL_CIT_PAB_SMILES)
    assert p.failure_reason is None
    assert p.full_mol_2d is not None
    assert set(p.payload_atoms).isdisjoint(p.stub_atoms)
    assert set(p.payload_atoms).isdisjoint(p.spacer_atoms)
    assert set(p.stub_atoms).isdisjoint(p.spacer_atoms)
    # Sum of partition sizes = total heavy atoms of the full ADC
    total = (len(p.payload_atoms) + len(p.stub_atoms) + len(p.spacer_atoms))
    assert total == p.full_mol_2d.GetNumAtoms()
    # Stub anchor is a carbon, payload anchor is a nitrogen for this ADC
    assert p.full_mol_2d.GetAtomWithIdx(p.stub_anchor).GetSymbol() == "C"
    assert p.full_mol_2d.GetAtomWithIdx(p.payload_anchor).GetSymbol() == "N"


def test_compute_partition_fails_cleanly_on_thiol_linker():
    # Legacy: no maleimide ring → partition should refuse, no crash.
    p = compute_partition(MMAE_SMILES, "NCCCC[C@H](NC(=O)CCS)C(=O)O")
    assert p.failure_reason == "no_maleimide_stub_matched"


# ---------------------------------------------------------------------------
# NEW main path: compute_adc_partition (payload-anchored)
# ---------------------------------------------------------------------------

def test_compute_adc_partition_succeeds_on_mc_val_cit_pab_mmae():
    adc = _load_mc_val_cit_pab_mmae_adc()
    if adc is None:
        pytest.skip("source splits not present")
    p = compute_adc_partition(adc, MMAE_SMILES)
    assert p.failure_reason is None
    # MMAE has 51 heavy atoms — anchor-based partition must preserve them all
    assert len(p.payload_atoms) == 51
    # Disjoint partition that covers the whole LP
    assert (set(p.payload_atoms) | set(p.stub_atoms) | set(p.spacer_atoms)
            == set(range(p.full_mol_2d.GetNumAtoms())))
    assert (set(p.payload_atoms) & set(p.stub_atoms)) == set()
    assert (set(p.payload_atoms) & set(p.spacer_atoms)) == set()
    assert (set(p.stub_atoms) & set(p.spacer_atoms)) == set()
    # Stub anchor is the alpha-C; payload anchor is the boundary junction
    assert p.full_mol_2d.GetAtomWithIdx(p.stub_anchor).GetSymbol() == "C"
    assert p.full_mol_2d.GetAtomWithIdx(p.payload_anchor).GetSymbol() == "N"


def test_compute_adc_partition_handles_missing_adc():
    p = compute_adc_partition("", MMAE_SMILES)
    assert p.failure_reason == "no_adc_smiles"


def test_compute_adc_partition_handles_no_match_payload():
    adc = _load_mc_val_cit_pab_mmae_adc()
    if adc is None:
        pytest.skip("source splits not present")
    # Trifluorophosphate (P, F atoms not present in MMAE/MCVCPAB)
    p = compute_adc_partition(adc, "OP(F)(F)F")
    assert p.failure_reason is not None
    assert p.failure_reason.startswith("boundary:")


# ---------------------------------------------------------------------------
# Conformer embedding produces > 1 distinct conformer with diversity
# ---------------------------------------------------------------------------

def test_embed_multiple_conformers_returns_diverse_set():
    p = compute_partition(MMAE_SMILES, MC_VAL_CIT_PAB_SMILES)
    assert p.failure_reason is None
    mol_3d, kept, _ = embed_multiple_conformers(p.full_mol_2d, n_confs=5, seed=0)
    assert mol_3d is not None
    assert len(kept) >= 2, "expected at least 2 distinct conformers"
    # Coordinates must actually differ between conformers (not duplicates)
    conf_a = mol_3d.GetConformer(kept[0]).GetPositions()
    conf_b = mol_3d.GetConformer(kept[1]).GetPositions()
    rms = ((conf_a - conf_b) ** 2).mean() ** 0.5
    assert rms > 0.01, f"conformers should differ; got RMS={rms}"


# ---------------------------------------------------------------------------
# Per-conformer extraction
# ---------------------------------------------------------------------------

def test_per_conformer_extraction_parity_and_components():
    p = compute_partition(MMAE_SMILES, MC_VAL_CIT_PAB_SMILES)
    mol_3d, kept, _ = embed_multiple_conformers(p.full_mol_2d, n_confs=3, seed=0)
    ex = extract_conformer_example(
        mol_3d, kept[0], p,
        source_uuid="test", canonical_linker_name="MC-Val-Cit-PAB",
        canonical_payload_name="MMAE",
    )
    assert ex is not None
    assert len(Chem.GetMolFrags(ex.frag_mol)) == 2
    assert len(Chem.GetMolFrags(ex.link_mol)) == 1
    # Anchor atoms must be in-bounds and consistent with the partition
    assert 0 <= ex.payload_anchor_0based < ex.frag_mol.GetNumAtoms()
    assert 0 <= ex.stub_anchor_0based < ex.frag_mol.GetNumAtoms()
    assert ex.frag_mol.GetAtomWithIdx(ex.stub_anchor_0based).GetSymbol() == "C"


def test_two_conformers_share_topology_but_differ_in_3d():
    p = compute_partition(MMAE_SMILES, MC_VAL_CIT_PAB_SMILES)
    mol_3d, kept, _ = embed_multiple_conformers(p.full_mol_2d, n_confs=4, seed=0)
    assert len(kept) >= 2
    ex_a = extract_conformer_example(mol_3d, kept[0], p, "src", "L", "P")
    ex_b = extract_conformer_example(mol_3d, kept[1], p, "src", "L", "P")
    # Same canonical SMILES (topology preserved)
    assert Chem.MolToSmiles(ex_a.frag_mol) == Chem.MolToSmiles(ex_b.frag_mol)
    assert Chem.MolToSmiles(ex_a.link_mol) == Chem.MolToSmiles(ex_b.link_mol)
    # ...but different coords
    pa = ex_a.frag_mol.GetConformer().GetPositions()
    pb = ex_b.frag_mol.GetConformer().GetPositions()
    assert ((pa - pb) ** 2).sum() > 0


# ---------------------------------------------------------------------------
# Linker-level split — no source linker may appear in both partitions
# ---------------------------------------------------------------------------

def test_linker_level_split_is_disjoint():
    uuids = [f"src_{i:03d}" for i in range(36)]
    train, val = linker_level_split(uuids, val_fraction=0.17, seed=42)
    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == len(uuids)
    assert len(val) >= 1


def test_linker_level_split_is_deterministic():
    uuids = [f"src_{i:03d}" for i in range(36)]
    t1, v1 = linker_level_split(uuids, val_fraction=0.17, seed=42)
    t2, v2 = linker_level_split(uuids, val_fraction=0.17, seed=42)
    assert t1 == t2 and v1 == v2


def test_linker_level_split_different_seeds_give_different_splits():
    uuids = [f"src_{i:03d}" for i in range(36)]
    _, v1 = linker_level_split(uuids, val_fraction=0.17, seed=42)
    _, v2 = linker_level_split(uuids, val_fraction=0.17, seed=99)
    assert set(v1) != set(v2)


# ---------------------------------------------------------------------------
# End-to-end on the real dataset on disk
# ---------------------------------------------------------------------------

AUG_DIR = ROOT / "data" / "processed" / "difflinker_trainset_v2_incl_aug"


@pytest.fixture(scope="module")
def aug_split():
    train_tbl = AUG_DIR / "adc_cys_train_table.csv"
    val_tbl = AUG_DIR / "adc_cys_val_table.csv"
    if not train_tbl.exists() or not val_tbl.exists():
        pytest.skip("augmented dataset not built yet")
    train = pd.read_csv(train_tbl)
    val = pd.read_csv(val_tbl)
    return train, val


def test_no_source_linker_leakage_between_train_and_val(aug_split):
    """The hard constraint: no source linker can appear in both partitions."""
    train, val = aug_split
    overlap = set(train["source_uuid"]) & set(val["source_uuid"])
    assert not overlap, f"LEAKAGE — source linkers in both splits: {overlap}"


def test_every_source_linker_has_multiple_conformers(aug_split):
    train, val = aug_split
    for tbl, name in [(train, "train"), (val, "val")]:
        counts = tbl["source_uuid"].value_counts()
        # No linker should be a singleton — the whole point of augmentation
        assert (counts > 1).all(), (
            f"{name} has singleton source linkers: "
            f"{counts[counts == 1].to_dict()}"
        )


def test_train_and_val_sdfs_align_with_tables(aug_split):
    train, val = aug_split
    for tbl, prefix in [(train, "adc_cys_train"), (val, "adc_cys_val")]:
        frag = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_frag.sdf"),
                                        removeHs=False, sanitize=False))
        link = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_link.sdf"),
                                        removeHs=False, sanitize=False))
        assert len(frag) == len(link) == len(tbl)


def test_every_frag_has_two_components_aug(aug_split):
    train, val = aug_split
    for prefix in ("adc_cys_train", "adc_cys_val"):
        frag = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_frag.sdf"),
                                        removeHs=False, sanitize=False))
        for i, m in enumerate(frag):
            assert m is not None, f"{prefix} row {i}: frag parse failed"
            assert len(Chem.GetMolFrags(m)) == 2, (
                f"{prefix} row {i}: expected 2 frag components, got "
                f"{len(Chem.GetMolFrags(m))}"
            )


def test_atom_parity_per_row_aug(aug_split):
    train, val = aug_split
    for tbl, prefix in [(train, "adc_cys_train"), (val, "adc_cys_val")]:
        frag = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_frag.sdf"),
                                        removeHs=False, sanitize=False))
        link = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_link.sdf"),
                                        removeHs=False, sanitize=False))
        for i, row in tbl.iterrows():
            full = Chem.MolFromSmiles(row["full_adc_smiles"])
            assert frag[i].GetNumAtoms() + link[i].GetNumAtoms() == full.GetNumAtoms()


def test_anchor_columns_in_bounds_aug(aug_split):
    train, val = aug_split
    for tbl, prefix in [(train, "adc_cys_train"), (val, "adc_cys_val")]:
        frag = list(Chem.SDMolSupplier(str(AUG_DIR / f"{prefix}_frag.sdf"),
                                        removeHs=False, sanitize=False))
        for i, row in tbl.iterrows():
            n = frag[i].GetNumAtoms()
            assert 0 <= row["anchor_1"] < n
            assert 0 <= row["anchor_2"] < n
            assert row["anchor_1"] != row["anchor_2"]

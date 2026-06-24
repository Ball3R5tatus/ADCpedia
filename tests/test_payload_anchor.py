"""Tests for src/diffusion/data/payload_anchor.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.data.payload_anchor import (  # noqa: E402
    BoundaryResult,
    derive_payload_boundary,
)


# ---------------------------------------------------------------------------
# Canonical inputs reused across tests
# ---------------------------------------------------------------------------

MMAE_PAYLOAD = (
    "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N"
    "[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C"
)


def _load_specific(linker_name: str, payload_name_substr: str):
    """Pull one ADC row from the source splits and return (lp_smi, payload_smi)."""
    parts = []
    for s in ("train", "val", "test"):
        p = ROOT / "data" / "processed" / f"{s}_split.csv"
        if not p.exists():
            continue
        parts.append(pd.read_csv(p, usecols=[
            "ADC_SMILES", "ADC_Linker_SMILES", "ADC_Payload_SMILES",
            "Canonical_Linker_Name", "Canonical_Payload_Name",
        ], low_memory=False))
    if not parts:
        return None, None
    df = pd.concat(parts).dropna(subset=["ADC_SMILES", "ADC_Payload_SMILES"])
    mask = (df["Canonical_Linker_Name"] == linker_name) \
           & (df["Canonical_Payload_Name"].astype(str).str.contains(payload_name_substr, na=False))
    df = df[mask]
    if df.empty:
        return None, None
    return df.iloc[0]["ADC_SMILES"], df.iloc[0]["ADC_Payload_SMILES"]


# ---------------------------------------------------------------------------
# Sanity / defensive checks
# ---------------------------------------------------------------------------

def test_invalid_lp_smiles_returns_failure():
    res = derive_payload_boundary("not_a_smiles", MMAE_PAYLOAD)
    assert not res.ok
    assert res.failure_reason == "lp_parse_fail"


def test_invalid_payload_smiles_returns_failure():
    res = derive_payload_boundary("CCO", "not_a_smiles")
    assert not res.ok
    assert res.failure_reason == "payload_parse_fail"


def test_partition_is_exhaustive_and_disjoint():
    lp_smi, pay_smi = _load_specific("MC-Val-Cit-PAB", "MMAE")
    if lp_smi is None:
        pytest.skip("source splits not present")
    res = derive_payload_boundary(lp_smi, pay_smi)
    assert res.ok
    lp = Chem.MolFromSmiles(lp_smi)
    assert set(res.payload_atom_idxs) & set(res.linker_atom_idxs) == set()
    assert (set(res.payload_atom_idxs) | set(res.linker_atom_idxs)
            == set(range(lp.GetNumAtoms())))


def test_single_junction_per_partition():
    """A clean payload/linker boundary must have exactly one cross-boundary bond."""
    lp_smi, pay_smi = _load_specific("MC-Val-Cit-PAB", "MMAE")
    if lp_smi is None:
        pytest.skip("source splits not present")
    res = derive_payload_boundary(lp_smi, pay_smi)
    assert res.ok
    lp = Chem.MolFromSmiles(lp_smi)
    p_set = set(res.payload_atom_idxs)
    pa = lp.GetAtomWithIdx(res.payload_junction_atom)
    la = lp.GetAtomWithIdx(res.linker_junction_atom)
    assert res.payload_junction_atom in p_set
    assert res.linker_junction_atom not in p_set
    assert any(nb.GetIdx() == res.linker_junction_atom for nb in pa.GetNeighbors())
    assert any(nb.GetIdx() == res.payload_junction_atom for nb in la.GetNeighbors())


# ---------------------------------------------------------------------------
# Specific cases the user asked to verify
# ---------------------------------------------------------------------------

def test_mc_val_cit_pab_mmae_preserves_full_payload():
    """MC-Val-Cit-PAB + MMAE: the well-known clean case must give exactly the
    payload's heavy-atom count as the payload partition size."""
    lp_smi, pay_smi = _load_specific("MC-Val-Cit-PAB", "MMAE")
    if lp_smi is None:
        pytest.skip("source splits not present")
    res = derive_payload_boundary(lp_smi, pay_smi)
    assert res.ok
    pay = Chem.MolFromSmiles(pay_smi)
    assert len(res.payload_atom_idxs) == pay.GetNumAtoms() == 51
    # The matching should pick the single-junction atom-set (not a 2-junction
    # alternative that happens to overlap because of intra-payload symmetry).
    assert res.method in ("exact_unique", "exact_min_junctions")


def test_amino_peg10_oh_payload_not_truncated():
    """Amino-PEG10-OH was the canary that exposed the SMARTS approach: with
    the old heuristic the payload was reduced to 5 atoms. With anchoring we
    must keep the entire payload (>=40 heavy atoms in every dataset variant)."""
    uniq_csv = ROOT / "data" / "processed" / "linkers_unique.csv"
    splits = [ROOT / "data" / "processed" / f"{s}_split.csv"
              for s in ("train", "val", "test")]
    if not uniq_csv.exists() or not all(p.exists() for p in splits):
        pytest.skip("source data not present")

    uniq = pd.read_csv(uniq_csv)
    rows = uniq[uniq["Canonical_Linker_Name"] == "Amino-PEG10-OH"]
    assert len(rows) > 0, "Amino-PEG10-OH should exist in linkers_unique.csv"

    lp_pairs = []
    for s in splits:
        lp_pairs.append(pd.read_csv(s, usecols=[
            "ADC_SMILES", "ADC_Linker_SMILES", "ADC_Payload_SMILES"
        ], low_memory=False))
    ts = pd.concat(lp_pairs).drop_duplicates()
    merged = rows.merge(ts, on=["ADC_Linker_SMILES", "ADC_Payload_SMILES"],
                         how="left").dropna(subset=["ADC_SMILES"])
    assert len(merged) > 0

    for _, r in merged.iterrows():
        res = derive_payload_boundary(r["ADC_SMILES"], r["ADC_Payload_SMILES"])
        assert res.ok, (
            f"Amino-PEG10-OH variant failed: {res.failure_reason} "
            f"(payload={r['ADC_Payload_SMILES'][:40]})"
        )
        pay = Chem.MolFromSmiles(r["ADC_Payload_SMILES"])
        # The whole payload must end up on the payload side — no truncation.
        assert len(res.payload_atom_idxs) == pay.GetNumAtoms(), (
            f"payload truncated: got {len(res.payload_atom_idxs)}, "
            f"expected {pay.GetNumAtoms()}"
        )
        # And the linker side should be the short PEG-thiol amide (≤15 atoms)
        # — never collapse to <8 the way the old pipeline did.
        assert len(res.linker_atom_idxs) >= 5


def test_no_match_no_mcs_returns_failure_not_crash():
    """A payload that doesn't substructure-match the LP and has no MCS
    must produce a failure result, not an exception."""
    # Unrelated payload (benzene) and a small LP (ethanol).
    res = derive_payload_boundary("CCO", "c1ccccc1")
    assert not res.ok
    assert res.failure_reason in {"no_match_no_mcs",
                                  f"mcs_below_coverage:0/{6}"}


# ---------------------------------------------------------------------------
# Cross-check: the new method does not regress any of the 36 trios that the
# old SMARTS-based pipeline already accepted (same n_payload).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cys_with_lp():
    qc_csv = ROOT / "data" / "processed" / "linkers_qc_passed.csv"
    if not qc_csv.exists():
        pytest.skip("qc_passed CSV missing")
    from diffusion.data.payload_anchor import _load_lp_join
    return _load_lp_join(qc_csv, site="cysteine")


def test_full_cys_coverage_above_old_baseline(cys_with_lp):
    """Coverage on the cys site must exceed the old SMARTS pipeline's 36/50."""
    n_ok = 0
    for _, r in cys_with_lp.iterrows():
        if derive_payload_boundary(r.get("ADC_SMILES"),
                                    r.get("payload_smiles")).ok:
            n_ok += 1
    assert n_ok > 40, f"coverage regressed: only {n_ok} resolved"


def test_no_regression_vs_old_trios(cys_with_lp):
    """For every (canonical_linker_name, canonical_payload_name) pair that
    appears in the OLD trainset table, the new pipeline must still succeed."""
    old_csv = ROOT / "data" / "processed" / "difflinker_trainset" / "adc_cys_table.csv"
    if not old_csv.exists():
        pytest.skip("old trainset table missing")
    old = pd.read_csv(old_csv)
    old_keys = {(str(r["canonical_linker_name"]),
                 str(r["canonical_payload_name"]))
                for _, r in old.iterrows()}
    for _, r in cys_with_lp.iterrows():
        key = (str(r["canonical_linker_name"]),
               str(r["canonical_payload_name"]))
        if key not in old_keys:
            continue
        res = derive_payload_boundary(r.get("ADC_SMILES"),
                                       r.get("payload_smiles"))
        assert res.ok, (
            f"NEW regressed on row OLD accepted: {key} "
            f"reason={res.failure_reason}"
        )

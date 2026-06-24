"""Tests for src/diffusion/data/linker_qc.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.data.linker_qc import (  # noqa: E402
    audit,
    derive_site_from_ab_pattern,
    derive_site_from_name,
    qc_row,
    resolve_site,
)


def _row(**kwargs):
    defaults = {
        "linker_smiles_original": "",
        "linker_smiles_starred": "",
        "site_normalized": "other",
        "conjugation_terminal_smarts_matched": "",
        "payload_attachment_smarts_matched": "",
        "canonical_linker_name": "test",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# Per-criterion unit checks
# ---------------------------------------------------------------------------

def test_clean_mc_val_cit_pab_passes():
    r = _row(
        linker_smiles_original=("CC(C)[C@H](NC(=O)CCCCCN1C(=O)CC(S)C1=O)C(=O)N"
                                "[C@@H](CCCNC(N)=O)C(=O)Nc2ccc(COC(=O)O)cc2"),
        linker_smiles_starred=("*C(=O)OCc1ccc(NC(=O)[C@H](CCCNC(N)=O)NC(=O)[C@@H]"
                               "(NC(=O)CCCCCN2C(=O)CC(*)C2=O)C(C)C)cc1"),
        site_normalized="cysteine",
        conjugation_terminal_smarts_matched="succinimide_thioether_4c",
        payload_attachment_smarts_matched="pabc_carbonate_oh",
    )
    assert qc_row(r).reasons == []


def test_unresolved_decomposition_flagged():
    r = _row(linker_smiles_starred="")
    assert "unresolved_decomposition" in qc_row(r).reasons


def test_unresolved_nan_starred_flagged():
    r = _row(linker_smiles_starred=float("nan"))
    assert "unresolved_decomposition" in qc_row(r).reasons


def test_fragment_too_small_flagged():
    # 6 heavy atoms after stripping 2 dummies — under the 8 threshold.
    r = _row(
        linker_smiles_original="CC(S)CCC(N)=O",
        linker_smiles_starred="*C(=O)CCC(*)C",
        site_normalized="cysteine",
        conjugation_terminal_smarts_matched="terminal_thiol_on_sp3",
    )
    assert "fragment_too_small" in qc_row(r).reasons


def test_star_on_aromatic_flagged():
    # Dummy attached directly to an aromatic carbon.
    r = _row(
        linker_smiles_original="Nc1ccc(C)cc1CCCCCCCC",
        linker_smiles_starred="*c1ccc(C)cc1CCCCCCCC",
        site_normalized="other",
    )
    assert "star_on_aromatic" in qc_row(r).reasons


def test_size_collapse_flagged():
    # Original has 30 heavy atoms, starred has 10 (incl. dummies) → 33%.
    r = _row(
        linker_smiles_original="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        linker_smiles_starred="CCCCCCCC**",
        site_normalized="other",
    )
    assert "size_collapse" in qc_row(r).reasons


def test_site_smarts_annotation_conflict_with_no_name_is_flagged():
    # Name does not match any keyword → name uninformative.
    # SMARTS says cysteine, annotation says lysine → conflict → flag.
    r = _row(
        linker_smiles_original="CCCCCCCCCCCCS",
        linker_smiles_starred="CCCCCCCCCCC*",
        site_normalized="lysine",
        conjugation_terminal_smarts_matched="terminal_thiol_on_sp3",
        canonical_linker_name="Amino-PEG10-OH",
    )
    res = qc_row(r)
    assert "site_terminal_mismatch" in res.reasons
    assert res.site_source == "conflict"


def test_ambiguous_pattern_with_cys_annotation_keeps_annotation():
    # terminal_carboxylic_acid_ab → SMARTS uninformative; with no informative
    # name we fall back to the annotation as last resort (source='annotation').
    r = _row(
        linker_smiles_original="CCCCCCCCCCCC(=O)O",
        linker_smiles_starred="CCCCCCCCCCC(=O)*",
        site_normalized="cysteine",
        conjugation_terminal_smarts_matched="terminal_carboxylic_acid_ab",
        canonical_linker_name="Some-Symmetric-Diacid",
    )
    res = qc_row(r)
    assert "site_terminal_mismatch" not in res.reasons
    assert res.site_source == "annotation"
    assert res.site_final == "cysteine"


def test_derive_site_unambiguous_mappings():
    assert derive_site_from_ab_pattern("succinimide_thioether_4c") == "cysteine"
    assert derive_site_from_ab_pattern("free_maleimide") == "cysteine"
    assert derive_site_from_ab_pattern("disulfide_terminal") == "cysteine"
    assert derive_site_from_ab_pattern("terminal_thiol_on_sp3") == "cysteine"
    assert derive_site_from_ab_pattern("lys_primary_amine") == "lysine"
    assert derive_site_from_ab_pattern("lys_primary_amide") == "lysine"
    assert derive_site_from_ab_pattern("nhs_ester") == "lysine"
    assert derive_site_from_ab_pattern("terminal_carboxylic_acid_ab") is None
    assert derive_site_from_ab_pattern("") is None
    assert derive_site_from_ab_pattern(None) is None


def test_site_other_filled_from_smarts():
    # When annotation is the catch-all 'other' and the name is uninformative,
    # an unambiguous SMARTS fills the slot in.
    r = _row(
        linker_smiles_original="CCCCCCCCCCCCS",
        linker_smiles_starred="CCCCCCCCCCC*",
        site_normalized="other",
        conjugation_terminal_smarts_matched="terminal_thiol_on_sp3",
        canonical_linker_name="Some-Unknown-Linker",
    )
    res = qc_row(r)
    assert res.reasons == []
    assert res.site_source == "smarts"
    assert res.site_final == "cysteine"


def test_site_glycan_smarts_conflict_is_flagged():
    # An informative SMARTS that contradicts a non-empty annotation = conflict.
    r = _row(
        linker_smiles_original="CCCCCCCCCCCCS",
        linker_smiles_starred="CCCCCCCCCCC*",
        site_normalized="glycan",
        conjugation_terminal_smarts_matched="terminal_thiol_on_sp3",
        canonical_linker_name="Some-Unknown-Linker",
    )
    res = qc_row(r)
    assert "site_terminal_mismatch" in res.reasons
    assert res.site_source == "conflict"


# ---------------------------------------------------------------------------
# Canonical-name based site derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("MC-Val-Cit-PAB", "cysteine"),
    ("MC-Gly-Gly-Phe-Gly", "cysteine"),
    ("Mal-PEG4-Val-Cit-PAB-PNP", "cysteine"),
    ("Mal-amido-PEG8-val-gly-PAB-OH", "cysteine"),
    ("5-Maleimidovaleric acid", "cysteine"),
    ("CL2A", "cysteine"),
    ("CL2E", "cysteine"),
    ("MCC", "lysine"),
    ("SMCC", "lysine"),
    ("Bis-SS-C3-NHS ester", "lysine"),
    ("AcBut", "lysine"),
    ("sulfo-SMCC", "lysine"),
    ("BCN-PEG1-Val-Cit-PABC-OH", "click_chemistry"),
    ("DBCO-PEG4-Propionic-Val-Cit-PAB", "click_chemistry"),
    ("Propargyl-PEG2-amine", "click_chemistry"),
    ("Boc-NH-PEG3-C2-triazole-DBCO-PEG4-VC-PAB-DMEA", "click_chemistry"),
    ("Amino-PEG10-OH", None),
    ("Ala-CO-amide-C4-Boc", None),
    ("", None),
    (None, None),
])
def test_derive_site_from_name(name, expected):
    assert derive_site_from_name(name) == expected


@pytest.mark.parametrize("name,ab,annot,expected_site,expected_source", [
    # Name wins outright
    ("CL2A", "lys_primary_amine", "lysine", "cysteine", "name"),
    ("CL2A", "succinimide_thioether_4c", "cysteine", "cysteine", "name"),
    ("MC-Gly-Gly-Phe-Gly", "lys_primary_amine", "cysteine", "cysteine", "name"),
    ("Mal-PEG4-Val-Cit-PAB-PNP", "lys_primary_amide", "cysteine", "cysteine", "name"),
    ("Bis-SS-C3-NHS ester", "terminal_thiol_on_sp3", "lysine", "lysine", "name"),
    ("AcBut", "terminal_thiol_on_sp3", "lysine", "lysine", "name"),
    # Name is click → keep click_chemistry
    ("BCN-PEG1-Val-Cit-PABC-OH", "succinimide_thioether_4c", "cysteine", "click_chemistry", "name"),
    # No name info → fall through to SMARTS if annotation agrees or is empty
    ("Some-Unknown", "succinimide_thioether_4c", "cysteine", "cysteine", "annotation"),
    ("Some-Unknown", "succinimide_thioether_4c", "other", "cysteine", "smarts"),
    # SMARTS conflict with non-empty annotation, no name → None / conflict
    ("Amino-PEG10-OH", "terminal_thiol_on_sp3", "lysine", None, "conflict"),
    # SMARTS ambiguous → use annotation (even when it's the catch-all 'other')
    ("Some-Unknown", "terminal_carboxylic_acid_ab", "cysteine", "cysteine", "annotation"),
    ("Some-Unknown", "terminal_carboxylic_acid_ab", "other", "other", "annotation"),
])
def test_resolve_site(name, ab, annot, expected_site, expected_source):
    site, source = resolve_site(name, ab, annot)
    assert site == expected_site
    assert source == expected_source




def test_free_maleimide_passes_cys_check():
    # 'free_maleimide' contains 'maleimide' substring → matches cys keyword.
    r = _row(
        linker_smiles_original="O=C(O)CCCCCN1C(=O)C=CC1=O",
        linker_smiles_starred="O=C(O)CCCCCN1C(=O)C(*)=CC1(=*)",
        site_normalized="cysteine",
        conjugation_terminal_smarts_matched="free_maleimide",
        payload_attachment_smarts_matched="terminal_carboxylic_acid",
    )
    reasons = qc_row(r).reasons
    assert "site_terminal_mismatch" not in reasons


# ---------------------------------------------------------------------------
# Integration tests — must run on the decomposed CSV produced by linker_prep
# ---------------------------------------------------------------------------

DECOMPOSED = ROOT / "data" / "processed" / "linkers_decomposed.csv"


@pytest.fixture(scope="module")
def audited():
    if not DECOMPOSED.exists():
        pytest.skip(f"{DECOMPOSED} not present; run linker_prep first")
    df = pd.read_csv(DECOMPOSED)
    passed, flagged, _, reannotations = audit(df)
    return df, passed, flagged, reannotations


def test_amino_peg10_oh_flagged(audited):
    df, _passed, flagged, _reann = audited
    names = flagged["canonical_linker_name"].astype(str).tolist()
    assert "Amino-PEG10-OH" in names, "Amino-PEG10-OH must be flagged"
    # At least one of its rows must be flagged for fragment_too_small.
    sub = flagged[flagged["canonical_linker_name"] == "Amino-PEG10-OH"]
    assert any("fragment_too_small" in r for r in sub["qc_reasons"]), \
        "Amino-PEG10-OH must trip fragment_too_small"


def test_bcn_peg1_flagged(audited):
    _df, _passed, flagged, _reann = audited
    names = flagged["canonical_linker_name"].astype(str).tolist()
    assert any("BCN-PEG1" in n for n in names), "BCN-PEG1 variants must be flagged"


def test_audit_partitions_input(audited):
    df, passed, flagged, _reann = audited
    assert len(passed) + len(flagged) == len(df)
    # Passed rows must have all four fields populated.
    for _, r in passed.iterrows():
        assert isinstance(r["linker_smiles_starred"], str) and r["linker_smiles_starred"]
        assert isinstance(r["conjugation_terminal_smarts_matched"], str)


def test_qc_reasons_column_only_in_flagged(audited):
    _df, passed, flagged, _reann = audited
    assert "qc_reasons" not in passed.columns
    assert "qc_reasons" in flagged.columns
    assert "site_corrected" in passed.columns
    assert "site_corrected" not in flagged.columns


def test_cl2a_all_variants_resolve_to_cysteine(audited):
    """Every CL2A row in the dataset must end up with site_final='cysteine',
    decided by the canonical name (regardless of SMARTS / annotation)."""
    _df, passed, flagged, _reann = audited
    cl2a_passed = passed[passed["canonical_linker_name"] == "CL2A"]
    assert len(cl2a_passed) > 0, "no CL2A rows survived QC at all"
    for _, r in cl2a_passed.iterrows():
        assert r["site_final"] == "cysteine"
        assert r["site_source"] == "name"
    # And no CL2A row should be flagged for a site mismatch — name is decisive.
    cl2a_flagged = flagged[flagged["canonical_linker_name"] == "CL2A"]
    for _, r in cl2a_flagged.iterrows():
        assert "site_terminal_mismatch" not in r["qc_reasons"]


def test_specific_named_linkers_resolve_correctly(audited):
    """User-specified cross-checks: name-derived site must beat SMARTS."""
    _df, passed, _flagged, _reann = audited
    expectations = [
        ("MC-Gly-Gly-Phe-Gly", "cysteine"),
        ("Mal-PEG4-Val-Cit-PAB-PNP", "cysteine"),
        ("Bis-SS-C3-NHS ester", "lysine"),
        ("AcBut", "lysine"),
    ]
    for name, expected_site in expectations:
        rows = passed[passed["canonical_linker_name"] == name]
        if rows.empty:
            continue  # may have been hard-rejected by another criterion
        for _, r in rows.iterrows():
            assert r["site_final"] == expected_site, (
                f"{name}: expected site_final={expected_site} got "
                f"{r['site_final']} (source={r['site_source']})"
            )
            assert r["site_source"] == "name"


def test_amino_peg10_oh_uniformly_rejected(audited):
    """Amino-PEG10-OH: name uninformative, so even the variant with enough
    heavy atoms must be rejected — its SMARTS-derived site conflicts with
    the dataset annotation, and we refuse to silently pick."""
    _df, passed, flagged, _reann = audited
    passed_apeg = passed[passed["canonical_linker_name"] == "Amino-PEG10-OH"]
    assert len(passed_apeg) == 0, (
        f"Amino-PEG10-OH must not appear in passed; got {len(passed_apeg)} rows: "
        f"{passed_apeg[['linker_smiles_original','site_final','site_source']].to_dict('records')}"
    )

"""Tests for src/diffusion/data/linker_prep.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.data.linker_prep import (  # noqa: E402
    CANONICAL_SITES,
    decompose_linker,
    normalize_site,
)


# ---------------------------------------------------------------------------
# Site normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("cysteine", "cysteine"),
        ("Cysteine", "cysteine"),
        ("Cystiene", "cysteine"),       # typo
        ("Custeine", "cysteine"),       # typo
        ("cystein", "cysteine"),        # truncated
        ("Cysteine?", "cysteine"),      # punctuation
        ("Cysteine ", "cysteine"),      # trailing space
        ("Lysine", "lysine"),
        ("Glutamine", "glutamine"),
        ("Glutomine", "glutamine"),     # typo
        ("Arginine", "arginine"),
        ("6-N3-GalNAc", "glycan"),
        ("Fc glycan", "glycan"),
        ("p-azidomethyl-L-phenylalanine", "click_chemistry"),
        ("Lysine: The C-terminal lysine on the heavy chain ...", "lysine"),
        ("LLQGA tag inserted", "glutamine"),
        ("", "other"),
        (None, "other"),
        (float("nan"), "other"),
        ("some unknown weirdness", "other"),
    ],
)
def test_normalize_site(raw, expected):
    assert normalize_site(raw) == expected


def test_canonical_sites_complete():
    # every normalized value lands in CANONICAL_SITES
    samples = [
        "Cysteine", "Cystiene", "Lysine", "Glutomine", "Arginine",
        "6-N3-GalNAc", "p-azidomethyl-L-phenylalanine", "", None,
    ]
    for s in samples:
        assert normalize_site(s) in CANONICAL_SITES


# ---------------------------------------------------------------------------
# Attachment-point detection
# ---------------------------------------------------------------------------

# MC-Val-Cit-PAB: cysteine succinimide-thioether + PABC carbonate.
MC_VAL_CIT_PAB = (
    "CC(C)[C@H](NC(=O)CCCCCN1C(=O)CC(S)C1=O)C(=O)N[C@@H](CCCNC(N)=O)"
    "C(=O)Nc2ccc(COC(=O)O)cc2"
)
# Val-cit-PAB-OH (lysine variant): Lys side-chain primary amine + PABC.
VAL_CIT_PAB_OH_LYS = (
    "CC(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CCCNC(N)=O)C(=O)"
    "Nc1ccc(COC(=O)O)cc1)C(C)C"
)
# 5-Maleimidovaleric acid: aromatic-drawn maleimide-thiol + terminal COOH.
MALEIMIDOVAL = "O=C(O)CCCCCCn1c(=O)cc(S)c1=O"
# Mc-Leu-Gly-Arg: succinimide-thioether + terminal carboxylic acid.
MC_LEU_GLY_ARG = (
    "CC(C)[C@H](NC(=O)[C@H](C(C)C)N(C)C(=O)CCCCCN1C(=O)CC(S)C1=O)C(=O)O"
)
# Garbage SMILES.
BAD_SMILES = "this is not a smiles"


def test_mc_val_cit_pab_full_resolution():
    from rdkit import Chem
    d = decompose_linker(MC_VAL_CIT_PAB)
    assert d.unresolved_reason is None, f"expected resolved, got {d.unresolved_reason}"
    assert d.ab_pattern == "succinimide_thioether_4c"
    assert d.pl_pattern == "pabc_carbonate_oh"
    assert d.starred_smiles is not None
    assert d.starred_smiles.count("*") == 2
    # The Ab anchor atom must be the placeholder sulfur (the future Cys-S).
    m = Chem.MolFromSmiles(MC_VAL_CIT_PAB)
    assert m.GetAtomWithIdx(d.ab_anchor_atom).GetSymbol() == "S"
    # The payload anchor must be the terminal oxygen of the PABC carbonate.
    pl_atom = m.GetAtomWithIdx(d.pl_anchor_atom)
    assert pl_atom.GetSymbol() == "O"
    assert pl_atom.GetDegree() == 1


def test_lysine_linker_picks_primary_amine():
    d = decompose_linker(VAL_CIT_PAB_OH_LYS)
    assert d.unresolved_reason is None
    assert d.ab_pattern == "lys_primary_amine"
    assert d.pl_pattern == "pabc_carbonate_oh"
    # The matched Ab anchor should be a nitrogen (it's the Lys e-amine)
    from rdkit import Chem
    m = Chem.MolFromSmiles(VAL_CIT_PAB_OH_LYS)
    assert m.GetAtomWithIdx(d.ab_anchor_atom).GetSymbol() == "N"


def test_aromatic_maleimide_matches_succinimide_thioether():
    d = decompose_linker(MALEIMIDOVAL)
    assert d.ab_pattern == "succinimide_thioether_4c"
    assert d.pl_pattern == "terminal_carboxylic_acid"
    assert d.unresolved_reason is None


def test_mc_leu_gly_arg_carboxylic_acid_payload():
    d = decompose_linker(MC_LEU_GLY_ARG)
    assert d.ab_pattern == "succinimide_thioether_4c"
    assert d.pl_pattern == "terminal_carboxylic_acid"


def test_bad_smiles_returns_unresolved_not_crash():
    d = decompose_linker(BAD_SMILES)
    assert d.unresolved_reason == "rdkit_parse_failure"
    assert d.starred_smiles is None


def test_anchor_atoms_distinct():
    d = decompose_linker(MC_VAL_CIT_PAB)
    assert d.ab_anchor_atom != d.pl_anchor_atom


def test_starred_smiles_parses_back():
    from rdkit import Chem
    d = decompose_linker(MC_VAL_CIT_PAB)
    m = Chem.MolFromSmiles(d.starred_smiles)
    assert m is not None
    dummies = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(dummies) == 2

"""Tests for src/diffusion/data/difflinker_trainset.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.data.difflinker_trainset import (  # noqa: E402
    build_full_adc,
    find_maleimide_stub_atoms,
    find_payload_reactive,
    find_pl_terminal,
    process_row,
    process_adc_row,
    find_maleimide_stub_in_linker_subset,
    find_click_stub_in_linker_subset,
    find_lysine_stub_in_linker_subset,
    _split_stub_spacer,
    load_cys_adcs_with_lp,
)


# ---------------------------------------------------------------------------
# Helpers + tests for the multi-site (click / lysine) stub detection.
# Synthetic molecules give deterministic atom indices so we can assert the
# stub/spacer partition directly without depending on the 3D embedding.
# ---------------------------------------------------------------------------

def _sets_from_payload_smarts(smiles, payload_smarts):
    """Build (mol, payload_set, linker_set) treating the SMARTS-matched ring
    as the payload and everything else as the linker."""
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    q = Chem.MolFromSmarts(payload_smarts)
    match = m.GetSubstructMatch(q)
    assert match, f"payload SMARTS {payload_smarts} not found in {smiles}"
    payload = set(match)
    linker = set(range(m.GetNumAtoms())) - payload
    return m, payload, linker


def _assert_valid_stub(m, stub, payload, linker):
    """The stub must be inside the linker, the anchor (last) must border the
    spacer, and (payload | stub | spacer) must partition the molecule."""
    assert stub is not None and len(stub) >= 1
    stub_set = set(stub)
    assert stub_set <= linker
    spacer = linker - stub_set
    assert spacer, "empty spacer"
    anchor = stub[-1]
    nbr_in_spacer = any(nb.GetIdx() in spacer
                        for nb in m.GetAtomWithIdx(anchor).GetNeighbors())
    assert nbr_in_spacer, "anchor does not border the spacer"
    assert payload | stub_set | spacer == set(range(m.GetNumAtoms()))


def test_click_triazole_stub_partition():
    # benzene(payload) - CC(spacer) - 1,2,3-triazole - N-methyl(antibody cap)
    smi = "Cn1cc(CCc2ccccc2)nn1"
    m, payload, linker = _sets_from_payload_smarts(smi, "c1ccccc1")
    name, stub = find_click_stub_in_linker_subset(m, linker, payload)
    assert name == "triazole"
    _assert_valid_stub(m, stub, payload, linker)
    # the triazole ring (3 N's) must be inside the stub
    n_in_stub = sum(1 for i in stub if m.GetAtomWithIdx(i).GetSymbol() == "N")
    assert n_in_stub >= 3


def test_click_alkyne_handle_stub_partition():
    # terminal propargyl alkyne(antibody) - O - CC(spacer) - phenol(payload)
    smi = "C#CCOCCc1ccc(O)cc1"
    m, payload, linker = _sets_from_payload_smarts(smi, "c1ccc(O)cc1")
    name, stub = find_click_stub_in_linker_subset(m, linker, payload)
    assert name == "alkyne_handle"
    _assert_valid_stub(m, stub, payload, linker)


def test_lysine_amide_terminal_stub_partition():
    # antibody amide cap - CCC(spacer) - aniline-N(payload junction) ; the
    # terminal acetamide is the NHS-derived antibody stub.
    smi = "CC(=O)NCCCC(=O)Nc1ccccc1"
    m, payload, linker = _sets_from_payload_smarts(smi, "c1ccccc1")
    name, stub = find_lysine_stub_in_linker_subset(m, linker, payload)
    assert name == "amide"
    _assert_valid_stub(m, stub, payload, linker)


def test_split_stub_spacer_internal_group_keeps_spacer_connected():
    # An INTERNAL triazole: stub must absorb the antibody-side branch so the
    # payload-side spacer stays a single connected fragment.
    smi = "Cn1cc(CCc2ccccc2)nn1"
    m, payload, linker = _sets_from_payload_smarts(smi, "c1ccccc1")
    qt = Chem.MolFromSmarts("[#6]1~[#6]~[#7]~[#7]~[#7]1")
    core = list(m.GetSubstructMatch(qt))
    stub = _split_stub_spacer(m, core, linker, payload)
    assert stub is not None
    spacer = linker - set(stub)
    # spacer connected
    rw = Chem.RWMol(m)
    for idx in sorted(set(range(m.GetNumAtoms())) - spacer, reverse=True):
        rw.RemoveAtom(idx)
    assert len(Chem.GetMolFrags(rw.GetMol())) == 1


@pytest.mark.parametrize("site,minimum", [("click_chemistry", 4), ("lysine", 6)])
def test_real_multisite_partition_counts(site, minimum):
    """Integration: on the real QC-passed rows, the new stub detection should
    partition at least `minimum` distinct LPs for each site (2D only, no
    embedding). Skips if the data file is absent."""
    from diffusion.data.difflinker_augment import compute_adc_partition
    qc = Path("data/processed/linkers_qc_passed.csv")
    if not qc.exists():
        pytest.skip("QC data not available")
    df = load_cys_adcs_with_lp(qc, site=site)
    df = df[df["ADC_SMILES"].notna()].drop_duplicates("ADC_SMILES")
    ok = sum(
        compute_adc_partition(str(r["ADC_SMILES"]), str(r["payload_smiles"]),
                              site=site).failure_reason is None
        for _, r in df.iterrows()
    )
    assert ok >= minimum, f"{site}: only {ok} partitions succeeded"


# ---------------------------------------------------------------------------
# Fixtures: real ADC entries used as ground truth
# ---------------------------------------------------------------------------

MMAE_SMILES = (
    "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N"
    "[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C"
)
MC_VAL_CIT_PAB_SMILES = (
    "CC(C)[C@H](NC(=O)CCCCCN1C(=O)CC(S)C1=O)C(=O)N[C@@H](CCCNC(N)=O)"
    "C(=O)Nc2ccc(COC(=O)O)cc2"
)
MAL_AMIDO_PEG8_VAL_GLY_PAB_OH = (
    "CC(C)[C@H](NC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCNC(=O)CCN1C(=O)CC(S)"
    "C1=O)C(=O)N[C@@H](C)C(=O)Nc2ccc(COC(=O)O)cc2"
)
MC_VAL_ALA_OH = "C[C@H](NC(=O)[C@H](C)NC(=O)CCCCC(=O)NCCN1C(=O)CCC(S)C1=O)C(=O)O"
FREE_MAL = "CC(C)[C@H](NC(=O)CCOCCOCCNC(=O)CCn1c(=O)ccc1=O)C(=O)O"


# ---------------------------------------------------------------------------
# Pattern unit tests
# ---------------------------------------------------------------------------

def test_find_pl_terminal_pabc_carbonate():
    m = Chem.MolFromSmiles(MC_VAL_CIT_PAB_SMILES)
    name, leaving_idx, bonding_idx = find_pl_terminal(m)
    assert name == "pabc_carbonate_oh"
    assert m.GetAtomWithIdx(leaving_idx).GetSymbol() == "O"
    assert m.GetAtomWithIdx(leaving_idx).GetTotalNumHs() == 1
    assert m.GetAtomWithIdx(leaving_idx).GetDegree() == 1
    assert m.GetAtomWithIdx(bonding_idx).GetSymbol() == "C"
    # bonding C is the carbonate carbonyl: 3 oxygens (=O + -O-CH2-Ph + terminal -OH)
    nbrs = [n.GetSymbol() for n in m.GetAtomWithIdx(bonding_idx).GetNeighbors()]
    assert nbrs.count("O") == 3


def test_find_pl_terminal_carboxylic_acid():
    m = Chem.MolFromSmiles(MC_VAL_ALA_OH)
    name, leaving_idx, bonding_idx = find_pl_terminal(m)
    assert name == "terminal_carboxylic_acid"
    assert m.GetAtomWithIdx(leaving_idx).GetSymbol() == "O"
    assert m.GetAtomWithIdx(bonding_idx).GetSymbol() == "C"


def test_find_payload_reactive_mmae_secondary_amine():
    m = Chem.MolFromSmiles(MMAE_SMILES)
    name, idx = find_payload_reactive(m)
    assert name == "secondary_aliphatic_amine"
    a = m.GetAtomWithIdx(idx)
    assert a.GetSymbol() == "N"
    assert a.GetTotalNumHs() == 1
    # Should be bonded to the alpha-C and the N-methyl C
    nbrs_h = [n.GetTotalNumHs() for n in a.GetNeighbors()]
    assert 3 in nbrs_h  # the CH3 of N-Me-Val


@pytest.mark.parametrize("smiles,expected_pattern", [
    (MC_VAL_CIT_PAB_SMILES, "succinimide_thioether_4c"),
    (MAL_AMIDO_PEG8_VAL_GLY_PAB_OH, "succinimide_thioether_4c"),
    (MC_VAL_ALA_OH, "succinimide_thioether_5c"),
    (FREE_MAL, "free_maleimide"),
])
def test_find_maleimide_stub_atoms(smiles, expected_pattern):
    m = Chem.MolFromSmiles(smiles)
    name, atoms = find_maleimide_stub_atoms(m)
    assert name == expected_pattern
    # Stub must include the ring N + at least one C=O + alpha-C
    syms = [m.GetAtomWithIdx(i).GetSymbol() for i in atoms]
    assert syms.count("N") >= 1
    assert syms.count("O") >= 2  # two ring carbonyls
    # The last index is the alpha-C by construction
    alpha = m.GetAtomWithIdx(atoms[-1])
    assert alpha.GetSymbol() == "C"
    assert not alpha.IsInRing()


def test_find_maleimide_stub_missing():
    # A linker with no maleimide ring at all → returns (None, None)
    m = Chem.MolFromSmiles("CCCCCC(=O)O")
    name, atoms = find_maleimide_stub_atoms(m)
    assert name is None and atoms is None


# ---------------------------------------------------------------------------
# build_full_adc atom-count parity
# ---------------------------------------------------------------------------

def test_build_full_adc_atom_count():
    linker = Chem.MolFromSmiles(MC_VAL_CIT_PAB_SMILES)
    payload = Chem.MolFromSmiles(MMAE_SMILES)
    _, leaving_idx, bonding_idx = find_pl_terminal(linker)
    _, payload_reactive_idx = find_payload_reactive(payload)
    full, _, _ = build_full_adc(
        linker, payload,
        linker_bonding_idx=bonding_idx,
        linker_leaving_idx=leaving_idx,
        payload_reactive_idx=payload_reactive_idx,
    )
    # full = linker + payload - 1 (leaving group removed)
    assert full.GetNumAtoms() == linker.GetNumAtoms() + payload.GetNumAtoms() - 1
    # And the result must parse back as a valid molecule
    smi = Chem.MolToSmiles(full)
    assert Chem.MolFromSmiles(smi) is not None


# ---------------------------------------------------------------------------
# End-to-end process_row
# ---------------------------------------------------------------------------

def _row(linker, payload, name="MC-Val-Cit-PAB", payload_name="MMAE"):
    return pd.Series({
        "canonical_linker_name": name,
        "canonical_payload_name": payload_name,
        "payload_smiles": payload,
        "linker_smiles_original": linker,
    })


def test_process_row_mc_val_cit_pab_mmae_end_to_end():
    row = _row(MC_VAL_CIT_PAB_SMILES, MMAE_SMILES)
    res, frag, link = process_row(row, seed=0)
    assert res.failure_reason is None
    assert frag is not None and link is not None
    # frag has 2 disconnected components (payload + stub)
    assert len(Chem.GetMolFrags(frag)) == 2
    # link is a single connected spacer
    assert len(Chem.GetMolFrags(link)) == 1
    # parity: frag + link = full ADC
    full = Chem.MolFromSmiles(res.full_adc_smiles)
    assert frag.GetNumAtoms() + link.GetNumAtoms() == full.GetNumAtoms()
    # Stub anchor is a CARBON (the alpha-C)
    stub_anchor = frag.GetAtomWithIdx(res.stub_anchor_0based)
    assert stub_anchor.GetSymbol() == "C"
    assert not stub_anchor.IsInRing()
    # Payload anchor is a nitrogen (MMAE's N-Me-Val secondary amine)
    payload_anchor = frag.GetAtomWithIdx(res.payload_anchor_0based)
    assert payload_anchor.GetSymbol() == "N"


def test_process_row_invalid_smiles_no_crash():
    row = _row("not a valid smiles", MMAE_SMILES)
    res, frag, link = process_row(row, seed=0)
    assert res.failure_reason == "linker_parse_fail"
    assert frag is None and link is None


def test_process_row_no_maleimide_skips():
    # A cysteine-conjugated linker with terminal free thiol only — no maleimide
    no_mal = "NCCCC[C@H](NC(=O)CCS)C(=O)Nc1ccc(COC(=O)O)cc1"
    row = _row(no_mal, MMAE_SMILES, name="custom-no-maleimide")
    res, frag, link = process_row(row, seed=0)
    assert res.failure_reason == "no_maleimide_stub_matched"
    assert frag is None and link is None


def test_stub_anchor_is_always_alpha_c_on_real_set():
    """Across every successfully processed cysteine row of the real CSV, the
    stub anchor must be an sp3 carbon outside the maleimide ring."""
    csv = ROOT / "data" / "processed" / "linkers_qc_passed.csv"
    if not csv.exists():
        pytest.skip("source CSV not present in this checkout")
    df = pd.read_csv(csv)
    df = df[df["site_final"] == "cysteine"]
    checked = 0
    for _, row in df.iterrows():
        res, frag, _ = process_row(row, seed=0)
        if res.failure_reason is not None or frag is None:
            continue
        a = frag.GetAtomWithIdx(res.stub_anchor_0based)
        assert a.GetSymbol() == "C", f"stub anchor symbol = {a.GetSymbol()}"
        assert not a.IsInRing(), "stub anchor must be exocyclic"
        checked += 1
    assert checked >= 30, f"only {checked} rows verified — expected ≥30"


# ---------------------------------------------------------------------------
# Integration with the persisted dataset on disk
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NEW main path: payload-anchored process_adc_row
# ---------------------------------------------------------------------------

def _load_mc_val_cit_pab_mmae_adc_smiles():
    """Pull the MC-Val-Cit-PAB + MMAE ADC_SMILES from the source splits."""
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


def test_process_adc_row_mc_val_cit_pab_mmae():
    adc = _load_mc_val_cit_pab_mmae_adc_smiles()
    if adc is None:
        pytest.skip("source splits not present")
    row = pd.Series({
        "canonical_linker_name": "MC-Val-Cit-PAB",
        "canonical_payload_name": "MMAE",
        "payload_smiles": MMAE_SMILES,
        "ADC_SMILES": adc,
    })
    res, frag, link = process_adc_row(row, seed=0)
    assert res.failure_reason is None, f"failed with {res.failure_reason}"
    # MMAE payload has 51 heavy atoms — payload anchoring must preserve them all
    assert res.n_payload_atoms == 51
    # Frag is payload + maleimide stub = 2 disconnected components
    assert len(Chem.GetMolFrags(frag)) == 2
    assert len(Chem.GetMolFrags(link)) == 1
    # Parity: frag heavy atoms + link heavy atoms = LP heavy atoms
    lp = Chem.MolFromSmiles(adc)
    assert frag.GetNumAtoms() + link.GetNumAtoms() == lp.GetNumAtoms()
    # Stub anchor is the alpha-C (exocyclic sp3 carbon adjacent to the ring N)
    stub_a = frag.GetAtomWithIdx(res.stub_anchor_0based)
    assert stub_a.GetSymbol() == "C"
    assert not stub_a.IsInRing()
    # Payload anchor is the N-Me-Val secondary amine N for MMAE
    pa = frag.GetAtomWithIdx(res.payload_anchor_0based)
    assert pa.GetSymbol() == "N"


def test_process_adc_row_no_adc_smiles_fails_gracefully():
    row = pd.Series({
        "canonical_linker_name": "X", "canonical_payload_name": "Y",
        "payload_smiles": MMAE_SMILES, "ADC_SMILES": "nan",
    })
    res, frag, link = process_adc_row(row)
    assert res.failure_reason == "no_adc_smiles"
    assert frag is None and link is None


def test_process_adc_row_unrelated_payload_fails_at_boundary():
    """A payload SMILES whose atoms cannot be found in the LP must fail at
    the boundary step (not crash). NB: any SMARTS substructure of the LP
    will match (this is intentional — we trust the dataset payload). The
    failure must be triggered by a payload that's chemically disjoint."""
    adc = _load_mc_val_cit_pab_mmae_adc_smiles()
    if adc is None:
        pytest.skip("source splits not present")
    row = pd.Series({
        "canonical_linker_name": "X", "canonical_payload_name": "Y",
        # Trifluorophosphate — P + F atoms are NOT present in MC-Val-Cit-PAB+MMAE,
        # so neither substructure nor MCS can match.
        "payload_smiles": "OP(F)(F)F",
        "ADC_SMILES": adc,
    })
    res, frag, link = process_adc_row(row)
    assert res.failure_reason is not None
    assert res.failure_reason.startswith("boundary:")
    assert frag is None and link is None


def test_maleimide_stub_filtered_by_linker_subset():
    """find_maleimide_stub_in_linker_subset must REJECT a maleimide ring whose
    atoms straddle the payload/linker boundary."""
    # Linker = MC-Val-Cit-PAB carries a succinimide-thioether stub.
    adc = _load_mc_val_cit_pab_mmae_adc_smiles()
    if adc is None:
        pytest.skip("source splits not present")
    lp = Chem.MolFromSmiles(adc)
    n = lp.GetNumAtoms()
    # Pass an empty linker subset → must refuse.
    name, atoms = find_maleimide_stub_in_linker_subset(lp, set())
    assert name is None and atoms is None
    # Pass the full LP as the "linker subset" → finds the ring normally.
    name, atoms = find_maleimide_stub_in_linker_subset(lp, set(range(n)))
    assert name is not None and atoms is not None


def test_load_cys_adcs_with_lp_returns_adc_smiles_column():
    qc_csv = ROOT / "data" / "processed" / "linkers_qc_passed.csv"
    if not qc_csv.exists():
        pytest.skip("qc_passed CSV missing")
    df = load_cys_adcs_with_lp(qc_csv, site="cysteine")
    assert "ADC_SMILES" in df.columns
    # Most cysteine rows must have an ADC_SMILES via the join
    assert df["ADC_SMILES"].notna().mean() > 0.8


# ---------------------------------------------------------------------------
# Legacy on-disk dataset checks (the trios produced by the old build)
# ---------------------------------------------------------------------------

TRAINSET_DIR = ROOT / "data" / "processed" / "difflinker_trainset"


@pytest.fixture(scope="module")
def trainset():
    table_path = TRAINSET_DIR / "adc_cys_table.csv"
    if not table_path.exists():
        pytest.skip("trainset has not been built yet")
    table = pd.read_csv(table_path)
    frag = list(Chem.SDMolSupplier(str(TRAINSET_DIR / "adc_cys_frag.sdf"),
                                    removeHs=False, sanitize=False))
    link = list(Chem.SDMolSupplier(str(TRAINSET_DIR / "adc_cys_link.sdf"),
                                    removeHs=False, sanitize=False))
    return table, frag, link


def test_table_and_sdfs_aligned(trainset):
    table, frag, link = trainset
    assert len(table) == len(frag) == len(link)


def test_every_frag_has_two_components(trainset):
    _, frag, _ = trainset
    for i, m in enumerate(frag):
        assert m is not None, f"frag row {i} failed to parse"
        assert len(Chem.GetMolFrags(m)) == 2, f"row {i} has != 2 components"


def test_every_link_has_one_component(trainset):
    _, _, link = trainset
    for i, m in enumerate(link):
        assert m is not None
        assert len(Chem.GetMolFrags(m)) == 1, f"link row {i} not connected"


def test_anchors_in_bounds_and_consistent(trainset):
    table, frag, link = trainset
    for i, row in table.iterrows():
        n = frag[i].GetNumAtoms()
        a1, a2 = int(row["anchor_1"]), int(row["anchor_2"])
        assert 0 <= a1 < n
        assert 0 <= a2 < n
        assert a1 != a2
        # Stub anchor must be a carbon (alpha-C of N-substituent)
        assert frag[i].GetAtomWithIdx(a2).GetSymbol() == "C"


def test_frag_link_parity_against_full_adc(trainset):
    """For each row, heavy-atom count of frag + link must equal the heavy-atom
    count of the recombined full ADC."""
    table, frag, link = trainset
    for i, row in table.iterrows():
        full = Chem.MolFromSmiles(row["full_adc_smiles"])
        assert full is not None
        assert frag[i].GetNumAtoms() + link[i].GetNumAtoms() == full.GetNumAtoms()


# ---------------------------------------------------------------------------
# On-disk dataset produced by the NEW payload-anchored builder
# ---------------------------------------------------------------------------

TRAINSET_V2_DIR = ROOT / "data" / "processed" / "difflinker_trainset_v2_incl"


@pytest.fixture(scope="module")
def trainset_v2():
    table_path = TRAINSET_V2_DIR / "adc_cys_table.csv"
    if not table_path.exists():
        pytest.skip("v2 trainset has not been built yet")
    table = pd.read_csv(table_path)
    frag = list(Chem.SDMolSupplier(str(TRAINSET_V2_DIR / "adc_cys_frag.sdf"),
                                    removeHs=False, sanitize=False))
    link = list(Chem.SDMolSupplier(str(TRAINSET_V2_DIR / "adc_cys_link.sdf"),
                                    removeHs=False, sanitize=False))
    return table, frag, link


def test_v2_coverage_above_old_baseline(trainset_v2):
    """The new payload-anchored pipeline must produce strictly more trios
    than the old SMARTS pipeline's 36."""
    table, _, _ = trainset_v2
    assert len(table) > 36


def test_v2_every_frag_has_two_components(trainset_v2):
    _, frag, _ = trainset_v2
    for i, m in enumerate(frag):
        assert m is not None
        assert len(Chem.GetMolFrags(m)) == 2, (
            f"row {i}: frag has {len(Chem.GetMolFrags(m))} components"
        )


def test_v2_every_link_is_connected(trainset_v2):
    _, _, link = trainset_v2
    for i, m in enumerate(link):
        assert m is not None
        assert len(Chem.GetMolFrags(m)) == 1, (
            f"row {i}: link not connected"
        )


def test_v2_atom_parity_against_lp(trainset_v2):
    """frag + link heavy atoms must equal the LP heavy atoms (the ground
    truth ADC_SMILES from the dataset)."""
    table, frag, link = trainset_v2
    for i, row in table.iterrows():
        lp = Chem.MolFromSmiles(row["adc_smiles"])
        assert lp is not None
        assert frag[i].GetNumAtoms() + link[i].GetNumAtoms() == lp.GetNumAtoms()


def test_v2_mmae_payload_preserved_in_full(trainset_v2):
    """Every MC-Val-Cit-PAB + MMAE row must have n_payload_atoms == 51
    (MMAE's heavy-atom count)."""
    table, _, _ = trainset_v2
    mmae = table[(table["canonical_linker_name"] == "MC-Val-Cit-PAB")
                  & (table["canonical_payload_name"] == "MMAE")]
    assert len(mmae) > 0
    assert (mmae["n_payload_atoms"] == 51).all()

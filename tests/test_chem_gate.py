"""Tests for src/diffusion/gate (Wave-0 task 0.2 — unified chemical-validity gate).

This is the test suite the two original ad-hoc scripts never had.  It pins the
behaviour that must be preserved exactly:

  * Level A (gate_ligand): sanitize / single_fragment / undefined_stereo /
    maleimide_state / druglike, with the original SMARTS + MW>1600 threshold.
  * Level B (gate_topology_valence): per-atom valence vs the MAX_VAL table,
    mirroring the §20.10 pentavalent-C bug and the §19.28 hypervalent-S bug.
  * 3D-strict (gate_usable3d): native_connected on the RAW perceived graph with
    NO MST closure, MMFF relaxability, and the ValCit/Urea motif chemistry.

Each failure test asserts the SPECIFIC reason code, not just overall failure.

Requires an RDKit-enabled interpreter (project env: amm_adc_10nm, rdkit 2022.x).
RDKit-dependent tests are skipped if rdkit is unimportable; the Level-B
(rdkit-free, pure-text) tests always run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.gate import (  # noqa: E402
    __version__,
    gate_all,
    gate_ligand,
    gate_topology_valence,
    gate_usable3d,
)
from diffusion.gate import chem_gate  # noqa: E402

# ---------------------------------------------------------------------------
# rdkit availability — Level A + 3D-strict need it; Level B does not.
# ---------------------------------------------------------------------------
try:
    import rdkit  # noqa: F401
    HAVE_RDKIT = True
except Exception:  # pragma: no cover
    HAVE_RDKIT = False

needs_rdkit = pytest.mark.skipif(not HAVE_RDKIT, reason="rdkit not importable")

# Real usable SDF shipped in the deliverable (validated: usable_3d == True).
USABLE_SDF = ROOT / "outputs" / "h4" / "deliverable" / "usable_sdf" / "output_13_MMAE_MC_.sdf"
# Real topologies: cand_5 is the §20.11-FIXED (valence-clean) build; cand_1 /
# cand_8 carry the §20.10 pentavalent conjugation carbon.
ITP_CLEAN = ROOT / "md" / "runs" / "cand_5" / "topol_Protein_chain_A.itp"
ITP_BUGGY_C = ROOT / "md" / "runs" / "cand_1" / "topol_Protein_chain_A.itp"


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
def test_version_string():
    assert __version__ == "1.1.0"


# ===========================================================================
# Level A — gate_ligand
# ===========================================================================
@needs_rdkit
def test_ligand_pass_clean_thiosuccinimide():
    """PASS: a clean valence-correct connected thiosuccinimide with fully
    defined stereo, closed maleimide, drug-like MW."""
    rec = gate_ligand("O=C1C[C@@H](SC)C(=O)N1C", name="clean")
    assert rec["ok"] is True
    assert rec["reasons"] == []
    assert rec["parsed"] is True
    assert rec["sanitizable"] is True
    assert rec["single_fragment"] is True
    assert rec["n_fragments"] == 1
    assert rec["undefined_stereo"] == 0
    assert rec["maleimide_state"] == "closed/none"
    assert rec["open_maleimide"] is False
    assert rec["druglike"] is True
    assert rec["flags"] == ["OK"]


@needs_rdkit
def test_ligand_pass_real_usable_sdf():
    """PASS: a real usable SDF from the deliverable parses as a single
    connected, sanitizable, stereo-defined molecule (Level-A view)."""
    if not USABLE_SDF.exists():
        pytest.skip(f"fixture SDF missing: {USABLE_SDF}")
    rec = gate_ligand(str(USABLE_SDF), name="output_13")
    assert rec["parsed"] is True
    assert rec["single_fragment"] is True
    assert rec["sanitizable"] is True


@needs_rdkit
def test_ligand_fail_native_connected_disconnected():
    """FAIL: a deliberately disconnected multi-fragment molecule (DiffLinker
    short-bond defect analogue) -> DISCONNECTED reason."""
    rec = gate_ligand("CCO.CCN", name="two-frags")
    assert rec["ok"] is False
    assert "DISCONNECTED" in rec["reasons"]
    assert rec["single_fragment"] is False
    assert rec["n_fragments"] == 2


@needs_rdkit
def test_ligand_fail_undefined_stereo():
    """FAIL: a molecule with an unassigned stereocenter (alanine SMILES with no
    @/@@) -> UNDEFINED-STEREO reason."""
    rec = gate_ligand("CC(N)C(=O)O", name="alanine-no-stereo")
    assert rec["ok"] is False
    assert "UNDEFINED-STEREO" in rec["reasons"]
    assert rec["undefined_stereo"] >= 1


@needs_rdkit
def test_ligand_fail_maleimide_state_open():
    """FAIL: an OPEN maleimide (unreacted C=C) where a closed thiosuccinimide is
    expected -> OPEN-MALEIMIDE reason, maleimide_state == 'open'."""
    rec = gate_ligand("O=C1C=CC(=O)N1C", name="open-maleimide")
    assert rec["ok"] is False
    assert "OPEN-MALEIMIDE" in rec["reasons"]
    assert rec["maleimide_state"] == "open"
    assert rec["open_maleimide"] is True


@needs_rdkit
def test_ligand_fail_unparseable():
    rec = gate_ligand("this is not smiles %%%", name="garbage")
    assert rec["ok"] is False
    assert "UNPARSEABLE" in rec["reasons"]


@needs_rdkit
def test_ligand_druglike_mw_threshold_preserved():
    """The MW>1600 drug-likeness threshold is preserved exactly: a small mol is
    druglike; nothing else trips the gate."""
    rec = gate_ligand("CCO")
    assert rec["druglike"] is True
    assert "MW>1600" not in rec["reasons"]


# ===========================================================================
# Level B — gate_topology_valence (rdkit-free pure-text parse)
# ===========================================================================

# §20.10 — pentavalent conjugation carbon: a c3 atom that gained the C-SG bond
# without losing one H -> 2 C + 2 H + S = 5 bonds (the ligand-side analogue of
# the HG bug).  GROMACS does not check valence; this gate must.
ITP_PENTAVALENT_C = """[ moleculetype ]
LIG 3

[ atoms ]
;  nr  type resnr residue atom cgnr charge mass
   1   c3   1   LIG   C3   1   0.0   12.01
   2   hc   1   LIG   H1   2   0.0   1.008
   3   hc   1   LIG   H2   3   0.0   1.008
   4   c3   1   LIG   CR1  4   0.0   12.01
   5   c3   1   LIG   CR2  5   0.0   12.01
   6   sg   1   CYX   SG   6   0.0   32.06

[ bonds ]
;  ai  aj  funct
   1   2   1
   1   3   1
   1   4   1
   1   5   1
   1   6   1

[ angles ]
   2   1   3   1
"""

# §19.28 — the conjugation S kept its thiol H *and* gained the ligand C-S bond.
# A real thioether S is 2-coordinate; the original MAX_VAL table allows S up to
# 6 bonds (sulfate/sulfonyl), so a merely 3-coordinate S is NOT flagged by the
# degree test (this is faithful to the original gate — see the dedicated
# not-flagged test below).  Here we mirror the hypervalent-S *class* with a
# genuinely over-valent S (degree 7 > MAX_VAL['S']=6) so the OVER-VALENT-S branch
# is asserted with the threshold preserved exactly.
ITP_HYPERVALENT_S = """[ moleculetype ]
CONJ 3

[ atoms ]
   1   sg   1   CYX   SG   1   0.0   32.06
   2   ct   1   CYX   CB   2   0.0   12.01
   3   c3   1   LIG   CL   3   0.0   12.01
   4   o    1   LIG   O1   4   0.0   16.0
   5   o    1   LIG   O2   5   0.0   16.0
   6   o    1   LIG   O3   6   0.0   16.0
   7   o    1   LIG   O4   7   0.0   16.0
   8   hs   1   CYX   HG   8   0.0   1.008

[ bonds ]
   1   2   1
   1   3   1
   1   4   1
   1   5   1
   1   6   1
   1   7   1
   1   8   1
"""

# A 3-coordinate conjugation S with the retained thiol H — the *literal* §19.28
# topology.  Under the preserved MAX_VAL['S']=6 this is NOT degree-over-valent,
# so the degree test alone misses it.  As of gate v1.1.0 the dedicated S-H bond
# detector flags it as RETAINED-THIOL-H (see test below).
ITP_S_3COORD_RETAINED_H = """[ atoms ]
   1   sg   1   CYX   SG   1   0.0   32.06
   2   ct   1   CYX   CB   2   0.0   12.01
   3   c3   1   LIG   CL   3   0.0   12.01
   4   hs   1   CYX   HG   4   0.0   1.008

[ bonds ]
   1   2   1
   1   3   1
   1   4   1
"""

ITP_VALID = """[ atoms ]
   1   c3   1   LIG   C1   1   0.0   12.01
   2   hc   1   LIG   H1   2   0.0   1.008
   3   hc   1   LIG   H2   3   0.0   1.008
   4   hc   1   LIG   H3   4   0.0   1.008
   5   sg   1   CYX   SG   5   0.0   32.06
   6   ct   1   CYX   CB   6   0.0   12.01

[ bonds ]
   1   2   1
   1   3   1
   1   4   1
   1   5   1
   5   6   1
"""


def test_topology_fail_pentavalent_carbon():
    """FAIL Level-B: pentavalent carbon (mirrors §20.10) -> OVER-VALENT-C and the
    offending atom is listed with degree 5 / max 4."""
    rec = gate_topology_valence(ITP_PENTAVALENT_C, name="penta")
    assert rec["ok"] is False
    assert "OVER-VALENT-C" in rec["reasons"]
    bad = rec["bad_atoms"]
    assert len(bad) == 1
    assert bad[0]["idx"] == 1
    assert bad[0]["element"] == "C"
    assert bad[0]["degree"] == 5
    assert bad[0]["max"] == 4
    assert bad[0]["type"] == "c3"


def test_topology_fail_hypervalent_sulfur():
    """FAIL Level-B: hypervalent sulfur (§19.28 class) -> OVER-VALENT-S with the
    threshold MAX_VAL['S']=6 preserved exactly."""
    rec = gate_topology_valence(ITP_HYPERVALENT_S, name="hyperS")
    assert rec["ok"] is False
    assert "OVER-VALENT-S" in rec["reasons"]
    bad = [b for b in rec["bad_atoms"] if b["element"] == "S"]
    assert len(bad) == 1
    assert bad[0]["idx"] == 1
    assert bad[0]["degree"] == 7
    assert bad[0]["max"] == 6
    assert bad[0]["type"] == "sg"
    # this fixture's S also bonds an H (idx 8), so v1.1.0 additionally flags it
    assert "RETAINED-THIOL-H" in rec["reasons"]


def test_topology_fail_retained_thiol_h():
    """FAIL Level-B (v1.1.0 hardening): the literal §19.28 topology — a
    3-coordinate conjugation S that kept its thiol proton (S-H bond) — is now
    flagged RETAINED-THIOL-H even though degree 3 <= MAX_VAL['S']=6, so the
    degree-only gate misses it.  bad_atoms stays empty (no degree over-valence);
    the S-H is reported under retained_thiol_h."""
    rec = gate_topology_valence(ITP_S_3COORD_RETAINED_H, name="s3")
    assert rec["ok"] is False
    assert "RETAINED-THIOL-H" in rec["reasons"]
    assert rec["bad_atoms"] == []          # not a degree over-valence
    assert len(rec["retained_thiol_h"]) == 1
    rt = rec["retained_thiol_h"][0]
    assert rt["s_idx"] == 1 and rt["h_idx"] == 4 and rt["s_type"] == "sg"


def test_topology_pass_valid():
    rec = gate_topology_valence(ITP_VALID, name="valid")
    assert rec["ok"] is True
    assert rec["reasons"] == []
    assert rec["bad_atoms"] == []


def test_topology_accepts_text_or_path(tmp_path):
    """gate_topology_valence accepts inline itp text or a file path identically."""
    p = tmp_path / "penta.itp"
    p.write_text(ITP_PENTAVALENT_C)
    via_text = gate_topology_valence(ITP_PENTAVALENT_C)
    via_path = gate_topology_valence(str(p))
    assert via_text["ok"] == via_path["ok"] is False
    assert via_text["reasons"] == via_path["reasons"]


def test_topology_real_clean_itp_passes():
    """Real §20.11-FIXED topology (cand_5) passes the valence gate."""
    if not ITP_CLEAN.exists():
        pytest.skip(f"fixture itp missing: {ITP_CLEAN}")
    rec = gate_topology_valence(str(ITP_CLEAN), name="cand_5")
    assert rec["ok"] is True
    assert rec["bad_atoms"] == []


def test_topology_real_buggy_itp_flagged():
    """Real §20.10-buggy topology (cand_1, atom 3325) is flagged OVER-VALENT-C —
    exactly the build-time guard described in §20.11."""
    if not ITP_BUGGY_C.exists():
        pytest.skip(f"fixture itp missing: {ITP_BUGGY_C}")
    rec = gate_topology_valence(str(ITP_BUGGY_C), name="cand_1")
    assert rec["ok"] is False
    assert "OVER-VALENT-C" in rec["reasons"]
    assert any(b["idx"] == 3325 and b["element"] == "C" for b in rec["bad_atoms"])


# ===========================================================================
# 3D-strict — gate_usable3d (real SDF)
# ===========================================================================
@needs_rdkit
def test_usable3d_real_usable_sdf():
    """A real usable SDF is native_connected (RAW graph, NO MST closure),
    MMFF-relaxable, carries an ADC cleavable motif, and is usable_3d."""
    if not USABLE_SDF.exists():
        pytest.skip(f"fixture SDF missing: {USABLE_SDF}")
    rec = gate_usable3d(str(USABLE_SDF))
    assert rec["parsed"] is True
    assert rec["native_connected"] is True
    assert rec["mmff_ok"] is True
    assert rec["has_valcit"] or rec["has_urea"]
    assert rec["usable_3d"] is True
    assert isinstance(rec["mmff_energy"], float)


@needs_rdkit
def test_usable3d_missing_file_fails_closed():
    rec = gate_usable3d(str(ROOT / "does" / "not" / "exist.sdf"))
    assert rec["parsed"] is False
    assert rec["native_connected"] is False
    assert rec["usable_3d"] is False


# ===========================================================================
# gate_all — fail-closed convenience
# ===========================================================================
def test_gate_all_no_input_fails_closed():
    out = gate_all()
    assert out["passed"] is False
    assert "NO-INPUT" in out["reasons"]


def test_gate_all_topology_only_buggy_fails():
    """gate_all with a buggy topology fails with a namespaced reason code."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".itp")
    os.close(fd)
    Path(path).write_text(ITP_PENTAVALENT_C)
    try:
        out = gate_all(topology=path)
        assert out["passed"] is False
        assert "topology:OVER-VALENT-C" in out["reasons"]
    finally:
        os.unlink(path)


@needs_rdkit
def test_gate_all_ligand_clean_passes():
    out = gate_all(ligand="O=C1C[C@@H](SC)C(=O)N1C")
    assert out["passed"] is True
    assert out["reasons"] == []


@needs_rdkit
def test_gate_all_ligand_disconnected_fails():
    out = gate_all(ligand="CCO.CCN")
    assert out["passed"] is False
    assert "ligand:DISCONNECTED" in out["reasons"]


@needs_rdkit
def test_gate_all_combined_pass(tmp_path):
    """Combined clean ligand + clean topology -> passed True."""
    p = tmp_path / "valid.itp"
    p.write_text(ITP_VALID)
    out = gate_all(ligand="O=C1C[C@@H](SC)C(=O)N1C", topology=str(p))
    assert out["passed"] is True
    assert out["reasons"] == []


# ===========================================================================
# Module-level invariants we must preserve
# ===========================================================================
def test_max_val_table_unchanged():
    """The MAX_VAL thresholds must match the original chem_validity_gate.py."""
    assert chem_gate.MAX_VAL == {
        'C': 4, 'N': 4, 'O': 2, 'H': 1, 'S': 6,
        'P': 5, 'F': 1, 'Cl': 1, 'Br': 1,
    }

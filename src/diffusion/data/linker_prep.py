"""Preprocess ADC linkers for DiffLinker training.

Two responsibilities:
  1. Normalize the messy free-text conjugation-site column to a clean category.
  2. Identify the two attachment points on each linker (Ab side, payload side)
     via SMARTS matching of known terminals, and emit a dummy-atom ``[*]`` SMILES.

The SMARTS patterns below were cataloged from real linker SMILES in
``data/processed/linkers_unique.csv`` (not invented in the abstract).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Conjugation-site normalization
# ---------------------------------------------------------------------------

CANONICAL_SITES = (
    "cysteine",
    "lysine",
    "glutamine",
    "arginine",
    "glycan",
    "click_chemistry",
    "other",
)

# Order matters: substring rules below this list run AFTER exact-key matches.
_EXACT_SYNONYMS = {
    # cysteine variants / typos
    "cysteine": "cysteine",
    "cystein": "cysteine",
    "custeine": "cysteine",
    "cystiene": "cysteine",
    "cysteine?": "cysteine",
    "cys": "cysteine",
    # lysine
    "lysine": "lysine",
    "lys": "lysine",
    # glutamine
    "glutamine": "glutamine",
    "glutomine": "glutamine",
    "gln": "glutamine",
    # arginine
    "arginine": "arginine",
    "arg": "arginine",
    # glycan-based site-specific conjugation
    "6-n3-galnac": "glycan",
    "fc glycan": "glycan",
    "glycan": "glycan",
    # click-chemistry unnatural amino acids
    "p-azidomethyl-l-phenylalanine": "click_chemistry",
    "azido": "click_chemistry",
    "click": "click_chemistry",
    "dbco": "click_chemistry",
    "bcn": "click_chemistry",
}

# substrings checked when no exact match — useful for long sentences like
# "Lysine: The C-terminal lysine on the heavy chain ..."
_SUBSTRING_RULES = [
    ("glycan", "glycan"),
    ("galnac", "glycan"),
    ("azido", "click_chemistry"),
    ("click", "click_chemistry"),
    ("dbco", "click_chemistry"),
    ("bcn", "click_chemistry"),
    ("transglutaminase", "glutamine"),
    ("llqga", "glutamine"),
    ("glutamine", "glutamine"),
    ("glutomine", "glutamine"),
    ("lysine", "lysine"),
    ("cysteine", "cysteine"),
    ("cystein", "cysteine"),
    ("cystiene", "cysteine"),
    ("custeine", "cysteine"),
    ("arginine", "arginine"),
]


def normalize_site(raw) -> str:
    """Map a raw site annotation to one of CANONICAL_SITES.

    NaN / empty / unrecognized → 'other'.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "other"
    s = str(raw).strip().lower()
    if not s:
        return "other"
    if s in _EXACT_SYNONYMS:
        return _EXACT_SYNONYMS[s]
    for key, target in _SUBSTRING_RULES:
        if key in s:
            return target
    return "other"


# ---------------------------------------------------------------------------
# 2. SMARTS catalogue for linker terminals
# ---------------------------------------------------------------------------
#
# Each entry: (name, smarts, anchor_index)
#   - anchor_index: position in the SMARTS atom list whose matched mol-atom
#     becomes the dummy-atom replacement target.
#
# Ab-side patterns: searched first. Ordered most-specific to least-specific.
# Payload-side: searched second on a different mol-atom than the Ab anchor.

ANTIBODY_PATTERNS = [
    # Succinimide-thioether (post-maleimide-Cys Michael adduct). Bond between
    # the two ring CH atoms may be drawn aromatic (`cc`) or aliphatic — use `~`.
    # S can be D1 (free placeholder) or D2 (drawn with the full Cys residue
    # attached, e.g. CL2A lysine variant).
    # Anchor index points to the placeholder S atom (the future Cys-S).
    ("succinimide_thioether_4c", "[#7]1[#6](=O)[#6]~[#6]([#16;D1,D2])[#6]1=O", 5),
    # 6-membered glutarimide-thioether variant (rare): N1C(=O)CCC(S)C1=O.
    ("succinimide_thioether_5c", "[#7]1[#6](=O)[#6][#6]~[#6]([#16;D1,D2])[#6]1=O", 6),
    # Free maleimide ring (intact, no Cys yet). Anchor = unsubstituted vinyl
    # carbon (H1) — that's where Cys-S will add.
    ("free_maleimide", "[#7;D3;R]1[#6;R](=O)[#6;R;H1]=,:[#6;R][#6;R]1=O", 3),
    # Disulfide terminal: -S-S(H or fragment). Anchor = terminal S (D1).
    ("disulfide_terminal", "[#16;D1][#16;D2]", 0),
    # Free terminal thiol on alkyl carbon — bare Cys-attachment placeholder.
    ("terminal_thiol_on_sp3", "[#16;H1;D1][CX4]", 0),
    # Primary amine attached to sp3 carbon — Lys-attachment placeholder.
    # The [CX4] guard avoids matching citrulline/urea/guanidinium NH2.
    ("lys_primary_amine", "[#7;H2;D1][CX4]", 0),
    # Lys post-conjugation primary amide: -C(=O)NH2 where NH2 is the placeholder
    # for the Lys-CH2 that formed the amide. Matches MCC-lysine `NC(=O)C2CCC...`.
    ("lys_primary_amide", "[#7;H2;D1][#6;X3]=O", 0),
    # N-hydroxysuccinimide ester drawn at the Ab-reactive terminus.
    ("nhs_ester", "[#6](=O)[#8;D2][#7]1[#6](=O)[#6][#6][#6]1=O", 2),
    # Haloacetamide (Cys-reactive electrophile).
    ("haloacetamide", "[Cl,Br,I][#6;X4][#6](=O)[#7]", 0),
    # Fallback: terminal carboxylic acid. Used for symmetric diacid linkers
    # (e.g. Ala-CO-amide-C4-Boc) where both ends are -COOH. PL matching will
    # find the OTHER -COOH OH on a different atom.
    ("terminal_carboxylic_acid_ab", "[#6](=O)[#8;H1;D1]", 2),
]


PAYLOAD_PATTERNS = [
    # PABC dangling carbonate -COC(=O)O-[H]. Anchor = terminal carbonate O.
    ("pabc_carbonate_oh", "c[CH2][#8][#6](=O)[#8;H1;D1]", 5),
    # PABC carbamate to a small leaving group (-COC(=O)N...). Anchor = the N.
    ("pabc_carbamate_n", "c[CH2][#8][#6](=O)[#7;H1,H2]", 5),
    # Terminal anilide -C(=O)-NH-Ar (PAB minus carbonate cap). Anchor = aryl C.
    ("terminal_anilide", "[#6](=O)[#7;H1;D2][c;R]", 3),
    # Para-amino aromatic (terminal aniline like Nc2ccc(...)cc2). Anchor = N.
    ("terminal_aryl_amine", "[c;R][#7;H2;D1]", 1),
    # Terminal carboxylic acid (-COOH). Anchor = terminal OH.
    ("terminal_carboxylic_acid", "[#6](=O)[#8;H1;D1]", 2),
    # Terminal primary amide (-C(=O)NH2). Anchor = terminal N.
    ("terminal_primary_amide", "[#6](=O)[#7;H2;D1]", 2),
    # NHS ester at payload-side too.
    ("nhs_ester_pl", "[#6](=O)[#8;D2][#7]1[#6](=O)[#6][#6][#6]1=O", 2),
    # Terminal sp3 alcohol (-CH2-OH).
    ("terminal_alcohol_sp3", "[CX4][#8;H1;D1]", 1),
    # Bare primary amine on sp3 (fallback if no other PL pattern fires).
    ("terminal_amine_sp3", "[#7;H2;D1][CX4]", 0),
]


@dataclass
class LinkerDecomposition:
    linker_smiles: str
    starred_smiles: Optional[str]
    ab_pattern: Optional[str]
    ab_anchor_atom: Optional[int]
    pl_pattern: Optional[str]
    pl_anchor_atom: Optional[int]
    unresolved_reason: Optional[str]


def _compile(patterns):
    out = []
    for name, smarts, anchor in patterns:
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            raise ValueError(f"Bad SMARTS '{smarts}' for pattern {name}")
        out.append((name, q, anchor))
    return out


_AB_QUERIES = _compile(ANTIBODY_PATTERNS)
_PL_QUERIES = _compile(PAYLOAD_PATTERNS)


def _find_first_match(mol, queries, forbidden_atoms=()):
    """Return (name, anchor_atom_idx, match_tuple) for the first pattern that
    matches `mol` on an atom not in `forbidden_atoms`."""
    forbidden = set(forbidden_atoms)
    for name, q, anchor in queries:
        for match in mol.GetSubstructMatches(q):
            anchor_atom = match[anchor]
            if anchor_atom in forbidden:
                continue
            return name, anchor_atom, match
    return None, None, None


def _replace_atom_with_dummy(mol: Chem.Mol, anchor_idx: int) -> Chem.Mol:
    """Return a new mol where atom `anchor_idx` is replaced by a dummy '*'.

    The atom's bonds are preserved; only its element is rewritten. This keeps
    DiffLinker's expected topology: the dummy marks the attachment, the rest of
    the linker is intact.
    """
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(anchor_idx)
    atom.SetAtomicNum(0)  # dummy
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetFormalCharge(0)
    atom.SetIsAromatic(False)
    return rw.GetMol()


def decompose_linker(smiles: str) -> LinkerDecomposition:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return LinkerDecomposition(
            linker_smiles=smiles,
            starred_smiles=None,
            ab_pattern=None,
            ab_anchor_atom=None,
            pl_pattern=None,
            pl_anchor_atom=None,
            unresolved_reason="rdkit_parse_failure",
        )

    ab_name, ab_anchor, ab_match = _find_first_match(mol, _AB_QUERIES)
    if ab_name is None:
        return LinkerDecomposition(
            linker_smiles=smiles,
            starred_smiles=None,
            ab_pattern=None,
            ab_anchor_atom=None,
            pl_pattern=None,
            pl_anchor_atom=None,
            unresolved_reason="no_ab_terminal_matched",
        )

    forbidden = set(ab_match) if ab_match else {ab_anchor}
    pl_name, pl_anchor, _pl_match = _find_first_match(mol, _PL_QUERIES, forbidden)
    if pl_name is None:
        return LinkerDecomposition(
            linker_smiles=smiles,
            starred_smiles=None,
            ab_pattern=ab_name,
            ab_anchor_atom=ab_anchor,
            pl_pattern=None,
            pl_anchor_atom=None,
            unresolved_reason="no_pl_terminal_matched",
        )

    # Replace both anchors with dummy atoms. Do it in one mol so indices stay
    # valid for both substitutions.
    starred = Chem.RWMol(mol)
    for idx in (ab_anchor, pl_anchor):
        a = starred.GetAtomWithIdx(idx)
        a.SetAtomicNum(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(starred)
        starred_smi = Chem.MolToSmiles(starred)
    except Exception as e:  # pragma: no cover — defensive
        return LinkerDecomposition(
            linker_smiles=smiles,
            starred_smiles=None,
            ab_pattern=ab_name,
            ab_anchor_atom=ab_anchor,
            pl_pattern=pl_name,
            pl_anchor_atom=pl_anchor,
            unresolved_reason=f"sanitize_failed:{e.__class__.__name__}",
        )

    return LinkerDecomposition(
        linker_smiles=smiles,
        starred_smiles=starred_smi,
        ab_pattern=ab_name,
        ab_anchor_atom=ab_anchor,
        pl_pattern=pl_name,
        pl_anchor_atom=pl_anchor,
        unresolved_reason=None,
    )


# ---------------------------------------------------------------------------
# 3. CLI entry point
# ---------------------------------------------------------------------------

INPUT_COLS = {
    "linker": "ADC_Linker_SMILES",
    "payload": "ADC_Payload_SMILES",
    "linker_name": "Canonical_Linker_Name",
    "payload_name": "Canonical_Payload_Name",
    "site": "Lysine_Cysteine_Glutamine_Other_Linker_Conjugation_Site",
}


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        smi = r[INPUT_COLS["linker"]]
        if not isinstance(smi, str) or not smi.strip():
            decomp = LinkerDecomposition(
                linker_smiles="",
                starred_smiles=None,
                ab_pattern=None,
                ab_anchor_atom=None,
                pl_pattern=None,
                pl_anchor_atom=None,
                unresolved_reason="empty_smiles",
            )
        else:
            decomp = decompose_linker(smi)
        rows.append({
            "payload_smiles": r[INPUT_COLS["payload"]],
            "linker_smiles_original": smi,
            "linker_smiles_starred": decomp.starred_smiles,
            "conjugation_terminal_smarts_matched": decomp.ab_pattern,
            "payload_attachment_smarts_matched": decomp.pl_pattern,
            "site_normalized": normalize_site(r.get(INPUT_COLS["site"])),
            "canonical_linker_name": r.get(INPUT_COLS["linker_name"]),
            "canonical_payload_name": r.get(INPUT_COLS["payload_name"]),
            "unresolved_reason": decomp.unresolved_reason,
        })
    return pd.DataFrame(rows)


def print_coverage_report(df_in: pd.DataFrame, df_out: pd.DataFrame, stream=sys.stdout):
    n = len(df_out)
    resolved = df_out["unresolved_reason"].isna().sum()
    print(f"\n=== Site normalization: before vs after ===", file=stream)
    raw_counts = df_in[INPUT_COLS["site"]].fillna("<NaN>").value_counts()
    norm_counts = df_out["site_normalized"].value_counts()
    print(f"Raw values ({len(raw_counts)} unique):", file=stream)
    for k, v in raw_counts.head(30).items():
        print(f"  {v:4d}  {k!r}", file=stream)
    print(f"Normalized:", file=stream)
    for k, v in norm_counts.items():
        print(f"  {v:4d}  {k}", file=stream)

    print(f"\n=== Attachment-point coverage ===", file=stream)
    print(f"Total linkers       : {n}", file=stream)
    print(f"Both anchors found  : {resolved}  ({resolved/n:.1%})", file=stream)
    print(f"Unresolved          : {n - resolved}", file=stream)

    ab_dist = df_out[df_out["unresolved_reason"].isna()]["conjugation_terminal_smarts_matched"].value_counts()
    pl_dist = df_out[df_out["unresolved_reason"].isna()]["payload_attachment_smarts_matched"].value_counts()
    print(f"\nAb-terminal SMARTS hits:", file=stream)
    for k, v in ab_dist.items():
        print(f"  {v:4d}  {k}", file=stream)
    print(f"\nPayload-attachment SMARTS hits:", file=stream)
    for k, v in pl_dist.items():
        print(f"  {v:4d}  {k}", file=stream)

    unresolved = df_out[df_out["unresolved_reason"].notna()]
    if len(unresolved):
        print(f"\n=== Unresolved linkers (manual triage needed) ===", file=stream)
        for _, r in unresolved.iterrows():
            print(f"  [{r['unresolved_reason']:30s}] "
                  f"{str(r['canonical_linker_name'])[:40]:40s}  "
                  f"site={r['site_normalized']:15s}  "
                  f"{r['linker_smiles_original'][:80]}", file=stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/linkers_unique.csv")
    ap.add_argument("--output", default="data/processed/linkers_decomposed.csv")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    df_in = pd.read_csv(in_path)
    df_out = process_dataframe(df_in)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print_coverage_report(df_in, df_out)
    print(f"\nWrote {out_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()

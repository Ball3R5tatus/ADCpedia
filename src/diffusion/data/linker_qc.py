"""Quality-control audit of ``data/processed/linkers_decomposed.csv``.

Splits the decomposed CSV into:
  * ``linkers_qc_passed.csv``  — decompositions that pass every check.
  * ``linkers_qc_flagged.csv`` — rejected rows with one or more reasons.

Rejection criteria (per the spec):
  1. ``fragment_too_small``      — starred molecule has < 8 heavy atoms
                                   (dummies and Hs excluded).
  2. ``star_on_aromatic``        — at least one ``[*]`` is bonded to an
                                   aromatic atom.
  3. ``size_collapse``           — starred heavy-atom count (incl. dummies)
                                   is < 50% of the original.
  4. ``site_terminal_mismatch``  — ``site_normalized`` and the Ab-side SMARTS
                                   name disagree (cys vs lys-chemistry).

Plus one implicit one we cannot ignore:
  0. ``unresolved_decomposition`` — the decomposition step produced no
                                    starred SMILES at all. These rows must be
                                    flagged before any of the above checks can
                                    run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)


MIN_HEAVY_ATOMS = 8
MIN_SIZE_RATIO = 0.50

# Pattern-name substrings considered consistent with each site. Substring match
# keeps the rule robust to future SMARTS additions (e.g. a new
# 'succinimide_*' variant is automatically accepted on the cys side).
CYS_AB_KEYWORDS = ("succinimide", "maleimide", "thiol", "disulfide")
LYS_AB_KEYWORDS = ("amine", "amide", "nhs")

# Explicit Ab-pattern → canonical site mapping. Used to *re-derive* the site
# from the chemistry when the dataset annotation disagrees. ``None`` means the
# pattern is intrinsically ambiguous (e.g. -COOH terminal can sit on either
# end of a symmetric diacid linker) and no derivation is possible.
AB_PATTERN_TO_SITE = {
    "succinimide_thioether_4c": "cysteine",
    "succinimide_thioether_5c": "cysteine",
    "free_maleimide": "cysteine",
    "terminal_thiol_on_sp3": "cysteine",
    "disulfide_terminal": "cysteine",
    "haloacetamide": "cysteine",
    "lys_primary_amine": "lysine",
    "lys_primary_amide": "lysine",
    "nhs_ester": "lysine",
    "terminal_carboxylic_acid_ab": None,
}

# Sites we are willing to *overwrite* with a re-derived value. Sites that
# describe a specific bioconjugation chemistry not modelled by our Ab-side
# SMARTS catalogue (glycan, click_chemistry, arginine) are left untouched —
# the SMARTS may have matched an irrelevant secondary terminal.
REANNOTATABLE_SITES = {"cysteine", "lysine", "glutamine", "other"}

# Canonical-name substring rules. The name encodes the conjugation chemistry
# almost perfectly in this dataset (much more reliably than the per-row
# annotation column, which is full of typos, blanks, and partial sentences,
# and also more reliably than the SMARTS, which occasionally matches a body
# residue rather than the actual terminal). Order matters: click is checked
# first because some click linkers also embed Mal/MC fragments; lysine is
# checked before cysteine so that 'MCC' / 'SMCC' do not get mis-classified
# by the cysteine 'mc-' / 'mal' rules.
NAME_KEYWORDS_CLICK = ("bcn", "dbco", "propargyl", "azido", "click")
NAME_KEYWORDS_LYS = ("mcc", "smcc", "nhs", "acbut", "sulfo")
NAME_KEYWORDS_CYS = ("mc-", "mal", "maleimid", "cl2a", "cl2e")


@dataclass
class QCResult:
    reasons: list = field(default_factory=list)
    site_final: object = None    # the resolved site string (always populated unless flagged)
    site_source: object = None   # 'name' | 'smarts' | 'annotation' | 'conflict' | 'no_info'

    @property
    def passed(self) -> bool:
        return not self.reasons


def _heavy_atom_count(mol: Chem.Mol, *, include_dummies: bool) -> int:
    count = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z == 1:
            continue  # explicit H
        if z == 0 and not include_dummies:
            continue
        count += 1
    return count


def _site_matches_ab_pattern(site: str, ab_pattern: str) -> bool:
    """Return True if the Ab-side SMARTS name is chemically consistent with
    the (normalized) conjugation site. Only cys/lys are checked; other sites
    are not enforced (glycan / click / glutamine / other / arginine pass)."""
    if not isinstance(ab_pattern, str) or not ab_pattern:
        return False
    name = ab_pattern.lower()
    if site == "cysteine":
        return any(k in name for k in CYS_AB_KEYWORDS)
    if site == "lysine":
        return any(k in name for k in LYS_AB_KEYWORDS)
    return True


def derive_site_from_ab_pattern(ab_pattern: str):
    """Re-derive the conjugation site from the matched Ab-side SMARTS name.

    Returns the canonical site (str) when the pattern maps unambiguously, or
    ``None`` when the pattern is intrinsically ambiguous / unknown.
    """
    if not isinstance(ab_pattern, str) or not ab_pattern:
        return None
    return AB_PATTERN_TO_SITE.get(ab_pattern, None)


def derive_site_from_name(name) -> object:
    """Re-derive the conjugation site from the canonical linker name.

    Returns 'cysteine' / 'lysine' / 'click_chemistry' when a keyword fires,
    else ``None``. The click bucket means "do not enforce cys/lys" — caller
    keeps the annotated site rather than overwriting with cys/lys."""
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.lower()
    if any(k in n for k in NAME_KEYWORDS_CLICK):
        return "click_chemistry"
    if any(k in n for k in NAME_KEYWORDS_LYS):
        return "lysine"
    if any(k in n for k in NAME_KEYWORDS_CYS):
        return "cysteine"
    return None


_EMPTY_ANNOTATIONS = ("", "other")


def _annotation_is_empty(annotation) -> bool:
    if annotation is None:
        return True
    if isinstance(annotation, float) and pd.isna(annotation):
        return True
    if isinstance(annotation, str) and annotation.strip().lower() in _EMPTY_ANNOTATIONS:
        return True
    return False


def resolve_site(name, ab_pattern, annotation):
    """Decide the final conjugation site for a row.

    Priority (high → low):
      1. Canonical linker NAME — most reliable; overrides everything.
         A click-chemistry name forces the click bucket (no cys/lys).
      2. Ab-side SMARTS — used only when the name is uninformative AND
         either (a) the annotation is empty/'other' (the SMARTS fills it
         in) or (b) the annotation agrees with the SMARTS.
         If the SMARTS contradicts a non-empty annotation, we refuse to
         silently pick a side — the row is flagged as a conflict.
      3. Original annotation — final fallback.

    Returns ``(site, source)``. ``site`` is None only when source='conflict'
    or source='no_info'; in both cases the caller should flag the row.
    """
    name_site = derive_site_from_name(name)
    if name_site is not None:
        return name_site, "name"

    smarts_site = derive_site_from_ab_pattern(ab_pattern)
    if smarts_site is not None:
        if _annotation_is_empty(annotation):
            return smarts_site, "smarts"
        if isinstance(annotation, str) and annotation.lower() == smarts_site:
            return annotation, "annotation"
        return None, "conflict"

    # SMARTS adds no information either way → trust the annotation if it
    # carries any. Otherwise give up.
    if isinstance(annotation, str) and annotation.strip():
        return annotation, "annotation"
    return None, "no_info"


def qc_row(row) -> QCResult:
    result = QCResult()
    starred = row.get("linker_smiles_starred")
    original = row.get("linker_smiles_original")
    site = row.get("site_normalized")
    ab_pattern = row.get("conjugation_terminal_smarts_matched")

    # ---- Criterion 0: missing decomposition --------------------------------
    if not isinstance(starred, str) or not starred.strip():
        result.reasons.append("unresolved_decomposition")
        return result

    starred_mol = Chem.MolFromSmiles(starred)
    if starred_mol is None:
        result.reasons.append("starred_parse_failure")
        return result

    # ---- Criterion 1: fragment_too_small -----------------------------------
    heavy_no_dummy = _heavy_atom_count(starred_mol, include_dummies=False)
    if heavy_no_dummy < MIN_HEAVY_ATOMS:
        result.reasons.append("fragment_too_small")

    # ---- Criterion 2: star_on_aromatic -------------------------------------
    star_aromatic = False
    for atom in starred_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        if atom.GetIsAromatic():
            star_aromatic = True
            break
        for nbr in atom.GetNeighbors():
            if nbr.GetIsAromatic():
                star_aromatic = True
                break
        if star_aromatic:
            break
    if star_aromatic:
        result.reasons.append("star_on_aromatic")

    # ---- Criterion 3: size_collapse ----------------------------------------
    if isinstance(original, str) and original.strip():
        orig_mol = Chem.MolFromSmiles(original)
        if orig_mol is not None:
            orig_heavy = _heavy_atom_count(orig_mol, include_dummies=False)
            starred_total = _heavy_atom_count(starred_mol, include_dummies=True)
            if orig_heavy > 0 and starred_total / orig_heavy < MIN_SIZE_RATIO:
                result.reasons.append("size_collapse")

    # ---- Criterion 4: site resolution (name > smarts > annotation) -------
    name = row.get("canonical_linker_name")
    final_site, source = resolve_site(name, ab_pattern, site)
    result.site_source = source
    if final_site is None:
        # 'conflict' (smarts ≠ annotation, name uninformative)
        # or 'no_info' (nothing decided anything).
        result.reasons.append("site_terminal_mismatch")
    else:
        result.site_final = final_site

    return result


def audit(df: pd.DataFrame):
    """Audit a decomposed-linkers DataFrame.

    Returns ``(passed_df, flagged_df, reasons_per_row, reannotations)`` where:
      - ``passed_df`` carries an extra ``site_corrected`` column (empty when
        the original site was kept).
      - ``flagged_df`` carries an extra ``qc_reasons`` column.
      - ``reannotations`` is a list of dicts describing every site override
        applied to a passing row.
    """
    reasons_per_row = []
    site_finals = []
    site_sources = []
    for _, row in df.iterrows():
        res = qc_row(row)
        reasons_per_row.append(res.reasons)
        site_finals.append(res.site_final)
        site_sources.append(res.site_source)

    df = df.copy()
    df["qc_reasons"] = [";".join(r) for r in reasons_per_row]
    df["site_final"] = site_finals
    df["site_source"] = site_sources
    # site_corrected: only populated when the final site differs from the
    # original annotation. Kept for downstream readability.
    df["site_corrected"] = [
        f if (isinstance(f, str) and f and f != s) else None
        for f, s in zip(site_finals, df["site_normalized"])
    ]

    mask_passed = df["qc_reasons"] == ""
    passed_cols_drop = ["qc_reasons"]
    flagged_cols_drop = ["site_final", "site_corrected"]  # source kept on flagged for triage
    passed = df[mask_passed].drop(columns=passed_cols_drop).reset_index(drop=True)
    flagged = df[~mask_passed].drop(columns=flagged_cols_drop).reset_index(drop=True)

    reannotations = []
    for _, r in passed.iterrows():
        if isinstance(r["site_corrected"], str) and r["site_corrected"]:
            reannotations.append({
                "canonical_linker_name": r.get("canonical_linker_name"),
                "linker_smiles_original": r.get("linker_smiles_original"),
                "ab_pattern": r.get("conjugation_terminal_smarts_matched"),
                "old_site": r["site_normalized"],
                "new_site": r["site_corrected"],
                "source": r["site_source"],
            })
    return passed, flagged, reasons_per_row, reannotations


def print_report(passed: pd.DataFrame, flagged: pd.DataFrame,
                 reasons_per_row, reannotations, stream=sys.stdout):
    total = len(passed) + len(flagged)
    print(f"\n=== Linker QC summary ===", file=stream)
    print(f"Total rows in   : {total}", file=stream)
    print(f"Passed          : {len(passed)} ({len(passed)/total:.1%})", file=stream)
    print(f"  of which reannotated: {len(reannotations)}", file=stream)
    print(f"Flagged         : {len(flagged)} ({len(flagged)/total:.1%})", file=stream)

    print(f"\nFlagged breakdown by reason:", file=stream)
    counter = Counter()
    for rs in reasons_per_row:
        for r in rs:
            counter[r] += 1
    for reason, n in counter.most_common():
        print(f"  {n:4d}  {reason}", file=stream)

    if reannotations:
        print(f"\nSite reannotations ({len(reannotations)}):", file=stream)
        by_source = Counter(r["source"] for r in reannotations)
        for src, n in by_source.most_common():
            print(f"  by source={src:12s}: {n}", file=stream)
        for r in reannotations:
            name = str(r["canonical_linker_name"])[:42]
            print(
                f"  site_reannotated: {str(r['old_site']):9s} -> {r['new_site']:16s}  "
                f"[source={r['source']:10s}] [ab={str(r['ab_pattern']):30s}] {name}",
                file=stream,
            )

    if len(flagged):
        print(f"\nIndividual flagged rows:", file=stream)
        for _, r in flagged.iterrows():
            name = str(r.get("canonical_linker_name", ""))[:42]
            site = str(r.get("site_normalized", ""))
            print(f"  [{r['qc_reasons']:50s}] {name:42s} site={site}", file=stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/linkers_decomposed.csv")
    ap.add_argument("--out-passed", default="data/processed/linkers_qc_passed.csv")
    ap.add_argument("--out-flagged", default="data/processed/linkers_qc_flagged.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    passed, flagged, reasons, reannotations = audit(df)

    Path(args.out_passed).parent.mkdir(parents=True, exist_ok=True)
    passed.to_csv(args.out_passed, index=False)
    flagged.to_csv(args.out_flagged, index=False)
    print_report(passed, flagged, reasons, reannotations)
    print(f"\nWrote {args.out_passed} ({len(passed)} rows)")
    print(f"Wrote {args.out_flagged} ({len(flagged)} rows)")


if __name__ == "__main__":
    main()

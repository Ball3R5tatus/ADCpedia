#!/usr/bin/env python3
"""
build_manifest.py -- Wave-0 task 0.1: frozen "dataset contract" manifest.

Emits ONE row per unique linker-payload (L/P) pair for the ADCpedia
molecular-diffusion ADC linker project, with a clean, typed schema that
downstream leave-one-payload-class-out (LOPO) splits and the Pareto sweep
will consume.

Row identity
------------
The unique L/P key is the *exact SMILES pair* (ADC_Linker_SMILES,
ADC_Payload_SMILES) from linkers_unique.csv (138 rows). Canonical
name pairs are NOT unique (only 67 distinct name pairs for 138 SMILES
pairs), so names are used only for the (approximate) trainset/anchor
join, never as the primary key.

Join strategy
-------------
  * SPINE        : linkers_unique.csv (138 rows; 138 distinct SMILES pairs).
  * decomposed   : linkers_decomposed.csv -- SMILES-exact join (all 138),
                   supplies site_normalized for every row.
  * qc_passed    : SMILES-exact join (89 rows) -- supplies site_final /
                   site_source (the most trustworthy site call).
  * qc_flagged   : SMILES-exact join (49 rows) -- the complement; only
                   site_normalized available there.
  * trainsets    : v3 multi-site (cys+lys+click) and v2 cysteine-only;
                   joined by (linker_name, payload_name, site) because that
                   is the only key they share with the spine. Anchors /
                   spacer sizes are taken as the MODAL value per group and
                   flagged ambiguous when the conformer-augmented rows
                   disagree (anchors are SMILES-index dependent and a name
                   pair maps to several distinct SMILES, so this is a best
                   effort -- see anchors_ambiguous / in_*_trainset flags).

Output
------
  data/processed/dataset_manifest.csv
  data/processed/dataset_manifest.json
Idempotent / deterministic: re-running overwrites with identical content.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# RDKit (optional but expected). If unavailable we still build the manifest
# but mark canonical_lp_smiles / stereo_defined as "unknown".
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:  # pragma: no cover
    HAVE_RDKIT = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
# repo root = .../ADCpedia (src/diffusion/data/build_manifest.py -> up 3 + 1)
REPO = HERE.parents[3]
DATA = REPO / "data" / "processed"

SPINE_CSV = DATA / "linkers_unique.csv"
DECOMP_CSV = DATA / "linkers_decomposed.csv"
QC_PASSED_CSV = DATA / "linkers_qc_passed.csv"
QC_FLAGGED_CSV = DATA / "linkers_qc_flagged.csv"

V3_TRAIN = DATA / "difflinker_trainset_v3_multi_site_aug" / "adc_cys_train_table.csv"
V3_VAL = DATA / "difflinker_trainset_v3_multi_site_aug" / "adc_cys_val_table.csv"
V2_TRAIN = DATA / "difflinker_trainset_v2_incl_aug_dedup" / "adc_cys_train_table.csv"
V2_VAL = DATA / "difflinker_trainset_v2_incl_aug_dedup" / "adc_cys_val_table.csv"

OUT_CSV = DATA / "dataset_manifest.csv"
OUT_JSON = DATA / "dataset_manifest.json"


# ===========================================================================
# Normalization maps + rules (documented, deterministic)
# ===========================================================================

# --- payload_class -------------------------------------------------------
# Maps raw Canonical_Payload_Name -> normalized payload family.
PAYLOAD_CLASS_MAP = {
    # Auristatins (tubulin inhibitors, dolastatin-10 analogues)
    "MMAE": "auristatin",
    "MMAF": "auristatin",
    "MMAD": "auristatin",
    "Dolastatin": "auristatin",
    # Camptothecin / topoisomerase-I (Dxd = exatecan deriv; SN-38 = irinotecan active)
    "Dxd derivative": "camptothecin",
    "SN-38": "camptothecin",
    # PBD dimers + anthracycline-PBD class warheads (DNA crosslinkers / alkylators)
    "PBD dimer": "PBD_anthracycline",
    "PNU-159682 derivative": "PBD_anthracycline",
    "DGN549 derivative": "PBD_anthracycline",
    "Aldoxorubicin": "PBD_anthracycline",  # doxorubicin (anthracycline) prodrug
    "Polyketomycin": "PBD_anthracycline",  # anthracycline-type DNA-binding antibiotic
    # Enediyne
    "Calicheamicin": "calicheamicin",
    # Maytansinoids (tubulin inhibitors)
    "DM1": "maytansinoid",
    "DM4": "maytansinoid",
    "Maytansine derivative": "maytansinoid",
    # Taxane
    "Paclitaxel derivative": "taxane",
    # Halichondrin / microtubule (eribulin)
    "Eribulin": "eribulin",
    # Tubulysin (tubulin inhibitor peptide)
    "Tubulysin derivative": "tubulysin",
    # Duocarmycin (DNA minor-groove alkylator)
    "Seco-DUBA": "duocarmycin",
    # RNA pol II inhibitor (amatoxin)
    "a-Amanitin": "amatoxin",
}


def map_payload_class(name) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "unknown"
    s = str(name).strip()
    if s == "" or s.lower() == "nan":
        return "unknown"
    if s in PAYLOAD_CLASS_MAP:
        return PAYLOAD_CLASS_MAP[s]
    return "other"


# --- conjugation_class ---------------------------------------------------
# Folds *all* spellings/typos of the raw site label -> normalized class.
# Order matters: longer / more specific keys first.
def map_conjugation_class(raw_site, pattern_stub=None) -> str:
    """Normalize a raw conjugation-site label (typo-laden) -> clean class.

    Classes: cysteine, lysine, glutamine, click, other.
    Falls back to pattern_stub hints when the site label is empty/other.
    """
    s = "" if raw_site is None else str(raw_site).strip().lower()

    # azide / click / glycan chemistry (check before generic letters)
    click_tokens = [
        "azid", "n3", "galnac", "glycan", "click", "triazol",
        "dbco", "bcn", "propargyl", "alkyne", "fc glycan",
    ]
    cys_tokens = [
        "cystein", "cystien", "custein", "custien", "cysteine", "cystiene",
        "cys", "thiol", "maleimid",
    ]
    lys_tokens = ["lysin", "lysine", "lys", "nhs", "amine"]
    gln_tokens = ["glutam", "glutom", "transglutamin", "llqg", "gln", "tg"]

    if s not in ("", "nan", "none"):
        if any(t in s for t in click_tokens):
            return "click"
        if any(t in s for t in gln_tokens):
            return "glutamine"
        if any(t in s for t in cys_tokens):
            return "cysteine"
        if any(t in s for t in lys_tokens):
            return "lysine"
        # explicit "other" annotation
        if s == "other":
            pass  # try pattern fallback below
        else:
            # unrecognised but non-empty -> try pattern, else other
            pass

    # Fallback to pattern_stub hint (e.g. succinimide_thioether -> cysteine)
    p = "" if pattern_stub is None else str(pattern_stub).strip().lower()
    if p and p not in ("nan", "none"):
        if "thioether" in p or "succinimide" in p or "maleim" in p or "thiol" in p:
            return "cysteine"
        if "amide" in p or "nhs" in p or "lys" in p:
            return "lysine"
        if "triazol" in p or "click" in p:
            return "click"

    return "other"


# --- linker_class --------------------------------------------------------
# Rules applied IN ORDER (first match wins) to Canonical_Linker_Name.
def map_linker_class(name) -> str:
    """Normalize Canonical_Linker_Name -> linker chemistry class.

    Classes: ValCitPAB, GGFG, ValAla, ValCit, PEG, disulfide,
             non-cleavable_MCC, click-handle, other.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "unknown"
    s = str(name).strip()
    sl = s.lower()
    if sl in ("", "nan"):
        return "unknown"

    # disulfide first (SS / disulfide) -- chemistry trumps anything else
    if re.search(r"\bss\b", sl) or "-ss-" in sl or "disulf" in sl or "bis-ss" in sl:
        return "disulfide"

    # Gly-Gly-Phe-Gly (tetrapeptide, Dxd ADCs)
    if "gly-gly-phe-gly" in sl or "ggfg" in sl:
        return "GGFG"

    # Val-Cit-PAB family (the canonical cleavable dipeptide-PABC)
    if ("val-cit-pab" in sl or "val cit pab" in sl or "vc-pab" in sl
            or re.search(r"\bvc\b", sl) and "pab" in sl):
        return "ValCitPAB"

    # Val-Ala (cleavable dipeptide, no/with PAB)
    if "val-ala" in sl or "val ala" in sl or re.search(r"\bva\b", sl):
        return "ValAla"

    # Val-Cit without PAB (still cleavable dipeptide)
    if "val-cit" in sl or "val cit" in sl or "vc-pab" in sl:
        return "ValCit"

    # non-cleavable thioether maleimidomethyl-cyclohexane (MCC)
    if "mcc" in sl:
        return "non-cleavable_MCC"

    # click handles (azide/alkyne/strained-alkyne reagents)
    if any(t in sl for t in ("propargyl", "dbco", "bcn", "azid", "triazol", "click", "n3")):
        return "click-handle"

    # PEG-dominant (no recognised cleavable motif above)
    if "peg" in sl:
        return "PEG"

    return "other"


# ===========================================================================
# RDKit helpers
# ===========================================================================
def canonical_smiles(smi: str):
    if not HAVE_RDKIT:
        return None
    if smi is None or (isinstance(smi, float) and pd.isna(smi)) or str(smi).strip() == "":
        return None
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def stereo_status(linker_smi: str, payload_smi: str) -> str:
    """Return 'yes' / 'no' / 'unknown'.

    'yes'  : zero UNassigned stereocentres across the L/P pair AND the
             payload itself has >=1 *defined* stereocentre.
    'no'   : at least one unassigned stereocentre somewhere in the pair.
    'unknown': RDKit missing or a SMILES failed to parse.
    """
    if not HAVE_RDKIT:
        return "unknown"

    def centres(smi):
        if smi is None or str(smi).strip() == "" or str(smi).lower() == "nan":
            return None
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        all_c = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
        assigned = Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False)
        n_all = len(all_c)
        n_assigned = len(assigned)
        n_unassigned = n_all - n_assigned
        return n_assigned, n_unassigned

    lc = centres(linker_smi)
    pc = centres(payload_smi)
    if lc is None or pc is None:
        return "unknown"
    l_assigned, l_unassigned = lc
    p_assigned, p_unassigned = pc
    total_unassigned = l_unassigned + p_unassigned
    if total_unassigned == 0 and p_assigned >= 1:
        return "yes"
    return "no"


# ===========================================================================
# slug helper
# ===========================================================================
def slugify(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "na"


# ===========================================================================
# Trainset aggregation (anchors / spacer sizes by name+site, modal value)
# ===========================================================================
def load_trainset(train_csv: Path, val_csv: Path, has_site: bool) -> pd.DataFrame:
    parts = []
    for p in (train_csv, val_csv):
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def aggregate_anchors(df: pd.DataFrame, has_site: bool) -> pd.DataFrame:
    """Collapse conformer-augmented rows to one row per (linker, payload[, site]).

    Numeric / pattern columns -> modal value; record whether the group's
    anchor/spacer values were ambiguous (more than one distinct value).
    """
    if df.empty:
        return pd.DataFrame()

    keys = ["canonical_linker_name", "canonical_payload_name"]
    if has_site and "site_final" in df.columns:
        keys = keys + ["site_final"]

    def mode_or_blank(series):
        vals = series.dropna().tolist()
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    rows = []
    for key, grp in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(keys, key))
        amb = False
        for col in ("anchor_1", "anchor_2", "n_stub_atoms",
                    "n_payload_atoms", "n_spacer_atoms"):
            if col in grp.columns:
                rec[col] = mode_or_blank(grp[col])
                if grp[col].dropna().nunique() > 1:
                    amb = True
        if "pattern_stub" in grp.columns:
            rec["pattern_stub"] = mode_or_blank(grp["pattern_stub"])
        rec["anchors_ambiguous"] = amb
        rec["n_conformers"] = len(grp)
        rows.append(rec)
    return pd.DataFrame(rows)


# ===========================================================================
# Main build
# ===========================================================================
def build() -> dict:
    if not SPINE_CSV.exists():
        sys.exit(f"FATAL: spine not found: {SPINE_CSV}")

    spine = pd.read_csv(SPINE_CSV)
    n_spine = len(spine)

    # ---- exact-SMILES join keys ----------------------------------------
    spine["_lp_key"] = (
        spine["ADC_Linker_SMILES"].fillna("") + "||" + spine["ADC_Payload_SMILES"].fillna("")
    )
    assert spine["_lp_key"].is_unique, "spine SMILES pairs are not unique!"

    # ---- decomposed (site_normalized for all rows) ---------------------
    decomp = pd.read_csv(DECOMP_CSV)
    decomp["_lp_key"] = (
        decomp["linker_smiles_original"].fillna("")
        + "||"
        + decomp["payload_smiles"].fillna("")
    )
    decomp_small = decomp[["_lp_key", "site_normalized", "linker_smiles_starred",
                           "unresolved_reason"]].rename(
        columns={"site_normalized": "site_norm_decomp"}
    ).drop_duplicates("_lp_key")

    # ---- qc_passed (site_final, site_source) --------------------------
    qcp = pd.read_csv(QC_PASSED_CSV)
    qcp["_lp_key"] = (
        qcp["linker_smiles_original"].fillna("") + "||" + qcp["payload_smiles"].fillna("")
    )
    qcp_small = qcp[["_lp_key", "site_final", "site_source", "site_normalized"]].rename(
        columns={"site_normalized": "site_norm_qcp"}
    ).drop_duplicates("_lp_key")

    # ---- qc_flagged (site_normalized only) ----------------------------
    qcf = pd.read_csv(QC_FLAGGED_CSV)
    qcf["_lp_key"] = (
        qcf["linker_smiles_original"].fillna("") + "||" + qcf["payload_smiles"].fillna("")
    )
    qcf_small = qcf[["_lp_key", "site_normalized"]].rename(
        columns={"site_normalized": "site_norm_qcf"}
    ).drop_duplicates("_lp_key")
    qcf_keys = set(qcf_small["_lp_key"])
    qcp_keys = set(qcp_small["_lp_key"])

    # ---- merge site sources onto spine --------------------------------
    m = spine.merge(decomp_small, on="_lp_key", how="left")
    m = m.merge(qcp_small, on="_lp_key", how="left")
    m = m.merge(qcf_small, on="_lp_key", how="left")
    m["qc_status"] = m["_lp_key"].apply(
        lambda k: "passed" if k in qcp_keys else ("flagged" if k in qcf_keys else "none")
    )

    # ---- trainset anchor aggregations ---------------------------------
    v3 = load_trainset(V3_TRAIN, V3_VAL, has_site=True)
    v2 = load_trainset(V2_TRAIN, V2_VAL, has_site=False)
    v3_agg = aggregate_anchors(v3, has_site=True)   # keyed name+site
    v2_agg = aggregate_anchors(v2, has_site=False)  # keyed name only (cys)

    # name-pair / name+site presence sets for trainset flags
    v3_namesite = set()
    if not v3.empty:
        for _, r in v3.iterrows():
            v3_namesite.add((r["canonical_linker_name"], r["canonical_payload_name"],
                             r.get("site_final")))
    v2_names = set()
    if not v2.empty:
        for _, r in v2.iterrows():
            v2_names.add((r["canonical_linker_name"], r["canonical_payload_name"]))

    # ---------------------------------------------------------------------
    # Build manifest rows
    # ---------------------------------------------------------------------
    site_raw_to_clean = Counter()          # (raw -> clean) counts
    rows = []
    seen_lp_ids = {}
    for _, r in m.iterrows():
        payload_name = r["Canonical_Payload_Name"]
        linker_name = r["Canonical_Linker_Name"]
        raw_site = r["Lysine_Cysteine_Glutamine_Other_Linker_Conjugation_Site"]
        linker_smi = r["ADC_Linker_SMILES"]
        payload_smi = r["ADC_Payload_SMILES"]

        payload_class = map_payload_class(payload_name)
        linker_class = map_linker_class(linker_name)

        # -- resolved site preference: site_final > qc/decomp normalized > raw
        site_final = r.get("site_final")
        site_norm = None
        for cand in (r.get("site_norm_qcp"), r.get("site_norm_qcf"), r.get("site_norm_decomp")):
            if isinstance(cand, str) and cand.strip() and cand.strip().lower() != "nan":
                site_norm = cand
                break

        # conjugation_class from the best available signal:
        # prefer the curated site_final / normalized label, fall back to raw.
        site_for_class = None
        for cand in (site_final, site_norm):
            if isinstance(cand, str) and cand.strip() and cand.strip().lower() not in ("nan", ""):
                site_for_class = cand
                break
        if site_for_class is None:
            site_for_class = raw_site

        conjugation_class = map_conjugation_class(site_for_class)

        # record raw->clean normalization of the *raw* label for the report
        raw_label = "<EMPTY>" if (raw_site is None or (isinstance(raw_site, float) and pd.isna(raw_site)) or str(raw_site).strip() == "") else str(raw_site).strip()
        raw_clean = map_conjugation_class(raw_site)
        site_raw_to_clean[(raw_label, raw_clean)] += 1

        # -- trainset anchor lookup -----------------------------------
        anchor_1 = anchor_2 = n_spacer = n_stub = n_payload = pattern_stub = None
        anchors_ambiguous = None
        in_multisite = (linker_name, payload_name, site_final) in v3_namesite
        # also try without site requirement (some spine rows lack site_final)
        in_multisite_any = any(
            (linker_name == a and payload_name == b)
            for (a, b, c) in v3_namesite
        )
        in_cys = (linker_name, payload_name) in v2_names

        # prefer v3 (multi-site) anchors keyed name+site, else name-only modal
        agg_hit = None
        if not v3_agg.empty:
            sel = v3_agg[
                (v3_agg["canonical_linker_name"] == linker_name)
                & (v3_agg["canonical_payload_name"] == payload_name)
            ]
            if "site_final" in v3_agg.columns and isinstance(site_final, str):
                sel_site = sel[sel["site_final"] == site_final]
                if len(sel_site):
                    sel = sel_site
            if len(sel):
                agg_hit = sel.iloc[0]
        if agg_hit is None and not v2_agg.empty:
            sel = v2_agg[
                (v2_agg["canonical_linker_name"] == linker_name)
                & (v2_agg["canonical_payload_name"] == payload_name)
            ]
            if len(sel):
                agg_hit = sel.iloc[0]

        if agg_hit is not None:
            def gi(col):
                v = agg_hit.get(col)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return None
                return int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v
            anchor_1 = gi("anchor_1")
            anchor_2 = gi("anchor_2")
            n_spacer = gi("n_spacer_atoms")
            n_stub = gi("n_stub_atoms")
            n_payload = gi("n_payload_atoms")
            pattern_stub = agg_hit.get("pattern_stub")
            anchors_ambiguous = bool(agg_hit.get("anchors_ambiguous"))

        # -- stub_anchor / payload_anchor mapping ---------------------
        # In the trainset, anchor_1 = stub (linker-side conjugation anchor),
        # anchor_2 = payload-side anchor (per difflinker_trainset convention).
        stub_anchor_atom = anchor_1
        payload_anchor_atom = anchor_2

        # -- true_linker_size: spacer heavy atoms is the best available --
        true_linker_size = n_spacer

        # -- RDKit canonicalization + stereo --------------------------
        if HAVE_RDKIT:
            can_l = canonical_smiles(linker_smi)
            can_p = canonical_smiles(payload_smi)
            if can_l is not None and can_p is not None:
                canonical_lp_smiles = can_l + "." + can_p
            elif can_l is not None:
                canonical_lp_smiles = can_l
            else:
                canonical_lp_smiles = str(linker_smi) + "." + str(payload_smi)
            stereo_defined = stereo_status(linker_smi, payload_smi)
        else:
            canonical_lp_smiles = "unknown"
            stereo_defined = "unknown"

        # -- stable lp_id slug (payload + linker; disambiguate dups) ---
        base_id = f"{slugify(payload_name)}__{slugify(linker_name)}"
        if base_id in seen_lp_ids:
            seen_lp_ids[base_id] += 1
            lp_id = f"{base_id}__{seen_lp_ids[base_id]:02d}"
        else:
            seen_lp_ids[base_id] = 1
            lp_id = base_id  # first occurrence keeps clean slug

        rows.append({
            "lp_id": lp_id,
            "payload_name": payload_name if isinstance(payload_name, str) else "",
            "payload_class": payload_class,
            "conjugation_class": conjugation_class,
            "linker_class": linker_class,
            "linker_name": linker_name if isinstance(linker_name, str) else "",
            "raw_site_label": raw_label,
            "site_resolved": site_for_class if isinstance(site_for_class, str) else "",
            "qc_status": r["qc_status"],
            "payload_anchor_atom": payload_anchor_atom if payload_anchor_atom is not None else "",
            "stub_anchor_atom": stub_anchor_atom if stub_anchor_atom is not None else "",
            "spacer_atoms": n_spacer if n_spacer is not None else "",
            "n_stub_atoms": n_stub if n_stub is not None else "",
            "n_payload_atoms": n_payload if n_payload is not None else "",
            "true_linker_size": true_linker_size if true_linker_size is not None else "",
            "pattern_stub": pattern_stub if isinstance(pattern_stub, str) else "",
            "anchors_ambiguous": anchors_ambiguous if anchors_ambiguous is not None else "",
            "in_cys_trainset": bool(in_cys),
            "in_multisite_trainset": bool(in_multisite or in_multisite_any),
            "linker_smiles": linker_smi,
            "payload_smiles": payload_smi,
            "canonical_lp_smiles": canonical_lp_smiles,
            "stereo_defined": stereo_defined,
            "source_type": "ADCpedia",
        })

    manifest = pd.DataFrame(rows)

    # dedupe lp_id safety (should already be unique)
    assert manifest["lp_id"].is_unique, "lp_id collisions remain!"

    # ---- write outputs -------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_CSV, index=False)
    manifest.to_json(OUT_JSON, orient="records", indent=2)

    return {
        "manifest": manifest,
        "n_spine": n_spine,
        "site_raw_to_clean": site_raw_to_clean,
        "have_rdkit": HAVE_RDKIT,
    }


# ===========================================================================
# Summary printing
# ===========================================================================
def print_summary(res: dict) -> None:
    mf = res["manifest"]
    n = len(mf)

    print("=" * 72)
    print("ADCpedia dataset-contract manifest  (Wave-0 task 0.1)")
    print("=" * 72)
    print(f"RDKit available     : {res['have_rdkit']}")
    print(f"Spine rows (input)  : {res['n_spine']}")
    print(f"Manifest L/P rows   : {n}")
    print(f"CSV  -> {OUT_CSV}")
    print(f"JSON -> {OUT_JSON}")

    print("\n--- payload_class_map (raw payload name -> normalized class) ---")
    for k, v in sorted(PAYLOAD_CLASS_MAP.items()):
        print(f"  {k:<26s} -> {v}")

    print("\n--- payload_class distribution ---")
    print(mf["payload_class"].value_counts().to_string())

    print("\n--- conjugation_class distribution ---")
    print(mf["conjugation_class"].value_counts().to_string())

    print("\n--- linker_class distribution ---")
    print(mf["linker_class"].value_counts().to_string())

    print("\n--- site-label normalization map (RAW -> clean : count) ---")
    # group by raw label, show the clean class + count
    by_raw = {}
    for (raw, clean), c in res["site_raw_to_clean"].items():
        by_raw.setdefault(raw, []).append((clean, c))
    for raw in sorted(by_raw, key=lambda x: (x == "<EMPTY>", x.lower())):
        for clean, c in sorted(by_raw[raw], key=lambda t: -t[1]):
            print(f"  {raw:<32s} -> {clean:<10s} : {c}")

    print("\n--- anchors / trainability ---")
    has_anchor = (mf["stub_anchor_atom"] != "") & (mf["payload_anchor_atom"] != "")
    print(f"  rows with both anchors      : {int(has_anchor.sum())} / {n}")
    print(f"  rows with spacer_atoms      : {int((mf['spacer_atoms'] != '').sum())} / {n}")
    print(f"  in_cys_trainset             : {int(mf['in_cys_trainset'].sum())}")
    print(f"  in_multisite_trainset       : {int(mf['in_multisite_trainset'].sum())}")
    amb = mf["anchors_ambiguous"].apply(lambda x: x is True or x == True)
    print(f"  anchors_ambiguous (flagged) : {int(amb.sum())}")

    print("\n--- stereo_defined distribution ---")
    print(mf["stereo_defined"].value_counts().to_string())

    print("\n--- qc_status distribution ---")
    print(mf["qc_status"].value_counts().to_string())

    # LOPO usability: payload classes with >= 5 examples
    print("\n--- leave-one-payload-class-out (LOPO) usability ---")
    vc = mf["payload_class"].value_counts()
    usable = vc[vc >= 5]
    print(f"  payload_classes with >=5 examples : {len(usable)}")
    print(usable.to_string())
    print(f"  total L/Ps in those classes        : {int(usable.sum())}")
    not_usable = vc[vc < 5]
    if len(not_usable):
        print(f"  classes with <5 (excluded from LOPO): {list(not_usable.index)}")

    print("\nDone.")


if __name__ == "__main__":
    res = build()
    print_summary(res)

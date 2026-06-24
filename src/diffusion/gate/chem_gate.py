#!/usr/bin/env python
"""Unified chemical-validity gate for the ADCpedia DiffLinker -> MD pipeline.

This module is the single, versioned, tested consolidation of the project's two
previously-separate ad-hoc gate scripts (Wave-0 task 0.2):

  * ``outputs/h4/eval_3dstrict.py`` — the "3D-strict usable" gate for generated
    DiffLinker outputs (Master doc §19.16 criterion).  Reproduced here as
    :func:`gate_usable3d`.
  * ``md/chem_validity_gate.py`` — the pre-MD ligand + topology gate.  Reproduced
    here as :func:`gate_ligand` (Level A, RDKit) and
    :func:`gate_topology_valence` (Level B, pure-text GROMACS .itp parse).

The chemistry / logic / thresholds / SMARTS of the originals are preserved
EXACTLY.  In particular:

  * ``native_connected`` is computed on the RAW obabel-perceived SDF graph with
    ``len(Chem.GetMolFrags(mol)) == 1`` and **NO MST closure / no post-hoc bond
    addition**.  This is a hard project requirement (Master §19.16).
  * ``neutralise`` zeroes radical electrons + clears NoImplicit before
    ``SanitizeMol`` so MMFF can parameterise the molecule and SMARTS can match;
    this fills implicit H WITHOUT changing heavy-atom topology, so it does not
    affect the raw-graph ``native_connected`` measured beforehand.
  * MMFF success = parameterisable AND minimises (maxIts=1000) to a finite
    energy < 1000 kcal/mol AND still a single fragment after ``RemoveHs``.  The
    strict convergence flag is logged but NOT required (large noisy geometries
    routinely need >1000 steps yet relax to sane energy).
  * Level-B valence uses the ``MAX_VAL`` table on the per-atom bond degree
    parsed from the ``[ atoms ]`` / ``[ bonds ]`` blocks of the itp text.

RDKit-version note: the project's RDKit is old (2022.03/2022.09).
``rdDetermineBonds`` is NOT used — bond perception relies entirely on the
obabel/SDF input, exactly as the originals did.  No API newer than rdkit
2022.03 is introduced.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# Level B: per-element maximum valence (degree).  Copied verbatim from
# md/chem_validity_gate.py.  rdkit is NOT needed for Level B, so it is imported
# lazily inside the rdkit-using functions only.
# ---------------------------------------------------------------------------
MAX_VAL = {'C': 4, 'N': 4, 'O': 2, 'H': 1, 'S': 6, 'P': 5, 'F': 1, 'Cl': 1, 'Br': 1}

# GAFF/AMBER atom-type first-letter -> element.  Copied verbatim from the
# original Level-B parser.
_GAFF_ELEM = {'c': 'C', 'n': 'N', 'o': 'O', 'h': 'H', 's': 'S', 'p': 'P', 'f': 'F'}


# ===========================================================================
# Level A — ligand validity (RDKit on SMILES / SDF)
#   wraps md/chem_validity_gate.py :: check_ligand
# ===========================================================================
def gate_ligand(src, name=''):
    """Level-A ligand gate (RDKit).  Wraps ``chem_validity_gate.check_ligand``.

    ``src`` may be a SMILES string, or a path ending in ``.sdf`` / ``.mol``.

    Returns a dict::

        {
          'name': str,
          'parsed': bool,
          'sanitizable': bool,
          'single_fragment': bool,
          'n_fragments': int,
          'undefined_stereo': int,        # number of unassigned stereocenters
          'maleimide_state': 'open' | 'closed/none',
          'open_maleimide': bool,
          'druglike': bool,               # MW <= 1600 (original sole MW gate)
          'props': {'MW','logP','rotB','TPSA'},
          'flags': [...],                 # original human-readable flag list
          'reasons': [...],               # machine reason codes for failures
          'ok': bool,
        }

    The flag strings, SMARTS (open maleimide ``O=C1C=CC(=O)N1``), drug-likeness
    threshold (MW > 1600) and stereo logic
    (``FindMolChiralCenters(includeUnassigned=True)`` counting ``'?'``) are
    identical to the original.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, FindMolChiralCenters
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    OPEN_MAL = Chem.MolFromSmarts('O=C1C=CC(=O)N1')  # unreacted maleimide (C=C intact)

    rec = dict(name=name, parsed=False, sanitizable=False, single_fragment=False,
               n_fragments=0, undefined_stereo=0, maleimide_state='closed/none',
               open_maleimide=False, druglike=True, props={}, flags=[],
               reasons=[], ok=False)

    m = (Chem.MolFromSmiles(src) if not src.endswith(('.sdf', '.mol'))
         else Chem.MolFromMolFile(src, sanitize=False))
    flags = []
    if m is None:
        rec['flags'] = ['UNPARSEABLE']
        rec['reasons'] = ['UNPARSEABLE']
        return rec
    rec['parsed'] = True

    try:
        Chem.SanitizeMol(m)
        sane = True
    except Exception as e:  # noqa: BLE001 — preserve original broad catch
        sane = False
        flags.append(f'NOT-SANITIZABLE ({str(e)[:40]})')
    rec['sanitizable'] = sane
    if not sane:
        rec['reasons'].append('NOT-SANITIZABLE')

    frags = Chem.GetMolFrags(m)
    rec['n_fragments'] = len(frags)
    rec['single_fragment'] = (len(frags) == 1)
    if len(frags) > 1:
        flags.append(f'DISCONNECTED ({len(frags)} fragments)')  # DiffLinker short-bond defect
        rec['reasons'].append('DISCONNECTED')

    try:
        undef = FindMolChiralCenters(m, useLegacyImplementation=False, includeUnassigned=True)
        n_undef = sum(1 for _, c in undef if c == '?')
        rec['undefined_stereo'] = n_undef
        if n_undef:
            flags.append(f'{n_undef} undefined stereocenter(s)')
            rec['reasons'].append('UNDEFINED-STEREO')
    except Exception:  # noqa: BLE001 — preserve original behaviour
        pass

    if m.HasSubstructMatch(OPEN_MAL):
        rec['open_maleimide'] = True
        rec['maleimide_state'] = 'open'
        flags.append('OPEN-MALEIMIDE (unreacted C=C — not a conjugated thiosuccinimide)')
        rec['reasons'].append('OPEN-MALEIMIDE')

    props = {'MW': round(Descriptors.MolWt(m), 0), 'logP': round(Descriptors.MolLogP(m), 1),
             'rotB': rdMolDescriptors.CalcNumRotatableBonds(m), 'TPSA': round(Descriptors.TPSA(m), 0)}
    rec['props'] = props
    if props['MW'] > 1600:
        rec['druglike'] = False
        flags.append(f"MW {props['MW']} > 1600 (very large)")
        rec['reasons'].append('MW>1600')

    rec['flags'] = flags or ['OK']
    rec['ok'] = (len(rec['reasons']) == 0)
    return rec


# ===========================================================================
# Level B — conjugate / topology valence (pure-text GROMACS .itp parse)
#   wraps md/chem_validity_gate.py :: check_topology_valence
# ===========================================================================
def gate_topology_valence(itp_path_or_text, name=''):
    """Level-B topology gate.  Wraps ``chem_validity_gate.check_topology_valence``.

    ``itp_path_or_text`` may be a path to a GROMACS ``.itp`` file OR the raw itp
    text itself.  (A heuristic distinguishes them: anything containing a newline
    or a ``[ ... ]`` directive is treated as inline text.)

    Returns a dict::

        {
          'name': str,
          'bad_atoms': [ {'idx','type','degree','element','max'} , ... ],
          'retained_thiol_h': [ {'s_idx','s_type','h_idx'} , ... ],
          'reasons': [...],     # e.g. ['OVER-VALENT-C','OVER-VALENT-S','RETAINED-THIOL-H']
          'ok': bool,
        }

    Degree-based valence parsing, element mapping and the ``MAX_VAL`` thresholds
    are identical to the original text parser (atom degree counted from
    ``[ bonds ]`` blocks, atom type -> element via first letter lower-cased);
    this catches the pentavalent C3 (§20.10).  ADDED in v1.1.0: an S-H bond
    detector that flags ``RETAINED-THIOL-H`` — the literal §19.28 bug where the
    conjugation sulfur kept its thiol proton (a 3-coordinate S that the degree
    test misses, since 3 <= MAX_VAL['S']=6).  A clean post-conjugation thioether
    has zero S-H bonds.  ``MAX_VAL`` is unchanged.
    """
    if _looks_like_text(itp_path_or_text):
        txt = itp_path_or_text
    else:
        with open(itp_path_or_text) as fh:
            txt = fh.read()

    atoms = {int(c[0]): c[1] for c in (l.split() for l in
             re.search(r'\[ *atoms *\]\s*\n(.*?)(?=^\[ )', txt, re.M | re.S).group(1).splitlines())
             if c and c[0].isdigit()}
    deg = {}
    bonds = []
    for blk in re.findall(r'^\[ *bonds *\]\s*\n(.*?)(?=^\[ |\Z)', txt, re.M | re.S):
        for l in blk.splitlines():
            c = l.split()
            if len(c) >= 2 and c[0].isdigit() and c[1].isdigit():
                a, b = int(c[0]), int(c[1])
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
                bonds.append((a, b))
    bad = []
    for idx, gafftype in atoms.items():
        el = _GAFF_ELEM.get(gafftype[0].lower())
        if el and idx in deg and deg[idx] > MAX_VAL.get(el, 4):
            bad.append((idx, gafftype, deg[idx], el))

    # §19.28 hardening: a post-conjugation thioether sulfur must carry NO
    # hydrogen.  The cysteine thiol proton (HG) must be removed when the C-SG
    # covalent bond forms; if it is retained the sulfur is 3-coordinate (Cbeta +
    # ligand C + H), which the degree test alone MISSES because 3 <= MAX_VAL['S']
    # = 6.  We therefore flag *any* S-H bond in the conjugate topology as
    # RETAINED-THIOL-H.  (The original gate caught the pentavalent C, not this;
    # this is the exact bug from Master §19.28.)  A clean thioether has 0 S-H.
    retained = []
    for a, b in bonds:
        ea = _GAFF_ELEM.get(atoms.get(a, ' ')[0].lower())
        eb = _GAFF_ELEM.get(atoms.get(b, ' ')[0].lower())
        if {ea, eb} == {'S', 'H'}:
            s_idx, h_idx = (a, b) if ea == 'S' else (b, a)
            retained.append((s_idx, atoms.get(s_idx, '?'), h_idx))

    reasons = [f'OVER-VALENT-{e}' for (_, _, _, e) in bad]
    if retained:
        reasons.append('RETAINED-THIOL-H')

    rec = dict(name=name,
               bad_atoms=[dict(idx=i, type=t, degree=d, element=e, max=MAX_VAL[e])
                          for (i, t, d, e) in bad],
               retained_thiol_h=[dict(s_idx=s, s_type=t, h_idx=h)
                                 for (s, t, h) in retained],
               reasons=reasons,
               ok=(len(bad) == 0 and len(retained) == 0))
    return rec


def _looks_like_text(s):
    """True if ``s`` should be treated as inline itp text rather than a path."""
    if not isinstance(s, str):
        return False
    if '\n' in s:
        return True
    if '[' in s and ']' in s:  # a single-line directive fragment
        return True
    return False


# ===========================================================================
# 3D-strict usable gate (DiffLinker generated SDFs)
#   wraps outputs/h4/eval_3dstrict.py
# ===========================================================================

# --- ADC cleavable-linker motif SMARTS (verbatim from eval_3dstrict.py) -----
# UREA / ureido: carbonyl C flanked by two N (citrulline ureido / urea spacer);
# does NOT match the imide carbonyls of maleimide/succinimide.
SMARTS_UREA = None
# VALCIT: citrulline side chain — ureido on a 3-carbon side chain.
SMARTS_VALCIT = None


def _ensure_smarts():
    """Lazily compile the motif SMARTS (needs rdkit)."""
    global SMARTS_UREA, SMARTS_VALCIT
    if SMARTS_UREA is None or SMARTS_VALCIT is None:
        from rdkit import Chem
        SMARTS_UREA = Chem.MolFromSmarts('[NX3][CX3](=[OX1])[NX3]')
        SMARTS_VALCIT = Chem.MolFromSmarts('[NX3][CX3](=[OX1])[NX3]CCC')


def neutralise(mol):
    """Fill valences / kill the radicals DiffLinker's atomic mode leaves, then
    sanitise.  Returns an RWMol or None.  Verbatim from eval_3dstrict.py.

    (SMARTS must run on the neutralised mol — radicals break substructure
    matching.  Neutralisation fills implicit H WITHOUT changing heavy-atom
    topology, so the raw-graph native_connected measured beforehand is
    unaffected.)
    """
    from rdkit import Chem
    try:
        m = Chem.Mol(mol)
        for a in m.GetAtoms():
            a.SetNumRadicalElectrons(0)
            a.SetNoImplicit(False)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def gate_usable3d(sdf):
    """3D-strict usable gate for one generated SDF.  Wraps
    ``eval_3dstrict.eval_one`` — logic preserved EXACTLY.

    A generation is usable_3d iff it is simultaneously:
      (a) NATIVELY connected — single fragment on the RAW obabel-perceived
          graph, NO MST closure / no post-hoc bonds,
      (b) MMFF-relaxable — survives MMFF94 minimisation to a finite, sane
          energy (< 1000 kcal/mol) without breaking topology, and
      (c) chemistry-bearing — carries an ADC cleavable motif (ValCit OR Urea).

    Returns the original per-mol record dict with keys:
    ``sdf, parsed, native_connected, mmff_ok, mmff_energy, mmff_converged,
    has_valcit, has_urea, usable_3d, smiles``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _ensure_smarts()

    rec = dict(sdf=os.path.basename(sdf), parsed=False, native_connected=False,
               mmff_ok=False, mmff_energy=None, mmff_converged=None,
               has_valcit=False, has_urea=False, usable_3d=False, smiles='')
    try:
        mol = next(iter(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)), None)
    except Exception:
        mol = None
    if mol is None:
        return rec
    rec['parsed'] = True

    # (a) NATIVE connectivity — single fragment on the obabel-perceived graph,
    # NO MST closure.  HARD project requirement.
    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return rec
    except Exception:
        return rec
    rec['native_connected'] = True

    # Neutralise once (radicals -> implicit H, no heavy-atom topology change).
    neu = neutralise(mol)

    # (b) MMFF relaxation must succeed with a sane energy.
    if neu is not None:
        try:
            mh = Chem.AddHs(neu, addCoords=True)
            mp = AllChem.MMFFGetMoleculeProperties(mh)
            ff = AllChem.MMFFGetMoleculeForceField(mh, mp) if mp is not None else None
            if ff is not None:
                converged = ff.Minimize(maxIts=1000)
                energy = ff.CalcEnergy()
                rec['mmff_converged'] = int(converged)
                rec['mmff_energy'] = round(float(energy), 2)
                if math.isfinite(energy) and energy < 1000 and \
                   len(Chem.GetMolFrags(Chem.RemoveHs(mh))) == 1:
                    rec['mmff_ok'] = True
        except Exception:
            pass

    # (c) ADC chemistry on the radical-neutralised molecule.
    if neu is not None:
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(neu))
            rec['smiles'] = smi
            recheck = Chem.MolFromSmiles(smi)
            if recheck is not None:
                rec['has_valcit'] = recheck.HasSubstructMatch(SMARTS_VALCIT)
                rec['has_urea'] = recheck.HasSubstructMatch(SMARTS_UREA)
        except Exception:
            pass

    rec['usable_3d'] = bool(rec['native_connected'] and rec['mmff_ok']
                            and (rec['has_valcit'] or rec['has_urea']))
    return rec


# ===========================================================================
# Fail-closed convenience
# ===========================================================================
def gate_all(ligand=None, topology=None, sdf=None):
    """Fail-closed convenience gate combining all available levels.

    Pass any subset of:
      * ``ligand``  — SMILES/SDF path for the Level-A ligand gate,
      * ``topology``— .itp path/text for the Level-B valence gate,
      * ``sdf``     — generated SDF path for the 3D-strict usable gate.

    Returns::

        {'passed': bool, 'reasons': [...], 'ligand': {...}|None,
         'topology': {...}|None, 'usable3d': {...}|None}

    Fail-closed: if NO level was requested, or any requested level reports a
    failure, ``passed`` is False.  Reason codes are namespaced by level
    (``ligand:`` / ``topology:`` / ``usable3d:``).
    """
    out = dict(passed=False, reasons=[], ligand=None, topology=None, usable3d=None)
    ran_any = False

    if ligand is not None:
        ran_any = True
        lr = gate_ligand(ligand)
        out['ligand'] = lr
        if not lr['ok']:
            out['reasons'] += [f'ligand:{r}' for r in lr['reasons']]

    if topology is not None:
        ran_any = True
        tr = gate_topology_valence(topology)
        out['topology'] = tr
        if not tr['ok']:
            out['reasons'] += [f'topology:{r}' for r in tr['reasons']]

    if sdf is not None:
        ran_any = True
        ur = gate_usable3d(sdf)
        out['usable3d'] = ur
        if not ur['usable_3d']:
            reasons = []
            if not ur['native_connected']:
                reasons.append('NOT-NATIVE-CONNECTED')
            if not ur['mmff_ok']:
                reasons.append('MMFF-FAIL')
            if not (ur['has_valcit'] or ur['has_urea']):
                reasons.append('NO-CLEAVABLE-MOTIF')
            out['reasons'] += [f'usable3d:{r}' for r in reasons]

    if not ran_any:
        out['reasons'].append('NO-INPUT')

    out['passed'] = ran_any and (len(out['reasons']) == 0)
    return out


# ===========================================================================
# CLI
# ===========================================================================
def _cmd_ligand(args):
    rec = gate_ligand(args.src, name=args.name or '')
    print(json.dumps(rec, indent=2))
    return 0 if rec['ok'] else 1


def _cmd_topology(args):
    rec = gate_topology_valence(args.itp, name=args.name or '')
    print(json.dumps(rec, indent=2))
    return 0 if rec['ok'] else 1


def _cmd_usable3d(args):
    target = args.path
    if os.path.isdir(target):
        sdfs = sorted(glob.glob(os.path.join(target, '*.sdf')))
        recs = [gate_usable3d(s) for s in sdfs]
        n = len(recs)
        summary = dict(dir=target, n=n)
        for k in ('parsed', 'native_connected', 'mmff_ok', 'has_valcit', 'has_urea', 'usable_3d'):
            c = sum(1 for r in recs if r[k])
            summary[k] = c
            summary[f'{k}_pct'] = round(100.0 * c / n, 1) if n else 0.0
        print(json.dumps(dict(summary=summary, per_mol=recs), indent=2))
        return 0 if (n and summary['usable_3d'] == n) else 1
    else:
        rec = gate_usable3d(target)
        print(json.dumps(rec, indent=2))
        return 0 if rec['usable_3d'] else 1


def build_parser():
    ap = argparse.ArgumentParser(
        prog='diffusion.gate.chem_gate',
        description='Unified chemical-validity gate (Level A ligand, Level B '
                    'topology valence, 3D-strict usable). Wraps eval_3dstrict + '
                    'chem_validity_gate.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_lig = sub.add_parser('ligand', help='Level-A ligand gate (SMILES or SDF) -> JSON')
    p_lig.add_argument('src', help='SMILES string or path to .sdf/.mol')
    p_lig.add_argument('name', nargs='?', default='', help='optional label')
    p_lig.set_defaults(func=_cmd_ligand)

    p_top = sub.add_parser('topology', help='Level-B per-atom valence gate (.itp) -> JSON')
    p_top.add_argument('itp', help='path to GROMACS .itp (or inline itp text)')
    p_top.add_argument('name', nargs='?', default='', help='optional label')
    p_top.set_defaults(func=_cmd_topology)

    p_u3d = sub.add_parser('usable3d', help='3D-strict usable gate (SDF file or dir) -> JSON')
    p_u3d.add_argument('path', help='path to an SDF file OR a directory of *.sdf')
    p_u3d.set_defaults(func=_cmd_usable3d)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

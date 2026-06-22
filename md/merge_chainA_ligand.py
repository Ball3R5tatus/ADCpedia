#!/usr/bin/env python
"""Merge ligand into Protein_chain_A moleculetype with a CHEMICALLY-CORRECT
covalent Cys-S-C thioether (maleimide-Cys conjugation).

The thiol S-H is CONSUMED when the thioether forms, so this script:
  a. deletes the conjugated Cys HG (thiol proton) and renumbers every atom
     after it (-1); ligand atoms are then spliced in at the shifted offset.
  b. removes every bonded term that referenced HG (SG-HG bond, CB-SG-HG angle,
     X-CB-SG-HG dihedrals) -- done generically by dropping any term containing
     the HG index.
  c. retypes the conjugated SG from SH (thiol) to S (thioether; same atom type
     amber99sb-ildn uses for the CYS2/CYX disulfide sulfur).
  d. folds HG's partial charge into SG (SG_q += HG_q). This is charge-CONSERVING
     -> the chain stays at the exact integer charge it had pre-merge, so the
     Cys+ligand junction is net-neutral by construction. (Resulting SG charge
     ~ -0.1034 e, within 0.005 e of amber99sb-ildn CYS2 SG = -0.1081 e.)
  e. C3 (ligand attachment atom) index is read GENERICALLY from restraint.json
     (C3_global - npro), never hardcoded -- each candidate has its own C3.

Ligand bonded sections are APPENDED as additional same-named sections (GROMACS
concatenates them). The C3-SG bond uses explicit params (no bondtype lookup).
Also shifts posre_Protein_chain_A.itp to match, and records HG_global_removed
in restraint.json so the coordinate step (strip_hg.py) drops the same atom.

Usage: python merge_chainA_ligand.py <cid>   (run on FRESH pre-merge inputs)
"""
import sys, json, re
cid = sys.argv[1]; run = f'/home/galeito/ADCpedia/md/runs/cand_{cid}'
r = json.load(open(f'{run}/restraint.json'))
SG_local = r['SG_global']               # SG is in chain A (first molecule) -> global == local
C3_off   = r['C3_global'] - r['npro']   # 1-based index of C3 WITHIN the ligand (generic)
chainA = f'{run}/topol_Protein_chain_A.itp'
ligf   = f'{run}/LIG_moltype.itp'
posreA = f'{run}/posre_Protein_chain_A.itp'

def sections(path):
    secs=[]; name=None; buf=[]
    for ln in open(path):
        m=re.match(r'\s*\[\s*(\w+)\s*\]', ln)
        if m: secs.append((name,buf)); name=m.group(1); buf=[ln]
        else: buf.append(ln)
    secs.append((name,buf)); return secs
def is_data(l): return bool(l.strip()) and not l.lstrip().startswith((';','[','#'))
def rows(buf): return [l for l in buf if is_data(l)]

A = sections(chainA)
atoms_buf = next(b for n,b in A if n=='atoms')

# --- locate SG and its thiol proton HG (same residue, atom name HG) ---
sg_resnr=None; sg_charge=None
for l in rows(atoms_buf):
    p=l.split()
    if int(p[0])==SG_local:
        assert p[4]=='SG', f'atom {SG_local} is {p[4]!r}, not SG'
        sg_resnr=p[2]; sg_charge=float(p[6])
assert sg_charge is not None, f'SG atom {SG_local} not found in chain A [atoms]'
hg_local=None; hg_charge=0.0
for l in rows(atoms_buf):
    p=l.split()
    if p[2]==sg_resnr and p[4]=='HG':
        hg_local=int(p[0]); hg_charge=float(p[6])
        assert p[1] in ('HS','H'), f'HG type {p[1]!r} unexpected for a thiol proton'
assert hg_local is not None, 'conjugated Cys HG (thiol proton) not found -- already merged?'
assert hg_local > SG_local, 'expected HG to follow SG so SG index is preserved'
sg_new_charge = round(sg_charge + hg_charge, 4)

def remap(i):
    """index after deleting HG: HG -> None (drop), >HG -> -1, else unchanged."""
    if i==hg_local: return None
    return i-1 if i>hg_local else i

# --- rebuild protein [atoms]: drop HG, retype+recharge SG, renumber ---
new_atoms=[]
for l in atoms_buf:
    if not is_data(l): new_atoms.append(l); continue
    p=l.split()[:8]                       # nr type resnr residue atom cgnr charge mass
    nr=int(p[0])
    if nr==hg_local: continue             # (d/a) delete the thiol proton
    p[0]=str(remap(nr))                   # (a) renumber
    if nr==SG_local:
        p[1]='S'                          # (c) SH -> thioether S
        p[6]=f'{sg_new_charge:.4f}'       # (d) fold HG charge into SG
    new_atoms.append('  '+'  '.join(p)+'\n')
nA=sum(1 for l in new_atoms if is_data(l))   # protein atom count after HG removal

# --- (b) drop HG-containing bonded terms + renumber protein bonded sections ---
NCOL={'bonds':2,'pairs':2,'angles':3,'dihedrals':4,'position_restraints':1}
def xform_protein(buf,ncol):
    out=[]
    for l in buf:
        if not is_data(l): out.append(l); continue
        p=l.split()
        mapped=[remap(int(p[k])) for k in range(ncol)]
        if any(v is None for v in mapped): continue   # term touched HG -> remove it
        for k in range(ncol): p[k]=str(mapped[k])
        out.append('  '+'  '.join(p)+'\n')
    return out

# --- ligand pieces ---
Lig=sections(ligf)

# (f) LIGAND-SIDE valence fix: the attach carbon C3 is parametrized as a CH2
#     (saturated succinimide). A real thiosuccinimide is -CH(S)-, so adding the
#     S-C bond WITHOUT deleting an H made C3 PENTAVALENT. Delete ONE H bonded to
#     C3 (symmetric to the HG deletion) and fold its charge into C3 (charge-
#     conserving -> ligand stays net-integer). lmap renumbers the ligand locally.
lig_rows = rows(next(b for n,b in Lig if n=='atoms'))
lig_atom = {int(l.split()[0]): l.split() for l in lig_rows}
assert lig_atom[C3_off][1][0] in 'cC', f'ligand C3 (atom {C3_off}) is type {lig_atom[C3_off][1]}, not carbon'
def _isH(p): return p[1][0] in 'hH' or (len(p)>=8 and abs(float(p[7])-1.008)<0.3)
lig_bond_rows = rows(next((b for n,b in Lig if n=='bonds'), []))
Hn=[]
for l in lig_bond_rows:
    a,b=int(l.split()[0]),int(l.split()[1])
    if a==C3_off and _isH(lig_atom[b]): Hn.append(b)
    if b==C3_off and _isH(lig_atom[a]): Hn.append(a)
assert Hn, f'no H bonded to ligand C3 (atom {C3_off}) -- cannot fix valence'
hH=max(Hn); hH_charge=float(lig_atom[hH][6])
def lmap(i): return None if i==hH else (i-1 if i>hH else i)

def renum_lig(buf,ncol):                 # drop terms touching the deleted H, lmap, then +nA
    out=[]
    for l in buf:
        if not is_data(l): out.append(l); continue
        p=l.split(); mp=[lmap(int(p[k])) for k in range(ncol)]
        if any(v is None for v in mp): continue
        for k in range(ncol): p[k]=str(mp[k]+nA)
        out.append('  '+'  '.join(p)+'\n')
    return out
def renum_lig_atoms(buf):                # delete C3-H, fold its charge into C3, lmap, +nA
    out=[]
    for l in buf:
        if not is_data(l): out.append(l); continue
        p=l.split(); nr=int(p[0])
        if nr==hH: continue
        if nr==C3_off: p[6]=f'{float(p[6])+hH_charge:.6f}'
        p[0]=str(lmap(nr)+nA); p[2]=str(int(p[2])+1000)
        if len(p)>=6: p[5]=str(int(p[5])+nA)
        out.append('  '+'  '.join(p)+'\n')
    return out

lig_atom_rows=renum_lig_atoms(rows(next(b for n,b in Lig if n=='atoms')))
append=[]
for name,buf in Lig:
    if name in ('moleculetype','atoms',None): continue
    if name in NCOL:
        append.append('\n'); append.append(f'[ {name} ]\n')
        append += renum_lig(rows(buf), NCOL[name])

C3_local=nA+lmap(C3_off)
append += ['\n[ bonds ]\n','; covalent Cys-SG -- ligand C (maleimide-Cys thioether)\n',
           f'  {SG_local}  {C3_local}  1  0.18100  185769.6\n']

# --- write chain A itp (atoms replaced; protein bonded sections transformed) ---
out=[]
for name,buf in A:
    if name=='atoms':                               # protein atoms + spliced ligand atoms
        li=max(i for i,l in enumerate(new_atoms) if l.strip())
        out += new_atoms[:li+1] + lig_atom_rows + new_atoms[li+1:]
    elif name in NCOL:          out+=xform_protein(buf, NCOL[name])
    else:                       out+=buf            # moleculetype, trailing #ifdef POSRES include
out+=append
open(chainA,'w').writelines(out)
print(f'chainA: protein {nA} atoms (HG {hg_local} deleted) + {len(lig_atom_rows)} ligand atoms = {nA+len(lig_atom_rows)}')
print(f'   SG {SG_local}: type SH->S, charge {sg_charge:+.4f} -> {sg_new_charge:+.4f} (folded HG {hg_charge:+.4f})')
print(f'   covalent bond: SG {SG_local} -- C3 {C3_local} (ligand offset {C3_off})')

# --- shift protein position restraints to the new numbering (HG is a proton: not listed) ---
pr=sections(posreA); out=[]
for name,buf in pr:
    out += xform_protein(buf,1) if name=='position_restraints' else buf
open(posreA,'w').writelines(out)
print('posre_Protein_chain_A.itp renumbered')

# --- record removed atoms so the coordinate step strips the same ones ---
r['HG_global_removed']=hg_local                 # Cys thiol proton (chain-A protein block)
r['C3_H_removed_lig']=hH                         # ligand-local 1-based index of the deleted C3 hydrogen
r['nlig_merged']=len(lig_atom_rows)              # ligand atom count AFTER H removal (analysis uses this, not nlig)
json.dump(r, open(f'{run}/restraint.json','w'), indent=2)
print(f'   ligand C3 (off {C3_off}): deleted 1 H (lig atom {hH}, q {hH_charge:+.4f} folded into C3) -> tetravalent CH(S)')

# --- topol.top: drop standalone LIG (now merged into chain A) ---
top=[l for l in open(f'{run}/topol.top') if 'LIG_moltype.itp' not in l
     and l.strip()!='; ligand moleculetype' and not re.match(r'\s*LIG\s+1\s*$',l)]
open(f'{run}/topol.top','w').writelines(top)
print('topol.top patched (LIG merged into chain A); HG_global_removed='+str(hg_local)+' written to restraint.json')

# --- FAIL-CLOSED valence gate: refuse to produce a topology with any over-valent
#     atom (would have auto-caught the pentavalent C3 §20.10 and over-coordinated
#     SG §20.0). Pure-text check; no rdkit needed. ---
from chem_validity_gate import check_topology_valence, MAX_VAL
_, bad = check_topology_valence(chainA)
if bad:
    sys.stderr.write('\n*** VALENCE GATE FAILED — over-valent atom(s) in merged chain A: ***\n')
    for i, t, d, e in bad:
        sys.stderr.write(f'   atom {i} (type {t}) has {d} bonds (max {MAX_VAL[e]} for {e})\n')
    sys.stderr.write('Refusing to emit a chemically-invalid topology. Fix the conjugation step.\n')
    sys.exit(1)
print('valence gate: OK — no over-valent atom in merged chain A')

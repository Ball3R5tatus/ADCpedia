#!/usr/bin/env python
"""Merge pdb2gmx protein + acpype ligand into a GROMACS complex; locate C3...SG
restraint atoms by geometry. Run inside a cand run dir after pdb2gmx + acpype.
Usage: python gmx_assemble.py <cid>
"""
import sys, json, os, math
cid = sys.argv[1]   # str cid (e.g. "vedotin") or numeric; used only for paths + restraint.json
run = f'/home/galeito/ADCpedia/md/runs/cand_{cid}'
acp = f'/home/galeito/ADCpedia/md/lig_params/cand_{cid}_run/LIG.acpype'
info = json.load(open('/home/galeito/ADCpedia/md/lig_params/posed_info.json'))[str(cid)]
C3_lig0 = info['C3_idx_in_posed_sdf']  # 0-based in ligand

def read_gro(p):
    L = open(p).read().splitlines()
    n = int(L[1]); atoms = L[2:2+n]; box = L[2+n]
    return L[0], n, atoms, box

# --- protein ---
ptitle, npro, patoms, pbox = read_gro(f'{run}/prot.gro')
# --- ligand ---
ltitle, nlig, latoms, lbox = read_gro(f'{acp}/LIG_GMX.gro')

def coords(line):  # gro fixed cols: resid(5)resname(5)name(5)idx(5) x,y,z (8.3 each)
    x=float(line[20:28]); y=float(line[28:36]); z=float(line[36:44]); return (x,y,z)
def aname(line): return line[10:15].strip()

# SG atoms in protein + their coords
sg = [(i, coords(patoms[i])) for i in range(npro) if aname(patoms[i])=='SG']
c3 = coords(latoms[C3_lig0])
def d2(a,b): return sum((a[k]-b[k])**2 for k in range(3))
sg_best = min(sg, key=lambda t: d2(t[1], c3))
SG_global = sg_best[0] + 1                  # 1-based, protein is first in [molecules]
C3_global = npro + C3_lig0 + 1              # 1-based
dist = math.sqrt(d2(sg_best[1], c3))*10     # nm->Angstrom
assert aname(latoms[C3_lig0]).startswith('C'), f'C3 atom is {aname(latoms[C3_lig0])}, not carbon'

# --- merged complex.gro ---
with open(f'{run}/complex.gro','w') as f:
    f.write('complex protein+LIG\n'); f.write(f'{npro+nlig}\n')
    for a in patoms: f.write(a+'\n')
    for a in latoms: f.write(a+'\n')
    f.write(pbox+'\n')

# --- split LIG_GMX.itp into atomtypes + moleculetype(rest) ---
itp = open(f'{acp}/LIG_GMX.itp').read().splitlines()
at_lines, rest, mode = [], [], None
i=0
while i < len(itp):
    ln = itp[i]
    if ln.strip().startswith('[') and 'atomtypes' in ln:
        mode='at'; i+=1
        while i < len(itp) and not (itp[i].strip().startswith('[') and 'atomtypes' not in itp[i]):
            at_lines.append(itp[i]); i+=1
        continue
    else:
        rest.append(ln); i+=1
with open(f'{run}/LIG_atomtypes.itp','w') as f: f.write('[ atomtypes ]\n'+'\n'.join(at_lines)+'\n')
with open(f'{run}/LIG_moltype.itp','w') as f: f.write('\n'.join(rest)+'\n')

# --- patch topol.top ---
top = open(f'{run}/topol.top').read().splitlines()
out=[]; done_at=False; done_mol=False
for ln in top:
    out.append(ln)
    if (not done_at) and 'forcefield.itp' in ln:
        out.append('; ligand atom types'); out.append('#include "LIG_atomtypes.itp"'); done_at=True
for j,ln in enumerate(out):
    if ln.strip().startswith('[') and 'system' in ln.lower() and not done_mol:
        out.insert(j, '#include "LIG_moltype.itp"'); out.insert(j, '; ligand moleculetype'); done_mol=True
        break
# posre for ligand under POSRES
moltext=open(f'{run}/LIG_moltype.itp').read()
if 'posre_LIG.itp' not in moltext and os.path.exists(f'{acp}/posre_LIG.itp'):
    os.system(f'cp {acp}/posre_LIG.itp {run}/posre_LIG.itp')
    with open(f'{run}/LIG_moltype.itp','a') as f:
        f.write('\n#ifdef POSRES\n#include "posre_LIG.itp"\n#endif\n')
# add LIG to [ molecules ]
for j,ln in enumerate(out):
    if ln.strip().lower().startswith('[ molecules'):
        pass
out.append('LIG                 1')
open(f'{run}/topol.top','w').write('\n'.join(out)+'\n')

json.dump({'cid':cid,'npro':npro,'nlig':nlig,'C3_global':C3_global,'SG_global':SG_global,
           'C3_SG_dist_A':round(dist,2),'SG_resid_atomname':aname(patoms[sg_best[0]])},
          open(f'{run}/restraint.json','w'), indent=2)
print(f'protein {npro} + ligand {nlig} = {npro+nlig} atoms')
print(f'C3_global={C3_global}  SG_global={SG_global}  C3-SG={dist:.2f} A')
print('wrote complex.gro, patched topol.top, restraint.json')

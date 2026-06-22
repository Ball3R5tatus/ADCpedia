#!/usr/bin/env python3
"""Cys214 conjugation-site microenvironment (Sebastian-Perez axis): which residues
surround the site, is there a PROXIMAL BASIC residue that could catalyse succinimide
ring-opening hydrolysis (-> self-stabilising maleimide, Lyon 2014), local pKa (PROPKA),
and SG solvent accessibility. Static analysis on the apo receptor."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, re
import MDAnalysis as mda

REC='/home/galeito/ADCpedia/md/runs/cand_vedotin/receptor.pdb'
PKA='/tmp/rec.pka'
u=mda.Universe(REC)
sg=u.select_atoms('segid A and resid 214 and name SG')
SG=sg.positions[0]
print(f'Cys214 SG @ {SG.round(2)} (chain A)')

BASIC={'LYS','ARG','HIS'}; ACID={'ASP','GLU'}
# residues with ANY atom within cutoff of SG (exclude Cys214 itself)
def near(cut):
    sel=u.select_atoms(f'byres (around {cut} (segid A and resid 214 and name SG))')
    res={}
    for a in sel.atoms:
        key=(a.segid,a.resname,a.resid)
        d=float(np.linalg.norm(a.position-SG))
        if key not in res or d<res[key]: res[key]=d
    res.pop(('A','CYS',214),None)
    return dict(sorted(res.items(), key=lambda kv: kv[1]))

# parse PROPKA pKa (format: "   RES  NNN C   pKa   model")
pka={}
for l in open(PKA):
    m=re.match(r'\s+([A-Z]{3})\s+(\d+)\s+([A-Z])\s+([\d.]+)\s+([\d.]+)\s*$', l)
    if m: pka[(m.group(3),m.group(1),int(m.group(2)))]=float(m.group(4))
cys_pka=pka.get(('A','CYS',214))
print(f'PROPKA pKa(Cys214 SG) = {cys_pka}  (model 9.0; a depressed pKa => more reactive thiolate)')

print('\n=== residues within 8 A of SG (nearest atom-atom) ===')
print(f'{"res":>12}{"minDist(A)":>12}{"class":>8}{"pKa":>8}')
basics_near=[]
for (seg,rn,ri),d in near(8.0).items():
    cls='BASIC' if rn in BASIC else ('acid' if rn in ACID else '')
    p=pka.get((seg,rn,ri),'')
    print(f'{rn+str(ri)+"/"+seg:>12}{d:>12.2f}{cls:>8}{("" if p=="" else f"{p:.2f}"):>8}')
    if rn in BASIC: basics_near.append((rn,ri,d,pka.get((seg,rn,ri))))

print('\n=== nearest BASIC residues to SG (Sebastian-Perez catalyst candidates) ===')
allbasic=[]
for (seg,rn,ri),d in near(12.0).items():
    if rn in BASIC: allbasic.append((rn,ri,d,pka.get((seg,rn,ri))))
for rn,ri,d,p in allbasic[:6]:
    print(f'  {rn}{ri}: {d:.2f} A   pKa={p}')

# verdict heuristic
nb=allbasic[0] if allbasic else None
print('\n=== Sebastian-Perez read ===')
if nb and nb[2] <= 7.0:
    print(f'  Proximal basic residue {nb[0]}{nb[1]} at {nb[2]:.1f} A -> CAN catalyse succinimide hydrolysis (self-stabilising)')
elif nb and nb[2] <= 12.0:
    print(f'  Nearest basic {nb[0]}{nb[1]} at {nb[2]:.1f} A -> too far for direct catalysis (>7 A); weak/no self-stabilisation expected')
else:
    print('  No basic residue within 12 A -> no catalytic self-stabilisation at this site')

#!/usr/bin/env python3
"""Build + pose the AUTHENTIC vedotin (mc-vc-PAB-MMAE) drug-linker as a positive
scale control, using the SAME conjugated representation + posing logic as the
candidates (make_posed_ligands.py). Only the ligand differs; scaffold reused.

conjugated form = maleimide reacted with a methylthiol Cys-mimic -> thiosuccinimide
where S binds C3 (the succinimide attach carbon) and a CH3 stub (methylC). The pose
step removes S+methylC -> C3 becomes the succinimide CH2, exactly like candidates.
Multi-conformer: pose every ETKDG conformer x 18 axis rotations, keep min-clash.
"""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda

STR='/home/galeito/ADCpedia/structures'; OUT='/home/galeito/ADCpedia/md/lig_params'
CID='vedotin'
# AUTH maleimide form, maleimide N4C(=O)C=CC4=O -> thiosuccinimide-S-CH3 N4C(=O)CC(SC)C4=O
CONJ=('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@H](C)'
      '[C@H](C2=CC=CC=C2)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)N(C)C(=O)'
      'OCC3=CC=C(C=C3)NC(=O)[C@H](CCCNC(=O)N)NC(=O)[C@H](C(C)C)NC(=O)CCCCCN4C(=O)CC(SC)C4=O')

mol=Chem.MolFromSmiles(CONJ); assert mol is not None
print('conjugated formula:', rdMolDescriptors.CalcMolFormula(mol),
      ' formal charge:', Chem.GetFormalCharge(mol), ' (expect C69H109N11O15S, 0)')

# tag original indices so we can track C3 after atom removal
for a in mol.GetAtoms(): a.SetIntProp('orig', a.GetIdx())
molH=Chem.AddHs(mol)
p=AllChem.ETKDGv3(); p.randomSeed=0xC0FFEE; p.numThreads=0
cids=list(AllChem.EmbedMultipleConfs(molH, numConfs=40, params=p))
AllChem.MMFFOptimizeMoleculeConfs(molH, numThreads=0, maxIters=2000)
print(f'embedded {len(cids)} conformers')

# --- conjugation atoms (heavy-atom topology) ---
Si=[a.GetIdx() for a in molH.GetAtoms() if a.GetSymbol()=='S'][0]
nb=[n.GetIdx() for n in molH.GetAtomWithIdx(Si).GetNeighbors()]
methylC=[j for j in nb if sum(1 for n2 in molH.GetAtomWithIdx(j).GetNeighbors() if n2.GetAtomicNum()>1)==1][0]
C3=[j for j in nb if j!=methylC][0]
print(f'S={Si} methylC={methylC} (Cys stub) C3={C3} (succinimide attach)')

# --- target Cys (shared scaffold) ---
tc=json.load(open(f'{STR}/target_cys.json'))
rec=mda.Universe(f'{STR}/1N8Z_clean.pdb')
SG=rec.select_atoms(f"segid {tc['chain']} and resid {tc['resid']} and name SG").positions[0].astype(float)
cenP=rec.select_atoms('protein').center_of_geometry().astype(float)
outward=SG-cenP; outward/=np.linalg.norm(outward)
near=rec.select_atoms('not name H*').positions.astype(float); near=near[np.linalg.norm(near-SG,axis=1)<35]

def rod_align(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if c>0.99999: return np.eye(3)
    if c<-0.99999:
        ax=np.cross(a,[1,0,0]); ax=ax if np.linalg.norm(ax)>1e-3 else np.cross(a,[0,1,0]); ax/=np.linalg.norm(ax); return -np.eye(3)+2*np.outer(ax,ax)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx/(1+c)
def axis_rot(ax,th):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(th),np.sin(th); x,y,z=ax
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],[y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],[z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])

# capped (succinimide) molecule template: remove S + methylC + their Hs
def make_capped(conf_id):
    m=Chem.Mol(molH, False, conf_id)         # copy with single conformer conf_id
    # keep only that conformer
    keep=m.GetConformer(conf_id)
    m2=Chem.RWMol(molH)
    drop={Si, methylC}
    for j in (Si, methylC):
        for n2 in molH.GetAtomWithIdx(j).GetNeighbors():
            if n2.GetAtomicNum()==1: drop.add(n2.GetIdx())
    for idx in sorted(drop, reverse=True): m2.RemoveAtom(idx)
    capped=m2.GetMol()
    # set capped conformer = the chosen conformer's surviving coords
    survivors=[i for i in range(molH.GetNumAtoms()) if i not in drop]
    src=molH.GetConformer(conf_id)
    cc=Chem.Conformer(capped.GetNumAtoms())
    for new_i, old_i in enumerate(survivors): cc.SetAtomPosition(new_i, src.GetAtomPosition(old_i))
    capped.RemoveAllConformers(); capped.AddConformer(cc, assignId=True)
    Chem.SanitizeMol(capped)
    C3n=[a.GetIdx() for a in capped.GetAtoms() if a.HasProp('orig') and a.GetIntProp('orig')==C3][0]
    return capped, C3n

def pose(capped, C3n):
    pos=capped.GetConformer().GetPositions()
    heavy=[a.GetIdx() for a in capped.GetAtoms() if a.GetAtomicNum()>1]
    C3p=pos[C3n]; cenL=pos[heavy].mean(0)
    P=(pos-C3p)@rod_align(cenL-C3p,outward).T; tgt=SG+1.8*outward; P=P+tgt
    kh=np.array(heavy); best=None
    for th in np.linspace(0,2*np.pi,18,endpoint=False):
        Pr=(P-tgt)@axis_rot(outward,th).T+tgt
        d=np.linalg.norm(Pr[kh][:,None]-near[None],axis=2)
        sev=int((d<1.6).sum()); cl=int((d<2.2).sum())
        if best is None or (sev,cl)<(best[0],best[1]): best=(sev,cl,Pr)
    return best

best=None
for ci in cids:
    capped,C3n=make_capped(ci)
    sev,cl,Pf=pose(capped,C3n)
    if best is None or (sev,cl)<(best[0],best[1]):
        best=(sev,cl,Pf,capped,C3n)
sev,cl,Pf,capped,C3n=best
conf=capped.GetConformer()
for i in range(capped.GetNumAtoms()): conf.SetAtomPosition(i, Pf[i].tolist())
cappedH=Chem.AddHs(capped, addCoords=True)
C3h=[a.GetIdx() for a in cappedH.GetAtoms() if a.HasProp('orig') and a.GetIntProp('orig')==C3][0]
Chem.MolToMolFile(cappedH, f'{OUT}/cand_{CID}_posed.sdf')
c3sg=float(np.linalg.norm(Pf[C3n]-SG))
heavy=[a.GetIdx() for a in capped.GetAtoms() if a.GetAtomicNum()>1]
info_entry={'output':'vedotin_Adcetris','C3_idx_in_posed_sdf':C3h,'n_heavy':len(heavy),
            'n_total':cappedH.GetNumAtoms(),'c3_sg':round(c3sg,2),'severe':sev,'soft':cl,
            'SG_xyz':SG.tolist(),'formal_charge':Chem.GetFormalCharge(cappedH)}
# merge into posed_info.json (keep candidate entries)
pj=f'{OUT}/posed_info.json'; info=json.load(open(pj))
info[CID]=info_entry; json.dump(info, open(pj,'w'), indent=2)
print(f'POSED vedotin: {cappedH.GetNumAtoms()} atoms ({len(heavy)} heavy), C3 idx={C3h}, '
      f'C3-SG={c3sg:.2f} A, severe={sev} soft={cl}, charge={Chem.GetFormalCharge(cappedH)}')
print(f'-> {OUT}/cand_{CID}_posed.sdf ; posed_info["{CID}"] written')

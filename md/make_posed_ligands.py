import os, json, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda
STR='/home/galeito/ADCpedia/structures'; CJ='/home/galeito/ADCpedia/docking/conjugated_ligands_stereo'
OUT='/home/galeito/ADCpedia/md/lig_params'; os.makedirs(OUT,exist_ok=True)
tc=json.load(open(f'{STR}/target_cys.json'))
rec=mda.Universe(f'{STR}/1N8Z_clean.pdb')
cys=rec.select_atoms(f"segid {tc['chain']} and resid {tc['resid']}")
SG=cys.select_atoms('name SG').positions[0].astype(float)
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

PRIM={4:'output_486',5:'output_345',1:'output_219',8:'output_106'}  # cand_4 first (pilot)
info={}
for cid,outp in PRIM.items():
    mol=Chem.MolFromMolFile(f'{CJ}/candidate_{cid}.sdf', removeHs=False)
    Si=[a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol()=='S'][0]
    nb=[n.GetIdx() for n in mol.GetAtomWithIdx(Si).GetNeighbors()]
    methylC=[j for j in nb if sum(1 for n2 in mol.GetAtomWithIdx(j).GetNeighbors() if n2.GetAtomicNum()>1)==1][0]
    C3=[j for j in nb if j!=methylC][0]
    # edit: remove S and methylC -> C3 capped with implicit H (succinimide CH2)
    em=Chem.RWMol(mol)
    for idx in sorted([Si,methylC], reverse=True): em.RemoveAtom(idx)
    capped=em.GetMol(); Chem.SanitizeMol(capped)
    # C3 index shifts after removal; recompute via atom map: track C3 by tagging before removal
    # redo with tagging for safety
    mol2=Chem.MolFromMolFile(f'{CJ}/candidate_{cid}.sdf', removeHs=False)
    for a in mol2.GetAtoms(): a.SetIntProp('orig',a.GetIdx())
    em=Chem.RWMol(mol2)
    for idx in sorted([Si,methylC], reverse=True): em.RemoveAtom(idx)
    capped=em.GetMol(); Chem.SanitizeMol(capped)
    C3n=[a.GetIdx() for a in capped.GetAtoms() if a.GetIntProp('orig')==C3][0]
    # pose heavy atoms: align C3->ligand-centroid axis to outward, place C3 at SG+1.8*outward, clash-min rotate
    pos=capped.GetConformer().GetPositions()
    heavy=[a.GetIdx() for a in capped.GetAtoms() if a.GetAtomicNum()>1]
    C3p=pos[C3n]; cenL=pos[heavy].mean(0)
    P=(pos-C3p)@rod_align(cenL-C3p,outward).T; tgt=SG+1.8*outward; P=P+tgt
    kh=np.array(heavy); best=None
    for th in np.linspace(0,2*np.pi,18,endpoint=False):
        Pr=(P-tgt)@axis_rot(outward,th).T+tgt; d=np.linalg.norm(Pr[kh][:,None]-near[None],axis=2)
        sev=int((d<1.6).sum()); cl=int((d<2.2).sum())
        if best is None or (sev,cl)<(best[0],best[1]): best=(sev,cl,Pr)
    sev,cl,Pf=best
    conf=capped.GetConformer()
    for i in range(capped.GetNumAtoms()): conf.SetAtomPosition(i, Pf[i].tolist())
    # add Hs with coords
    cappedH=Chem.AddHs(capped, addCoords=True)
    C3h=[a.GetIdx() for a in cappedH.GetAtoms() if a.HasProp('orig') and a.GetIntProp('orig')==C3][0]
    Chem.MolToMolFile(cappedH, f'{OUT}/cand_{cid}_posed.sdf')
    c3sg=float(np.linalg.norm(Pf[C3n]-SG))
    info[cid]={'output':outp,'C3_idx_in_posed_sdf':C3h,'n_heavy':len(heavy),'n_total':cappedH.GetNumAtoms(),'c3_sg':round(c3sg,2),'severe':sev,'soft':cl,'SG_xyz':SG.tolist(),'formal_charge':Chem.GetFormalCharge(cappedH)}
    print(f"cand_{cid} ({outp}): posed SDF {cappedH.GetNumAtoms()} atoms, C3 idx={C3h}, C3-SG={c3sg:.2f}A, severe={sev} soft={cl}, charge={Chem.GetFormalCharge(cappedH)}")
json.dump(info, open(f'{OUT}/posed_info.json','w'), indent=2)
print("wrote posed SDFs + posed_info.json (C3 indices + SG coords for restraint)")

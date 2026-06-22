import os, json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import MDAnalysis as mda
STR='/home/galeito/ADCpedia/structures'; CJ='/home/galeito/ADCpedia/docking/conjugated_ligands_stereo'
OUT='/home/galeito/ADCpedia/md/complexes_stereo'; os.makedirs(OUT,exist_ok=True)
tc=json.load(open(f'{STR}/target_cys.json'))
rec=mda.Universe(f'{STR}/1N8Z_clean.pdb')
cys=rec.select_atoms(f"segid {tc['chain']} and resid {tc['resid']}")
SG=cys.select_atoms('name SG').positions[0].astype(float)
cenP=rec.select_atoms('protein').center_of_geometry().astype(float)
outward=SG-cenP; outward/=np.linalg.norm(outward)
near=rec.select_atoms('not name H*').positions.astype(float); near=near[np.linalg.norm(near-SG,axis=1)<35]
MMAE=Chem.MolFromSmiles('CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)N[C@H](C)[C@@H](O)c2ccccc2)OC)N(C)C(=O)[C@@H](NC(=O)[C@@H](NC)C(C)C)C(C)C')
Chem.AssignStereochemistry(MMAE,cleanIt=True,force=True)
qry=Chem.MolFromSmiles(Chem.MolToSmiles(MMAE,isomericSmiles=False)); mc=MMAE.GetSubstructMatch(qry)
core_cip={i:MMAE.GetAtomWithIdx(mc[i]).GetProp('_CIPCode') for i in range(qry.GetNumAtoms()) if MMAE.GetAtomWithIdx(mc[i]).HasProp('_CIPCode')}
def rod_align(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if c>0.99999: return np.eye(3)
    if c<-0.99999:
        ax=np.cross(a,[1,0,0]); ax=ax if np.linalg.norm(ax)>1e-3 else np.cross(a,[0,1,0]); ax/=np.linalg.norm(ax); return -np.eye(3)+2*np.outer(ax,ax)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx/(1+c)
def axis_rot(ax,th):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(th),np.sin(th); x,y,z=ax
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],[y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],[z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])
def lig_universe(coords, els):
    n=len(coords); u=mda.Universe.empty(n,n_residues=1,atom_resindex=np.zeros(n,int),trajectory=True)
    u.add_TopologyAttr('names',[f'{e}{i}' for i,e in enumerate(els)]); u.add_TopologyAttr('elements',list(els))
    u.add_TopologyAttr('resnames',['LIG']); u.add_TopologyAttr('resids',[999]); u.add_TopologyAttr('segids',['X'])
    u.add_TopologyAttr('record_types',['HETATM']*n); u.atoms.positions=coords; return u
PRIM={8:'output_106',1:'output_219',5:'output_345',4:'output_486'}
rows=[]; manifest=[]
for cid,outp in PRIM.items():
    sdf=f'{CJ}/candidate_{cid}.sdf'; mol=Chem.MolFromMolFile(sdf,removeHs=False)
    pos=mol.GetConformer().GetPositions()
    Si=[a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol()=='S'][0]
    nb=[n.GetIdx() for n in mol.GetAtomWithIdx(Si).GetNeighbors()]
    methylC=[j for j in nb if sum(1 for n2 in mol.GetAtomWithIdx(j).GetNeighbors() if n2.GetAtomicNum()>1)==1][0]
    C3=[j for j in nb if j!=methylC][0]; drop={Si,methylC}
    for j in (Si,methylC):
        for n2 in mol.GetAtomWithIdx(j).GetNeighbors():
            if n2.GetAtomicNum()==1: drop.add(n2.GetIdx())
    keep_heavy=[i for i in range(mol.GetNumAtoms()) if i not in drop and mol.GetAtomWithIdx(i).GetAtomicNum()>1]
    P=pos.copy(); C3p=P[C3]; cenL=P[keep_heavy].mean(0)
    P=(P-C3p)@rod_align(cenL-C3p,outward).T; tgt=SG+1.8*outward; P=P+tgt
    best=None; kh=np.array(keep_heavy)
    for th in np.linspace(0,2*np.pi,18,endpoint=False):
        Pr=(P-tgt)@axis_rot(outward,th).T+tgt; d=np.linalg.norm(Pr[kh][:,None]-near[None],axis=2)
        sev=int((d<1.6).sum()); cl=int((d<2.2).sum())
        if best is None or (sev,cl)<(best[0],best[1]): best=(sev,cl,Pr)
    sev,cl,Pf=best; c3sg=float(np.linalg.norm(Pf[C3]-SG)); valid=(sev<=2 and 1.4<=c3sg<=2.3)
    els=[mol.GetAtomWithIdx(i).GetSymbol() for i in keep_heavy]
    lu=lig_universe(Pf[kh], els); mrg=mda.Merge(rec.atoms, lu.atoms)
    outpdb=f'{OUT}/complex_{cid}.pdb'; mrg.atoms.write(outpdb)
    # verify LIG stereo preserved (rigid transform should preserve; check anyway via the source SDF since PDB has no bond orders)
    Chem.AssignStereochemistryFrom3D(mol); m2=Chem.RemoveHs(mol); Chem.AssignStereochemistry(m2,cleanIt=True,force=True)
    mq=m2.GetSubstructMatch(qry)
    cipok=sum(1 for i,c in core_cip.items() if m2.GetAtomWithIdx(mq[i]).HasProp('_CIPCode') and m2.GetAtomWithIdx(mq[i]).GetProp('_CIPCode')==c)
    rows.append((cid,outp,round(c3sg,2),sev,cl,'valid' if valid else 'CLASH',f'{cipok}/10'))
    manifest.append({'complex':f'complex_{cid}.pdb','candidate_id':cid,'output':outp,'c3_sg_dist':round(c3sg,2),'severe_clashes':sev,'soft_clashes':cl,'valid':bool(valid),'n_ligand_heavy':len(keep_heavy),'mmae_cip_match':f'{cipok}/10','stereo':'canonical-MMAE-imposed'})
json.dump({'target_cys':tc,'note':'stereo-corrected primary rebuild (Phase 2.7 Etape 0 case-3)','complexes':manifest},open(f'{OUT}/manifest.json','w'),indent=2)
print(f"{'cid':>4}{'output':>12}{'C3-SG':>7}{'sev':>4}{'soft':>5}{'status':>7}{'MMAE-CIP':>9}")
for r in rows: print(f"{r[0]:>4}{r[1]:>12}{str(r[2]):>7}{r[3]:>4}{r[4]:>5}{r[5]:>7}{r[6]:>9}")
print(f"VALID: {sum(r[5]=='valid' for r in rows)}/{len(rows)}")

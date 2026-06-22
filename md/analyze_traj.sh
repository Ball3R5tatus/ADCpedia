#!/bin/bash
# Phase 2.7 Etape 10 — trajectory analysis. Arg: cid
# Produces RMSD (receptor CA + ligand), RMSF, C3-SG covalent-bond trace,
# ligand-receptor contacts, ligand SASA. Output -> md/analysis/cand_<cid>/
#
# NOTE: production xtc is compressed to the "Complex" group (protein+ligand only,
# compressed-x-grps=Complex), so we analyse against a Complex-matched topology
# (complex.tpr, built here via convert-tpr) — NOT prod.tpr (full solvated system).
# Atom indices are computed generically from restraint.json + the merged chain-A
# itp (the conjugated-Cys HG was removed, so ligand/C3 indices shift vs pre-fix).
set -e
source /usr/local/gromacs/bin/GMXRC
cid=$1; run=~/ADCpedia/md/runs/cand_$cid; out=~/ADCpedia/md/analysis/cand_$cid
mkdir -p $out; cd $run

NDXFULL=vis_fix.ndx     # full-system index containing the "Complex" group
[ -f "$NDXFULL" ] || { echo "ERROR: $NDXFULL (Complex group) not found"; exit 1; }

# --- Complex-matched topology to match the compressed xtc ---
printf "Complex\n" | gmx convert-tpr -s prod.tpr -n $NDXFULL -o complex.tpr >/dev/null 2>&1

# --- generic atom indices (valid in the Complex subset: protein+ligand come first) ---
read C3 SG LIGSTART LIGEND < <(python3 -c "
import json,re
r=json.load(open('restraint.json'))
itp=open('topol_Protein_chain_A.itp').read()
m=re.search(r'\[ *atoms *\]\s*\n(.*?)(?=^\[ )', itp, re.M|re.S)
nA=sum(1 for l in m.group(1).splitlines() if l.split() and l.split()[0].isdigit())
nlig=r.get('nlig_merged', r['nlig']); nprot=nA-nlig   # nlig_merged != nlig when a C3-H was removed (valence fix)
print(nprot+(r['C3_global']-r['npro']), r['SG_global'], nprot+1, nA)")
echo "indices: LIG ${LIGSTART}-${LIGEND}  C3=${C3}  SG=${SG}"

# --- index groups (dynamic numbering: append LIG/C3/SG after the defaults) ---
NG=$(printf 'q\n' | gmx make_ndx -f complex.tpr -o /dev/null 2>&1 | grep -cE '^ *[0-9]+ +[A-Za-z]')
gmx make_ndx -f complex.tpr -o ana.ndx >/dev/null 2>&1 <<NDX
a ${LIGSTART}-${LIGEND}
name $NG LIG
a ${C3}
name $((NG+1)) C3
a ${SG}
name $((NG+2)) SG
q
NDX

# PBC correction (multi-chain complex): make molecules whole, THEN remove jumps so the
# Fab+HER2 assembly stays in one image — otherwise chains drift to opposite box sides and
# RMSD fitting explodes (-pbc mol/-center alone is NOT enough for a multi-chain assembly).
FIT=$out/prod_fit.xtc
printf "System\n" | gmx trjconv -s complex.tpr -f prod.xtc      -pbc whole  -o $out/.whole.xtc >/dev/null 2>&1
printf "System\n" | gmx trjconv -s complex.tpr -f $out/.whole.xtc -pbc nojump -o $FIT          >/dev/null 2>&1
rm -f $out/.whole.xtc

# 1. receptor CA RMSD (fit to t0)
echo -e "C-alpha\nC-alpha" | gmx rms -s complex.tpr -f $FIT -o $out/ca_rmsd.xvg -tu ns >/dev/null 2>&1
# 2. ligand RMSD (fit on protein, measure ligand)
echo -e "Protein\nLIG" | gmx rms -s complex.tpr -f $FIT -n ana.ndx -o $out/lig_rmsd.xvg -tu ns >/dev/null 2>&1
# 3. RMSF per residue (CA)
echo "C-alpha" | gmx rmsf -s complex.tpr -f $FIT -o $out/rmsf.xvg -res >/dev/null 2>&1
# 4. C3-SG covalent bond distance over time (sanity: should stay ~0.18 nm)
gmx distance -s complex.tpr -f $FIT -select "atomnr ${SG} plus atomnr ${C3}" -oall $out/c3sg_dist.xvg -tu ns >/dev/null 2>&1
# 5. ligand SASA
echo "LIG" | gmx sasa -s complex.tpr -f $FIT -n ana.ndx -o $out/lig_sasa.xvg -tu ns >/dev/null 2>&1
# 6. ligand-receptor min distance + contacts (<0.4 nm)
echo -e "LIG\nProtein" | gmx mindist -s complex.tpr -f $FIT -n ana.ndx -od $out/lig_prot_mindist.xvg -on $out/lig_prot_ncontacts.xvg -d 0.4 -tu ns >/dev/null 2>&1

# summary
python3 - "$out" <<'PY'
import sys,glob,os
out=sys.argv[1]
def load(f):
    xs=[];ys=[]
    for l in open(f):
        if l[0] in '#@': continue
        p=l.split();
        if len(p)>=2: xs.append(float(p[0])); ys.append(float(p[1]))
    return xs,ys
import statistics as st
def summ(name,f,scale=1):
    if not os.path.exists(f): print(f"  {name}: (missing)"); return
    x,y=load(f); y=[v*scale for v in y]
    half=[v for t,v in zip(x,y) if t>=x[-1]/2]
    print(f"  {name}: last={y[-1]:.3f} mean(2nd half)={st.mean(half):.3f} min={min(y):.3f} max={max(y):.3f}")
print("=== cand summary ===")
summ("CA RMSD (nm)", f"{out}/ca_rmsd.xvg")
summ("ligand RMSD (nm)", f"{out}/lig_rmsd.xvg")
summ("C3-SG dist (nm)", f"{out}/c3sg_dist.xvg")
summ("ligand SASA (nm^2)", f"{out}/lig_sasa.xvg")
summ("lig-prot contacts", f"{out}/lig_prot_ncontacts.xvg")
PY
echo "### analysis done -> $out"

#!/bin/bash
# Box -> solvate -> ions -> C3..SG restraint -> EM -> NVT -> NPT. Arg: cid
set -e
source /usr/local/gromacs/bin/GMXRC
cid=$1; cd ~/ADCpedia/md/runs/cand_$cid
MDP=~/ADCpedia/md/mdp
C3=$(python3 -c "import json;print(json.load(open('restraint.json'))['C3_global'])")
SG=$(python3 -c "import json;print(json.load(open('restraint.json'))['SG_global'])")
echo "### restraint C3=$C3 SG=$SG"
gmx editconf -f merged.gro -o box.gro -bt dodecahedron -d 1.2 -c >editconf.log 2>&1
gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro >solvate.log 2>&1
gmx grompp -f $MDP/em.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 3 >grompp_ions.log 2>&1
echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15 >genion.log 2>&1
# add distance-restraint "covalent" tether (func 6 = harmonic, no exclusions)

echo "### EM"
gmx grompp -f $MDP/em.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 3 >grompp_em.log 2>&1
gmx mdrun -deffnm em -nb gpu >mdrun_em.log 2>&1
echo "### NVT"
gmx grompp -f $MDP/nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 3 >grompp_nvt.log 2>&1
gmx mdrun -deffnm nvt -nb gpu -pme gpu -bonded gpu >mdrun_nvt.log 2>&1
echo "### NPT"
gmx grompp -f $MDP/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 3 >grompp_npt.log 2>&1
gmx mdrun -deffnm npt -nb gpu -pme gpu -bonded gpu >mdrun_npt.log 2>&1
echo "### SETUP DONE for cand_$cid"
tail -2 genion.log; grep -iE "Potential Energy|Steepest" mdrun_em.log | tail -2 || true

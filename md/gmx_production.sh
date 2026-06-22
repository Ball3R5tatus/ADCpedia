#!/bin/bash
# 50 ns production NPT. Arg: cid
set -e
source /usr/local/gromacs/bin/GMXRC
cid=$1; cd ~/ADCpedia/md/runs/cand_$cid
MDP=~/ADCpedia/md/mdp
gmx grompp -f $MDP/prod.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 3 >grompp_prod.log 2>&1
gmx mdrun -deffnm prod -nb gpu -pme gpu -bonded gpu -update cpu >mdrun_prod.log 2>&1
echo "### PRODUCTION DONE cand_$cid"

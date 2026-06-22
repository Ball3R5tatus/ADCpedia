#!/bin/bash
# Post-acpype build + equilibration for the vedotin positive control (cid=vedotin).
# Mirrors the corrected cand_5 pipeline EXACTLY: assemble -> merge (thioether+valence
# fix+fail-closed valence GATE) -> build_coords -> box/solvate/ions -> EM -> NVT ->
# NPT(npt_eq.mdp,-update cpu) -> vis_fix.ndx (Complex=Protein|MOL). Stops before production.
set -e
source /usr/local/gromacs/bin/GMXRC
MD=/home/galeito/ADCpedia/md; MDP=$MD/mdp; cid=vedotin; RUN=$MD/runs/cand_$cid
cd $RUN
log(){ echo "[$(date '+%F %T')] $*"; }

log "1) assemble protein + ligand"
python3 $MD/gmx_assemble.py $cid
log "2) merge ligand into chain A (thioether + valence fix + GATE)"
python3 $MD/merge_chainA_ligand.py $cid          # fail-closed valence gate (exits 1 if over-valent)
log "3) coordinate reorder + strip HG/C3-H"
python3 $MD/build_coords.py $cid

log "4) box -> solvate -> ions"
gmx editconf -f merged_noHG.gro -o box.gro -bt dodecahedron -d 1.2 -c >editconf.log 2>&1
gmx solvate -cp box.gro -cs spc216.gro -p topol.top -o solv.gro >solvate.log 2>&1
gmx grompp -f $MDP/em.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 3 >grompp_ions.log 2>&1
echo SOL | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15 >genion.log 2>&1

log "5) EM"
gmx grompp -f $MDP/em.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 2 >grompp_em.log 2>&1
gmx mdrun -deffnm em -nb gpu >mdrun_em.log 2>&1
log "6) NVT"
gmx grompp -f $MDP/nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2 >grompp_nvt.log 2>&1
gmx mdrun -deffnm nvt -nb gpu -pme gpu -bonded gpu >mdrun_nvt.log 2>&1
log "7) NPT (npt_eq.mdp, -update cpu)"
gmx grompp -f $MDP/npt_eq.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2 >grompp_npt.log 2>&1
gmx mdrun -deffnm npt -nb gpu -pme gpu -bonded gpu -update cpu >mdrun_npt.log 2>&1

log "8) vis_fix.ndx (Complex = Protein | MOL)"
NG=$(printf 'q\n' | gmx make_ndx -f npt.gro -o /dev/null 2>&1 | grep -cE '^ *[0-9]+ +[A-Za-z]')
gmx make_ndx -f npt.gro -o vis_fix.ndx >make_ndx.log 2>&1 <<NDX
"Protein" | "MOL"
name $NG Complex
q
NDX

log "=== BUILD + EQUILIBRATION DONE (vedotin) ==="
echo "--- LINCS warnings across EM/NVT/NPT (expect 0) ---"
grep -ci "lincs warn" mdrun_em.log mdrun_nvt.log mdrun_npt.log 2>/dev/null || true
echo "--- restraint.json ---"; cat restraint.json
echo "--- Complex group size ---"; python3 -c "import re;t=open('vis_fix.ndx').read();b=re.split(r'\[\s*(.*?)\s*\]',t)[1:];g={b[i].strip():len(b[i+1].split()) for i in range(0,len(b),2)};print('Complex =',g.get('Complex'),' Protein =',g.get('Protein'),' MOL =',g.get('MOL'))"

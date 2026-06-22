#!/bin/bash
# Vedotin positive control: rep#2 + rep#3 (rep#1 already done + integrity-passed).
# Same design as the candidate replicate chain: grompp from npt.gro WITHOUT -t
# (gen_vel=yes, distinct seed, continuation=no -> independent velocities). Idempotent:
# skips finished reps, resumes via -cpi, -cpt 5 (WSL-crash resilience).
set -e
source /usr/local/gromacs/bin/GMXRC
MDP=/home/galeito/ADCpedia/md/mdp; RUN=/home/galeito/ADCpedia/md/runs/cand_vedotin
GPU="-nb gpu -pme gpu -bonded gpu -update gpu -pin on -cpt 5"
cd $RUN
log(){ echo "[$(date '+%F %T')] $*" | tee -a vedotin_chain.log; }

build_run(){  # mdp deffnm
  if [ -f "$2.gro" ]; then log "$2 already complete -- skip"; return; fi
  [ -f "$2.tpr" ] || { log "grompp $2 (from npt.gro, $(basename $1))"; \
    gmx grompp -f "$1" -c npt.gro -p topol.top -n vis_fix.ndx -o "$2.tpr" -maxwarn 2 >"grompp_$2.log" 2>&1; }
  if [ -f "$2.cpt" ]; then log "RESUME $2 (-cpi)"; gmx mdrun -deffnm "$2" -cpi "$2.cpt" $GPU >"mdrun_$2.log" 2>&1
  else                    log "START  $2";        gmx mdrun -deffnm "$2"               $GPU >"mdrun_$2.log" 2>&1; fi
  log "$2 DONE"
}

log "=== VEDOTIN rep#2 + rep#3 chain START ==="
build_run $MDP/prod_rep2.mdp prod_rep2
build_run $MDP/prod_rep3.mdp prod_rep3
log "=== VEDOTIN N=3 COMPLETE (rep#1 + rep#2 + rep#3) ==="

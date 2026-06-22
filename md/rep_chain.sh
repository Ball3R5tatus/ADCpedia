#!/bin/bash
# Replicate-chain resume driver (post-WSL-crash, 2026-06-17).
# IDEMPOTENT: skips any replicate whose final .gro already exists; resumes via -cpi
# whenever a checkpoint is present; only grompp's a tpr that is missing.
#   Step 1: cand_4 rep#3  -- RESUME from prod_rep3.cpt (~27.5/50 ns done)
#   Step 2: cand_5 rep#2  -- fresh, prod_rep2.mdp (gen_seed 22222, vel regenerated)
#   Step 3: cand_5 rep#3  -- fresh, prod_rep3.mdp (gen_seed 33333, vel regenerated)
# Each replicate grompp's from npt.gro WITHOUT -t (true independent velocities).
set -e
source /usr/local/gromacs/bin/GMXRC
MDP=/home/galeito/ADCpedia/md/mdp
RUNS=/home/galeito/ADCpedia/md/runs
GPU="-nb gpu -pme gpu -bonded gpu -update gpu -pin on -cpt 5"   # -cpt 5: checkpoint every 5 min (WSL keeps crashing -> minimise loss)
CHAINLOG=/home/galeito/ADCpedia/md/rep_chain.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$CHAINLOG"; }

build(){  # dir mdp deffnm  -- build tpr if missing
  cd "$1"
  [ -f "$3.tpr" ] || { log "grompp $3 (from npt.gro, $2)"; \
    gmx grompp -f "$2" -c npt.gro -p topol.top -n vis_fix.ndx -o "$3.tpr" -maxwarn 2 >"grompp_${3}.log" 2>&1; }
}
run_md(){  # dir deffnm  -- resume if cpt exists, else fresh
  cd "$1"
  if [ -f "$2.cpt" ]; then log "RESUME $2 in $(basename "$1") (-cpi)"; gmx mdrun -deffnm "$2" -cpi "$2.cpt" $GPU >"mdrun_${2}.log" 2>&1
  else                       log "START  $2 in $(basename "$1")";        gmx mdrun -deffnm "$2"               $GPU >"mdrun_${2}.log" 2>&1; fi
}

log "=== CHAIN START (resume after WSL crash) ==="

# Step 1 -- cand_4 rep#3 (resume)
if [ -f "$RUNS/cand_4/prod_rep3.gro" ]; then log "cand_4 rep#3 already complete -- skip"
else run_md "$RUNS/cand_4" prod_rep3; log "cand_4 rep#3 DONE"; fi

# Step 2 -- cand_5 rep#2
if [ -f "$RUNS/cand_5/prod_rep2.gro" ]; then log "cand_5 rep#2 already complete -- skip"
else build "$RUNS/cand_5" "$MDP/prod_rep2.mdp" prod_rep2; run_md "$RUNS/cand_5" prod_rep2; log "cand_5 rep#2 DONE"; fi

# Step 3 -- cand_5 rep#3
if [ -f "$RUNS/cand_5/prod_rep3.gro" ]; then log "cand_5 rep#3 already complete -- skip"
else build "$RUNS/cand_5" "$MDP/prod_rep3.mdp" prod_rep3; run_md "$RUNS/cand_5" prod_rep3; log "cand_5 rep#3 DONE"; fi

log "=== REPLICATE CHAIN COMPLETE (cand_4 reps 1-3 + cand_5 reps 1-3) ==="

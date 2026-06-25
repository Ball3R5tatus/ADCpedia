#!/usr/bin/env bash
# Background watchdog for the GB CpHMD GPU titration. Keys on mdinfo's NSTEP
# (flushed every ntpr, reliable) NOT mdout (buffered -> false stalls). EXITS
# notifying the parent on: STALL (NSTEP frozen >5min while pmemd alive = WSL
# GPU hang), DONE, or pmemd died before completion.
cd /home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4
STALL=300
last_step="x"; last_change=$(date +%s)
while true; do
  if grep -q GB_CPHMD_GPU_TITRATION_DONE titration.log 2>/dev/null; then
    echo "EVENT=DONE  $(date +%H:%M:%S)"; exit 0
  fi
  if ! pgrep -f "pmemd.*-cpin" >/dev/null 2>&1; then
    sleep 15
    grep -q GB_CPHMD_GPU_TITRATION_DONE titration.log 2>/dev/null && { echo "EVENT=DONE $(date +%H:%M:%S)"; exit 0; }
    pgrep -f "pmemd.*-cpin" >/dev/null 2>&1 || { echo "EVENT=PMEMD_GONE (not done) $(date +%H:%M:%S)"; tail -3 titration.log; exit 2; }
  fi
  step=$(grep "NSTEP =" mdinfo 2>/dev/null | tail -1 | sed -nE 's/.*NSTEP =[ ]*([0-9]+).*/\1/p' | head -1)
  now=$(date +%s)
  if [ "$step" != "$last_step" ]; then last_step="$step"; last_change=$now; fi
  age=$(( now - last_change ))
  if [ "$age" -gt "$STALL" ]; then
    echo "EVENT=STALL  mdinfo_step=${step:-?}  frozen=${age}s  $(date +%H:%M:%S)"; exit 3
  fi
  sleep 60
done

#!/usr/bin/env bash
# GB CpHMD GPU titration dashboard — cand_4 (pH ladder 3->7, 2 ns each).
#   bash titration_dashboard.sh          (single snapshot)
#   bash titration_live.sh               (auto-refresh every 5 s)
# Detects STALLS: a running pH whose .out hasn't been written in >5 min is
# flagged (pmemd.cuda can hang under WSL when another CUDA context shares the GPU).
OUT=/home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4
LOG=/home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4/titration.log
PHS="3 4 5 6 7"; NSTLIM=1000000; NPER=2; STALL=300   # stall threshold (s)

bar(){ local p=$1; local n=$((p/10)); local i; printf '['; for((i=0;i<10;i++)); do [ $i -lt $n ] && printf '#' || printf '.'; done; printf ']'; }

echo   "=================================================================="
echo   "  GB CpHMD GPU titration  -  cand_4   (pH 3->7, 2 ns each, GPU)"
echo   "  $(date '+%H:%M:%S  %Y-%m-%d')"
echo   "=================================================================="
alive=$(pgrep -fc "pmemd.cuda" 2>/dev/null); [ "${alive:-0}" -gt 0 ] && eng="RUNNING" || eng="idle/done"
echo   "  engine: pmemd.cuda ($eng)    GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
gprocs=""; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do gprocs="$gprocs $(ps -o comm= -p $p 2>/dev/null)($p)"; done
echo   "  GPU contexts:${gprocs:- none}    (a 2nd CUDA context can hang pmemd.cuda under WSL)"
echo   "------------------------------------------------------------------"
printf "  %-4s %-10s %-13s %-9s %s\n" "pH" "status" "progress" "step" "out-age"
donec=0; curfrac=0; stalled=0
for ph in $PHS; do
  o=$OUT/gb_cph_gpu_pH${ph}.out
  if [ ! -f "$o" ]; then printf "  %-4s %-10s %-13s\n" "$ph" "pending" "$(bar 0)"; continue; fi
  last=$(grep "NSTEP =" "$o" 2>/dev/null | tail -1)
  step=$(echo "$last" | sed -nE 's/.*NSTEP =[ ]*([0-9]+).*/\1/p' | head -1); step=${step:-0}
  if grep -q "wall time" "$o" 2>/dev/null; then
    printf "  %-4s %-10s %-13s %-9s\n" "$ph" "DONE" "$(bar 100) 100%" "-"; donec=$((donec+1)); continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$o" 2>/dev/null || date +%s) ))
  pct=$(( step*100/NSTLIM ))
  if [ "$age" -gt "$STALL" ] && [ "$eng" = "RUNNING" ]; then
    printf "  %-4s %-10s %s %3d%% %-9s %ss  <-- STALL?\n" "$ph" "STALLED!" "$(bar $pct)" "$pct" "$step" "$age"; stalled=1
  elif [ "$age" -gt "$STALL" ]; then
    printf "  %-4s %-10s %s %3d%% %-9s %ss  (engine stopped)\n" "$ph" "stopped" "$(bar $pct)" "$pct" "$step" "$age"
  else
    printf "  %-4s %-10s %s %3d%% %-9s %ss\n" "$ph" "running" "$(bar $pct)" "$pct" "$step" "$age"
  fi
  curfrac=$(awk "BEGIN{print $step/$NSTLIM}")
done
echo   "------------------------------------------------------------------"
overall=$(awk "BEGIN{printf \"%.0f\",($donec+$curfrac)/5*100}")
donens=$(awk "BEGIN{printf \"%.3f\",($donec+$curfrac)*$NPER}")
remns=$(awk "BEGIN{printf \"%.2f\",(5-$donec-$curfrac)*$NPER}")
start=$(head -1 "$LOG" 2>/dev/null | grep -oE '[0-9]{2}:[0-9]{2}' | head -1)
rate="?"; etah="?"
if [ -n "$start" ] && awk "BEGIN{exit !($donens>0)}"; then
  emin=$(( ( $(date +%s) - $(date -d "$start" +%s) ) / 60 )); [ $emin -le 0 ] && emin=1
  rate=$(awk "BEGIN{printf \"%.1f\",$donens/($emin/1440.0)}")
  etah=$(awk "BEGIN{printf \"%.1f\",$remns/($rate/24)}")
fi
echo   "  overall: $donec/5 pH done  ->  ~${overall}%   |   done ${donens} ns, remaining ~${remns} ns"
echo   "  avg rate ~${rate} ns/day  ->  ETA ~${etah} h   (NB: .out buffered/coarse; rate is a lower bound)"
[ "$stalled" = 1 ] && { echo; echo "  *** WARNING: STALL DETECTED -> pmemd.cuda likely hung (GPU context conflict)."; echo "      Free the GPU of other CUDA contexts (auto_trading/gmx), kill + relaunch."; }
ls $OUT/pka_gpu_pH*.dat >/dev/null 2>&1 && { echo "  ---- pKa (cphstats) ----"; for f in $OUT/pka_gpu_pH*.dat; do echo "   $(basename $f): $(grep -iE 'pKa|Pred' $f 2>/dev/null|head -2|tr '\n' ' ')"; done; }
[ "$donec" -eq 5 ] && echo "  *** TITRATION COMPLETE ***"
echo   "=================================================================="

#!/usr/bin/env bash
# GB CpHMD GPU titration dashboard — cand_4 (pH ladder 3->7, 2 ns each).
# Usage:  bash /home/galeito/titration_dashboard.sh
#   live: watch -n 30 bash /home/galeito/titration_dashboard.sh
OUT=/home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4
LOG=/home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4/titration.log
PHS="3 4 5 6 7"; NSTLIM=1000000; NPER=2   # 2 ns/pH

bar(){ local p=$1 n=$((p/10)) i; printf '['; for((i=0;i<10;i++)); do [ $i -lt $n ] && printf '#' || printf '.'; done; printf ']'; }

echo   "=================================================================="
echo   "  GB CpHMD GPU titration  -  cand_4   (pH 3->7, 2 ns each, GPU)"
echo   "  $(date '+%H:%M:%S  %Y-%m-%d')"
echo   "=================================================================="
# engine / GPU
alive=$(pgrep -fc "pmemd.cuda" 2>/dev/null); [ "${alive:-0}" -gt 0 ] && eng="RUNNING" || eng="idle/done"
echo   "  engine: pmemd.cuda ($eng)    GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
nsday=$(grep -h "ns/day" $OUT/gb_cph_gpu_pH*.out 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1); nsday=${nsday:-50}
echo   "------------------------------------------------------------------"
printf "  %-4s %-9s %-14s %-8s %s\n" "pH" "status" "progress" "step" "T(K)"
donec=0; curfrac=0
for ph in $PHS; do
  o=$OUT/gb_cph_gpu_pH${ph}.out
  if [ ! -f "$o" ]; then printf "  %-4s %-9s %-14s\n" "$ph" "pending" "$(bar 0)"; continue; fi
  last=$(grep "NSTEP =" "$o" 2>/dev/null | tail -1)
  step=$(echo "$last" | sed -nE 's/.*NSTEP =[ ]*([0-9]+).*/\1/p' | head -1); step=${step:-0}
  temp=$(echo "$last" | sed -nE 's/.*TEMP\(K\) =[ ]*([0-9.]+).*/\1/p' | head -1)
  if grep -q "wall time" "$o" 2>/dev/null; then
    printf "  %-4s %-9s %-14s %-8s %s\n" "$ph" "DONE" "$(bar 100) 100%" "-" "-"; donec=$((donec+1))
  else
    pct=$(( step*100/NSTLIM ))
    printf "  %-4s %-9s %s %3d%% %-8s %s\n" "$ph" "running" "$(bar $pct)" "$pct" "$step" "${temp:-..}"
    curfrac=$(awk "BEGIN{print $step/$NSTLIM}")
  fi
done
echo   "------------------------------------------------------------------"
overall=$(awk "BEGIN{printf \"%.0f\",($donec+$curfrac)/5*100}")
remns=$(awk "BEGIN{printf \"%.2f\",(5-$donec-$curfrac)*$NPER}")
donens=$(awk "BEGIN{printf \"%.3f\",($donec+$curfrac)*$NPER}")
# LIVE throughput from elapsed wall-clock since the driver started (1st log timestamp)
start=$(head -1 "$LOG" 2>/dev/null | grep -oE '[0-9]{2}:[0-9]{2}' | head -1)
rate=$nsday; etah="?"
if [ -n "$start" ] && awk "BEGIN{exit !($donens>0)}"; then
  emin=$(( ( $(date +%s) - $(date -d "$start" +%s) ) / 60 )); [ $emin -le 0 ] && emin=1
  rate=$(awk "BEGIN{printf \"%.1f\",$donens/($emin/1440.0)}")
  etah=$(awk "BEGIN{printf \"%.1f\",$remns/($rate/24)}")
fi
echo   "  overall: $donec/5 pH done  ->  ~${overall}%   |   done ${donens} ns, remaining ~${remns} ns"
echo   "  LIVE rate: ~${rate} ns/day (observed)   ->   ETA ~${etah} h"
# show pKa if cphstats results already exist
if ls $OUT/pka_gpu_pH*.dat >/dev/null 2>&1; then
  echo "  ---- pKa (cphstats, partial) ----"
  for f in $OUT/pka_gpu_pH*.dat; do echo "   $(basename $f): $(grep -iE 'pKa|Pred' $f 2>/dev/null | head -3 | tr '\n' ' ')"; done
fi
[ "$donec" -eq 5 ] && echo "  *** TITRATION COMPLETE -> generating pKa via cphstats ***"
echo   "=================================================================="

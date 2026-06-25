#!/usr/bin/env bash
# Resume-capable GB CpHMD GPU titration driver. Survives the WSL 2-CUDA-context
# pmemd.cuda hang (stochastic: NSTEP freezes while pmemd spins the GPU) WITHOUT
# touching the user's auto_trading: frequent restarts (ntwr=10000=20ps) + an
# integrated stall detector (NSTEP frozen > STALL s -> kill) + resume from the
# last restart, continuing protonation from the last cprestrt. Per-pH cpout is
# written in segments; cphstats combines them at the end.
#   CAND=cand_4 bash run_gb_resume.sh
set -u
source /home/galeito/Amber26/pmemd26/amber.sh
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
PMEMD=/home/galeito/Amber26/pmemd26/bin/pmemd.cuda
CAND=${CAND:-cand_4}
PHS=${PHS:-"3 4 5 6 7"}
DT=0.002; TARGET_PS=${TARGET_PS:-2000}; STALL=${STALL:-120}; MAXSEG=${MAXSEG:-80}
TMPL=/home/galeito/ADCpedia/experiments/wave2/cphmd/gb_cph_resume.tmpl
D=/home/galeito/ADCpedia/outputs/wave2/cphmd/$CAND
cd "$D"

step_now(){ grep "NSTEP =" mdinfo 2>/dev/null | tail -1 | sed -nE 's/.*NSTEP =[ ]*([0-9]+).*/\1/p' | head -1; }
# accumulated sim time (ps): read from mdinfo (ASCII, format-independent; on a
# resume it still holds the just-killed segment's last TIME, and irest=1
# continues that time). Robust whether the restart is ASCII or NetCDF.
cur_time(){ grep "TIME(PS) =" mdinfo 2>/dev/null | tail -1 | sed -nE 's/.*TIME\(PS\) =[ ]*([0-9.eE+]+).*/\1/p' | head -1; }

echo "[$(date +%H:%M:%S)] RESUME-DRIVER start  cand=$CAND  phs='$PHS'  target=${TARGET_PS}ps  stall=${STALL}s"
for ph in $PHS; do
  out=gb_cph_gpu_pH${ph}.out
  if [ -f "$out" ] && grep -q "wall time" "$out" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] pH $ph already complete -> skip"; continue
  fi
  echo "[$(date +%H:%M:%S)] === pH $ph ==="
  rm -f cont_pH${ph}.rst cont_pH${ph}.cpin cpout_gpu_pH${ph}_seg* 2>/dev/null
  seg=0
  while :; do
    seg=$((seg+1))
    if [ "$seg" -gt "$MAXSEG" ]; then echo "[$(date +%H:%M:%S)]  pH$ph GAVE UP after $MAXSEG segments"; break; fi
    if [ -f cont_pH${ph}.rst ]; then
      crd=cont_pH${ph}.rst; cpi=cont_pH${ph}.cpin; ir=1; nx=5
      cur=$(cur_time); cur=${cur:-0}
    else
      crd=gb_min.rst; cpi=cpin_gb; ir=0; nx=1; cur=0
    fi
    nstlim=$(awk "BEGIN{n=($TARGET_PS-$cur)/$DT; printf \"%d\", (n<0?0:n)}")
    if [ "$nstlim" -le 0 ]; then echo "[$(date +%H:%M:%S)]  pH$ph at target (${cur}ps)"; break; fi
    echo "[$(date +%H:%M:%S)]  pH$ph seg$seg: from ${cur}ps, nstlim=$nstlim, irest=$ir"
    sed -e "s/@PH@/$ph/" -e "s/@IREST@/$ir/" -e "s/@NTX@/$nx/" -e "s/@NSTLIM@/$nstlim/" "$TMPL" > seg_pH${ph}.in
    $PMEMD -O -i seg_pH${ph}.in -p complex_gb.prmtop -c "$crd" \
        -o "$out" -r cont_pH${ph}.rst \
        -cpin "$cpi" -cpout cpout_gpu_pH${ph}_seg${seg} -cprestrt cont_pH${ph}.cpin &
    pm=$!
    last="x"; lastchg=$(date +%s)
    while kill -0 $pm 2>/dev/null; do
      sleep 15
      st=$(step_now); now=$(date +%s)
      if [ "$st" != "$last" ]; then last="$st"; lastchg=$now; fi
      if [ $((now-lastchg)) -gt $STALL ]; then
        echo "[$(date +%H:%M:%S)]  !! STALL pH$ph seg$seg @NSTEP=${st:-?} -> kill+resume"
        kill -9 $pm 2>/dev/null; sleep 3; break
      fi
    done
    wait $pm 2>/dev/null || true
    if grep -q "wall time" "$out" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)]  pH$ph completed cleanly (seg$seg)"; break
    fi
    [ -f cont_pH${ph}.rst ] && echo "[$(date +%H:%M:%S)]  pH$ph resume from $(cur_time)ps" \
                            || echo "[$(date +%H:%M:%S)]  pH$ph hung before 1st restart -> retry from scratch"
  done
  echo "[$(date +%H:%M:%S)] pH $ph DONE"
done
echo "[$(date +%H:%M:%S)] GB_CPHMD_GPU_TITRATION_DONE"

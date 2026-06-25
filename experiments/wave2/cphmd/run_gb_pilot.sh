#!/bin/bash
set -e
export AMBERHOME=/home/galeito/amber_env; export PATH=$AMBERHOME/bin:$PATH
cd /home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4
echo "[$(date +%H:%M)] GB minimization..."
sander -O -i gb_min.in -p complex_gb.prmtop -c complex_gb.inpcrd -o gb_min.out -r gb_min.rst
echo "[$(date +%H:%M)] min done. CpHMD pH ladder..."
for ph in 4 5 6 7; do
  echo "[$(date +%H:%M)]   pH $ph ..."
  sed "s/@PH@/$ph/" gb_cph.tmpl > gb_cph_pH${ph}.in
  sander -O -i gb_cph_pH${ph}.in -p complex_gb.prmtop -c gb_min.rst \
     -o gb_cph_pH${ph}.out -r gb_cph_pH${ph}.rst \
     -cpin cpin_gb -cpout cpout_pH${ph} -cprestrt cprestrt_pH${ph}
  cphstats -i cpin_gb cpout_pH${ph} > pka_pH${ph}.dat 2>/dev/null
  echo "[$(date +%H:%M)]   pH $ph done -> pka_pH${ph}.dat"
done
echo "[$(date +%H:%M)] GB CpHMD PILOT COMPLETE"

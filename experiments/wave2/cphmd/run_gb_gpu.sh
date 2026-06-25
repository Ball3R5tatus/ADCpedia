#!/bin/bash
set -e
source /home/galeito/Amber26/pmemd26/amber.sh
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
PMEMD=/home/galeito/Amber26/pmemd26/bin/pmemd.cuda
cd /home/galeito/ADCpedia/outputs/wave2/cphmd/cand_4
for ph in 3 4 5 6 7; do
  echo "[$(date +%H:%M)] GB CpHMD GPU pH $ph (2 ns)..."
  sed "s/@PH@/$ph/" gb_cph_gpu.tmpl > gb_cph_gpu_pH${ph}.in
  $PMEMD -O -i gb_cph_gpu_pH${ph}.in -p complex_gb.prmtop -c gb_min.rst \
     -o gb_cph_gpu_pH${ph}.out -r gb_cph_gpu_pH${ph}.rst \
     -cpin cpin_gb -cpout cpout_gpu_pH${ph} -cprestrt cprestrt_gpu_pH${ph}
  echo "[$(date +%H:%M)] pH $ph done -> cpout_gpu_pH${ph}"
done
echo "GB_CPHMD_GPU_TITRATION_DONE"

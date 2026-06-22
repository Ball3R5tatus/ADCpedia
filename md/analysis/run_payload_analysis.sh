#!/usr/bin/env bash
# Payload (MMAE-core) self-fitted RMSD + Rg for cand_1 and cand_4.
set -e
source /home/galeito/miniconda3/etc/profile.d/conda.sh
conda activate gmx
RUNS=/home/galeito/ADCpedia/md/runs
ANA=/home/galeito/ADCpedia/md/analysis

for c in cand_1 cand_4; do
  echo "######## $c ########"
  cd $RUNS/$c
  # 1) make molecules whole across PBC (uses bonds from tpr)
  echo 0 | gmx trjconv -s prod.tpr -f prod.xtc -n ana_mmae.ndx -pbc whole \
       -o whole.xtc >/dev/null 2>&1
  # 2) payload self-fitted RMSD (fit MMAE, RMSD MMAE) -> internal conformational drift
  echo "MMAE MMAE" | gmx rms -s prod.tpr -f whole.xtc -n ana_mmae.ndx \
       -fit rot+trans -tu ns -o $ANA/$c/mmae_self_rmsd.xvg >/dev/null 2>&1
  # 3) payload radius of gyration (compactness/extension)
  echo "MMAE" | gmx gyrate -s prod.tpr -f whole.xtc -n ana_mmae.ndx \
       -o $ANA/$c/mmae_rg.xvg >/dev/null 2>&1
  rm -f whole.xtc
  echo "  -> $ANA/$c/mmae_self_rmsd.xvg"
  echo "  -> $ANA/$c/mmae_rg.xvg"
  echo "  last RMSD: $(grep -vE '^[@#]' $ANA/$c/mmae_self_rmsd.xvg | tail -1)"
  echo "  last Rg:   $(grep -vE '^[@#]' $ANA/$c/mmae_rg.xvg | tail -1)"
done
echo "DONE"

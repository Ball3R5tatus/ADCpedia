#!/usr/bin/env bash
# Full payload (MMAE-core) diagnostic for ONE candidate, end to end.
# Usage: bash payload_diag.sh <cid>           (e.g. 5 -> cand_5)
# Steps: analyze_traj.sh (complex.tpr, ana.ndx, prod_fit.xtc, contacts, lig_rmsd)
#        -> MMAE index group -> self-fitted RMSD + Rg on the Complex trajectory.
set -e
cid=$1
RUN=/home/galeito/ADCpedia/md/runs/cand_$cid
OUT=/home/galeito/ADCpedia/md/analysis/cand_$cid
MD=/home/galeito/ADCpedia/md
PYADC=/home/galeito/miniconda3/envs/adc_docking/bin/python
mkdir -p $OUT

# 1) base trajectory analysis (idempotent: skip if already present)
if [ ! -f $OUT/lig_prot_ncontacts.xvg ] || [ ! -f $RUN/ana.ndx ] || [ ! -f $OUT/prod_fit.xtc ]; then
  echo ">> analyze_traj.sh $cid"
  bash $MD/analyze_traj.sh $cid
else
  echo ">> base analysis already present, skipping analyze_traj.sh"
fi

# 2) MMAE-core index group (RDKit + MDAnalysis, adc_docking env)
echo ">> extract MMAE group"
$PYADC $MD/analysis/extract_mmae.py $cid

# 3) payload self-fitted RMSD + Rg on the PBC-corrected Complex trajectory
source /usr/local/gromacs/bin/GMXRC
FIT=$OUT/prod_fit.xtc
echo "MMAE MMAE" | gmx rms -s $RUN/complex.tpr -f $FIT -n $RUN/ana_mmae.ndx \
     -fit rot+trans -tu ns -o $OUT/mmae_self_rmsd.xvg >/dev/null 2>&1
echo "MMAE" | gmx gyrate -s $RUN/complex.tpr -f $FIT -n $RUN/ana_mmae.ndx \
     -o $OUT/mmae_rg.xvg >/dev/null 2>&1
echo ">> cand_$cid payload analysis done"
echo "   last self-RMSD (nm): $(grep -vE '^[@#]' $OUT/mmae_self_rmsd.xvg | tail -1)"
echo "   last Rg (nm):        $(grep -vE '^[@#]' $OUT/mmae_rg.xvg | tail -1)"

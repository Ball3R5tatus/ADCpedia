#!/bin/bash
# MMAE-core self-fitted RMSD for one replicate of a candidate.
# Reuses complex.tpr + ana_mmae.ndx (51-atom MMAE group, identical across reps of a
# candidate); only the compressed-Complex xtc changes. Self-fit (rot+trans) on MMAE,
# RMSD on MMAE => internal conformational drift of the payload core vs t=0.
# Usage: bash selfrmsd_reps.sh <cid> <deffnm>     e.g.  4 prod_rep2
set -e
source /usr/local/gromacs/bin/GMXRC
cid=$1; deffnm=$2
RUN=/home/galeito/ADCpedia/md/runs/cand_$cid
OUT=/home/galeito/ADCpedia/md/analysis/cand_$cid
mkdir -p "$OUT"; cd "$RUN"
tmp=$(mktemp -u)_whole.xtc
printf "System\n"      | gmx trjconv -s complex.tpr -f "$deffnm.xtc" -pbc whole -o "$tmp" >/dev/null 2>&1
printf "MMAE\nMMAE\n"  | gmx rms -s complex.tpr -f "$tmp" -n ana_mmae.ndx -fit rot+trans -tu ns \
                          -o "$OUT/mmae_selfrmsd_$deffnm.xvg" >/dev/null 2>&1
rm -f "$tmp"
echo "cand_$cid $deffnm -> $OUT/mmae_selfrmsd_$deffnm.xvg"

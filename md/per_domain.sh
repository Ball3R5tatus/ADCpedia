#!/bin/bash
# Per-domain self-fitted CA RMSD (Fab L/H + HER2 I-IV) + DSSP for a finished run.
# Distinguishes inter-domain motion (OK) from unfolding. Arg: cid
# Uses analysis/cand_<cid>/{complex.tpr,prod_fit.xtc} from analyze_traj.sh.
set -e
source /usr/local/gromacs/bin/GMXRC
cid=$1; run=~/ADCpedia/md/runs/cand_$cid; out=~/ADCpedia/md/analysis/cand_$cid
cd $run
# generic boundaries: protein chain A=3249 (HG removed), B=3256, C=9149; ligand=nlig after chain A
read CHB0 CHB1 CHC0 CHC1 NL < <(python3 -c "
import json; r=json.load(open('restraint.json')); nl=r.get('nlig_merged', r['nlig'])
chA=3249; b0=chA+nl+1; b1=chA+nl+3256; c0=b1+1; c1=b1+9149
print(b0,b1,c0,c1,nl)")
echo "domains: chainA_prot 1-3249 | ligand 3250-$((3249+NL)) | chainB ${CHB0}-${CHB1} | chainC ${CHC0}-${CHC1}"

S=$run/complex.tpr; F=$out/prod_fit.xtc
sel() { gmx select -s $S -on /tmp/d_$1.ndx -select "$2" >/dev/null 2>&1; printf "0\n0\n" | gmx rms -s $S -f $F -n /tmp/d_$1.ndx -o $out/dom_$1.xvg -tu ns >/dev/null 2>&1; }
sel chA "name CA and atomnr 1 to 3249"
sel chB "name CA and atomnr ${CHB0} to ${CHB1}"
sel chC "name CA and atomnr ${CHC0} to ${CHC1}"
# HER2 sub-domains (resid within chain C atom range)
sel hI   "name CA and atomnr ${CHC0} to ${CHC1} and resid 1 to 195"
sel hII  "name CA and atomnr ${CHC0} to ${CHC1} and resid 196 to 319"
sel hIII "name CA and atomnr ${CHC0} to ${CHC1} and resid 320 to 488"
sel hIV  "name CA and atomnr ${CHC0} to ${CHC1} and resid 489 to 607"
# DSSP (secondary structure preservation), sampled every 1 ns
gmx dssp -s $S -f $F -o $out/dssp.dat -dt 1000 >/dev/null 2>&1

python3 - "$out" <<'PY'
import sys, statistics as st
out=sys.argv[1]
def rms(tag):
    try:
        ys=[float(l.split()[1])*10 for l in open(f'{out}/dom_{tag}.xvg') if l[0] not in '#@']
        h=ys[len(ys)//2:]; return st.mean(h),max(ys)
    except: return None
labels=[('chA','Light chain (VL+CL)'),('chB','Heavy chain (VH+CH1)'),('chC','HER2 ECD (whole)'),
        ('hI','  HER2 dom I'),('hII','  HER2 dom II'),('hIII','  HER2 dom III'),('hIV','  HER2 dom IV')]
print("=== per-domain CA RMSD (self-fitted) ===")
for tag,lab in labels:
    r=rms(tag)
    if r: print(f"  {lab:24s}: mean={r[0]:.2f} A  max={r[1]:.2f} A")
# DSSP structured fraction
lines=[l.strip() for l in open(f'{out}/dssp.dat') if l.strip() and l[0] not in '#@=']
fr=[100*sum(c in 'HGIEB' for c in ln)/len(ln) for ln in lines if len(ln)>100]
if fr: print(f"=== DSSP structured %: first={fr[0]:.1f} last={fr[-1]:.1f} mean={st.mean(fr):.1f} std={st.pstdev(fr):.2f}")
PY

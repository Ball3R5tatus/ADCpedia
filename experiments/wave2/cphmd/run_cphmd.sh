#!/usr/bin/env bash
# =====================================================================
# AMBER constant-pH MD scaffold for the ADC conjugate (Cys214 thiosuccinimide).
#
# Rigorous, per-linker version of the §25.6 static-PROPKA pilot: a pH ladder of
# explicit-solvent CpHMD gives each candidate's local pKa for the titratable
# residues around Cys214 (the §20.13 / §25.6 observable), dynamically.
#
# ENGINE DECISION (Master §25.4): AMBER chosen over the GROMACS-constantph fork
# and CHARMM because it needs ZERO reparametrization — the system is AMBER-native
# (protein amber99sb-ildn≈ff14SB; ligand GAFF2/AM1-BCC; acpype already wrote
# md/lig_params/<cand>/LIG.acpype/LIG_AC.frcmod + LIG_bcc_gaff2.mol2). The fork
# is CHARMM36m-only; CHARMM CpHMD needs CGenFF — both = full reparametrization.
#
# STATUS: scaffold — AmberTools is NOT installed here (the blocker). Nothing
# below is validated by execution. Install first:
#     conda create -n ambertools -c conda-forge ambertools=24
#     conda activate ambertools
# then resolve the two per-candidate covalent values (see EDIT-ME) and run.
#
# Usage:
#   conda activate ambertools
#   CAND=cand_4 CYS_RESNUM=214 LIG_ATTACH=C3 PH_LADDER="4 5 6 7 8" \
#     bash experiments/wave2/cphmd/run_cphmd.sh
# =====================================================================
set -euo pipefail

ADC=/home/galeito/ADCpedia
CAND="${CAND:-cand_4}"
RECEPTOR="${RECEPTOR:-$ADC/structures/1N8Z_clean.pdb}"
LIGDIR="$ADC/md/lig_params/${CAND}_run/LIG.acpype"
LIG_MOL2="${LIG_MOL2:-$LIGDIR/LIG_bcc_gaff2.mol2}"
LIG_FRCMOD="${LIG_FRCMOD:-$LIGDIR/LIG_AC.frcmod}"
PH_LADDER="${PH_LADDER:-4 5 6 7 8}"        # pH points → cphstats fits the pKa
OUT="$ADC/outputs/wave2/cphmd/$CAND"

# ---- EDIT-ME (per-candidate covalent values) -----------------------
# CYS_RESNUM : the conjugation cysteine's residue number AS NUMBERED BY tleap
#   (sequential over the combined system — NOT necessarily the PDB 214). After a
#   first `loadpdb`, check with: parmed -p complex.prmtop -> `printResidues :CYS`.
# LIG_ATTACH : the ligand's attach-carbon ATOM NAME in LIG_bcc_gaff2.mol2
#   (the C that bonds to SG; the conjugation H was already removed when the
#   maleimide-Cys adduct topology was built — restraint.json C3_H_removed_lig).
CYS_RESNUM="${CYS_RESNUM:?set CYS_RESNUM (tleap residue number of the conjugation Cys)}"
LIG_ATTACH="${LIG_ATTACH:?set LIG_ATTACH (ligand attach-carbon atom name in the mol2)}"
# --------------------------------------------------------------------

# fail-closed: AmberTools must be present
for tool in tleap cpinutil.py cphstats; do
  command -v "$tool" >/dev/null 2>&1 || { echo "MISSING: $tool — install ambertools (see header). No run."; exit 2; }
done
MD=$(command -v pmemd.cuda || command -v pmemd || command -v sander)  # GPU if licensed, else CPU sander (free)
echo "using MD engine: $MD"
mkdir -p "$OUT"; cd "$OUT"

# --- 1. build the covalent conjugate in tleap -----------------------
# NOTE: the receptor PDB must have constant-pH residue names (ASP->AS4, GLU->GL4,
# HIS->HIP) so cpinutil can titrate them. Do this rename once on a copy:
#   sed -E 's/ ASP / AS4 /; s/ GLU / GL4 /; s/ HIS / HIP /' $RECEPTOR > rec_cph.pdb
REC_CPH="$OUT/rec_cph.pdb"
sed -E 's/(.{17})ASP/\1AS4/; s/(.{17})GLU/\1GL4/; s/(.{17})HIS/\1HIP/' "$RECEPTOR" > "$REC_CPH"

cat > build.tleap <<EOF
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams $LIG_FRCMOD
LIG = loadmol2 $LIG_MOL2
REC = loadpdb $REC_CPH
complex = combine { REC LIG }
# covalent thioether: Cys SG -- ligand attach carbon (HG + one ligand H pre-removed)
bond complex.$CYS_RESNUM.SG complex.LIG.$LIG_ATTACH
solvateoct complex TIP3PBOX 12.0
addionsrand complex Na+ 0 Cl- 0          # neutralize
addionsrand complex Na+ 0 Cl- 0          # (then add 0.15 M with a second pass / use addions counts)
saveamberparm complex complex.prmtop complex.inpcrd
savepdb complex complex_built.pdb
quit
EOF
tleap -f build.tleap > tleap.log 2>&1 || { echo "tleap failed — see $OUT/tleap.log (check CYS_RESNUM / LIG_ATTACH / covalent valence)"; exit 1; }

# --- 2. cpin (titratable residues; default = all standard titratable) -----
# Restrict to the proximal residues from select_titratable.py if desired (-resnum).
cpinutil.py -p complex.prmtop -o cpin -op complex_cph.prmtop > cpin.log 2>&1

# --- 3. mdin templates (explicit-solvent constant pH) ---------------
cat > min.in <<'EOF'
minimisation
 &cntrl
  imin=1, maxcyc=10000, ncyc=2000, ntb=1, cut=10.0,
  ntr=1, restraintmask='@CA,C,N & !:WAT,Na+,Cl-', restraint_wt=5.0,
 &end
EOF
cat > heat.in <<'EOF'
heat 0->300K NVT, restrained
 &cntrl
  imin=0, irest=0, ntx=1, nstlim=50000, dt=0.002,
  ntc=2, ntf=2, cut=10.0, ntb=1, ntp=0,
  ntt=3, gamma_ln=5.0, tempi=0.0, temp0=300.0,
  ntr=1, restraintmask='@CA,C,N & !:WAT,Na+,Cl-', restraint_wt=2.0,
  ntpr=1000, ntwx=5000, ntwr=5000, ig=-1,
 &end
EOF
cat > equil.in <<'EOF'
equilibrate NPT
 &cntrl
  imin=0, irest=1, ntx=5, nstlim=250000, dt=0.002,
  ntc=2, ntf=2, cut=10.0, ntt=3, gamma_ln=5.0, temp0=300.0,
  ntb=2, ntp=1, barostat=2, pres0=1.0, taup=2.0,
  ntpr=1000, ntwx=5000, ntwr=5000, ig=-1,
 &end
EOF
# prod template — @PH@ substituted per ladder point
cat > prod_cph.tmpl <<'EOF'
constant-pH production (explicit solvent) at pH @PH@
 &cntrl
  imin=0, irest=1, ntx=5, nstlim=2500000, dt=0.002,    ! 5 ns; extend for convergence
  ntc=2, ntf=2, cut=10.0, ntt=3, gamma_ln=5.0, temp0=300.0,
  ntb=1, ntp=0,                                        ! NVT for CpHMD production
  icnstph=2, ntcnstph=100, ntrelax=100, solvph=@PH@, saltcon=0.15,
  ntpr=5000, ntwx=10000, ntwr=50000, ig=-1,
 &end
EOF

# --- 4. min -> heat -> equil (standard) -----------------------------
$MD -O -i min.in   -p complex_cph.prmtop -c complex.inpcrd -ref complex.inpcrd -o min.out  -r min.rst
$MD -O -i heat.in  -p complex_cph.prmtop -c min.rst        -ref min.rst        -o heat.out -r heat.rst -x heat.nc
$MD -O -i equil.in -p complex_cph.prmtop -c heat.rst                            -o equil.out -r equil.rst -x equil.nc

# --- 5. constant-pH production over the pH ladder -------------------
for ph in $PH_LADDER; do
  sed "s/@PH@/$ph/g" prod_cph.tmpl > prod_pH${ph}.in
  $MD -O -i prod_pH${ph}.in -p complex_cph.prmtop -c equil.rst \
      -o prod_pH${ph}.out -r prod_pH${ph}.rst -x prod_pH${ph}.nc \
      -cpin cpin -cpout cpout_pH${ph} -cprestrt cprestrt_pH${ph}
  cphstats -i cpin cpout_pH${ph} > pkastats_pH${ph}.dat
done

echo "DONE. Per-residue protonation per pH in $OUT/pkastats_pH*.dat"
echo "Fit fraction-protonated vs pH (Henderson-Hasselbalch) per residue -> pKa."
echo "Compare the proximal-residue pKa ACROSS candidates = the per-linker signal."

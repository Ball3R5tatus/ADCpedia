#!/usr/bin/env bash
# Build the AMBER covalent GB-CpHMD conjugates for the remaining pilot
# candidates (cand_5, cand_vedotin), replicating the VALIDATED cand_4 recipe
# (BUILD_VALIDATED.md / §25.4). CPU-light: tleap + cpinutil + parmed QC, no MD.
#
# Shared, candidate-independent (same Cys214 site, same receptor conformation so
# the per-linker pKa comparison is controlled): rec_cph.pdb (CYX + proximal
# GL4-213/AS4-122/GL4-219) and junction.frcmod (S-c5 thioether; both attach
# carbons are GAFF type c5, so the cand_4 junction applies verbatim).
#
# Per-candidate (resolved from the acpype mol2 + restraint.json):
#   cand_5     : attach C55, remove H81  -> C55 charge -0.072700
#   cand_vedotin: attach C67, remove H107 -> C67 charge -0.073950
set -e
export AMBERHOME=/home/galeito/amber_env; export PATH=$AMBERHOME/bin:$PATH
ROOT=/home/galeito/ADCpedia
CPH=$ROOT/outputs/wave2/cphmd
SRC=$CPH/cand_4              # template (rec_cph.pdb + junction.frcmod live here)
LP=$ROOT/md/lig_params

build_one() {
  local cand=$1 attachC=$2 remH=$3 newq=$4 ligdir=$5
  local d=$CPH/$cand
  echo "==================================================================="
  echo " BUILD $cand   attachC=$attachC  removeH=$remH  newCharge=$newq"
  echo "==================================================================="
  mkdir -p "$d"
  cp -f "$SRC/rec_cph.pdb"     "$d/rec_cph.pdb"
  cp -f "$SRC/junction.frcmod" "$d/junction.frcmod"
  cat > "$d/build_gb.tleap" <<EOF
source leaprc.protein.ff14SB
source leaprc.constph
source leaprc.gaff2
loadamberparams $LP/$ligdir/LIG.acpype/LIG_AC.frcmod
loadamberparams $d/junction.frcmod
LIG = loadmol2 $LP/$ligdir/LIG.acpype/LIG_bcc_gaff2.mol2
remove LIG LIG.1.$remH
set LIG.1.$attachC charge $newq
REC = loadpdb $d/rec_cph.pdb
complex = combine { REC LIG }
bond complex.214.SG complex.1042.$attachC
saveamberparm complex $d/complex_gb.prmtop $d/complex_gb.inpcrd
quit
EOF
  tleap -f "$d/build_gb.tleap" > "$d/build_gb.log" 2>&1
  if [ ! -s "$d/complex_gb.prmtop" ]; then
    echo " !! tleap FAILED — tail of log:"; tail -25 "$d/build_gb.log"; return 1
  fi
  echo " tleap OK -> complex_gb.prmtop"
  grep -iE 'FATAL|ERROR|WARNING' "$d/build_gb.log" | grep -viE 'leaprc|Welcome' | head -8 || true
  # cpin (3 proximal carboxylates auto-detected as GL4/AS4), igb=2 to match production
  cpinutil.py -p "$d/complex_gb.prmtop" -o "$d/cpin_gb" -igb 2 > "$d/cpin_gb.log" 2>&1 \
    && echo " cpin_gb OK" || { echo " !! cpinutil FAILED:"; tail -15 "$d/cpin_gb.log"; return 1; }
  # ---- valence / charge QC (parmed) ----
  echo " --- QC (parmed) ---"
  parmed -p "$d/complex_gb.prmtop" <<PQC 2>/dev/null | grep -iE 'atom|charge|net|bond' | head -30
printBonds :214@SG
printBonds :1042@$attachC
netCharge
netCharge :1042
PQC
}

build_one cand_5        C55 H81  -0.072700 cand_5_run
build_one cand_vedotin  C67 H107 -0.073950 cand_vedotin_run
echo; echo "ALL PILOT CONJUGATES BUILT."

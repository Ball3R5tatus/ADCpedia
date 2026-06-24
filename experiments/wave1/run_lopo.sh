#!/usr/bin/env bash
# =====================================================================
# LOPO (leave-one-payload-class-out) GPU runner — turn-key.
#
# Tests generalization vs interpolation: for each held-out payload class C,
#   (held)   train a model with C removed, generate C-payload linkers
#   (indist) train the full model (all classes),  generate C-payload linkers
# Comparable usable-3D% => generalizes ; collapse => interpolates.
#
# Prereqs (no GPU):
#   PYTHONPATH=src python -m diffusion.data.build_split lopo \
#       --classes auristatin,camptothecin,maytansinoid,PBD_anthracycline,duocarmycin,eribulin --seed 0
#   PYTHONPATH=src python -m diffusion.data.build_split full --seed 0
#   PYTHONPATH=src python experiments/wave1/export_lopo_inputs.py
#
# Run from the difflinker_gpu env:
#   conda activate difflinker_gpu && bash experiments/wave1/run_lopo.sh
# =====================================================================
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-disabled}"  # headless: no wandb key/TTY

# ---- EDIT-ME paths -------------------------------------------------
ADC=/home/galeito/ADCpedia
DL=/home/galeito/tools/DiffLinker
GEOM_CKPT=$DL/models/geom_difflinker.ckpt
LINKER_SIZE=20
N_SAMPLES=500
DEVICE_TRAIN=gpu
DEVICE_GEN=cuda
SEED=0
# -------------------------------------------------------------------

SPLITS=$ADC/data/processed/splits
OUT=$ADC/outputs/wave1_lopo
RESULTS=$OUT/results.csv
MAP=$ADC/data/processed/difflinker_inputs/lopo_inputs_map.csv
mkdir -p "$OUT"
[ -f "$MAP" ] || { echo "missing $MAP — run export_lopo_inputs.py first"; exit 1; }

# train a split's config, then generate from an arbitrary input SDF, gate-score under <label>
train_and_gen () {
  local split="$1" input_sdf="$2" label="$3" train_table="$4"
  local cfg="$SPLITS/$split/$split.yml"
  [ -f "$cfg" ] || { echo "!! missing $cfg"; return 1; }
  local ck="$DL/checkpoints/$split"
  if [ ! -d "$ck" ] || [ -z "$(ls "$ck"/*.ckpt 2>/dev/null | grep -v 'epoch=00')" ]; then
    rm -rf "$ck"; mkdir -p "$ck"; cp "$GEOM_CKPT" "$ck/${split}_epoch=00.ckpt"
    ( cd "$DL" && python train_difflinker.py --config "$cfg" --device "$DEVICE_TRAIN" )
  else
    echo "  ($split already trained — reusing checkpoints)"
  fi
  local best; best=$(ls -t "$ck"/*.ckpt | grep -v 'epoch=00.ckpt' | head -1)
  local gdir="$OUT/$label/gen"; mkdir -p "$gdir"
  ( cd "$DL" && python generate.py --fragments "$input_sdf" --model "$best" \
      --linker_size "$LINKER_SIZE" --output "$gdir" --n_samples "$N_SAMPLES" --device "$DEVICE_GEN" )
  PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" eval \
      --gen-dir "$gdir" --label "$label" --results "$RESULTS" \
      --train-smiles-csv "$train_table" --smiles-col molecule
}

FULL="full_s${SEED}"
FULL_TABLE="$SPLITS/$FULL/geom_${FULL}_train_table.csv"

# iterate held-out classes from the input map (skip header)
tail -n +2 "$MAP" | while IFS=, read -r cls pname input_sdf anchors npay reason; do
  [ -z "$cls" ] && continue
  split="lopo_${cls}_s${SEED}"
  abs_input="$ADC/$input_sdf"
  echo "================  $cls  ================"
  # in-distribution baseline: full model generating this class's input
  train_and_gen "$FULL" "$abs_input" "indist_${cls}" "$FULL_TABLE"
  # held-out: LOPO model (class removed) generating the same input
  train_and_gen "$split" "$abs_input" "held_${cls}" "$SPLITS/$split/geom_${split}_train_table.csv"
done

PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" plot-lopo \
    --results "$RESULTS" --out "$OUT/lopo_generalization.png" --metric usable_pct

echo "DONE. results: $RESULTS  figure: $OUT/lopo_generalization.png"

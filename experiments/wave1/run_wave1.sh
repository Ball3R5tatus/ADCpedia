#!/usr/bin/env bash
# =====================================================================
# Wave-1 GPU runner — Pareto cysteine-fraction sweep (and, optionally, LOPO).
#
# For EACH split directory under data/processed/splits/ it:
#   1. seeds a fresh resume checkpoint from the GEOM checkpoint,
#   2. fine-tunes DiffLinker (train_difflinker.py, the config in the split dir),
#   3. generates N linkers on a FIXED payload input,
#   4. scores them with the unified validity gate (eval_sweep.py),
# then renders the Pareto frontier figure.
#
# Splits must be built FIRST (no GPU):
#   PYTHONPATH=src python -m diffusion.data.build_split pareto --fractions 50,66,75,85,100 --seed 0
#   # for multi-seed rigor (recommended, see Finding 9), also: --seed 1 ; --seed 2
#
# Run this from the difflinker_gpu conda env:
#   conda activate difflinker_gpu
#   bash experiments/wave1/run_wave1.sh
# =====================================================================
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-disabled}"  # headless: no wandb key/TTY

# ---- EDIT-ME paths -------------------------------------------------
ADC=/home/galeito/ADCpedia
DL=/home/galeito/tools/DiffLinker
GEOM_CKPT=$DL/models/geom_difflinker.ckpt
GEN_INPUT=$ADC/data/processed/difflinker_inputs/MMAE_MC-Val-Cit-PAB_0.sdf
LINKER_SIZE=20            # fixed operational size (sweet spot 20-25)
N_SAMPLES=500            # >=500 for Wilson CIs that mean something (Finding 9)
DEVICE_TRAIN=gpu          # train_difflinker.py --device
DEVICE_GEN=cuda           # generate.py --device
SPLIT_GLOB="pareto_*"     # set to "lopo_*" to run the LOPO arm instead
# -------------------------------------------------------------------

SPLITS=$ADC/data/processed/splits
OUT=$ADC/outputs/wave1
RESULTS=$OUT/results.csv
mkdir -p "$OUT"

for cfgdir in "$SPLITS"/$SPLIT_GLOB; do
  [ -d "$cfgdir" ] || { echo "no splits match $SPLIT_GLOB — build them first"; exit 1; }
  split=$(basename "$cfgdir")
  cfg="$cfgdir/$split.yml"
  [ -f "$cfg" ] || { echo "!! missing config $cfg — skipping"; continue; }
  echo "================  $split  ================"

  # 1. fresh resume seed (a single .ckpt so find_last_checkpoint picks the GEOM weights)
  ck="$DL/checkpoints/$split"
  rm -rf "$ck"; mkdir -p "$ck"
  cp "$GEOM_CKPT" "$ck/${split}_epoch=00.ckpt"

  # 2. fine-tune (run from the DiffLinker repo; config has relative checkpoints/logs)
  ( cd "$DL" && python train_difflinker.py --config "$cfg" --device "$DEVICE_TRAIN" )

  # 3. pick a checkpoint to generate from.
  #    NOTE: this takes the most-recent epoch checkpoint. With early-stopping
  #    (patience 20) that is ~20 epochs past the val-loss minimum, which is fine
  #    for generation quality. If your ModelCheckpoint filenames encode val loss,
  #    refine this to pick the minimum instead.
  best=$(ls -t "$ck"/*.ckpt | grep -v "epoch=00.ckpt" | head -1 || true)
  [ -z "$best" ] && best="$ck/${split}_epoch=00.ckpt"
  echo "generating from: $best"

  # 4. generate on the FIXED payload input (same input for every split)
  gdir="$OUT/$split/gen"
  mkdir -p "$gdir"
  ( cd "$DL" && python generate.py --fragments "$GEN_INPUT" --model "$best" \
      --linker_size "$LINKER_SIZE" --output "$gdir" --n_samples "$N_SAMPLES" \
      --device "$DEVICE_GEN" )

  # 5. gate eval (reuses src/diffusion/gate; novelty vs this split's training set)
  PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" eval \
      --gen-dir "$gdir" --label "$split" --results "$RESULTS" \
      --train-smiles-csv "$cfgdir/geom_${split}_train_table.csv" --smiles-col molecule
done

# 6. Pareto figure (averages replicates per cysteine fraction)
PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" plot \
    --results "$RESULTS" --out "$OUT/pareto_frontier.png"

echo "DONE. results: $RESULTS   figure: $OUT/pareto_frontier.png"

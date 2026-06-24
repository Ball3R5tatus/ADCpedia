#!/usr/bin/env bash
# =====================================================================
# Wave-2 benchmark — lock the generation numbers with a multi-seed,
# size-swept, n>=500, Wilson-CI table across the surviving configs.
#
# A *confirmation* engine (Finding 9), not a discovery engine: it pins the
# Pareto/connectivity ceiling with CIs instead of single noisy checkpoints.
#
# Reuses the Pareto-built training splits as config proxies:
#   pareto_f100_s* = cysteine-only ; pareto_f066_s* = multi-site ;
#   pareto_f085_s* = curriculum.  Plus a zero-shot (GEOM, no training) baseline.
# Adds the SIZE dimension (15/20/25/30) the Pareto sweep fixed at 20.
#
# Prereqs (no GPU): build the splits for every seed you list, e.g.
#   for s in 0 1 2; do PYTHONPATH=src python -m diffusion.data.build_split \
#       pareto --fractions 66,85,100 --seed $s ; done
#
# Run from the difflinker_gpu env:
#   conda activate difflinker_gpu && bash experiments/wave2/run_benchmark.sh
# =====================================================================
set -euo pipefail

# wandb requires an API key + TTY; disable it for headless training.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ---- EDIT-ME -------------------------------------------------------
ADC=/home/galeito/ADCpedia
DL=/home/galeito/tools/DiffLinker
GEOM_CKPT=$DL/models/geom_difflinker.ckpt
GEN_INPUT=$ADC/data/processed/difflinker_inputs/MMAE_MC-Val-Cit-PAB_0.sdf
N_SAMPLES=500
DEVICE_TRAIN=gpu
DEVICE_GEN=cuda
SEEDS="${SEEDS:-0 1 2}"
SIZES="${SIZES:-15 20 25 30}"
# config label -> pareto split base (without seed)
declare -A CONFIG=( [cysonly]=pareto_f100 [multisite]=pareto_f066 [curriculum]=pareto_f085 )
# -------------------------------------------------------------------

SPLITS=$ADC/data/processed/splits
OUT=$ADC/outputs/wave2
RESULTS=$OUT/results.csv
mkdir -p "$OUT"

gate_eval () {  # gen_dir label train_table
  PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" eval \
      --gen-dir "$1" --label "$2" --results "$RESULTS" \
      --train-smiles-csv "$3" --smiles-col molecule
}

for seed in $SEEDS; do
  # ---- trained configs ----
  for name in "${!CONFIG[@]}"; do
    split="${CONFIG[$name]}_s${seed}"
    cfg="$SPLITS/$split/$split.yml"
    [ -f "$cfg" ] || { echo "!! missing $cfg (build the split) — skipping"; continue; }
    ck="$DL/checkpoints/$split"
    if [ -z "$(ls "$ck"/*.ckpt 2>/dev/null | grep -v 'epoch=00' || true)" ]; then
      rm -rf "$ck"; mkdir -p "$ck"; cp "$GEOM_CKPT" "$ck/${split}_epoch=00.ckpt"
      ( cd "$DL" && python train_difflinker.py --config "$cfg" --device "$DEVICE_TRAIN" )
    fi
    best=$(ls -t "$ck"/*.ckpt | grep -v 'epoch=00.ckpt' | head -1 || true)
    [ -z "$best" ] && { echo "!! $split produced no trained checkpoint (training failed?) — aborting"; exit 1; }
    for sz in $SIZES; do
      gdir="$OUT/${name}_s${seed}_sz${sz}/gen"; mkdir -p "$gdir"
      ( cd "$DL" && python generate.py --fragments "$GEN_INPUT" --model "$best" \
          --linker_size "$sz" --output "$gdir" --n_samples "$N_SAMPLES" --device "$DEVICE_GEN" )
      gate_eval "$gdir" "bench_${name}_sz${sz}_s${seed}" "$SPLITS/$split/geom_${split}_train_table.csv"
    done
  done
  # ---- zero-shot baseline (GEOM checkpoint, no fine-tune) ----
  for sz in $SIZES; do
    gdir="$OUT/zeroshot_s${seed}_sz${sz}/gen"; mkdir -p "$gdir"
    ( cd "$DL" && python generate.py --fragments "$GEN_INPUT" --model "$GEOM_CKPT" \
        --linker_size "$sz" --output "$gdir" --n_samples "$N_SAMPLES" --device "$DEVICE_GEN" ) \
      || echo "  (zero-shot size $sz failed — expected at large ADC sizes, §12)"
    [ -n "$(ls "$gdir"/*.sdf 2>/dev/null)" ] && \
      gate_eval "$gdir" "bench_zeroshot_sz${sz}_s${seed}" "$SPLITS/pareto_f100_s${seed}/geom_pareto_f100_s${seed}_train_table.csv"
  done
done

PYTHONPATH="$ADC/src" python "$ADC/experiments/wave1/eval_sweep.py" table \
    --results "$RESULTS" --out "$OUT/benchmark.md"
echo "DONE. results: $RESULTS  table: $OUT/benchmark.md"

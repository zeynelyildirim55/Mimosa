#!/bin/bash
#SBATCH --job-name=Mimosa_MILtrain
#SBATCH --output=logs/mil/Mimosa_MILtrain_%j.txt
#SBATCH --partition=GPU_nvgtx1080
#SBATCH --exclude=gpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# One MIL round: train one epoch on ROUND_DIR/selection_train (unless SKIP_TRAIN=1), then score the
# validation subset and write ROUND_DIR/val_metrics.json.
#   MODEL_IN=... ROUND_DIR=runs/mil/round1 OPT_IN=... sbatch --export=ALL,MODEL_IN,ROUND_DIR,OPT_IN archive/scripts/run_mil_train.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
ROOT=$HOME/Mimosa
PAIRS=data/custom/pairs_train_val.tsv
# GPUs are not allocated by Slurm here: take an idle card, or the least busy one if none is idle
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sort -t, -k3 -n -k2 -n | head -1 | cut -d, -f1)
echo "$(hostname) GPU $CUDA_VISIBLE_DEVICES"
set -e
mkdir -p "$ROOT/$ROUND_DIR"
MODEL="$ROOT/$MODEL_IN"
if [ "${SKIP_TRAIN:-0}" != 1 ] && [ "${RANDOM_SELECTION:-0}" != 1 ]; then
    # helper jobs outside the dependency chain may still be finishing the last chunks
    for _ in $(seq 240); do
        [ "$(ls "$ROOT/$ROUND_DIR/selection_train"/chunk_*.tsv 2>/dev/null | wc -l)" -ge "${NCHUNKS:-400}" ] && break
        sleep 30
    done
fi
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    OPT=(); [ -n "$OPT_IN" ] && OPT=(--optimizer-in "$ROOT/$OPT_IN")
    [ "${RANDOM_SELECTION:-0}" = 1 ] && OPT+=(--random-selection)
    OPT+=(--epochs "${EPOCHS:-1}")
    python -u "$ROOT/training/mil_train.py" --model-in "$MODEL" --model-out "$ROOT/$ROUND_DIR/model.pth" \
        --optimizer-out "$ROOT/$ROUND_DIR/optimizer.pth" "${OPT[@]}" \
        --selection-dir "$ROOT/$ROUND_DIR/selection_train" --pairs "$ROOT/$PAIRS" \
        --features "$ROOT/data/custom/mil_features/train"
    MODEL="$ROOT/$ROUND_DIR/model.pth"
fi
python -u "$ROOT/training/mil_select.py" --model "$MODEL" --pairs "$ROOT/$PAIRS" \
    --features "$ROOT/data/custom/mil_features/val" --out-dir "$ROOT/$ROUND_DIR/val_selection" --nchunks 20 --threads 4
python -u "$ROOT/training/mil_metrics.py" --selection-dir "$ROOT/$ROUND_DIR/val_selection" --out "$ROOT/$ROUND_DIR/val_metrics.json"

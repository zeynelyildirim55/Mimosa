#!/bin/bash
#SBATCH --job-name=Mimosa_Predict
#SBATCH --output=logs/predict/Mimosa_Predict_%A_%a.txt
#SBATCH --partition=CPU_midmem,CPU_lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=08:00:00

# Gene-level test with the original sliding-window procedure (step size 5), one shard per task:
#   MODEL=runs/site40_long/model_concate_112.pth SPLIT=test_unseen_pair NSHARDS=40 \
#     sbatch --array=0-39 --export=ALL,MODEL,SPLIT,NSHARDS archive/scripts/run_predict_site40.slurm
# Then the same script with MERGE=1 combines the shards and writes the metrics.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=$HOME/Mimosa
# PAIRS, SEQS and OUT_DIR can be overridden for other inputs (e.g. the cross-evaluations)
PAIRS="${PAIRS:-$ROOT/data/custom/pairs_$SPLIT.tsv}"
SEQS="${SEQS:-$ROOT/data/custom/sequences.fa}"
OUT_DIR="${OUT_DIR:-$ROOT/results/$(basename "$(dirname "$MODEL")")/$SPLIT}"
mkdir -p "$OUT_DIR"
COMMON=(--pairs "$PAIRS" --sequences "$SEQS" --model "$ROOT/$MODEL" --out "$OUT_DIR/pred" --stepsize 5)

if [ "${MERGE:-0}" = 1 ]; then
    python -u "$ROOT/training/predict_site40.py" "${COMMON[@]}" --merge
else
    python -u "$ROOT/training/predict_site40.py" "${COMMON[@]}" --threads 1 \
        --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
fi

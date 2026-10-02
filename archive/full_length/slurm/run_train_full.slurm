#!/bin/bash
#SBATCH --job-name=Mimosa_TrainFull
#SBATCH --output=Mimosa_TrainFull_%A_%a.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:45:00

# One task at a time; each one resumes where the previous stopped:
#   sbatch --array=1-20%1 run_train_full.slurm --amp --epochs 2
# Override the environment or output directory with MIMOSA_ENV / OUT_DIR.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${MIMOSA_ENV:-mimosa2}"

# GPUs are not allocated by Slurm on this cluster, so pick the least busy card
if [ -z "$CUDA_VISIBLE_DEVICES" ] && command -v nvidia-smi >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "selected GPU $CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
fi

OUT="${OUT_DIR:-runs/full_length}"
if [ -f "$OUT/DONE" ]; then
    echo "training already finished ($OUT/DONE), task ${SLURM_ARRAY_TASK_ID} exits"
    exit 0
fi

python -u training/train_full.py --out-dir "$OUT" --max-hours "${MAX_HOURS:-11.0}" "$@"

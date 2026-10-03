#!/bin/bash
#SBATCH --job-name=Mimosa
#SBATCH --partition=GPU_nvgtx1080
# GPU_nvrtx4090 is the default partition; gpu1 (GPU_nvgtx1080) is used while both 4090 nodes are busy (approved).
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

# Gene-level prediction with the original CPU procedure (get_cts + kmers_predict, step 5), one shard per task.
# Submit from the repo root:
#   MODEL=runs/site40_v2/model_concate_117.pth SPLIT=valid_unseen_pair NSHARDS=48 \
#     sbatch --array=0-47 --export=ALL,MODEL,SPLIT,NSHARDS slurm/predict_cpu.slurm
# then the same with MERGE=1 (no --array) merges the shards and writes the metrics.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/Mimosa"
D=data/custom_v2
OUT_DIR="results/$(basename "$(dirname "${MODEL:?}")")/${SPLIT:?}/cpu"
mkdir -p "$OUT_DIR"
COMMON=(--pairs "$D/pairs_$SPLIT.tsv" --sequences "$D/sequences.fa" --model "$MODEL" --out "$OUT_DIR/pred" --stepsize 5)
if [ "${MERGE:-0}" = 1 ]; then
    python -u training/predict_site40.py "${COMMON[@]}" --merge
else
    python -u training/predict_site40.py "${COMMON[@]}" --threads 1 --shard "$SLURM_ARRAY_TASK_ID" --nshards "${NSHARDS:?}"
fi

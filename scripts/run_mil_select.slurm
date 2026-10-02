#!/bin/bash
#SBATCH --job-name=Mimosa_MILsel
#SBATCH --output=logs/mil/Mimosa_MILsel_%j.txt
#SBATCH --partition=GPU_nvgtx1080
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=11:50:00

# Scores all windows with MODEL and keeps each pair's best window; one worker per idle GPU on the
# node. Workers share OUT through chunk claims, so several jobs (other nodes, restarts) can help.
#   MODEL=... PAIRS=... FEATURES=... OUT=... NCHUNKS=400 sbatch --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS scripts/run_mil_select.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
ROOT=$HOME/Mimosa
: "${MODEL:?MODEL not set}" "${PAIRS:?PAIRS not set}" "${FEATURES:?FEATURES not set}" "${OUT:?OUT not set}" "${NCHUNKS:?NCHUNKS not set}"
ngpu=$(nvidia-smi -L | wc -l)
started=0
for i in $(seq 0 $((ngpu - 1))); do
    used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 1000 ] && [ "$util" -lt 10 ]; then
        CUDA_VISIBLE_DEVICES=$i python -u "$ROOT/training/mil_select.py" --model "$ROOT/$MODEL" --pairs "$ROOT/$PAIRS" \
            --features "$ROOT/$FEATURES" --out-dir "$ROOT/$OUT" --nchunks "$NCHUNKS" \
            --start $((i * NCHUNKS / ngpu)) $EXTRA &
        started=$((started + 1))
    fi
done
echo "$(hostname): started $started workers"
[ "$started" -eq 0 ] && exit 1
wait

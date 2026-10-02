#!/bin/bash
#SBATCH --job-name=Mimosa_MILhelp
#SBATCH --output=logs/mil/Mimosa_MILhelp_%A_%a.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:50:00

# Waits for an idle RTX 4090 on this node and uses it to help with whichever MIL round is currently
# selecting (chunks are claimed, so it shares the work with the gpu1 job). Idles between checks.
#   FIRST_ROUND=7 LAST_ROUND=9 sbatch --array=1-3%1 --export=ALL,FIRST_ROUND,LAST_ROUND scripts/run_mil_helper.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
ROOT=$HOME/Mimosa
BASE=$ROOT/runs/mil_scratch
FIRST=${FIRST_ROUND:-7}; LAST=${LAST_ROUND:-9}; NCHUNKS=400
chunks() { ls "$BASE/round$1/selection_train"/chunk_*.tsv 2>/dev/null | wc -l; }

current_round() {   # first round whose selection is unfinished and whose input model is complete
    for r in $(seq "$FIRST" "$LAST"); do
        m="$BASE/round$((r - 1))/model.pth"
        if [ "$(chunks $r)" -lt "$NCHUNKS" ] && [ -f "$m" ] && [ $(( $(date +%s) - $(stat -c %Y "$m") )) -gt 60 ]; then
            echo "$r"; return
        fi
    done
}

idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 1000 && $3 < 10 {print $1}'
}

while true; do
    if [ "$(chunks $LAST)" -ge "$NCHUNKS" ]; then echo "$(date) all selections done"; exit 0; fi
    r=$(current_round)
    [ "${DRY:-0}" = 1 ] && { echo "current round: ${r:-none}"; exit 0; }
    gpus=$(idle_gpus)
    if [ -z "$r" ] || [ -z "$gpus" ]; then sleep 300; continue; fi
    echo "$(date) $(hostname): round $r, idle GPUs: $(echo $gpus)"
    for i in $gpus; do
        CUDA_VISIBLE_DEVICES=$i python -u "$ROOT/training/mil_select.py" --model "$BASE/round$((r - 1))/model.pth" \
            --pairs "$ROOT/data/custom/pairs_train_val.tsv" --features "$ROOT/data/custom/mil_features/train" \
            --out-dir "$BASE/round$r/selection_train" --nchunks $NCHUNKS --start $(( (200 + 97 * i) % NCHUNKS )) &
    done
    wait
    echo "$(date) round $r: this node's workers finished ($(chunks $r)/$NCHUNKS chunks done)"
done

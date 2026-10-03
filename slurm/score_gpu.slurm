#!/bin/bash
#SBATCH --job-name=Mimosa
#SBATCH --partition=GPU_nvgtx1080
# GPU_nvrtx4090 is the default partition; gpu1 (GPU_nvgtx1080) is used while both 4090 nodes are busy (approved).
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# Gene-level prediction on GPU from precomputed window features (same features and decision rule as
# kmers_predict/decision_for_whole: a pair is positive when its best window scores above 0.5).
# One worker per idle GPU; then metrics on all rows (unprocessable = negative) and on processable rows.
# Submit from the repo root:
#   MODEL=runs/site40_v2/model_concate_117.pth SPLIT=valid_unseen_pair FEATURES=data/custom_v2/mil_features/valid_unseen_pair \
#     NCHUNKS=40 sbatch --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
cd "$HOME/Mimosa"
D=data/custom_v2
: "${MODEL:?}" "${SPLIT:?}" "${FEATURES:?}" "${NCHUNKS:?}"
OUT="results/$(basename "$(dirname "$MODEL")")/$SPLIT/gpu"
mkdir -p "$OUT"
ngpu=$(nvidia-smi -L | wc -l)
started=0
for i in $(seq 0 $((ngpu - 1))); do
    used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 1000 ] && [ "$util" -lt 10 ]; then
        CUDA_VISIBLE_DEVICES=$i python -u training/mil_select.py --model "$MODEL" --pairs "$D/pairs_$SPLIT.tsv" \
            --features "$FEATURES" --sequences "$D/sequences.fa" --out-dir "$OUT" --nchunks "$NCHUNKS" \
            --start $((i * NCHUNKS / ngpu)) &
        started=$((started + 1))
    fi
done
echo "$(hostname): started $started workers"
[ "$started" -eq 0 ] && exit 1
wait
n=$(ls "$OUT"/chunk_*.tsv 2>/dev/null | wc -l)
[ "$n" -ne "$NCHUNKS" ] && { echo "incomplete: $n of $NCHUNKS chunks"; exit 1; }
python -u training/mil_metrics.py --selection-dir "$OUT" --pairs "$D/pairs_$SPLIT.tsv" --out "$OUT/metrics.json"

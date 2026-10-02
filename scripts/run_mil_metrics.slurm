#!/bin/bash
#SBATCH --job-name=Mimosa_MILmetrics
#SBATCH --output=logs/mil/Mimosa_MILmetrics_%j.txt
#SBATCH --partition=CPU_midmem,CPU_lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00

# Gene-level metrics of a finished selection directory; fails if chunks are missing.
#   OUT=runs/mil_scratch/test/test_unseen_pair PAIRS=data/custom/pairs_test_unseen_pair.tsv NCHUNKS=40 sbatch ... scripts/run_mil_metrics.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
ROOT=$HOME/Mimosa
: "${OUT:?OUT not set}" "${PAIRS:?PAIRS not set}" "${NCHUNKS:?NCHUNKS not set}"
n=$(ls "$ROOT/$OUT"/chunk_*.tsv 2>/dev/null | wc -l)
if [ "$n" -ne "$NCHUNKS" ]; then echo "selection incomplete: $n of $NCHUNKS chunks"; exit 1; fi
python -u "$ROOT/training/mil_metrics.py" --selection-dir "$ROOT/$OUT" --pairs "$ROOT/$PAIRS" --out "$ROOT/$OUT/metrics.json"

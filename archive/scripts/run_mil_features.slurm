#!/bin/bash
#SBATCH --job-name=Mimosa_MILfeat
#SBATCH --output=logs/mil/Mimosa_MILfeat_%A_%a.txt
#SBATCH --partition=CPU_midmem,CPU_lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00

# Window features for the MIL experiment, one shard per array task:
#   PAIRS=... NAME=train SPLIT=train NSHARDS=200 sbatch --array=0-199%50 --export=ALL,PAIRS,NAME,SPLIT,NSHARDS,EVERY archive/scripts/run_mil_features.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=$HOME/Mimosa
EXTRA=()
[ -n "$SPLIT" ] && EXTRA+=(--split "$SPLIT")
python -u "$ROOT/training/mil_features.py" --pairs "$ROOT/$PAIRS" --sequences "$ROOT/data/custom/sequences.fa" \
    --out-prefix "$ROOT/data/custom/mil_features/$NAME" --every "${EVERY:-1}" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" "${EXTRA[@]}"

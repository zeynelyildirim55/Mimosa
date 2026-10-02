#!/bin/bash
#SBATCH --job-name=Mimosa_Pairing
#SBATCH --output=Mimosa_Pairing_%A_%a.txt
#SBATCH --partition=CPU_midmem,CPU_lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

# usage: NSHARDS=40 sbatch --array=0-39 --export=ALL,NSHARDS run_pairing.slurm [extra args]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

python -u training/precompute_pairing.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-1}" "$@"

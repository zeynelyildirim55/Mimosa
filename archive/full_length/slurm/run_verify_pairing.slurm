#!/bin/bash
#SBATCH --job-name=Mimosa_VerifyCache
#SBATCH --output=Mimosa_VerifyCache_%j.txt
#SBATCH --partition=CPU_midmem,CPU_lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

python -u training/verify_pairing_cache.py "$@"

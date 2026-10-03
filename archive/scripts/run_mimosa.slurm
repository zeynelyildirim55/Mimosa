#!/bin/bash
#SBATCH --job-name=Mimosa_Test
#SBATCH --output=Mimosa_Log_%j.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --mem-per-cpu=20000
#SBATCH --cpus-per-task=8

set -x

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

python -u training/Mimosa.py

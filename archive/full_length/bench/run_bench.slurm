#!/bin/bash
#SBATCH --job-name=Mimosa_Bench
#SBATCH --output=Mimosa_Bench_%j.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${MIMOSA_ENV:-mimosa}"

python -u bench/bench_gpu.py "$@"

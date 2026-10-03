#!/bin/bash
#SBATCH --job-name=Mimosa_Test
#SBATCH --output=Mimosa_Log_%A_Test%a.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --mem-per-cpu=20000
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

python -u training/Mimosa.py --mode test --stepsize 5 --threads 8 \
    --data data/miRAW_Test${SLURM_ARRAY_TASK_ID}.txt

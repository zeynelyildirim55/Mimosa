#!/bin/bash
#SBATCH --job-name=Mimosa_Prep
#SBATCH --output=Mimosa_Prep_%j.txt
#SBATCH --partition=CPU_himem
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

python -u training/prepare_dataset.py "$@"

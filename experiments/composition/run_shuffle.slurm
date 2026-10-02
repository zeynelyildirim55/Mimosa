#!/bin/bash
#SBATCH --job-name=Mimosa_MirnaShuffle
#SBATCH --output=logs/predict/Mimosa_MirnaShuffle_%j.txt
#SBATCH --partition=CPU_himem
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
python -u experiments/composition/mirna_shuffle.py

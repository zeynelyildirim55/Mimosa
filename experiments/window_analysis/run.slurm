#!/bin/bash
#SBATCH --job-name=Mimosa_WindowAnalysis
#SBATCH --output=logs/predict/Mimosa_WindowAnalysis_%j.txt
#SBATCH --partition=CPU_himem
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:30:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
python -u experiments/window_analysis/window_analysis.py

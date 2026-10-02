#!/bin/bash
#SBATCH --job-name=Mimosa_Train_site40
#SBATCH --output=logs/training/Mimosa_Train_site40_%j.txt
#SBATCH --partition=CPU_midmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=23:00:00

# Original Mimosa training (perform_train, unchanged: 40 epochs, batch 256, lr 1e-4) on the
# 40-nt site rows of our dataset. Checkpoints (model_concate_<epoch>.pth) go to runs/site40/.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa

# RUN_DIR and EPOCHS can be overridden: RUN_DIR=site40_long EPOCHS=120 sbatch ...
ROOT=$HOME/Mimosa
RUN_DIR="${RUN_DIR:-site40}"
mkdir -p "$ROOT/runs/$RUN_DIR"
cd "$ROOT/runs/$RUN_DIR"
python -u "$ROOT/training/Mimosa.py" --mode train \
    --data "$ROOT/data/custom/site40_Train_Validation.txt" \
    --epochs "${EPOCHS:-40}" \
    --threads "${SLURM_CPUS_PER_TASK:-8}"

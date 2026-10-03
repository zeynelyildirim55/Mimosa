#!/bin/bash
#SBATCH --job-name=Mimosa
#SBATCH --partition=CPU_midmem
# Original training code is CPU-only; CPU_midmem (24 cores, same hardware as the earlier 120-epoch run) approved by the user.
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

# Experiment A: original Mimosa training (perform_train unchanged: batch 256, lr 1e-4, seed 1234), 120 epochs,
# validation = 40-nt-site rows of the data_with_negatives valid splits. Mimosa.py saves model_concate_<epoch>.pth
# whenever the validation loss improves, so the last saved checkpoint is the selected model.
# Submit from the repo root: sbatch slurm/train_site40.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
ROOT=$HOME/Mimosa
mkdir -p "$ROOT/runs/site40_v2"
cd "$ROOT/runs/site40_v2"
python -u "$ROOT/training/Mimosa.py" --mode train \
    --data "$ROOT/data/custom_v2/site40_Train_ValidSplits.txt" \
    --epochs 120 --threads "$SLURM_CPUS_PER_TASK"

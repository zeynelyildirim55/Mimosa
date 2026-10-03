#!/bin/bash
#SBATCH --job-name=Mimosa
#SBATCH --partition=GPU_nvgtx1080
# GPU_nvrtx4090 is the default partition; gpu1 (GPU_nvgtx1080) is used while both 4090 nodes are busy (approved).
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# Dataset preparation for the validation-split experiments (data/custom_v2):
# all usable training_chunks rows go to training (--val-frac 0); model selection uses data_with_negatives valid splits.
# Submit from the repo root: sbatch slurm/prepare.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
cd "$HOME/Mimosa"
python -u training/prepare_dataset.py --val-frac 0 --out-dir data/custom_v2

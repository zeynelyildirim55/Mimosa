#!/bin/bash
#SBATCH --job-name=Mimosa_MILshuffle
#SBATCH --output=logs/mil/Mimosa_MILshuffle_%j.txt
#SBATCH --partition=CPU_himem
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# DATASET=miraw|ours sbatch --export=ALL,DATASET experiments/composition/run_mil_shuffle.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
: "${DATASET:?DATASET not set}"
python -u experiments/composition/mil_shuffle.py --model "${MODEL:-runs/mil_scratch/round9/model.pth}" --name "${NAME:-mil_r9}" \
    --dataset "$DATASET" --threads "${SLURM_CPUS_PER_TASK:-8}"

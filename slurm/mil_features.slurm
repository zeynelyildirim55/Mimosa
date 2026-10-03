#!/bin/bash
#SBATCH --job-name=Mimosa
#SBATCH --partition=GPU_nvgtx1080
# GPU_nvrtx4090 is the default partition; gpu1 (GPU_nvgtx1080) is used while both 4090 nodes are busy (approved).
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

# Window features of every processable row of one valid/test split, one shard per array task (no subsampling).
# Submit from the repo root:
#   NAME=valid_unseen_source NSHARDS=100 sbatch --array=0-99%30 --export=ALL,NAME,NSHARDS slurm/mil_features.slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/Mimosa"
D=data/custom_v2
python -u training/mil_features.py --pairs "$D/pairs_${NAME:?}.tsv" --sequences "$D/sequences.fa" \
    --out-prefix "$D/mil_features/$NAME" --shard "$SLURM_ARRAY_TASK_ID" --nshards "${NSHARDS:?}"

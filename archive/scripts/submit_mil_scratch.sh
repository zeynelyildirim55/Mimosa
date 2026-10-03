#!/bin/bash
# MIL chain for a model trained from scratch (same initial model as the original perform_train):
#   round 0: validation of the untrained model
#   round 1: one epoch on a random window per pair
#   round r: best-window selection with model r-1 (gpu1 1080 Ti + a 4090 node), one epoch, validation
# Usage: archive/scripts/submit_mil_scratch.sh <train-features job id> [rounds]
set -e
cd ~/Mimosa
TRAIN_FEAT=$1; ROUNDS=${2:-6}
BASE=runs/mil_scratch
PAIRS=data/custom/pairs_train_val.tsv
FEATURES=data/custom/mil_features/train
J0=$(MODEL_IN=$BASE/model_init.pth ROUND_DIR=$BASE/round0 SKIP_TRAIN=1 sbatch --parsable \
     --export=ALL,MODEL_IN,ROUND_DIR,SKIP_TRAIN archive/scripts/run_mil_train.slurm)
echo "round 0 (validation of the untrained model): $J0"
T=$(MODEL_IN=$BASE/model_init.pth ROUND_DIR=$BASE/round1 RANDOM_SELECTION=1 OPT_IN= sbatch --parsable \
    --dependency=afterok:$TRAIN_FEAT --export=ALL,MODEL_IN,ROUND_DIR,RANDOM_SELECTION,OPT_IN archive/scripts/run_mil_train.slurm)
echo "round 1 (random window per pair): train+validation $T"
for r in $(seq 2 $ROUNDS); do
    PREV=$BASE/round$((r - 1))
    OUT=$BASE/round$r/selection_train
    S1=$(MODEL=$PREV/model.pth PAIRS=$PAIRS FEATURES=$FEATURES OUT=$OUT NCHUNKS=400 EXTRA= sbatch --parsable \
         --dependency=afterok:$T --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    S2=$(MODEL=$PREV/model.pth PAIRS=$PAIRS FEATURES=$FEATURES OUT=$OUT NCHUNKS=400 EXTRA= sbatch --parsable \
         --dependency=afterok:$T --partition=GPU_nvrtx4090 --cpus-per-task=8 --mem=32G \
         --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    T=$(MODEL_IN=$PREV/model.pth ROUND_DIR=$BASE/round$r OPT_IN=$PREV/optimizer.pth RANDOM_SELECTION=0 sbatch --parsable \
        --dependency=afterany:$S1:$S2 --export=ALL,MODEL_IN,ROUND_DIR,OPT_IN,RANDOM_SELECTION archive/scripts/run_mil_train.slurm)
    echo "round $r: selection $S1 (1080 Ti) + $S2 (4090), train+validation $T"
done

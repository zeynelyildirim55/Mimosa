#!/bin/bash
# Submits the whole MIL chain with Slurm dependencies:
#   round 0: validation of the starting model (the 120-epoch site model)
#   round r: selection on the training pairs with model r-1 (gpu1 1080 Ti + a 4090 node sharing the
#            work), then one training epoch + validation
# Usage: archive/scripts/submit_mil_rounds.sh <train-features job id> <val-features job id> [rounds]
set -e
cd ~/Mimosa
TRAIN_FEAT=$1; VAL_FEAT=$2; ROUNDS=${3:-3}
M0=runs/site40_long/model_concate_112.pth
PAIRS=data/custom/pairs_train_val.tsv
FEATURES=data/custom/mil_features/train
J0=$(MODEL_IN=$M0 ROUND_DIR=runs/mil/round0 SKIP_TRAIN=1 sbatch --parsable --dependency=afterok:$VAL_FEAT \
     --export=ALL,MODEL_IN,ROUND_DIR,SKIP_TRAIN archive/scripts/run_mil_train.slurm)
echo "round 0 (validation of the start model): $J0"
PREV_MODEL=$M0; PREV_OPT=""; DEP="afterok:$TRAIN_FEAT"
for r in $(seq 1 $ROUNDS); do
    OUT=runs/mil/round$r/selection_train
    S1=$(MODEL=$PREV_MODEL PAIRS=$PAIRS FEATURES=$FEATURES OUT=$OUT NCHUNKS=400 EXTRA= sbatch --parsable \
         --dependency=$DEP --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    S2=$(MODEL=$PREV_MODEL PAIRS=$PAIRS FEATURES=$FEATURES OUT=$OUT NCHUNKS=400 EXTRA= sbatch --parsable \
         --dependency=$DEP --partition=GPU_nvrtx4090 --cpus-per-task=8 --mem=32G \
         --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    T=$(MODEL_IN=$PREV_MODEL ROUND_DIR=runs/mil/round$r OPT_IN=$PREV_OPT sbatch --parsable \
        --dependency=afterany:$S1:$S2 --export=ALL,MODEL_IN,ROUND_DIR,OPT_IN archive/scripts/run_mil_train.slurm)
    echo "round $r: selection $S1 (1080 Ti) + $S2 (4090), train+validation $T"
    PREV_MODEL=runs/mil/round$r/model.pth; PREV_OPT=runs/mil/round$r/optimizer.pth; DEP="afterok:$T"
done

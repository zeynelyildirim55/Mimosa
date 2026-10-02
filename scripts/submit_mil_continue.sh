#!/bin/bash
# Continues a finished MIL chain: rounds FIRST..LAST, each = best-window selection with the previous
# round's model, EPOCHS epochs on that selection (optimizer state carried over), validation.
# Usage: scripts/submit_mil_continue.sh <first round> <last round> <epochs per round> [base dir]
set -e
cd ~/Mimosa
FIRST=$1; LAST=$2; EPOCHS=$3; BASE=${4:-runs/mil_scratch}
export PAIRS=data/custom/pairs_train_val.tsv FEATURES=data/custom/mil_features/train NCHUNKS=400 EXTRA= EPOCHS
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency=afterok:$AFTER)
for r in $(seq $FIRST $LAST); do
    PREV=$BASE/round$((r - 1))
    [ -f "$PREV/model.pth" ] || [ ${#DEP[@]} -gt 0 ] || { echo "missing $PREV/model.pth"; exit 1; }
    export MODEL=$PREV/model.pth OUT=$BASE/round$r/selection_train
    S1=$(sbatch --parsable "${DEP[@]}" --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA scripts/run_mil_select.slurm)
    S2=$(sbatch --parsable "${DEP[@]}" --partition=GPU_nvrtx4090 --cpus-per-task=8 --mem=32G \
         --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA scripts/run_mil_select.slurm)
    export MODEL_IN=$PREV/model.pth ROUND_DIR=$BASE/round$r OPT_IN=$PREV/optimizer.pth RANDOM_SELECTION=0
    T=$(sbatch --parsable --dependency=afterany:$S1:$S2 \
        --export=ALL,MODEL_IN,ROUND_DIR,OPT_IN,RANDOM_SELECTION,EPOCHS scripts/run_mil_train.slurm)
    echo "round $r: selection $S1 (1080 Ti) + $S2 (4090), train ($EPOCHS epochs) + validation $T"
    DEP=(--dependency=afterok:$T)
done

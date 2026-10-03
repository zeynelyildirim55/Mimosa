#!/bin/bash
# Gene-level test of a MIL model on the three test splits, one split after the other
# (workers only use idle GPUs, so overlapping splits would find no free card).
# Usage: archive/scripts/submit_mil_test.sh <model path relative to ~/Mimosa> <output dir relative to ~/Mimosa>
set -e
cd ~/Mimosa
MODEL=$1; BASE=$2
PREV=""
for spec in "test_unseen_pair 40" "test_unseen_source 200" "test_unseen_target 200"; do
    set -- $spec; S=$1; N=$2
    export MODEL PAIRS=data/custom/pairs_$S.tsv FEATURES=data/custom/mil_features/$S OUT=$BASE/$S NCHUNKS=$N EXTRA=
    DEP=(); [ -n "$PREV" ] && DEP=(--dependency=afterany:$PREV)
    A=$(sbatch --parsable "${DEP[@]}" --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    B=$(sbatch --parsable "${DEP[@]}" --partition=GPU_nvrtx4090 --cpus-per-task=8 --mem=32G \
        --export=ALL,MODEL,PAIRS,FEATURES,OUT,NCHUNKS,EXTRA archive/scripts/run_mil_select.slurm)
    M=$(sbatch --parsable --dependency=afterany:$A:$B --export=ALL,OUT,PAIRS,NCHUNKS archive/scripts/run_mil_metrics.slurm)
    echo "$S: selection $A (1080 Ti) + $B (4090), metrics $M"
    PREV=$A:$B
done

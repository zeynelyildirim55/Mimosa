# Walkthrough

Running log of the validation-split experiments (AGENTS.md compliant). Earlier work (random 7% holdout, test seen several times) is described in README.md and the reports; its test numbers are superseded by the runs below.

## Decisions (agreed with the user, 2026-10-02)

- Model selection uses the `data_with_negatives` valid splits (valid_unseen_pair/source/target); the test splits are evaluated once per experiment.
- Training uses all usable `training_chunks` rows (no random holdout).
- Experiment A (paper method): retrain the 40-nt site model, 120 epochs; selection by site-level validation loss as in the paper, on the 40-nt-site rows of the valid splits (Mimosa.py keeps the checkpoint with the lowest validation loss).
- Experiment B (multiple-instance learning) is dropped: it changes the paper's training procedure and the professor asked to keep the paper's method. The earlier MIL report stays as exploratory work (selection on the random holdout, test seen twice). The valid-split window features (jobs 13519–13521) are still computed in case it is revisited.
- Rows Mimosa cannot process (sequence missing from rna.fa, miRNA outside 10–30 nt, non-ACGU) cannot be fed to the model. Test results are reported on all rows (these rows counted as negative) and on the processable rows.
- Test breakdown decided before seeing any test result (2026-10-02): metrics on all rows (main result), and additionally on the 40-nt-site rows and on the gene rows, each also on processable rows; all from the single test run (a 40-nt target gives exactly one window, so its gene-level prediction is the site-level prediction).
- Compute: GPU_nvrtx4090 first; gpu1 (GPU_nvgtx1080) while both 4090 nodes are busy.
- Sequences are read from `~/fasta_files/rna.fa`.

## Commands

All jobs are submitted from the repository root; Slurm writes `Mimosa_<jobid>.txt` to the root. The logs of the jobs below were then moved to `logs/site40_v2/`.

```bash
sbatch slurm/prepare.slurm    # job 13518: training/prepare_dataset.py --val-frac 0 --out-dir data/custom_v2
python training/make_site40_valsplits.py    # data/custom_v2/site40_Train_ValidSplits.txt: train 39,770 site rows, val 6,374 (valid-split 40-nt sites)
NAME=valid_unseen_pair   NSHARDS=10  sbatch --array=0-9     --export=ALL,NAME,NSHARDS slurm/mil_features.slurm   # job 13519
NAME=valid_unseen_source NSHARDS=100 sbatch --array=0-99%27 --export=ALL,NAME,NSHARDS slurm/mil_features.slurm   # job 13520
NAME=valid_unseen_target NSHARDS=100 sbatch --array=0-99%27 --export=ALL,NAME,NSHARDS slurm/mil_features.slurm   # job 13521
sbatch slurm/train_site40.slurm    # job 13557: Experiment A, Mimosa.py --mode train, 120 epochs, CPU_midmem, runs/site40_v2
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=valid_unseen_pair NSHARDS=48 sbatch --array=0-47 --export=ALL,MODEL,SPLIT,NSHARDS slurm/predict_cpu.slurm   # job 13746: CPU reference for the CPU/GPU check
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=valid_unseen_pair FEATURES=data/custom_v2/mil_features/valid_unseen_pair NCHUNKS=40 sbatch --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm   # job 13747: GPU scoring for the same check
python experiments/cpu_gpu_check/compare.py --pairs data/custom_v2/pairs_valid_unseen_pair.tsv --cpu results/site40_v2/valid_unseen_pair/cpu --gpu results/site40_v2/valid_unseen_pair/gpu   # 0 of 11,281 decisions differ
# single test run of the selected model (GPU path; test features from data/custom, identical pair files), chained:
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=test_unseen_pair   FEATURES=data/custom/mil_features/test_unseen_pair   NCHUNKS=40  sbatch --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm   # job 13795
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=test_unseen_source FEATURES=data/custom/mil_features/test_unseen_source NCHUNKS=200 sbatch --dependency=afterok:13795 --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm   # job 13796
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=test_unseen_target FEATURES=data/custom/mil_features/test_unseen_target NCHUNKS=200 sbatch --dependency=afterok:13796 --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm   # job 13797
for s in test_unseen_pair test_unseen_source test_unseen_target; do python training/metrics_by_kind.py --selection-dir results/site40_v2/$s/gpu --pairs data/custom_v2/pairs_$s.tsv --out results/site40_v2/$s/metrics_by_kind.json; done
```

Notes:
- prepare_dataset.py on ~/fasta_files/rna.fa produced the same eval pair files and sequences.fa as the earlier run, so the existing test window features (data/custom/mil_features/test_unseen_*) are reused.
- Job 13557 finished in 8 h 18 min. Selected checkpoint: runs/site40_v2/model_concate_117.pth (epoch 118, lowest validation loss 0.4784; site-level validation acc 0.830). Validation loss had flattened after about epoch 36 (0.4826), so no further epochs were run.
- Test is run on GPU (user decision) after checking on valid_unseen_pair that GPU and CPU predictions agree.
- CPU/GPU check on valid_unseen_pair: 0 of 11,281 gene-level decisions differ, so the test uses the GPU path. Gene-level validation result of the selected model on valid_unseen_pair: accuracy 0.514 (processable rows), 0.523 (all rows); specificity 0.099.

## Results (single test run, selected model runs/site40_v2/model_concate_117.pth)

Accuracy / specificity; all rows = unprocessable rows counted as negative (main result).

| Split | All rows | Processable | 40-nt sites (all rows) | Gene rows (all rows) |
|---|---|---|---|---|
| test_unseen_pair | 0.517 / 0.168 | 0.513 / 0.118 | 0.818 / 0.857 (n=625) | 0.512 / 0.158 |
| test_unseen_source | 0.518 / 0.169 | 0.514 / 0.118 | 0.834 / 0.874 (n=6,083) | 0.513 / 0.158 |
| test_unseen_target | 0.514 / 0.165 | 0.511 / 0.112 | 0.864 / 0.826 (n=5,593) | 0.510 / 0.156 |

Full metrics: results/site40_v2/<split>/metrics_by_kind.json and results/site40_v2/<split>/gpu/metrics.json.

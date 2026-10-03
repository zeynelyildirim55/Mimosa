# Mimosa
## Advancing microRNA Target Site Prediction with Transformer and Base-Pairing Patterns
This repository contains code and data for Mimosa, a tool designed to recognize miRNA targets, including both canonical and elusive non-canonical sites. It uses the Transformer framework and integrates base-pairing patterns to capture in-depth attributes. Mimosa has demonstrated superior performance compared to existing deep learning models and can be applied across diverse species. As an independent tool, Mimosa requires no third-party software and holds promise for facilitating miRNA-specific target identification and exploring intricate small RNA regulatory network.

![A_graph](https://github.com/biyueeee/Mimosa/assets/104138625/be4e0b88-acff-4db1-a933-be122751dd3a)

## About this branch
This branch retrains Mimosa on a miRNA–mRNA dataset from our lab with the paper's method: the 40-nt architecture, features, training procedure and gene-level test rule are the authors'. The report [report/Mimosa_report.pdf](report/Mimosa_report.pdf) describes the data, the training, the test and the results; [WALKTHROUGH.md](WALKTHROUGH.md) lists every decision and command in order.

- **Training**: the 39,770 pairs of `training_chunks` whose target is a 40-nt site (Mimosa's training input), original `perform_train`, 120 epochs.
- **Model selection**: lowest validation loss, as in the paper, on the 40-nt-site rows of the `data_with_negatives` validation splits (`valid_unseen_pair/source/target`). Selected: epoch 118.
- **Test**: the three `data_with_negatives` test splits, evaluated once with the selected model.

| Model | Test data | Accuracy | Recall | Specificity |
|---|---|---|---|---|
| Published model | paper's 10 miRAW test sets | 0.757 | 0.933 | 0.580 |
| Trained on our data (paper's method) | `test_unseen_pair/source/target`, all rows | 0.514–0.518 | 0.864–0.867 | 0.165–0.169 |
| | only the rows whose target is a 40-nt site | 0.818–0.864 | 0.782–0.903 | 0.826–0.874 |

The model recognises binding sites in unseen pairs, but at gene level (whole transcripts, a pair is positive if any 40-nt window scores above 0.5) it calls most pairs positive.

## Repository layout
| Path | Content |
|---|---|
| `training/` | Original `Mimosa.py`, `utils.py`, `model_mimosa.pth`; added `prepare_dataset.py`, `make_site40_valsplits.py`, `predict_site40.py`, `mil_features.py` / `mil_select.py` / `mil_common.py` (GPU scoring), `mil_metrics.py`, `metrics_by_kind.py` |
| `slurm/` | Slurm jobs of this run; each has its usage in the header. Submit from the repository root |
| `experiments/cpu_gpu_check/` | Check that GPU scoring gives the same decisions as the original CPU code |
| `results/site40_v2/` | Metrics of the validation check and of the test |
| `runs/site40_v2/` | The selected model `model_concate_117.pth` |
| `report/` | `Mimosa_report.pdf` and `make_report_v2.py`, which builds it |
| `WALKTHROUGH.md` | Decisions, commands and results of this run |
| `archive/` | Earlier attempts and their scripts (not used in this run) |
| `data/`, `web/` | Original miRAW files and web server code |

Other folders under `runs/`, `results/`, `experiments/`, `logs/` and `report/` belong to earlier experiments and are not part of the report.

## Changes to `training/Mimosa.py`
Command-line interface (`--mode`, `--data`, `--stepsize`, `--epochs`, `--threads`); precision is reported next to average precision (the original code printed average precision as PPV); the commented-out `torch.save` is enabled so training keeps the model with the lowest validation loss. Architecture, features, loss, optimiser, seed and test procedure are unchanged; training runs on CPU as in the original code.

## Reproducing
The lab dataset and everything derived from it (`data/custom_v2/`) are **not** in this repository. Dataset preparation expects the raw files at `~/training_chunks`, `~/data_with_negatives` and `~/fasta_files/rna.fa`.

Checking the released model on the paper's data needs no extra files:
```bash
python training/Mimosa.py --mode test --data data/miRAW_Test0.txt --stepsize 5
```

On the cluster, from the repository root (exact commands and job ids in [WALKTHROUGH.md](WALKTHROUGH.md)):
```bash
sbatch slurm/prepare.slurm                       # data/custom_v2/: all usable training rows, full validation and test splits
python training/make_site40_valsplits.py         # training file: training sites + 40-nt sites of the validation splits
sbatch slurm/train_site40.slurm                  # original training, 120 epochs, keeps the lowest-validation-loss model
NAME=test_unseen_pair NSHARDS=10 sbatch --array=0-9 --export=ALL,NAME,NSHARDS slurm/mil_features.slurm   # window features
MODEL=runs/site40_v2/model_concate_117.pth SPLIT=test_unseen_pair FEATURES=data/custom_v2/mil_features/test_unseen_pair \
    NCHUNKS=40 sbatch --export=ALL,MODEL,SPLIT,FEATURES,NCHUNKS slurm/score_gpu.slurm                    # gene-level test
python training/metrics_by_kind.py --selection-dir results/site40_v2/test_unseen_pair/gpu \
    --pairs data/custom_v2/pairs_test_unseen_pair.tsv                                                   # metrics by target type
```
`slurm/predict_cpu.slurm` runs the same gene-level test with the original CPU code.

## Requirements
- Python 3.9
- pytorch 1.13.1
- scikit-learn 0.24.2
- numpy 1.23.5 (newer NumPy breaks scikit-learn 0.24.2)
- pandas

## Online
Users can explore Mimosa in this repository or access Mimosa via the online web server at: http://monash.bioweb.cloud.edu.au/Mimosa/.


## Copyright
This project is free to use for non-commercial purposes.

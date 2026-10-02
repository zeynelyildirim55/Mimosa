# Mimosa
## Advancing microRNA Target Site Prediction with Transformer and Base-Pairing Patterns
This repository contains code and data for Mimosa, a tool designed to recognize miRNA targets, including both canonical and elusive non-canonical sites. It uses the Transformer framework and integrates base-pairing patterns to capture in-depth attributes. Mimosa has demonstrated superior performance compared to existing deep learning models and can be applied across diverse species. As an independent tool, Mimosa requires no third-party software and holds promise for facilitating miRNA-specific target identification and exploring intricate small RNA regulatory network.

![A_graph](https://github.com/biyueeee/Mimosa/assets/104138625/be4e0b88-acff-4db1-a933-be122751dd3a)

## About this branch
This branch reproduces Mimosa and retrains it on a miRNA–mRNA dataset from our lab, keeping the paper's 40-nt architecture, features and test rule. Two reports describe everything that was done:

- [report/Mimosa_reproduction_report.pdf](report/Mimosa_reproduction_report.pdf): reproduction of the paper, changes to the code, the data, training with the paper's method, gene-level tests and diagnostic experiments.
- [report/Mimosa_MIL_report.pdf](report/Mimosa_MIL_report.pdf): a second experiment that changes only the training procedure (multiple-instance training), so that all gene-level pairs can be used.

| Model | Test data | Accuracy | Recall | Specificity |
|---|---|---|---|---|
| Published model | paper's 10 miRAW test sets | 0.757 | 0.933 | 0.580 |
| Trained on 40-nt sites (paper's method) | our 3 test splits | 0.51–0.52 | 0.88 | 0.14 |
| Multiple-instance training (round 9) | our 3 test splits | 0.76–0.77 | 0.78–0.80 | 0.75 |

Our test splits are `test_unseen_pair`, `test_unseen_source` and `test_unseen_target` (816k pairs; new miRNA, new target, or both new). The published model also scores at chance on them (accuracy 0.499), and on miRAW its decisions follow the miRNA rather than the target, which our splits do not allow (reports, section on diagnostics).

## Repository layout
| Path | Content |
|---|---|
| `training/` | Original `Mimosa.py`, `utils.py`, `model_mimosa.pth`; added `prepare_dataset.py`, `predict_site40.py` and `mil_*.py` |
| `scripts/` | Slurm jobs and submit scripts; each has its usage in the header. Submit from the repository root |
| `experiments/` | Diagnostic experiments: `cross_eval`, `window_analysis`, `composition` |
| `results/` | Metrics of the tests and of every training round |
| `runs/` | The three final models: 40 epochs, 120 epochs, multiple-instance round 9 |
| `logs/` | Logs of the paper reproduction, dataset preparation and training |
| `report/` | The two reports and the scripts that build them |
| `archive/full_length/` | An abandoned attempt without the 40-nt limit (code only) |
| `data/`, `web/` | Original miRAW files and web server code |

## Changes to `training/Mimosa.py`
Command-line interface (`--mode`, `--data`, `--stepsize`, `--epochs`, `--threads`); precision is reported next to average precision (the original code printed average precision as PPV); the commented-out `torch.save` is enabled so training keeps the best model. Architecture, features, loss, optimiser and test procedure are unchanged.

## Reproducing
The lab dataset and everything derived from it (`data/custom/`) are **not** in this repository. Dataset preparation expects the raw files at `~/training_chunks`, `~/data_with_negatives` and `~/fasta_files.zip`.

Paper reproduction with the released model needs no extra data:
```bash
python training/Mimosa.py --mode test --data data/miRAW_Test0.txt --stepsize 5
```

On the cluster, from the repository root:
```bash
sbatch scripts/run_prepare.slurm          # builds data/custom/
sbatch scripts/run_train_site40.slurm     # original training on the 40-nt sites (RUN_DIR, EPOCHS to override)
sbatch scripts/run_predict_site40.slurm   # gene-level test, one shard per array task (see the header)
```
Multiple-instance experiment: `scripts/run_mil_features.slurm` (window features, once), `scripts/submit_mil_scratch.sh` (rounds 1–6), `scripts/submit_mil_continue.sh` (further rounds) and `scripts/submit_mil_test.sh` (gene-level test). The reports list the settings that were used.

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

# Archived: Mimosa without the 40-nt limit

Not used. This was an attempt to train Mimosa on full-length targets (gene-level
miRNA–mRNA pairs) instead of 40-nt binding sites. It was dropped because it departs
from the paper's method; the project follows the original 40-nt architecture instead.

Contents:
- `training/` model (masked pooling, 4,096-position encoding), dataset with padding and
  length-bucketed batching, base-pairing features (the authors' Smith_Waterman on the
  full transcript, or an ungapped seed scan), resumable training, prediction, and the
  pairing-cache precompute/verification scripts
- `slurm/` job scripts (pairing cache array, resumable training array, cache verification)
- `bench/` GPU benchmark script and probes
- `data/` pairing cache for pairs_train_val.tsv (40 shards) and the memmapped sequence index
- `runs/pilot_gpu1/` 264-step pilot (loss stayed at chance level)
- `logs/` Slurm logs of the jobs above

Measured on a GTX 1080 Ti (PyTorch 2.5, memory-efficient attention, fp32): ~3.8
samples/s on real data, about 7.4 days per epoch over 2.4M pairs.

To run anything here again, move the scripts back into `training/`: they import
`utils.py` from there and each other by module name.

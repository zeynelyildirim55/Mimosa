'''Does the MIL model's call follow the target or the miRNA identity?

Same design as mirna_shuffle.py, applied to the MIL model, on two datasets:
  miraw: the paper's test sets, where we already found the authors' model tracks miRNA identity
  ours:  our own test_unseen_pair, the benchmark this model was actually evaluated on

Targets are kept and miRNAs are swapped between positive and negative pairs. If the call
tracks the target, results should not flip when the miRNA is swapped. The true label of a
swapped pair is unknown; only the direction the call moves matters.
'''
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.expanduser('~/Mimosa')
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'window_analysis'))
from Mimosa import get_cts  # noqa: E402
from predict_site40 import load_model, read_fasta, read_pairs  # noqa: E402
from utils import decision_for_whole, reverse_seq  # noqa: E402
from window_analysis import features  # noqa: E402

PER_GROUP = 150
DATASETS = {
    'miraw': (f'{ROOT}/experiments/cross_eval/pairs_B_miraw_unique.tsv',
              f'{ROOT}/experiments/cross_eval/miraw_sequences.fa'),
    'ours': (f'{ROOT}/data/custom/pairs_test_unseen_pair.tsv', f'{ROOT}/data/custom/sequences.fa'),
}


def run(model, seqs, rows, seed, tag):
    rng = random.Random(seed)
    pos, neg = [r for r in rows if r['label'] == 1], [r for r in rows if r['label'] == 0]
    pos_sample, neg_sample = rng.sample(pos, PER_GROUP), rng.sample(neg, PER_GROUP)
    groups = {
        'negative target + its own miRNA':       [(r['target_id'], r['mirna_id']) for r in neg_sample],
        'negative target + positive-pair miRNA': [(r['target_id'], rng.choice(pos)['mirna_id']) for r in neg_sample],
        'positive target + its own miRNA':       [(r['target_id'], r['mirna_id']) for r in pos_sample],
        'positive target + negative-pair miRNA': [(r['target_id'], rng.choice(neg)['mirna_id']) for r in pos_sample],
    }
    t0 = time.time()
    results = {}
    for gname, pairs in groups.items():
        calls = []
        for target, mirna_id in pairs:
            mirna = seqs[mirna_id].upper().replace('T', 'U')
            kmers = get_cts(reverse_seq(seqs[target].upper().replace('T', 'U')), 5)
            f1, f2, f3, f4 = features(kmers, mirna)
            with torch.no_grad():
                calls.append(decision_for_whole(model(f1, f2, f3, f4).numpy().tolist()))
        results[gname] = float(np.mean(calls))
        print(f'[{tag}] [{time.strftime("%H:%M:%S")}] {gname}: {results[gname]:.3f} '
              f'({(time.time()-t0)/60:.1f} min)', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--name', default='mil')
    ap.add_argument('--dataset', choices=list(DATASETS), default='miraw')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model = load_model(args.model)
    model.eval()
    pairs_path, fasta_path = DATASETS[args.dataset]
    seqs = read_fasta(fasta_path)
    rows = [r for r in read_pairs(pairs_path) if r['status'] == 'ok']
    results = run(model, seqs, rows, args.seed, f'{args.name}/{args.dataset}')

    print(f'\nshare of pairs called positive ({PER_GROUP} targets per group), model={args.name}, data={args.dataset}')
    for gname, v in results.items():
        print(f'{gname:40s}{v:>10.3f}')


if __name__ == '__main__':
    main()

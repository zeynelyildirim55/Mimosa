'''Does the model's call follow the miRNA or the target? (miRAW test sets)

Targets are kept and miRNAs are swapped between positive and negative pairs. If the call
tracks the miRNA, negative targets paired with miRNAs taken from positive pairs will be
called positive about as often as real positives, and vice versa. The true label of a
swapped pair is unknown; only the direction the call moves matters.
'''
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
from window_analysis import MODELS, features  # noqa: E402

PER_GROUP = 150


def main():
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    models = {name: load_model(path) for name, path in MODELS.items()}
    seqs = read_fasta(f'{ROOT}/experiments/cross_eval/miraw_sequences.fa')
    rows = [r for r in read_pairs(f'{ROOT}/experiments/cross_eval/pairs_B_miraw_unique.tsv') if r['status'] == 'ok']
    rng = random.Random(1)
    pos, neg = [r for r in rows if r['label'] == 1], [r for r in rows if r['label'] == 0]
    pos_sample, neg_sample = rng.sample(pos, PER_GROUP), rng.sample(neg, PER_GROUP)
    groups = {
        'negative target + its own miRNA':       [(r['target_id'], r['mirna_id']) for r in neg_sample],
        'negative target + positive-pair miRNA': [(r['target_id'], rng.choice(pos)['mirna_id']) for r in neg_sample],
        'positive target + its own miRNA':       [(r['target_id'], r['mirna_id']) for r in pos_sample],
        'positive target + negative-pair miRNA': [(r['target_id'], rng.choice(neg)['mirna_id']) for r in pos_sample],
    }
    results = {}
    t0 = time.time()
    for gname, pairs in groups.items():
        calls = {m: [] for m in models}
        for target, mirna_id in pairs:
            mirna = seqs[mirna_id].upper().replace('T', 'U')
            kmers = get_cts(reverse_seq(seqs[target].upper().replace('T', 'U')), 5)
            fea = features(kmers, mirna)
            for mname, model in models.items():
                with torch.no_grad():
                    calls[mname].append(decision_for_whole(model(*fea).numpy().tolist()))
        results[gname] = {m: float(np.mean(c)) for m, c in calls.items()}
        print(f'[{time.strftime("%H:%M:%S")}] {gname}: done ({(time.time()-t0)/60:.1f} min)', flush=True)

    print(f'\nshare of pairs called positive ({PER_GROUP} targets per group)')
    print(f'{"":40s}{"authors":>10s}{"ours":>10s}')
    for gname, r in results.items():
        print(f'{gname:40s}{r["authors"]:>10.3f}{r["ours"]:>10.3f}')


if __name__ == '__main__':
    main()

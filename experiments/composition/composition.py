'''Can the label be predicted from the target transcript alone, without looking at the miRNA?

If a miRNA-free model separates positives from negatives well, the benchmark carries a
transcript-level signal that a pair model can exploit instead of recognising binding sites.
'''
import itertools
import os
import random
import statistics
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.expanduser('~/Mimosa')
sys.path.insert(0, os.path.join(ROOT, 'training'))
from predict_site40 import read_fasta, read_pairs  # noqa: E402

DIMERS = [''.join(p) for p in itertools.product('ACGU', repeat=2)]


def features(seq):
    n = len(seq)
    mono = [seq.count(b) / n for b in 'ACGU']
    di = [sum(1 for i in range(n - 1) if seq[i:i + 2] == d) / (n - 1) for d in DIMERS]
    return [np.log(n), mono[1] + mono[2]] + mono + di


def analyse(name, pairs_path, fasta_path, limit=None):
    seqs = read_fasta(fasta_path)
    rows = [r for r in read_pairs(pairs_path) if r['status'] == 'ok']
    if limit:
        random.Random(0).shuffle(rows)
        rows = rows[:limit]
    targets_by_label = {0: set(), 1: set()}
    for r in rows:
        targets_by_label[r['label']].add(r['target_id'])
    shared = targets_by_label[0] & targets_by_label[1]
    # one row per (target, label); targets seen with both labels are dropped so the task is well defined
    items = [(t, l) for l in (0, 1) for t in targets_by_label[l] if t not in shared]
    X = np.array([features(seqs[t].upper().replace('T', 'U')) for t, _ in items])
    y = np.array([l for _, l in items])
    print(f'\n== {name}: {len(rows)} pairs, {len(targets_by_label[1])} positive / {len(targets_by_label[0])} negative '
          f'targets, {len(shared)} targets appear with both labels (dropped)')
    for label in (1, 0):
        L = [len(seqs[t]) for t, l in items if l == label]
        gc = [X[i][1] for i, (_, l) in enumerate(items) if l == label]
        print(f'   label {label}: n={len(L)}  length median={statistics.median(L):.0f}  GC mean={statistics.mean(gc):.3f}')
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for fname, cols in (('length only', [0]), ('length + GC', [0, 1]), ('length + mono + dinucleotides', slice(None))):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        auc = cross_val_score(clf, X[:, cols], y, cv=cv, scoring='roc_auc')
        acc = cross_val_score(clf, X[:, cols], y, cv=cv, scoring='balanced_accuracy')
        print(f'   {fname:32s} ROC-AUC={auc.mean():.3f}  balanced accuracy={acc.mean():.3f}')


analyse('miRAW test sets (paper)', f'{ROOT}/experiments/cross_eval/pairs_B_miraw_unique.tsv',
        f'{ROOT}/experiments/cross_eval/miraw_sequences.fa')
analyse('our test_unseen_pair (random 20,000 pairs)', f'{ROOT}/data/custom/pairs_test_unseen_pair.tsv',
        f'{ROOT}/data/custom/sequences.fa', limit=20000)

'''Does the miRNA alone predict the label on the miRAW test sets?

For each test pair, the only feature is how often that miRNA was a positive in the miRAW
training data the authors' model was trained on. A high ROC-AUC would mean a pair model can
score well by recognising the miRNA rather than the binding site.
'''
import collections
import os

from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser('~/Mimosa')
train = collections.defaultdict(lambda: [0, 0])  # miRNA sequence -> [negatives, positives]
for line in open(f'{ROOT}/data/miRAW_Train_Validation.txt').read().replace('\r', '').splitlines()[1:]:
    c = line.split('\t')
    train[c[1].upper().replace('T', 'U')][int(c[4])] += 1

mirna_of, rows = {}, []
for line in open(f'{ROOT}/experiments/cross_eval/miraw_sequences.fa').read().splitlines():
    if line.startswith('>'):
        name = line[1:]
    else:
        mirna_of[name] = line
for line in open(f'{ROOT}/experiments/cross_eval/pairs_B_miraw_unique.tsv').read().splitlines()[1:]:
    c = line.split('\t')
    rows.append((mirna_of[c[1]], int(c[3])))

pos_mirnas = {m for m, l in rows if l == 1}
neg_mirnas = {m for m, l in rows if l == 0}
print(f'test pairs: {sum(l for _, l in rows)} positive / {sum(1 - l for _, l in rows)} negative')
print(f'distinct miRNAs: {len(pos_mirnas)} in positives, {len(neg_mirnas)} in negatives, '
      f'{len(pos_mirnas & neg_mirnas)} in both')
for label, group in ((1, pos_mirnas), (0, neg_mirnas)):
    seen = [m for m in group if sum(train[m]) > 0]
    rate = [train[m][1] / sum(train[m]) for m in seen]
    n_train = [sum(train[m]) for m in seen]
    print(f'  label {label} miRNAs: {len(seen)}/{len(group)} appear in training; '
          f'mean training positive rate={sum(rate)/max(len(rate),1):.3f}; '
          f'mean training examples={sum(n_train)/max(len(n_train),1):.0f}')

score = [train[m][1] / sum(train[m]) if sum(train[m]) else 0.5 for m, _ in rows]
count = [sum(train[m]) for m, _ in rows]
labels = [l for _, l in rows]
print(f'ROC-AUC, training positive rate of the miRNA alone: {roc_auc_score(labels, score):.3f}')
print(f'ROC-AUC, number of training examples of the miRNA:  {roc_auc_score(labels, count):.3f}')

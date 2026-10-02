'''Side-by-side metrics for the two cross-evaluations (run after both merge jobs finish).'''
import glob
import os
import re
import statistics

from sklearn.metrics import confusion_matrix

ROOT = os.path.expanduser('~/Mimosa')
NAMES = ('accuracy', 'ppv', 'recall', 'specificity', 'f1', 'npv')


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    ppv = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': (tp + tn) / len(y), 'ppv': ppv, 'recall': rec,
            'specificity': tn / (tn + fp) if tn + fp else 0.0,
            'f1': 2 * ppv * rec / (ppv + rec) if ppv + rec else 0.0,
            'npv': tn / (tn + fn) if tn + fn else 0.0}


def predictions(pattern):
    out = {}
    for path in glob.glob(pattern):
        for line in open(path).read().splitlines()[1:]:
            r, y, p = line.split('\t')
            out[int(r)] = (int(y), int(p))
    return out


def row(name, m, n=None):
    return f'{name:44s}' + ''.join(f'{m[k]:>9.3f}' for k in NAMES) + (f'   n={n}' if n else '')


print(f'{"":44s}' + ''.join(f'{k[:8]:>9s}' for k in NAMES))

# ---- A: same 4,000 pairs of test_unseen_pair, two models ----
authors = predictions(f'{ROOT}/results/experiments/A_authors_model_on_test_unseen_pair/pred.*.tsv')
ours_all = predictions(f'{ROOT}/results/site40_long/test_unseen_pair/pred.*.tsv')
common = sorted(set(authors) & set(ours_all))
print('A) test_unseen_pair, 4,000-pair sample')
print(row('   authors\' model (model_mimosa.pth)', metrics([authors[r][0] for r in common], [authors[r][1] for r in common]), len(common)))
print(row('   our model (120 epochs)', metrics([ours_all[r][0] for r in common], [ours_all[r][1] for r in common]), len(common)))

# ---- B: the paper's 10 miRAW test sets, averaged over sets ----
ours = predictions(f'{ROOT}/results/experiments/B_our_model_on_miraw/pred.*.tsv')
sets = {}
for line in open(f'{ROOT}/experiments/cross_eval/miraw_membership.tsv').read().splitlines()[1:]:
    s, _, r = line.split('\t')
    sets.setdefault(int(s), []).append(int(r))
per_set = [metrics([ours[r][0] for r in rows], [ours[r][1] for r in rows]) for s, rows in sorted(sets.items())
           if all(r in ours for r in rows)]
authors_sets = []
for path in sorted(glob.glob(f'{ROOT}/logs/paper_reproduction/Mimosa_Log_11329_Test*.txt')):
    text = open(path).read()
    get = lambda label: float(re.search(label + r':\s+([0-9.]+)', text).group(1))
    authors_sets.append({'accuracy': get('Accuracy'), 'ppv': get(r'PPV \(Prec\)'), 'recall': get('Recall'),
                         'specificity': get('Specificity'), 'f1': get('F1 Score'), 'npv': get('NPV')})
mean = lambda ms: {k: statistics.mean(m[k] for m in ms) for k in NAMES}
print('B) the paper\'s 10 miRAW test sets (mean over sets)')
print(row('   authors\' model (paper reproduction)', mean(authors_sets), f'{len(authors_sets)} sets'))
if per_set:
    print(row('   our model (120 epochs)', mean(per_set), f'{len(per_set)} sets'))
else:
    print('   our model: predictions incomplete')

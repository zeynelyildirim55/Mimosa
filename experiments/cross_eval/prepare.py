'''Input files for the two cross-evaluations (both use the original sliding-window test):
  A: the authors' model on 4,000 random pairs of our test_unseen_pair (2,000 per label),
     keeping the original row_index so our model's existing predictions can be compared
  B: our model on the paper's 10 miRAW test sets; each distinct pair is predicted once
     (the 10 sets share their 548 negatives) and mapped back to its sets afterwards
'''
import os
import random

ROOT = os.path.expanduser('~/Mimosa')
OUT = os.path.join(ROOT, 'experiments', 'cross_eval')

# ---- A ----
rows = open(os.path.join(ROOT, 'data', 'custom', 'pairs_test_unseen_pair.tsv')).read().splitlines()
header, body = rows[0], [r.split('\t') for r in rows[1:]]
col = {c: i for i, c in enumerate(header.split('\t'))}
ok = [r for r in body if r[col['mimosa_status']] == 'ok']
rng = random.Random(0)
sample = []
for label in ('1', '0'):
    group = [r for r in ok if r[col['label']] == label]
    sample += rng.sample(group, 2000)
sample.sort(key=lambda r: int(r[col['row_index']]))
with open(os.path.join(OUT, 'pairs_A_test_unseen_pair_sample4000.tsv'), 'w') as f:
    f.write(header + '\n')
    f.writelines('\t'.join(r) + '\n' for r in sample)
print(f'A: {len(sample)} pairs sampled from {len(ok)} processable rows')

# ---- B ----
mirna_ids, target_ids, pairs, membership = {}, {}, {}, []
for s in range(10):
    lines = open(os.path.join(ROOT, 'data', f'miRAW_Test{s}.txt')).read().replace('\r', '').splitlines()[1:]
    for line_no, line in enumerate(lines, 1):
        c = line.split('\t')
        mi, tg, label = c[1].upper().replace('T', 'U'), c[3].upper().replace('T', 'U'), c[4]
        mid = mirna_ids.setdefault(mi, f'mi{len(mirna_ids) + 1}')
        tid = target_ids.setdefault(tg, f'tg{len(target_ids) + 1}')
        key = (mid, tid, label)
        if key not in pairs:
            pairs[key] = len(pairs) + 1
        membership.append((s, line_no, pairs[key]))
with open(os.path.join(OUT, 'miraw_sequences.fa'), 'w') as f:
    for seq, i in list(mirna_ids.items()) + list(target_ids.items()):
        f.write(f'>{i}\n{seq}\n')
seqs = {i: s for s, i in list(mirna_ids.items()) + list(target_ids.items())}
with open(os.path.join(OUT, 'pairs_B_miraw_unique.tsv'), 'w') as f:
    f.write('row_index\tmirna_id\ttarget_id\tlabel\ttarget_len\tmimosa_status\n')
    for (mid, tid, label), r in pairs.items():
        good = set(seqs[mid] + seqs[tid]) <= set('ACGU') and 10 <= len(seqs[mid]) <= 30
        f.write(f'{r}\t{mid}\t{tid}\t{label}\t{len(seqs[tid])}\t{"ok" if good else "not_processable"}\n')
with open(os.path.join(OUT, 'miraw_membership.tsv'), 'w') as f:
    f.write('set\tline\trow_index\n')
    f.writelines(f'{s}\t{l}\t{r}\n' for s, l, r in membership)
n_ok = sum(1 for l in open(os.path.join(OUT, 'pairs_B_miraw_unique.tsv')).read().splitlines()[1:] if l.endswith('\tok'))
print(f'B: {len(membership)} rows in 10 sets -> {len(pairs)} distinct pairs ({n_ok} processable), '
      f'{len(mirna_ids)} miRNAs, {len(target_ids)} targets')

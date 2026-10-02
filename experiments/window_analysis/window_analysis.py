'''How often does each model score an ordinary window above 0.5?

Mimosa calls a pair positive when at least one 40-nt window scores above 0.5, so the
per-window false-positive rate on transcripts without a real site decides gene-level
specificity. For the same pairs, both models score every window; the features are built
exactly as Mimosa.kmers_predict builds them, and the resulting gene-level call is checked
against kmers_predict itself.
'''
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.expanduser('~/Mimosa')
sys.path.insert(0, os.path.join(ROOT, 'training'))
import Mimosa  # noqa: E402
from Mimosa import get_cts, kmers_predict  # noqa: E402
from predict_site40 import load_model, read_fasta, read_pairs  # noqa: E402
from utils import (decision_for_whole, get_embedding, get_interaction_map_for_test,  # noqa: E402
                   get_interaction_map_for_test_short, reverse_seq)

OUT = os.path.join(ROOT, 'experiments', 'window_analysis')
MODELS = {'authors': os.path.join(ROOT, 'training', 'model_mimosa.pth'),
          'ours': os.path.join(ROOT, 'runs', 'site40_long', 'model_concate_112.pth')}
DATA = {'miraw': (os.path.join(ROOT, 'experiments', 'cross_eval', 'pairs_B_miraw_unique.tsv'),
                  os.path.join(ROOT, 'experiments', 'cross_eval', 'miraw_sequences.fa')),
        'ours': (os.path.join(ROOT, 'data', 'custom', 'pairs_test_unseen_pair.tsv'),
                 os.path.join(ROOT, 'data', 'custom', 'sequences.fa'))}
PER_LABEL = 100


def features(kmers, mirna):
    '''the four inputs exactly as Mimosa.kmers_predict builds them'''
    mirna = mirna + 'X' * (30 - len(mirna))
    fea1, fea2, fea3, fea4 = [], [], [], []
    for k in kmers:
        fea1.append(get_embedding(k))
        fea2.append(get_embedding(mirna))
        m, mi = (get_interaction_map_for_test_short if 'X' in k else get_interaction_map_for_test)(mirna, k)
        fea3.append(m)
        fea4.append(mi)
    return (torch.tensor(fea1, dtype=torch.long), torch.tensor(fea2), torch.tensor(fea3), torch.tensor(fea4))


def main():
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    models = {name: load_model(path) for name, path in MODELS.items()}
    rng = random.Random(0)
    records, checks = [], [0, 0]
    for dname, (pairs_path, fasta_path) in DATA.items():
        seqs = read_fasta(fasta_path)
        rows = [r for r in read_pairs(pairs_path) if r['status'] == 'ok']
        sample = []
        for label in (0, 1):
            sample += rng.sample([r for r in rows if r['label'] == label], PER_LABEL)
        t0 = time.time()
        for n, r in enumerate(sample, 1):
            mirna = seqs[r['mirna_id']].upper().replace('T', 'U')
            kmers = get_cts(reverse_seq(seqs[r['target_id']].upper().replace('T', 'U')), 5)
            fea = features(kmers, mirna)
            rec = {'data': dname, 'row_index': r['row_index'], 'label': r['label'], 'windows': len(kmers)}
            for mname, model in models.items():
                with torch.no_grad():
                    probs = model(*fea).numpy()
                rec[f'{mname}_pos_windows'] = int((probs[:, 1] > 0.5).sum())
                rec[f'{mname}_max_prob'] = float(probs[:, 1].max())
                rec[f'{mname}_call'] = decision_for_whole(probs.tolist())
                if n <= 10:  # consistency with the authors' kmers_predict
                    with torch.no_grad():
                        checks[0] += int(kmers_predict(kmers, mirna, model) == rec[f'{mname}_call'])
                    checks[1] += 1
            records.append(rec)
            if n % 50 == 0:
                print(f'[{time.strftime("%H:%M:%S")}] {dname}: {n}/{len(sample)} pairs ({(time.time()-t0)/n:.1f}s/pair)', flush=True)

    with open(os.path.join(OUT, 'windows.json'), 'w') as f:
        json.dump(records, f)
    print(f'\nconsistency with kmers_predict: {checks[0]}/{checks[1]} identical calls\n')
    print(f'{"data":6s} {"label":>5s} {"model":8s} {"windows":>8s} {"win>0.5":>9s} {"pos win/tx":>11s} '
          f'{"median":>7s} {"called+":>8s} {"max p":>6s}')
    for dname in DATA:
        for label in (0, 1):
            group = [r for r in records if r['data'] == dname and r['label'] == label]
            for mname in MODELS:
                win = sum(r['windows'] for r in group)
                pos = [r[f'{mname}_pos_windows'] for r in group]
                print(f'{dname:6s} {label:>5d} {mname:8s} {win/len(group):8.0f} {sum(pos)/win:9.4f} '
                      f'{np.mean(pos):11.1f} {np.median(pos):7.0f} {np.mean([r[f"{mname}_call"] for r in group]):8.3f} '
                      f'{np.mean([r[f"{mname}_max_prob"] for r in group]):6.3f}')


if __name__ == '__main__':
    main()

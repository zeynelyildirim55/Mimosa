'''Check a pairing cache written by precompute_pairing.py against the pairs file.'''
import argparse
import os
import random
import time

import numpy as np

from full_data import PairDataset
from full_features import decode, pairing_vectors, pairing_vectors_sw


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Verify a pairing cache')
    parser.add_argument('--pairs', default=os.path.join(base, 'data', 'custom', 'pairs_train_val.tsv'))
    parser.add_argument('--index', default=os.path.join(base, 'data', 'custom', 'seqindex'))
    parser.add_argument('--cache', default=os.path.join(base, 'data', 'custom', 'pairing_train_val'))
    parser.add_argument('--mode', default='sw', choices=['sw', 'scan'])
    parser.add_argument('--min-pairs', type=float, default=8)
    parser.add_argument('--samples', type=int, default=200, help='rows recomputed and compared')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    dataset = PairDataset(args.pairs, prefix=args.index, pairing=args.mode, min_pairs=args.min_pairs,
                          cache=args.cache)
    cache = dataset.cache
    print(f'rows in pairs file : {len(dataset)}', flush=True)
    print(f'rows in cache      : {len(cache.ordinals)}', flush=True)
    missing = set(r[5] for r in dataset.rows) - set(cache.ordinals.tolist())
    print(f'rows missing from cache: {len(missing)}', flush=True)
    duplicates = len(cache.ordinals) - len(set(cache.ordinals.tolist()))
    print(f'duplicate ordinals : {duplicates}', flush=True)

    bad_len = 0
    for i in range(len(dataset)):
        ordinal, target_len = dataset.rows[i][5], dataset.rows[i][3]
        j = np.searchsorted(cache.ordinals, ordinal)
        if j < len(cache.ordinals) and cache.ordinals[j] == ordinal:
            if int(cache.lengths[j]) != min(target_len, dataset.max_len):
                bad_len += 1
    print(f'length mismatches  : {bad_len}', flush=True)

    random.seed(args.seed)
    picks = random.sample(range(len(dataset)), min(args.samples, len(dataset)))
    same = 0
    for i in picks:
        item = dataset[i]
        mirna, mrna = item['mirna'], item['mrna']
        if args.mode == 'sw':
            live_m, live_mi = pairing_vectors_sw(decode(mirna), decode(mrna))
        else:
            live_m, live_mi = pairing_vectors(mirna, mrna, args.min_pairs)
        same += int(np.array_equal(live_m, item['pair_m']) and np.array_equal(live_mi, item['pair_mi']))
    print(f'recomputed rows identical: {same}/{len(picks)}', flush=True)
    ok = not missing and not duplicates and not bad_len and same == len(picks)
    print(f'VERDICT: {"cache is complete and correct" if ok else "PROBLEM FOUND"} ({time.time()-t0:.0f}s)', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

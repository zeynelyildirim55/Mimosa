'''Precompute the authors' window features for every pair of a pairs file (one shard per task).

For each pair: the target is reversed and cut with get_cts (step 5); every window gets
get_interaction_map_for_test (or _short for padded windows), exactly as kmers_predict does.
Shards are resumable: a restarted task keeps the pairs it already wrote.
'''
import argparse
import os
import time

import numpy as np

from Mimosa import get_cts
from mil_common import MIRNA_LEN, PACKED, pack
from predict_site40 import read_fasta
from utils import get_interaction_map_for_test, get_interaction_map_for_test_short, reverse_seq


def rows_of(path, split, ok_only):
    with open(path) as f:
        col = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        for ordinal, line in enumerate(f):
            c = line.rstrip('\n').split('\t')
            if split and c[col['split']] != split:
                continue
            if ok_only and 'mimosa_status' in col and c[col['mimosa_status']] != 'ok':
                continue
            yield ordinal, c[col['mirna_id']], c[col['target_id']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--sequences', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--split', default=None, help='train/val column filter')
    ap.add_argument('--every', type=int, default=1, help='keep every n-th selected row (validation subsample)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--stepsize', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    seqs = read_fasta(args.sequences)
    todo = [r for i, r in enumerate(rows_of(args.pairs, args.split, ok_only=True)) if i % args.every == 0]
    todo = [r for r in todo if r[0] % args.nshards == args.shard]
    if args.limit:
        todo = todo[:args.limit]

    bin_path, idx_path = f'{args.out_prefix}.{args.shard:03d}.bin', f'{args.out_prefix}.{args.shard:03d}.idx'
    done, offset = set(), 0
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            for line in f:
                if line.count('\t') == 2 and line.endswith('\n'):
                    o, off, n = map(int, line.split('\t'))
                    done.add(o); offset = max(offset, off + n)
        with open(bin_path, 'ab') as f:   # drop bytes of a pair that was cut off mid-write
            f.truncate(offset * PACKED)
        with open(idx_path) as f:
            complete = [l for l in f if l.count('\t') == 2 and l.endswith('\n')]
        with open(idx_path, 'w') as f:
            f.writelines(complete)
    todo = [r for r in todo if r[0] not in done]
    print(f'shard {args.shard}/{args.nshards}: {len(todo)} pairs to do, {len(done)} already done', flush=True)

    start = time.time()
    windows = 0
    with open(bin_path, 'ab') as fb, open(idx_path, 'a', buffering=1) as fi:
        for k, (ordinal, mid, tid) in enumerate(todo, 1):
            mirna = seqs[mid].upper().replace('T', 'U')
            mirna = mirna + 'X' * (MIRNA_LEN - len(mirna))          # as in kmers_predict
            kmers = get_cts(reverse_seq(seqs[tid].upper().replace('T', 'U')), args.stepsize)
            maps = np.zeros((len(kmers), 50), dtype=np.uint8)
            for j, kmer in enumerate(kmers):
                fn = get_interaction_map_for_test_short if 'X' in kmer else get_interaction_map_for_test
                m, mi = fn(mirna, kmer)
                maps[j, :40] = m
                maps[j, 40:] = mi[:10]
            fb.write(pack(maps))
            fb.flush()
            fi.write(f'{ordinal}\t{offset}\t{len(kmers)}\n')
            offset += len(kmers)
            windows += len(kmers)
            if k % 500 == 0:
                rate = k / (time.time() - start)
                print(f'  {k}/{len(todo)} pairs, {windows} windows, {rate:.2f} pairs/s, '
                      f'eta {(len(todo)-k)/rate/60:.0f} min', flush=True)
    print(f'done: {windows} windows in {(time.time()-start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()

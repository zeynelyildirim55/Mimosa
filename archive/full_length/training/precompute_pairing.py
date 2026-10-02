'''Cache the base-pairing vectors of one pairs file on disk.

Smith-Waterman costs ~120 ms per pair, which would dominate every epoch, so the vectors are
computed once here (sharded, one Slurm array task per shard) and read back by PairDataset.

Each shard writes <out-prefix>.<shard>.u8 (mRNA vector then the 30-byte miRNA vector, per row)
and <out-prefix>.<shard>.idx (row ordinal, byte offset, mRNA vector length).
'''
import argparse
import os
import time

from full_data import PairDataset
from full_features import MIRNA_LEN


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Precompute base-pairing vectors')
    parser.add_argument('--pairs', default=os.path.join(base, 'data', 'custom', 'pairs_train_val.tsv'))
    parser.add_argument('--index', default=os.path.join(base, 'data', 'custom', 'seqindex'))
    parser.add_argument('--out-prefix', default=os.path.join(base, 'data', 'custom', 'pairing_train_val'))
    parser.add_argument('--mode', default='sw', choices=['sw', 'scan'])
    parser.add_argument('--min-pairs', type=float, default=8, help='scan mode only')
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--split', default=None, help='train/val column filter, default: all rows')
    parser.add_argument('--all-statuses', action='store_true',
                        help='evaluation files: also process rows Mimosa cannot handle (default: only ok)')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    statuses = None if args.all_statuses else ('ok',)
    dataset = PairDataset(args.pairs, prefix=args.index, split=args.split, statuses=statuses,
                          max_len=args.max_len, min_pairs=args.min_pairs, pairing=args.mode,
                          limit=args.limit)
    todo = [i for i in range(len(dataset)) if dataset.rows[i][5] % args.nshards == args.shard]
    print(f'shard {args.shard}/{args.nshards}: {len(todo)} of {len(dataset)} rows', flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    data_path = f'{args.out_prefix}.{args.shard:03d}.u8'
    index_path = f'{args.out_prefix}.{args.shard:03d}.idx'
    start = time.time()
    offset = 0
    with open(data_path, 'wb') as data, open(index_path, 'w') as index:
        for n, i in enumerate(todo, 1):
            item = dataset[i]
            pair_m, pair_mi = item['pair_m'], item['pair_mi']
            data.write(pair_m.tobytes())
            data.write(pair_mi.tobytes())
            index.write(f'{dataset.rows[i][5]}\t{offset}\t{len(pair_m)}\n')
            offset += len(pair_m) + MIRNA_LEN
            if n % 2000 == 0:
                rate = n / (time.time() - start)
                print(f'  {n}/{len(todo)} rows, {rate:.1f} rows/s, eta {(len(todo)-n)/rate/60:.0f} min', flush=True)
    print(f'wrote {data_path} ({offset/2**20:.0f} MiB) in {(time.time()-start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()

'''Dataset for full-length training, reading the files written by prepare_dataset.py.

Sequences are stored once as a memmapped array of codes so that DataLoader workers share
them instead of each holding a copy of sequences.fa.
'''
import glob
import os
import sys

import numpy as np
import torch

from full_features import MIRNA_LEN, PAD, decode, encode, pairing_vectors, pairing_vectors_sw


def build_index(fasta_path, prefix):
    '''write <prefix>.u8 (all sequences encoded, concatenated) and <prefix>.idx (id, offset, length)'''
    offset = 0
    with open(fasta_path) as fasta, open(prefix + '.u8', 'wb') as data, open(prefix + '.idx', 'w') as index:
        name, parts = None, []

        def flush():
            nonlocal offset
            if name is None:
                return
            codes = encode(''.join(parts))
            data.write(codes.tobytes())
            index.write(f'{name}\t{offset}\t{len(codes)}\n')
            offset += len(codes)

        for line in fasta:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                name, parts = line[1:], []
            else:
                parts.append(line)
        flush()
    return offset


class PairingCache(torch.utils.data.Dataset if False else object):
    """Base-pairing vectors written by precompute_pairing.py, one entry per row ordinal."""

    def __init__(self, prefix):
        self.shards, ordinals, shard_ids, offsets, lengths = [], [], [], [], []
        paths = sorted(glob.glob(prefix + '.*.idx'))
        if not paths:
            raise FileNotFoundError(f'no pairing cache shards at {prefix}.*.idx')
        for shard, idx_path in enumerate(paths):
            self.shards.append(np.memmap(idx_path[:-4] + '.u8', dtype=np.uint8, mode='r'))
            with open(idx_path) as f:
                for line in f:
                    ordinal, offset, length = line.split('\t')
                    ordinals.append(int(ordinal))
                    shard_ids.append(shard)
                    offsets.append(int(offset))
                    lengths.append(int(length))
        order = np.argsort(np.asarray(ordinals, dtype=np.int64))
        self.ordinals = np.asarray(ordinals, dtype=np.int64)[order]
        self.shard_ids = np.asarray(shard_ids, dtype=np.int16)[order]
        self.offsets = np.asarray(offsets, dtype=np.int64)[order]
        self.lengths = np.asarray(lengths, dtype=np.int32)[order]

    def __contains__(self, ordinal):
        i = np.searchsorted(self.ordinals, ordinal)
        return i < len(self.ordinals) and self.ordinals[i] == ordinal

    def get(self, ordinal):
        i = np.searchsorted(self.ordinals, ordinal)
        if i >= len(self.ordinals) or self.ordinals[i] != ordinal:
            raise KeyError(f'row {ordinal} missing from the pairing cache')
        data = self.shards[self.shard_ids[i]]
        start, length = int(self.offsets[i]), int(self.lengths[i])
        return data[start:start + length], data[start + length:start + length + MIRNA_LEN]


class PairDataset(torch.utils.data.Dataset):
    '''rows of a pairs_*.tsv file; __getitem__ builds the four Mimosa inputs'''

    def __init__(self, pairs_tsv, prefix, split=None, statuses=('ok',), max_len=4096,
                 min_pairs=8, limit=0, pairing='sw', cache=None):
        self.codes = np.memmap(prefix + '.u8', dtype=np.uint8, mode='r')
        self.offsets = {}
        with open(prefix + '.idx') as index:
            for line in index:
                name, offset, length = line.rstrip('\n').split('\t')
                self.offsets[sys.intern(name)] = (int(offset), int(length))
        self.max_len = max_len
        self.min_pairs = min_pairs
        self.pairing = pairing
        self.cache = PairingCache(cache) if isinstance(cache, str) else cache
        self.rows = []
        with open(pairs_tsv) as f:
            columns = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
            for ordinal, line in enumerate(f):
                c = line.rstrip('\n').split('\t')
                if split is not None and c[columns['split']] != split:
                    continue
                if 'mimosa_status' in columns and statuses is not None and c[columns['mimosa_status']] not in statuses:
                    continue
                row_index = int(c[columns['row_index']]) if 'row_index' in columns else len(self.rows) + 1
                self.rows.append((sys.intern(c[columns['mirna_id']]), sys.intern(c[columns['target_id']]),
                                  int(c[columns['label']]), int(c[columns['target_len']]), row_index, ordinal))
                if limit and len(self.rows) >= limit:
                    break
        self.lengths = np.array([min(r[3], max_len) for r in self.rows], dtype=np.int32)

    def __len__(self):
        return len(self.rows)

    def sequence(self, name):
        offset, length = self.offsets[name]
        return self.codes[offset:offset + length]

    def pairing_for(self, mirna, mrna, ordinal):
        if self.cache is not None:
            pair_m, pair_mi = self.cache.get(ordinal)
            return np.ascontiguousarray(pair_m[:len(mrna)]), np.ascontiguousarray(pair_mi)
        if self.pairing == 'sw':
            return pairing_vectors_sw(decode(mirna), decode(mrna))
        return pairing_vectors(mirna, mrna, self.min_pairs)

    def __getitem__(self, i):
        mirna_id, target_id, label, _, row_index, ordinal = self.rows[i]
        mirna = np.full(MIRNA_LEN, PAD, dtype=np.uint8)
        raw = self.sequence(mirna_id)[:MIRNA_LEN]
        mirna[:len(raw)] = raw
        # 3'->5', as reverse_seq does in the original code
        mrna = np.ascontiguousarray(self.sequence(target_id)[::-1][:self.max_len])
        pair_m, pair_mi = self.pairing_for(mirna, mrna, ordinal)
        return {'mrna': mrna, 'mirna': mirna, 'pair_m': pair_m, 'pair_mi': pair_mi,
                'label': label, 'row_index': row_index}


def collate(batch):
    '''pad the mRNA side to the longest sequence in the batch and build the padding mask'''
    length = max(len(b['mrna']) for b in batch)
    n = len(batch)
    mrna = np.full((n, length), PAD, dtype=np.uint8)
    pair_m = np.zeros((n, length), dtype=np.uint8)
    mask = np.ones((n, length), dtype=bool)
    mirna = np.empty((n, MIRNA_LEN), dtype=np.uint8)
    pair_mi = np.empty((n, MIRNA_LEN), dtype=np.uint8)
    for i, b in enumerate(batch):
        L = len(b['mrna'])
        mrna[i, :L] = b['mrna']
        pair_m[i, :L] = b['pair_m']
        mask[i, :L] = False
        mirna[i] = b['mirna']
        pair_mi[i] = b['pair_mi']
    return {
        'fea1': torch.from_numpy(mrna).long(),
        'fea2': torch.from_numpy(mirna).long(),
        'fea3': torch.from_numpy(pair_m).long(),
        'fea4': torch.from_numpy(pair_mi).long(),
        'pad_mask': torch.from_numpy(mask),
        'label': torch.tensor([b['label'] for b in batch], dtype=torch.long),
        'row_index': torch.tensor([b['row_index'] for b in batch], dtype=torch.long),
    }


class TokenBatchSampler(torch.utils.data.Sampler):
    '''batches of similar length, sized by padded tokens instead of a fixed sample count'''

    def __init__(self, lengths, max_tokens=8192, max_batch=64, shuffle=True, pool=4096, seed=0):
        self.lengths = np.asarray(lengths)
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.pool = pool
        self.seed = seed
        self.epoch = 0
        self._batches = None

    def _build(self):
        order = np.arange(len(self.lengths))
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.pool):
            chunk = order[start:start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind='stable')]
            current, longest = [], 0
            for i in chunk:
                longest_next = max(longest, int(self.lengths[i]))
                if current and ((len(current) + 1) * longest_next > self.max_tokens
                                or len(current) + 1 > self.max_batch):
                    batches.append(current)
                    current, longest = [int(i)], int(self.lengths[i])
                else:
                    current.append(int(i))
                    longest = longest_next
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = None

    def batches_for(self, epoch):
        '''the batch list for one epoch; deterministic given (seed, epoch), so a resumed
        run can skip the batches the previous job already trained on'''
        self.set_epoch(epoch)
        self._batches = self._build()
        return self._batches

    def __iter__(self):
        if self._batches is None:
            self._batches = self._build()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._batches = self._build()
        return len(self._batches)

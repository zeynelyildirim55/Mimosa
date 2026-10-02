'''Shared helpers for the multiple-instance (MIL) experiment.

Window features are the authors' own (get_cts, get_interaction_map_for_test[_short]); this module
only stores them compactly and rebuilds the model inputs. Each window keeps its mRNA pairing map
(40 values) and the first 10 values of its miRNA pairing map (the rest are always 0); values are
0/1/2 and are packed four per byte, 13 bytes per window.
'''
import glob

import numpy as np

CODES = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': 4}       # as in utils.get_embedding
_TABLE = bytes(CODES.get(chr(i), 4) for i in range(256))
STEP = 5
WINDOW = 40
MIRNA_LEN = 30
PACKED = 13                                             # bytes per window (52 two-bit slots)
_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)


def encode(seq):
    return np.frombuffer(seq.encode().translate(_TABLE), dtype=np.uint8)


def window_codes(rev_codes, step=STEP):
    '''the windows get_cts produces, as an (n, 40) array of get_embedding codes'''
    if len(rev_codes) >= WINDOW:
        return np.lib.stride_tricks.sliding_window_view(rev_codes, WINDOW)[::step]
    padded = np.full(WINDOW, CODES['X'], dtype=np.uint8)
    padded[:len(rev_codes)] = rev_codes
    return padded[None, :]


def mirna_codes(mirna):
    '''the miRNA padded to 30 nt with X, as kmers_predict does'''
    return encode(mirna + 'X' * (MIRNA_LEN - len(mirna)))


def pack(maps):
    '''(n, 50) values in 0..2 -> n * 13 bytes'''
    n = len(maps)
    padded = np.zeros((n, PACKED * 4), dtype=np.uint8)
    padded[:, :50] = maps
    return (padded.reshape(n, PACKED, 4) << _SHIFTS).sum(axis=2, dtype=np.uint8).tobytes()


def unpack(buf, n):
    '''n * 13 bytes -> (n, 40) mRNA maps and (n, 30) miRNA maps'''
    b = np.frombuffer(buf, dtype=np.uint8).reshape(n, PACKED, 1)
    vals = ((b >> _SHIFTS) & 3).reshape(n, PACKED * 4)
    mi = np.zeros((n, MIRNA_LEN), dtype=np.uint8)
    mi[:, :10] = vals[:, 40:50]
    return vals[:, :40], mi


class FeatureStore:
    '''read-only access to the shards written by mil_features.py'''

    def __init__(self, prefix):
        self.data, ordinals, shards, offsets, counts = [], [], [], [], []
        paths = sorted(glob.glob(prefix + '.*.idx'))
        if not paths:
            raise FileNotFoundError(f'no feature shards at {prefix}.*.idx')
        for s, path in enumerate(paths):
            self.data.append(np.memmap(path[:-4] + '.bin', dtype=np.uint8, mode='r'))
            for line in open(path):
                if line.count('\t') == 2 and line.endswith('\n'):
                    o, off, n = line.split('\t')
                    ordinals.append(int(o)); shards.append(s); offsets.append(int(off)); counts.append(int(n))
        order = np.argsort(ordinals)
        self.ordinals = np.asarray(ordinals, dtype=np.int64)[order]
        self.shards = np.asarray(shards, dtype=np.int32)[order]
        self.offsets = np.asarray(offsets, dtype=np.int64)[order]
        self.counts = np.asarray(counts, dtype=np.int32)[order]

    def __contains__(self, ordinal):
        i = np.searchsorted(self.ordinals, ordinal)
        return i < len(self.ordinals) and self.ordinals[i] == ordinal

    def get(self, ordinal):
        i = np.searchsorted(self.ordinals, ordinal)
        if i >= len(self.ordinals) or self.ordinals[i] != ordinal:
            raise KeyError(ordinal)
        start, n = int(self.offsets[i]) * PACKED, int(self.counts[i])
        return unpack(self.data[self.shards[i]][start:start + n * PACKED], n)


def get_window(store, ordinal, j):
    '''maps of window j only (13 bytes read instead of the whole pair)'''
    i = np.searchsorted(store.ordinals, ordinal)
    if i >= len(store.ordinals) or store.ordinals[i] != ordinal or j >= store.counts[i]:
        raise KeyError((ordinal, j))
    start = (int(store.offsets[i]) + j) * PACKED
    m, mi = unpack(store.data[store.shards[i]][start:start + PACKED], 1)
    return m[0], mi[0]


def load_pairs(path):
    '''ordinal (0-based data row) -> (mirna_id, target_id, label, status)'''
    out = {}
    with open(path) as f:
        col = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        for ordinal, line in enumerate(f):
            c = line.rstrip('\n').split('\t')
            status = c[col['mimosa_status']] if 'mimosa_status' in col else 'ok'
            out[ordinal] = (c[col['mirna_id']], c[col['target_id']], int(c[col['label']]), status)
    return out


def rev_codes(seqs, target_id):
    '''target reversed to 3'->5' (reverse_seq) and encoded'''
    return encode(seqs[target_id].upper().replace('T', 'U'))[::-1].copy()

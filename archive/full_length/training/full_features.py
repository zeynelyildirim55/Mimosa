'''Base-pairing features for full-length targets.

Two modes, both producing the vectors the original code uses
(0 = unpaired, 1 = Watson-Crick, 2 = wobble):

  'sw'   (default) the authors' Smith_Waterman, aligning the miRNA seed against the whole
         transcript. get_interaction_map_for_test_short in utils.py already does exactly
         this for targets shorter than 40 nt, so it is the closest thing to the original.
         It marks one local alignment per pair and costs ~120 ms per pair, so the vectors
         are cached by precompute_pairing.py instead of being recomputed every epoch.

  'scan' an ungapped scan marking every offset where the seed forms at least `min_pairs`
         pairs. ~2 ms per pair, but the feature no longer matches the original.
'''
import numpy as np

from utils import Smith_Waterman

CODES = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'X': 4}
PAD = CODES['X']
SEED_LEN = 10
MIRNA_LEN = 30

_TABLE = bytes(CODES.get(chr(i), PAD) for i in range(256))

PAIR_TYPE = np.zeros((5, 5), dtype=np.uint8)
for _a, _b in (('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')):
    PAIR_TYPE[CODES[_a], CODES[_b]] = 1
for _a, _b in (('G', 'U'), ('U', 'G')):
    PAIR_TYPE[CODES[_a], CODES[_b]] = 2


def encode(seq):
    '''RNA string -> uint8 codes (A0 C1 G2 U3, everything else X4), as in get_embedding'''
    return np.frombuffer(seq.encode().translate(_TABLE), dtype=np.uint8)


def pairing_vectors_sw(mirna, mrna):
    '''seed vs. full transcript with the authors' Smith_Waterman; both arguments are strings'''
    _, pair_m, pair_mi = Smith_Waterman(mirna[:SEED_LEN], mrna)
    mirna_vec = np.zeros(MIRNA_LEN, dtype=np.uint8)
    mirna_vec[:SEED_LEN] = np.asarray(pair_mi, dtype=np.uint8)
    return np.asarray(pair_m, dtype=np.uint8), mirna_vec


def decode(codes):
    '''uint8 codes -> RNA string, for the functions in utils.py that work on strings'''
    return ''.join('ACGUX'[c] for c in codes)


def pairing_vectors(mirna_codes, mrna_codes, min_pairs=8, max_sites=256):
    '''mrna_codes must be in 3'->5' orientation (reversed), like reverse_seq in the original code.

    Returns (mrna pairing vector of len(mrna_codes), miRNA pairing vector of MIRNA_LEN).
    The miRNA vector comes from the best-scoring site, as only one alignment can be shown there.
    '''
    length = len(mrna_codes)
    mrna_vec = np.zeros(length, dtype=np.uint8)
    mirna_vec = np.zeros(MIRNA_LEN, dtype=np.uint8)
    if length < SEED_LEN:
        return mrna_vec, mirna_vec
    seed = mirna_codes[:SEED_LEN]
    windows = np.lib.stride_tricks.sliding_window_view(mrna_codes, SEED_LEN)
    types = PAIR_TYPE[seed[None, :], windows]
    score = (types == 1).sum(1) + 0.5 * (types == 2).sum(1)
    sites = np.flatnonzero(score >= min_pairs)
    if sites.size > max_sites:
        sites = sites[np.argsort(score[sites])[-max_sites:]]
    for offset in sites:
        segment = mrna_vec[offset:offset + SEED_LEN]
        np.maximum(segment, types[offset], out=segment)
    best = sites[np.argmax(score[sites])] if sites.size else int(np.argmax(score))
    mirna_vec[:SEED_LEN] = types[best]
    return mrna_vec, mirna_vec

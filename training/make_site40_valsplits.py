'''Build the site-level training file for model selection on the data_with_negatives valid splits.

train rows: site40_Train_Validation.txt of a prepare_dataset.py run with --val-frac 0 (all rows are "train")
val rows:   every row of pairs_valid_unseen_{pair,source,target}.tsv whose target is a 40-nt site and
            that Mimosa can process (mimosa_status == ok); no deduplication, no resampling

Output has the same format as data/miRAW_Train_Validation.txt, so Mimosa.py --mode train reads it unchanged.
'''
import argparse
import csv
import json
import os
import sys

VALID_SPLITS = ['valid_unseen_pair', 'valid_unseen_source', 'valid_unseen_target']


def load_fasta(path, needed):
    seqs, key = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                key = line[1:].strip()
                key = key if key in needed else None
            elif key is not None:
                seqs[key] = line.strip()
    return seqs


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-dir', default=os.path.join(base_dir, 'data', 'custom_v2'))
    parser.add_argument('--out', default=None, help='default: <data-dir>/site40_Train_ValidSplits.txt')
    args = parser.parse_args()
    out = args.out or os.path.join(args.data_dir, 'site40_Train_ValidSplits.txt')
    csv.field_size_limit(sys.maxsize)

    val_rows = []
    for name in VALID_SPLITS:
        with open(os.path.join(args.data_dir, f'pairs_{name}.tsv')) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r['target_kind'] == 'site40' and r['mimosa_status'] == 'ok':
                    val_rows.append((name, r['mirna_id'], r['target_id'], r['label']))
    seqs = load_fasta(os.path.join(args.data_dir, 'sequences.fa'), {r[1] for r in val_rows} | {r[2] for r in val_rows})

    counts = {}
    with open(out, 'w') as fo:
        with open(os.path.join(args.data_dir, 'site40_Train_Validation.txt')) as fi:
            fo.write(next(fi))
            n = {'0': 0, '1': 0}
            for line in fi:
                cols = line.rstrip('\n').split('\t')
                if cols[5] != 'train':
                    sys.exit(f'unexpected split {cols[5]!r}: run prepare_dataset.py with --val-frac 0')
                n[cols[4]] += 1
                fo.write(line)
            counts['train'] = n
        for name, rid, tid, label in val_rows:
            fo.write(f'{rid}\t{seqs[rid]}\t{tid}\t{seqs[tid]}\t{label}\tval\n')
            counts.setdefault(name, {'0': 0, '1': 0})[label] += 1
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

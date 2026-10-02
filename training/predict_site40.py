'''Gene-level prediction with the original Mimosa test procedure.

Calls the authors' own functions (get_cts, kmers_predict, decision_for_whole): the mRNA is
cut into 40-nt windows with the given step size, every window is scored, and the pair is
called positive if at least one window scores above 0.5. Nothing about the method changes
here; this file only reads our prepared files and writes the predictions out.

Rows whose mimosa_status is not "ok" (sequence missing, precursor miRNA longer than 30 nt)
cannot be processed; they are skipped here and reported separately by --merge.
'''
import argparse
import glob
import json
import os
import sys
import time

import torch

import Mimosa
from Mimosa import get_cts, kmers_predict
from utils import reverse_seq
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_model(path):
    '''the authors save the whole model object, pickled from __main__, so make the class
    reachable under that name before unpickling'''
    main = sys.modules['__main__']
    for name in ('Transformer', 'myDataset'):
        if not hasattr(main, name):
            setattr(main, name, getattr(Mimosa, name))
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model


def read_fasta(path):
    seqs, name, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(parts)
                name, parts = line[1:], []
            else:
                parts.append(line)
    if name is not None:
        seqs[name] = ''.join(parts)
    return seqs


def read_pairs(path):
    rows = []
    with open(path) as f:
        columns = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        for line in f:
            c = line.rstrip('\n').split('\t')
            rows.append({'row_index': int(c[columns['row_index']]), 'mirna_id': c[columns['mirna_id']],
                         'target_id': c[columns['target_id']], 'label': int(c[columns['label']]),
                         'status': c[columns['mimosa_status']]})
    return rows


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'n': len(y_true), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy_score(y_true, y_pred),
            'ppv': precision_score(y_true, y_pred, zero_division=0),
            'ap_paper_ppv': average_precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if tn + fp else 0.0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'npv': tn / (tn + fn) if tn + fn else 0.0}


def merge(args):
    predicted = {}
    files = sorted(glob.glob(args.out + '.*.tsv'))
    for path in files:
        with open(path) as f:
            next(f)
            for line in f:
                row_index, _, prediction = line.split('\t')
                predicted[int(row_index)] = int(prediction)
    rows = read_pairs(args.pairs)
    log(f'{len(files)} shard files, {len(predicted)} predictions, {len(rows)} rows in {os.path.basename(args.pairs)}')
    ok = [r for r in rows if r['status'] == 'ok']
    missing = [r['row_index'] for r in ok if r['row_index'] not in predicted]
    if missing:
        log(f'WARNING: {len(missing)} processable rows have no prediction (first: {missing[:5]})')
    report = {
        'pairs': args.pairs, 'model': args.model, 'stepsize': args.stepsize,
        'rows_total': len(rows), 'rows_processable': len(ok), 'rows_predicted': len(predicted),
        'subset_processable_only': metrics([r['label'] for r in ok if r['row_index'] in predicted],
                                           [predicted[r['row_index']] for r in ok if r['row_index'] in predicted]),
        'full_set_unprocessable_as_negative': metrics([r['label'] for r in rows],
                                                      [predicted.get(r['row_index'], 0) for r in rows]),
    }
    with open(args.out + '_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def predict(args):
    model = load_model(args.model)
    seqs = read_fasta(args.sequences)
    rows = [r for r in read_pairs(args.pairs) if r['status'] == 'ok']
    todo = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    if args.limit:
        todo = todo[:args.limit]
    log(f'shard {args.shard}/{args.nshards}: {len(todo)} of {len(rows)} processable rows')

    # resume: rows already written by an earlier run of this shard are skipped
    out_path = f'{args.out}.{args.shard:03d}.tsv'
    done = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            next(f)
            done = {int(line.split('\t')[0]) for line in f if line.count('\t') == 2 and line.endswith('\n')}
        log(f'resuming: {len(done)} rows already predicted')
    todo = [r for r in todo if r['row_index'] not in done]

    start = time.time()
    with open(out_path, 'a' if done else 'w', buffering=1) as out:
        if not done:
            out.write('row_index\tlabel\tprediction\n')
        for n, row in enumerate(todo, 1):
            # exactly what perform_test does per pair
            mirna = seqs[row['mirna_id']].upper().replace('T', 'U')
            mrna = seqs[row['target_id']].upper().replace('T', 'U')
            reverse_mrna = reverse_seq(mrna)
            kmers = get_cts(reverse_mrna, args.stepsize)
            # no_grad only stops autograd from keeping activations (4.4 GB -> 0.9 GB on
            # 4-kb targets); the authors' kmers_predict and its outputs are unchanged
            with torch.no_grad():
                prediction = 0 if kmers is None else kmers_predict(kmers, mirna, model)
            out.write(f'{row["row_index"]}\t{row["label"]}\t{prediction}\n')
            if n % 200 == 0:
                rate = n / (time.time() - start)
                log(f'  {n}/{len(todo)} pairs, {rate:.2f} pairs/s, eta {(len(todo)-n)/rate/60:.0f} min')
    log(f'wrote {out_path} in {(time.time()-start)/60:.1f} min')


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Gene-level prediction with the original Mimosa test path')
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--sequences', default=os.path.join(base, 'data', 'custom', 'sequences.fa'))
    parser.add_argument('--model', default=os.path.join(base, 'training', 'model_mimosa.pth'))
    parser.add_argument('--out', required=True, help='output prefix; shards are <out>.<shard>.tsv')
    parser.add_argument('--stepsize', type=int, default=5)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--merge', action='store_true', help='combine shard outputs and report metrics')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    merge(args) if args.merge else predict(args)


if __name__ == '__main__':
    main()

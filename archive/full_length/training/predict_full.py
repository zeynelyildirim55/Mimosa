'''Predict with a full-length Mimosa checkpoint on one prepared evaluation split.

Rows Mimosa cannot process (mimosa_status != ok) are reported both ways: excluded
(subset metrics) and counted as a negative prediction (full-set metrics).
'''
import argparse
import glob
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from full_data import PairDataset, TokenBatchSampler, collate
from full_model import FullLengthMimosa


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'n': len(y_true), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'acc': accuracy_score(y_true, y_pred),
            'ppv': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'spec': tn / (tn + fp) if tn + fp else 0.0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'npv': tn / (tn + fn) if tn + fn else 0.0}


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Evaluate a full-length Mimosa checkpoint')
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--index', default=os.path.join(base, 'data', 'custom', 'seqindex'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', help='TSV of per-row predictions (default: next to the checkpoint)')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--pairing-cache', help='prefix written by precompute_pairing.py for this split')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    state = torch.load(args.checkpoint, map_location=device)
    trained = state['args']
    model = FullLengthMimosa(hidden_size=trained['hidden'], num_layers=trained['layers'],
                             num_heads=trained['heads'], dropout=trained['dropout'],
                             max_len=trained['max_len'], pooling=trained.get('pooling', 'mean')).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    log(f'checkpoint epoch={state["epoch"]} val_loss={state["val_loss"]:.4f} device={device}')

    cache = args.pairing_cache if (args.pairing_cache and glob.glob(args.pairing_cache + '.*.idx')) else None
    dataset = PairDataset(args.pairs, prefix=args.index, statuses=('ok',), max_len=trained['max_len'],
                          min_pairs=trained['min_pairs'], pairing=trained.get('pairing', 'sw'),
                          cache=cache, limit=args.limit)
    sampler = TokenBatchSampler(dataset.lengths, args.max_tokens, args.max_batch, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                                         num_workers=args.workers)
    log(f'{len(dataset)} processable rows in {os.path.basename(args.pairs)}')

    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                        os.path.basename(args.pairs).replace('pairs_', 'pred_'))
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            fea = [batch[k].to(device) for k in ('fea1', 'fea2', 'fea3', 'fea4')]
            with torch.autocast(device.type, enabled=args.amp):
                logits = model(*fea, pad_mask=batch['pad_mask'].to(device))
            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu()
            for r, p, y in zip(batch['row_index'].tolist(), prob.tolist(), batch['label'].tolist()):
                rows.append((r, y, p, int(p > 0.5)))
            if (i + 1) % 200 == 0:
                log(f'  batch {i+1}/{len(loader)}')
    rows.sort()
    with open(out_path, 'w') as f:
        f.write('row_index\tlabel\tprob\tprediction\n')
        for r, y, p, pred in rows:
            f.write(f'{r}\t{y}\t{p:.6f}\t{pred}\n')
    log(f'wrote {out_path}')

    subset = metrics([r[1] for r in rows], [r[3] for r in rows])
    predicted = {r[0]: r[3] for r in rows}
    all_true, all_pred = [], []
    with open(args.pairs) as f:
        columns = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        for line in f:
            c = line.rstrip('\n').split('\t')
            row_index = int(c[columns['row_index']])
            all_true.append(int(c[columns['label']]))
            all_pred.append(predicted.get(row_index, 0))
    full = metrics(all_true, all_pred)
    report = {'checkpoint': args.checkpoint, 'pairs': args.pairs,
              'subset_processable_only': subset, 'full_set_unprocessable_as_negative': full}
    with open(out_path.replace('.tsv', '_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

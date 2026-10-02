'''One MIL training epoch: train on the selected (best-scoring) window of every pair.

Everything inside the loop is the original Deep_train: model outputs (softmax), CrossEntropyLoss
against one-hot float labels, Adam lr 1e-4 / weight decay 1e-5, batch 256, shuffled.
'''
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mil_common import FeatureStore, get_window, load_pairs, mirna_codes, rev_codes, window_codes
from predict_site40 import load_model, read_fasta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-in', required=True)
    ap.add_argument('--model-out', required=True)
    ap.add_argument('--optimizer-in')
    ap.add_argument('--optimizer-out', required=True)
    ap.add_argument('--selection-dir')
    ap.add_argument('--random-selection', action='store_true', help='pick a random window per pair instead')
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--sequences', default=os.path.expanduser('~/Mimosa/data/custom/sequences.fa'))
    ap.add_argument('--epochs', type=int, default=1, help='epochs on the same selection')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=500)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    store = FeatureStore(args.features)
    t0 = time.time()
    seqs, pairs = read_fasta(args.sequences), load_pairs(args.pairs)
    if args.random_selection:
        # first round of a model trained from scratch: a random window per pair (the argmax of an
        # untrained model is no better, and skips a full scoring pass)
        rng = np.random.default_rng(args.seed)
        selected = [(int(o), pairs[int(o)][2], int(rng.integers(c))) for o, c in zip(store.ordinals, store.counts)]
    else:
        selected = []
        for path in sorted(glob.glob(f'{args.selection_dir}/chunk_*.tsv')):
            for line in open(path).read().splitlines()[1:]:
                o, label, _, j, _ = line.split('\t')
                selected.append((int(o), int(label), int(j)))
        if len(selected) != len(store.ordinals):
            raise SystemExit(f'selection incomplete: {len(selected)} of {len(store.ordinals)} pairs')
    selected.sort()
    if args.limit:
        selected = selected[:args.limit]
    n = len(selected)
    X1 = np.empty((n, 40), np.uint8); X2 = np.empty((n, 30), np.uint8)
    X3 = np.empty((n, 40), np.uint8); X4 = np.empty((n, 30), np.uint8)
    Y = np.empty(n, np.int64)
    for k, (o, label, j) in enumerate(selected):
        mid, tid, _, _ = pairs[o]
        X1[k] = window_codes(rev_codes(seqs, tid))[j]
        X2[k] = mirna_codes(seqs[mid].upper().replace('T', 'U'))
        X3[k], X4[k] = get_window(store, o, j)
        Y[k] = label
    print(f'built {n} training windows in {(time.time()-t0)/60:.1f} min; positives {Y.mean():.3f}', flush=True)

    model = load_model(args.model_in).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer_in and os.path.exists(args.optimizer_in):
        optimizer.load_state_dict(torch.load(args.optimizer_in, map_location=device))
        print('optimizer state resumed', flush=True)

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)       # a new shuffle every epoch, as DataLoader(shuffle=True) does
        total, t0 = 0.0, time.time()
        for b, start in enumerate(range(0, n, args.batch), 1):
            idx = perm[start:start + args.batch]
            f = [torch.from_numpy(x[idx]).long().to(device) for x in (X1, X2, X3, X4)]
            y = torch.from_numpy(Y[idx]).to(device)
            target = torch.stack([1 - y, y], dim=1).float()      # one-hot, as read_data builds it
            outputs = model(*f)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if b % args.log_every == 0:
                print(f'  epoch {epoch} batch {b}/{(n + args.batch - 1) // args.batch} loss={total/b:.4f} '
                      f'{b*args.batch/(time.time()-t0):.0f} samples/s', flush=True)
        print(f'epoch done: epoch {epoch}/{args.epochs} mean loss {total/b:.4f} in {(time.time()-t0)/60:.1f} min', flush=True)
        torch.save(model, args.model_out + '.partial')          # checkpoint after every epoch
        torch.save(optimizer.state_dict(), args.optimizer_out + '.partial')
    os.replace(args.model_out + '.partial', args.model_out)
    os.replace(args.optimizer_out + '.partial', args.optimizer_out)
    print(f'saved {args.model_out}', flush=True)


if __name__ == '__main__':
    main()

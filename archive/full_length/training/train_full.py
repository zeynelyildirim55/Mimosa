'''Train Mimosa without the 40-nt limit on the prepared custom dataset.

Built for a Slurm job array that runs one task at a time (sbatch --array=1-N%1):
each task resumes the checkpoint the previous one left (including the position inside an
epoch), stops cleanly before the wall-clock limit, and writes a DONE file when all epochs
are finished so the remaining tasks exit immediately.
'''
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from full_data import PairDataset, TokenBatchSampler, build_index, collate
from full_model import FullLengthMimosa

METRIC_NAMES = ('acc', 'ppv', 'recall', 'spec', 'f1', 'npv')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'acc': accuracy_score(y_true, y_pred),
            'ppv': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'spec': tn / (tn + fp) if tn + fp else 0.0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'npv': tn / (tn + fn) if tn + fn else 0.0}


def forward(model, batch, device, criterion, amp, legacy_softmax):
    fea = [batch[k].to(device, non_blocking=True) for k in ('fea1', 'fea2', 'fea3', 'fea4')]
    pad_mask = batch['pad_mask'].to(device, non_blocking=True)
    label = batch['label'].to(device, non_blocking=True)
    with torch.autocast(device.type, enabled=amp):
        logits = model(*fea, pad_mask=pad_mask)
    scores = logits.float()
    if legacy_softmax:
        scores = torch.softmax(scores, dim=1)
    return criterion(scores, label), logits.detach(), label


def validate(model, loader, device, criterion, args):
    model.eval()
    total, batches, y_true, y_pred = 0.0, 0, [], []
    with torch.no_grad():
        for batch in loader:
            loss, logits, label = forward(model, batch, device, criterion, args.amp, args.legacy_softmax)
            total += loss.item()
            batches += 1
            y_true.extend(label.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())
    model.train()
    return total / max(batches, 1), metrics(y_true, y_pred)


def save_checkpoint(path, model, optimizer, scaler, state, args):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scaler': scaler.state_dict() if scaler else None,
               'torch_rng': torch.get_rng_state(), 'args': vars(args), **state}
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_history(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['epoch', 'batch', 'global_step', 'train_loss', 'val_loss']
                            + [f'val_{k}' for k in METRIC_NAMES])
        writer.writerow(row)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Train Mimosa on full-length targets')
    parser.add_argument('--pairs', default=os.path.join(base, 'data', 'custom', 'pairs_train_val.tsv'))
    parser.add_argument('--fasta', default=os.path.join(base, 'data', 'custom', 'sequences.fa'))
    parser.add_argument('--index', default=os.path.join(base, 'data', 'custom', 'seqindex'))
    parser.add_argument('--out-dir', default=os.path.join(base, 'runs', 'full_length'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=8192, help='padded nucleotides per batch')
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--max-len', type=int, default=4096)
    parser.add_argument('--pooling', default='mean', choices=['mean', 'max'])
    parser.add_argument('--pairing', default='sw', choices=['sw', 'scan'])
    parser.add_argument('--pairing-cache', default=os.path.join(base, 'data', 'custom', 'pairing_train_val'))
    parser.add_argument('--min-pairs', type=float, default=8)
    parser.add_argument('--legacy-softmax', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=20000)
    parser.add_argument('--save-every', type=int, default=500, help='checkpoint every N optimizer steps')
    parser.add_argument('--val-every', type=int, default=5000, help='validate every N steps (0 = epoch end only)')
    parser.add_argument('--max-hours', type=float, default=11.0, help='stop cleanly after this many hours')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    done_file = os.path.join(args.out_dir, 'DONE')
    if os.path.exists(done_file):
        log(f'training already finished ({done_file}), nothing to do')
        return 0

    deadline = time.time() + args.max_hours * 3600
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    log(f'device={device} torch={torch.__version__}'
        + (f' gpu={torch.cuda.get_device_name(0)}' if device.type == 'cuda' else ''))

    if not os.path.exists(args.index + '.u8'):
        log('building sequence index')
        log(f'indexed {build_index(args.fasta, args.index)} nucleotides')

    cache = args.pairing_cache if glob.glob(str(args.pairing_cache) + '.*.idx') else None
    log(f'pairing={args.pairing} cache={cache or "none (computed on the fly)"} pooling={args.pooling}')
    common = dict(prefix=args.index, max_len=args.max_len, min_pairs=args.min_pairs,
                  pairing=args.pairing, cache=cache)
    train_set = PairDataset(args.pairs, split='train', limit=args.limit, **common)
    val_set = PairDataset(args.pairs, split='val', limit=args.val_limit, **common)
    log(f'train rows={len(train_set)} val rows={len(val_set)}')

    model = FullLengthMimosa(hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads,
                             dropout=args.dropout, max_len=args.max_len, pooling=args.pooling).to(device)
    log(f'parameters={sum(p.numel() for p in model.parameters())}')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    state = {'epoch': 1, 'next_batch': 0, 'global_step': 0, 'best_val': float('inf')}
    last_path = os.path.join(args.out_dir, 'last.pth')
    if os.path.exists(last_path) and not args.no_resume:
        payload = torch.load(last_path, map_location=device)
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        if scaler and payload.get('scaler'):
            scaler.load_state_dict(payload['scaler'])
        if payload.get('torch_rng') is not None:
            torch.set_rng_state(payload['torch_rng'].cpu())
        state = {k: payload[k] for k in state}
        log(f'resumed from {last_path}: epoch {state["epoch"]}, batch {state["next_batch"]}, '
            f'step {state["global_step"]}, best val loss {state["best_val"]:.4f}')
    else:
        with open(os.path.join(args.out_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    train_sampler = TokenBatchSampler(train_set.lengths, args.max_tokens, args.max_batch, shuffle=True, seed=args.seed)
    val_sampler = TokenBatchSampler(val_set.lengths, args.max_tokens, args.max_batch, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=list(val_sampler), collate_fn=collate,
                                             num_workers=args.workers)
    history = os.path.join(args.out_dir, 'history.csv')
    model.train()

    for epoch in range(state['epoch'], args.epochs + 1):
        batches = train_sampler.batches_for(epoch)
        start = state['next_batch'] if epoch == state['epoch'] else 0
        log(f'epoch {epoch}/{args.epochs}: {len(batches)} batches, starting at {start}')
        loader = torch.utils.data.DataLoader(train_set, batch_sampler=batches[start:], collate_fn=collate,
                                             num_workers=args.workers, pin_memory=device.type == 'cuda')
        running, seen, t0 = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        stopped = False
        for i, batch in enumerate(loader, start=start):
            loss, _, _ = forward(model, batch, device, criterion, args.amp, args.legacy_softmax)
            scaled = loss / args.accum
            scaler.scale(scaled).backward() if scaler else scaled.backward()
            if (i + 1) % args.accum == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            seen += 1
            state['global_step'] += 1
            state['next_batch'] = i + 1

            if args.log_every and seen % args.log_every == 0:
                log(f'  epoch {epoch} batch {i+1}/{len(batches)} loss={running/seen:.4f} '
                    f'{seen/(time.time()-t0):.2f} batches/s')
            if args.save_every and state['global_step'] % args.save_every == 0:
                save_checkpoint(last_path, model, optimizer, scaler, state, args)
            if args.val_every and state['global_step'] % args.val_every == 0:
                val_loss, val_metrics = validate(model, val_loader, device, criterion, args)
                log(f'  validation at step {state["global_step"]}: loss={val_loss:.4f} '
                    + ' '.join(f'{k}={val_metrics[k]:.4f}' for k in METRIC_NAMES))
                append_history(history, [epoch, i + 1, state['global_step'], running / seen, val_loss]
                               + [val_metrics[k] for k in METRIC_NAMES])
                if val_loss < state['best_val']:
                    state['best_val'] = val_loss
                    save_checkpoint(os.path.join(args.out_dir, 'best.pth'), model, optimizer, scaler, state, args)
                    log(f'  saved best.pth (val loss {val_loss:.4f})')
                save_checkpoint(last_path, model, optimizer, scaler, state, args)
            if time.time() > deadline:
                state['epoch'] = epoch
                save_checkpoint(last_path, model, optimizer, scaler, state, args)
                log(f'wall-clock limit reached at epoch {epoch} batch {i+1}; checkpoint saved, '
                    f'the next array task will continue')
                stopped = True
                break
        if stopped:
            return 0

        val_loss, val_metrics = validate(model, val_loader, device, criterion, args)
        log(f'epoch {epoch} done: train loss={running/max(seen,1):.4f} val loss={val_loss:.4f} '
            + ' '.join(f'{k}={val_metrics[k]:.4f}' for k in METRIC_NAMES))
        append_history(history, [epoch, len(batches), state['global_step'], running / max(seen, 1), val_loss]
                       + [val_metrics[k] for k in METRIC_NAMES])
        state['epoch'] = epoch + 1
        state['next_batch'] = 0
        if val_loss < state['best_val']:
            state['best_val'] = val_loss
            save_checkpoint(os.path.join(args.out_dir, 'best.pth'), model, optimizer, scaler, state, args)
            log(f'saved best.pth (val loss {val_loss:.4f})')
        save_checkpoint(last_path, model, optimizer, scaler, state, args)

    with open(done_file, 'w') as f:
        f.write(f'finished {args.epochs} epochs at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    log('all epochs finished')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

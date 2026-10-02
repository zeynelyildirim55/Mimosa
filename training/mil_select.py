'''Score every window of every pair with a model and keep the best one (GPU, no gradients).

Used for three things: choosing the training window of each pair (MIL), validation and testing.
A pair is predicted positive when its best window scores above 0.5, which is exactly Mimosa's
decision_for_whole rule. Work is split into chunks that workers claim one at a time, so any
number of GPUs (and restarted jobs) can share one output directory.
'''
import argparse
import os
import socket
import time

import numpy as np
import torch

from mil_common import FeatureStore, load_pairs, mirna_codes, rev_codes, window_codes
from predict_site40 import load_model, read_fasta


def claim(out_dir, c, stale_hours):
    done = os.path.join(out_dir, f'chunk_{c:04d}.tsv')
    lock = os.path.join(out_dir, f'chunk_{c:04d}.claim')
    if os.path.exists(done):
        return False
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) > stale_hours * 3600 and not os.path.exists(done):
            os.remove(lock)   # left behind by a job that was killed
            return claim(out_dir, c, stale_hours)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--features', required=True, help='feature prefix written by mil_features.py')
    ap.add_argument('--sequences', default=os.path.expanduser('~/Mimosa/data/custom/sequences.fa'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--nchunks', type=int, default=400)
    ap.add_argument('--batch-windows', type=int, default=8192)
    ap.add_argument('--stale-hours', type=float, default=2.0)
    ap.add_argument('--start', type=int, default=0, help='first chunk to try (spreads workers out)')
    ap.add_argument('--max-chunks', type=int, default=0, help='debug: stop after this many chunks')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f'{socket.gethostname()}:{os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")}'
    model = load_model(args.model).to(device)
    model.eval()
    seqs = read_fasta(args.sequences)
    pairs = load_pairs(args.pairs)
    store = FeatureStore(args.features)
    chunks = np.array_split(np.arange(len(store.ordinals)), args.nchunks)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[{tag}] device={device} pairs={len(store.ordinals)} chunks={args.nchunks}', flush=True)

    n_done = 0
    for step in range(args.nchunks):
        c = (args.start + step) % args.nchunks
        if not claim(args.out_dir, c, args.stale_hours):
            continue
        t0, windows, rows = time.time(), 0, []
        batch, meta = [[], [], [], []], []

        def flush():
            f1, f2, f3, f4 = (torch.from_numpy(np.concatenate(x)).long().to(device) for x in batch)
            with torch.no_grad():
                prob = model(f1, f2, f3, f4)[:, 1].float().cpu().numpy()
            pos = 0
            for o, label, n in meta:
                p = prob[pos:pos + n]
                j = int(p.argmax())
                rows.append(f'{o}\t{label}\t{n}\t{j}\t{p[j]:.6f}\n')
                pos += n
            for x in batch:
                x.clear()
            meta.clear()

        for i in chunks[c]:
            o = int(store.ordinals[i])
            mid, tid, label, _ = pairs[o]
            m, mi = store.get(o)
            w = window_codes(rev_codes(seqs, tid))
            assert len(w) == len(m), (o, len(w), len(m))
            mc = np.repeat(mirna_codes(seqs[mid].upper().replace('T', 'U'))[None], len(w), axis=0)
            if meta and sum(n for _, _, n in meta) + len(w) > args.batch_windows:
                flush()
            for x, v in zip(batch, (w, mc, m, mi)):
                x.append(v)
            meta.append((o, label, len(w)))
            windows += len(w)
        if meta:
            flush()
        tmp = os.path.join(args.out_dir, f'.chunk_{c:04d}.tsv.{os.getpid()}')
        with open(tmp, 'w') as f:
            f.write('ordinal\tlabel\tn_windows\tbest_window\tbest_prob\n')
            f.writelines(rows)
        os.replace(tmp, os.path.join(args.out_dir, f'chunk_{c:04d}.tsv'))
        dt = time.time() - t0
        n_done += 1
        print(f'[{tag}] chunk {c}: {len(rows)} pairs, {windows} windows, {windows/dt:.0f} windows/s', flush=True)
        if args.max_chunks and n_done >= args.max_chunks:
            break
    print(f'[{tag}] finished, {n_done} chunks', flush=True)


if __name__ == '__main__':
    main()

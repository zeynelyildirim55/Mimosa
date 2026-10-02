'''Benchmark Mimosa training cost: 40-nt model (current) vs a variant without the 40-nt limit.'''
import argparse, os, subprocess, sys, time

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='tiny sizes on CPU, only checks that the code runs')
parser.add_argument('--wait-minutes', type=float, default=0,
                    help='poll this long for an idle GPU before giving up (0 = check once)')
args = parser.parse_args()


def idle_gpus():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    gpus = [[x.strip() for x in l.split(',')] for l in q.strip().splitlines()]
    return q, [g for g in gpus if int(g[2]) < 1000 and int(g[4]) < 10]


if not args.dry_run:
    listing, idle = idle_gpus()
    print(listing, flush=True)
    deadline = time.time() + args.wait_minutes * 60
    while not idle and time.time() < deadline:
        time.sleep(120)
        listing, idle = idle_gpus()
        if idle:
            print('a GPU became free:', listing, flush=True)
    if not idle:
        print('no idle GPU on this node (all have >1 GB used or >10% utilization); not running', flush=True)
        sys.exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = idle[0][0]
    print('using GPU', idle[0], flush=True)

import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'training'))
import Mimosa as M
from full_model import FullLengthMimosa

dev = torch.device('cpu' if args.dry_run else 'cuda')
if not args.dry_run:
    print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0),
          'capability', torch.cuda.get_device_capability(0), flush=True)


def batch(B, L):
    return (torch.randint(0, 4, (B, L), device=dev), torch.randint(0, 5, (B, 30), device=dev),
            torch.randint(0, 3, (B, L), device=dev), torch.randint(0, 3, (B, 30), device=dev),
            nn.functional.one_hot(torch.randint(0, 2, (B,), device=dev), 2).float())


def measure(model, B, L, amp, train, reps):
    '''seconds per step and peak GPU memory (GB); None if out of memory'''
    model.train(train)
    x1, x2, x3, x4, y = batch(B, L)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and not args.dry_run)
    crit = nn.CrossEntropyLoss()
    if not args.dry_run:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for r in range(reps + 2):
            if not args.dry_run: torch.cuda.synchronize()
            t = time.time()
            if train:
                with torch.autocast(dev.type, enabled=amp and not args.dry_run):
                    loss = crit(model(x1, x2, x3, x4).float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad(), torch.autocast(dev.type, enabled=amp and not args.dry_run):
                    model(x1, x2, x3, x4)
            if not args.dry_run: torch.cuda.synchronize()
            times.append(time.time() - t)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            del x1, x2, x3, x4, y
            if not args.dry_run: torch.cuda.empty_cache()
            return None
        raise
    mem = 0.0 if args.dry_run else torch.cuda.max_memory_allocated() / 2**30
    return sum(times[2:]) / reps, mem


def report(name, B, L, amp, res):
    if res is None:
        print(f'{name:28s} L={L:5d} B={B:5d} amp={amp!s:5s} OOM', flush=True)
    else:
        s, mem = res
        print(f'{name:28s} L={L:5d} B={B:5d} amp={amp!s:5s} {s*1000:9.1f} ms/step  {B/s:10.1f} samples/s  peak {mem:5.1f} GB', flush=True)


t_start = time.time()
small = args.dry_run
m40 = M.Transformer(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, output_size=2).to(dev)
for amp in (False, True):
    for B in ([4] if small else [256, 1024, 4096]):
        report('40nt train', B, 40, amp, measure(m40, B, 40, amp, True, 1 if small else 5))
    for B in ([4] if small else [4096, 16384]):
        report('40nt inference', B, 40, amp, measure(m40, B, 40, amp, False, 1 if small else 5))
del m40

lengths = [64] if small else [500, 1000, 2000, 2608, 4076]
full = FullLengthMimosa(max_len=4096, pooling='mean').to(dev)
for amp in (False, True):
    for L in lengths:
        for B in ([2] if small else [64, 32, 16, 8, 4, 2, 1]):
            res = measure(full, B, L, amp, True, 1 if small else 3)
            if res is not None or B == 1:
                report('no-limit train', B, L, amp, res)
            if res is not None:
                break
print(f'total benchmark time {time.time()-t_start:.0f}s', flush=True)

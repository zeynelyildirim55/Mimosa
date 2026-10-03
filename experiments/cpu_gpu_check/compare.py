'''Compare gene-level decisions of the original CPU procedure (predict_site40.py) and the GPU scoring
(mil_select.py) for the same model and split; run before using the GPU path for the test.

  python experiments/cpu_gpu_check/compare.py --pairs data/custom_v2/pairs_valid_unseen_pair.tsv \
      --cpu results/site40_v2/valid_unseen_pair/cpu --gpu results/site40_v2/valid_unseen_pair/gpu
'''
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--cpu', required=True, help='directory with predict_site40.py shards pred.*.tsv')
    ap.add_argument('--gpu', required=True, help='directory with mil_select.py chunk_*.tsv')
    args = ap.parse_args()

    with open(args.pairs) as f:
        col = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        row_index = [int(line.split('\t')[col['row_index']]) for line in f]

    cpu = {}
    for path in sorted(glob.glob(os.path.join(args.cpu, 'pred.*.tsv'))):
        with open(path) as f:
            next(f)
            for line in f:
                r, label, pred = line.split()
                cpu[int(r)] = (int(label), int(pred))

    gpu = {}
    for path in sorted(glob.glob(os.path.join(args.gpu, 'chunk_*.tsv'))):
        with open(path) as f:
            next(f)
            for line in f:
                ordinal, label, _, _, prob = line.split()
                gpu[row_index[int(ordinal)]] = (int(label), float(prob))

    common = cpu.keys() & gpu.keys()
    flips = sorted((r, cpu[r][1], gpu[r][1]) for r in common if cpu[r][1] != int(gpu[r][1] > 0.5))
    out = {'cpu_rows': len(cpu), 'gpu_rows': len(gpu), 'common_rows': len(common),
           'only_cpu': len(cpu.keys() - gpu.keys()), 'only_gpu': len(gpu.keys() - cpu.keys()),
           'label_mismatch': sum(cpu[r][0] != gpu[r][0] for r in common),
           'decision_differences': len(flips),
           'differences': [{'row_index': r, 'cpu_prediction': c, 'gpu_best_prob': p} for r, c, p in flips[:50]]}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

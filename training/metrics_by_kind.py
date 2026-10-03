'''Metrics of one scored split broken down by target kind (site40 = the target is a 40-nt site,
gene = the target is a transcript, raw_other = a raw sequence of another length), each on all rows
(unprocessable = negative) and on processable rows. The main result stays the one over all rows.

  python training/metrics_by_kind.py --selection-dir results/site40_v2/test_unseen_pair/gpu \
      --pairs data/custom_v2/pairs_test_unseen_pair.tsv --out results/site40_v2/test_unseen_pair/metrics_by_kind.json
'''
import argparse
import collections
import json

from mil_metrics import metrics, read_selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection-dir', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()
    sel = read_selection(args.selection_dir)

    groups = collections.defaultdict(lambda: {'all': ([], []), 'processable': ([], [])})
    with open(args.pairs) as f:
        col = {c: i for i, c in enumerate(next(f).rstrip('\n').split('\t'))}
        for ordinal, line in enumerate(f):
            c = line.rstrip('\n').split('\t')
            label = int(c[col['label']])
            pred = int(sel[ordinal][1] > 0.5) if ordinal in sel else 0
            for kind in ('all_kinds', c[col['target_kind']]):
                g = groups[kind]
                g['all'][0].append(label)
                g['all'][1].append(pred)
                if ordinal in sel:
                    g['processable'][0].append(label)
                    g['processable'][1].append(pred)

    report = {kind: {subset: metrics(*yp) for subset, yp in g.items() if yp[0]} for kind, g in groups.items()}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        open(args.out, 'w').write(text)


if __name__ == '__main__':
    main()

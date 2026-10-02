'''Gene-level metrics from a selection directory (pair positive <=> best window > 0.5).'''
import argparse
import glob
import json

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from mil_common import load_pairs


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {'n': len(y), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy_score(y, p), 'ppv': precision_score(y, p, zero_division=0),
            'ap_paper_ppv': average_precision_score(y, p), 'recall': recall_score(y, p, zero_division=0),
            'specificity': tn / (tn + fp) if tn + fp else 0.0, 'f1': f1_score(y, p, zero_division=0),
            'npv': tn / (tn + fn) if tn + fn else 0.0}


def read_selection(sel_dir):
    out = {}
    for path in glob.glob(f'{sel_dir}/chunk_*.tsv'):
        for line in open(path).read().splitlines()[1:]:
            o, label, _, _, prob = line.split('\t')
            out[int(o)] = (int(label), float(prob))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection-dir', required=True)
    ap.add_argument('--pairs', help='pairs file with mimosa_status: also report the full set (unprocessable = negative)')
    ap.add_argument('--out')
    args = ap.parse_args()
    sel = read_selection(args.selection_dir)
    report = {'selection_dir': args.selection_dir, 'pairs_scored': len(sel),
              'processable': metrics([v[0] for v in sel.values()], [int(v[1] > 0.5) for v in sel.values()])}
    if args.pairs:
        pairs = load_pairs(args.pairs)
        report['all_rows'] = metrics([v[2] for v in pairs.values()],
                                     [int(sel[o][1] > 0.5) if o in sel else 0 for o in pairs])
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        open(args.out, 'w').write(text)


if __name__ == '__main__':
    main()

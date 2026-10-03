'''Prepare the custom miRNA-mRNA dataset (training_chunks + data_with_negatives) for Mimosa.

Training rows (training_chunks):
  1. keep rna-rna rows with RNA_type=miRNA and target_RNA_type=mRNA
  2. deduplicate on (miRNA id, target id, label)
  3. resolve sequences from ~/fasta_files/rna.fa (T -> U)
  4. drop rows with a missing or non-ACGU sequence, a miRNA outside
     [--min-mirna-len, --max-mirna-len], a target shorter than 40 nt,
     a pair that occurs with both labels, or a pair that also occurs in a valid/test split
  5. random train/val split, stratified by label and target kind

Evaluation rows (valid_unseen_* / test_unseen_*):
  every row is kept in the original order (no deduplication, no filtering) so that Mimosa is
  evaluated on exactly the same rows as other models; mimosa_status tells whether Mimosa can
  process the row ("ok") or why not. Targets shorter than 40 nt are "ok" because Mimosa's
  test code pads them.

Outputs (in --out-dir):
  pairs_train_val.tsv          kept training rows (ids only, split column = train/val)
  pairs_<eval split>.tsv       all evaluation rows (ids only) with row_index and mimosa_status
  sequences.fa                 sequence of every id used by a kept/"ok" row
  site40_Train_Validation.txt  training rows whose target is already a 40-nt site,
                               same format as data/miRAW_Train_Validation.txt
  prep_report.json             row counts after each step
'''
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time

RAW_SEQ = re.compile(r'^[ACGTUacgtu]+$')
RNA_ALPHABET = set('ACGU')
EVAL_SPLITS = ['valid_unseen_pair', 'valid_unseen_source', 'valid_unseen_target',
               'test_unseen_pair', 'test_unseen_source', 'test_unseen_target']
PAIR_COLUMNS = ['mirna_id', 'target_id', 'label', 'split', 'target_kind',
                'data_type', 'neg_strategy', 'mirna_len', 'target_len']
EVAL_COLUMNS = ['row_index'] + PAIR_COLUMNS + ['mimosa_status']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def is_raw_seq(s):
    return RAW_SEQ.match(s) is not None


def to_rna(s):
    return s.upper().replace('T', 'U')


def target_kind(target_id):
    '''site40: the target id is itself a 40-nt binding site; gene: a transcript/gene id'''
    if is_raw_seq(target_id):
        return 'site40' if len(target_id) == 40 else 'raw_other'
    return 'gene'


def iter_mirna_mrna(path, report):
    '''yield (line number, record) for the rna-rna miRNA-mRNA rows of a jsonl file'''
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            report['lines'] += 1
            if '"miRNA"' not in line or '"mRNA"' not in line:
                continue
            d = json.loads(line)
            if (d.get('interaction_type') != 'rna-rna' or d.get('RNA_type') != 'miRNA'
                    or d.get('target_RNA_type') != 'mRNA'):
                continue
            yield line_no, d


def load_train_rows(paths, limit, report):
    '''keep miRNA-mRNA rows and deduplicate on (miRNA id, target id, label)'''
    rows = {}
    seen = 0
    for path in paths:
        for _, d in iter_mirna_mrna(path, report):
            seen += 1
            key = (sys.intern(d['RNA_id']), sys.intern(d['target_id']), int(d['interaction_label']))
            if key not in rows:
                rows[key] = (sys.intern(d.get('data_type', '')), sys.intern(d.get('neg_strategy', '')))
            if limit and seen >= limit:
                break
        if limit and seen >= limit:
            break
    report['mirna_mrna_rows'] = seen
    report['unique_rows'] = len(rows)
    return rows


def load_eval_rows(path, limit, report):
    '''keep every miRNA-mRNA row in file order: (line number, miRNA id, target id, label, data_type, neg_strategy)'''
    rows = []
    for line_no, d in iter_mirna_mrna(path, report):
        rows.append((line_no, sys.intern(d['RNA_id']), sys.intern(d['target_id']), int(d['interaction_label']),
                     sys.intern(d.get('data_type', '')), sys.intern(d.get('neg_strategy', ''))))
        if limit and len(rows) >= limit:
            break
    report['mirna_mrna_rows'] = len(rows)
    report['non_mirna_mrna_lines'] = report['lines'] - len(rows)
    return rows


def load_sequences(fasta_path, needed):
    '''stream a fasta file and keep only the ids in `needed`.
    Prefixed headers such as "miRBase:MIMAT0000062" also match their bare id;
    an exact header match always wins.'''
    exact, bare = {}, {}
    with open(fasta_path, encoding='utf-8') as raw:
        store, key, buf = None, None, []
        for line in raw:
            if line.startswith('>'):
                if store is not None:
                    store.setdefault(key, to_rna(''.join(buf)))
                parts = line[1:].split()
                header = parts[0] if parts else ''
                store, key, buf = None, None, []
                if header in needed:
                    store, key = exact, header
                elif ':' in header and header.split(':', 1)[1] in needed:
                    store, key = bare, header.split(':', 1)[1]
            elif store is not None:
                buf.append(line.strip())
        if store is not None:
            store.setdefault(key, to_rna(''.join(buf)))
    bare.update(exact)
    return bare


def make_checker(seq_of, min_mirna_len, max_mirna_len, min_target_len):
    '''validate each id once; returns (reason or None, sequence length or None)'''
    cache = {}

    def check(id_, role):
        k = (id_, role)
        if k not in cache:
            s = seq_of(id_)
            if s is None:
                cache[k] = (f'{role}_not_found', None)
            elif not set(s) <= RNA_ALPHABET:
                cache[k] = (f'{role}_non_acgu', len(s))
            elif role == 'mirna' and not min_mirna_len <= len(s) <= max_mirna_len:
                cache[k] = ('mirna_length', len(s))
            elif role == 'target' and len(s) < min_target_len:
                cache[k] = (f'target_shorter_than_{min_target_len}', len(s))
            else:
                cache[k] = (None, len(s))
        return cache[k]

    return check


def filter_train_rows(rows, check, report):
    kept = {}
    drops = collections.Counter()
    for (rid, tid, label), meta in rows.items():
        reason, mirna_len = check(rid, 'mirna')
        if reason is None:
            reason, target_len = check(tid, 'target')
        if reason is not None:
            drops[reason] += 1
            continue
        kept[(rid, tid, label)] = meta + (mirna_len, target_len)

    # a pair that occurs with both labels is ambiguous
    labels = collections.defaultdict(set)
    for rid, tid, label in kept:
        labels[(rid, tid)].add(label)
    conflicts = {pair for pair, ls in labels.items() if len(ls) > 1}
    if conflicts:
        before = len(kept)
        kept = {k: v for k, v in kept.items() if (k[0], k[1]) not in conflicts}
        drops['label_conflict'] = before - len(kept)

    report['dropped'] = dict(drops)
    report['kept_rows'] = len(kept)
    return kept


def mark_eval_rows(rows, check, report):
    marked = []
    status = collections.Counter()
    for line_no, rid, tid, label, data_type, neg_strategy in rows:
        mirna_reason, mirna_len = check(rid, 'mirna')
        target_reason, target_len = check(tid, 'target')
        s = mirna_reason or target_reason or 'ok'
        status[(label, s)] += 1
        marked.append((line_no, rid, tid, label, data_type, neg_strategy, mirna_len, target_len, s))

    keys = collections.Counter((r[1], r[2], r[3]) for r in rows)
    labels = collections.defaultdict(set)
    for rid, tid, label in keys:
        labels[(rid, tid)].add(label)
    n_ok = sum(n for (_, s), n in status.items() if s == 'ok')
    report['status'] = {f'label={l}|{s}': n for (l, s), n in sorted(status.items())}
    report['ok_rows'] = n_ok
    report['ok_fraction'] = round(n_ok / len(rows), 4) if rows else 0.0
    # informational only, these rows are kept
    report['duplicate_rows'] = sum(c - 1 for c in keys.values())
    report['pairs_with_both_labels'] = sum(len(ls) > 1 for ls in labels.values())
    return marked


def assign_split(keys, val_frac, seed):
    groups = collections.defaultdict(list)
    for k in keys:
        groups[(k[2], target_kind(k[1]))].append(k)
    rng = random.Random(seed)
    split = {}
    for g in sorted(groups):
        members = groups[g]
        rng.shuffle(members)
        n_val = round(len(members) * val_frac)
        for i, k in enumerate(members):
            split[k] = 'val' if i < n_val else 'train'
    return split


def write_tsv(path, columns, records):
    with open(path, 'w') as f:
        f.write('\t'.join(columns) + '\n')
        for rec in records:
            f.write('\t'.join('' if v is None else str(v) for v in rec) + '\n')


def main():
    home = os.path.expanduser('~')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Prepare the custom dataset for Mimosa')
    parser.add_argument('--chunks-dir', default=os.path.join(home, 'training_chunks'))
    parser.add_argument('--eval-dir', default=os.path.join(home, 'data_with_negatives', 'rna_rna', 'miRNA_mRNA'))
    parser.add_argument('--fasta', default=os.path.join(home, 'fasta_files', 'rna.fa'))
    parser.add_argument('--out-dir', default=os.path.join(base_dir, 'data', 'custom'))
    parser.add_argument('--val-frac', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-mirna-len', type=int, default=10,
                        help='Mimosa aligns the first 10 nt of the miRNA')
    parser.add_argument('--max-mirna-len', type=int, default=30,
                        help='Mimosa pads miRNAs to 30 nt')
    parser.add_argument('--limit', type=int, default=0,
                        help='debug only: stop after this many miRNA-mRNA rows per source')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report = {'args': vars(args)}

    chunk_files = sorted(glob.glob(os.path.join(args.chunks_dir, 'chunk_*.jsonl')))
    log(f'reading {len(chunk_files)} training chunk files')
    report['train'] = collections.Counter()
    train_rows = load_train_rows(chunk_files, args.limit, report['train'])
    log(f'train: {report["train"]["mirna_mrna_rows"]} miRNA-mRNA rows, {len(train_rows)} unique')

    eval_rows = {}
    for name in EVAL_SPLITS:
        report[name] = collections.Counter()
        eval_rows[name] = load_eval_rows(os.path.join(args.eval_dir, f'final_{name}.jsonl'), args.limit, report[name])
        log(f'{name}: {len(eval_rows[name])} rows')

    needed = {i for k in train_rows for i in k[:2] if not is_raw_seq(i)}
    needed |= {i for rows in eval_rows.values() for r in rows for i in r[1:3] if not is_raw_seq(i)}
    log(f'streaming rna.fa for {len(needed)} ids')
    seqs = load_sequences(args.fasta, needed)
    report['ids_needed'] = len(needed)
    report['ids_resolved'] = len(seqs)
    log(f'resolved {len(seqs)} of {len(needed)} ids')

    def seq_of(id_):
        return to_rna(id_) if is_raw_seq(id_) else seqs.get(id_)

    train_check = make_checker(seq_of, args.min_mirna_len, args.max_mirna_len, min_target_len=40)
    # Mimosa's test code pads targets shorter than 40 nt, so they stay predictable in the eval splits
    eval_check = make_checker(seq_of, args.min_mirna_len, args.max_mirna_len, min_target_len=1)

    train_kept = filter_train_rows(train_rows, train_check, report['train'])
    eval_pairs = {(r[1], r[2]) for rows in eval_rows.values() for r in rows}
    before = len(train_kept)
    train_kept = {k: v for k, v in train_kept.items() if (k[0], k[1]) not in eval_pairs}
    report['train']['dropped_overlap_with_eval'] = before - len(train_kept)
    report['train']['kept_rows'] = len(train_kept)

    split = assign_split(list(train_kept), args.val_frac, args.seed)
    write_tsv(os.path.join(args.out_dir, 'pairs_train_val.tsv'), PAIR_COLUMNS,
              ([rid, tid, label, split[(rid, tid, label)], target_kind(tid), dt, ns, ml, tl]
               for (rid, tid, label), (dt, ns, ml, tl) in train_kept.items()))
    c = collections.Counter((split[k], k[2], target_kind(k[1])) for k in train_kept)
    report['train_summary'] = {f'{s}|label={l}|{kind}': n for (s, l, kind), n in sorted(c.items())}
    used = {i for k in train_kept for i in k[:2]}

    for name in EVAL_SPLITS:
        marked = mark_eval_rows(eval_rows[name], eval_check, report[name])
        write_tsv(os.path.join(args.out_dir, f'pairs_{name}.tsv'), EVAL_COLUMNS,
                  ([line_no, rid, tid, label, name, target_kind(tid), dt, ns, ml, tl, s]
                   for line_no, rid, tid, label, dt, ns, ml, tl, s in marked))
        used |= {i for r in marked if r[8] == 'ok' for i in r[1:3]}
        log(f'{name}: ok_fraction={report[name]["ok_fraction"]}')

    with open(os.path.join(args.out_dir, 'sequences.fa'), 'w') as f:
        for i in sorted(used):
            f.write(f'>{i}\n{seq_of(i)}\n')
    report['sequences_written'] = len(used)

    with open(os.path.join(args.out_dir, 'site40_Train_Validation.txt'), 'w') as f:
        f.write('mirna_id\tmirna_seq\tmrna_id\tmrna_seq\tlabel\tsplit\n')
        for k in train_kept:
            rid, tid, label = k
            if target_kind(tid) == 'site40':
                f.write(f'{rid}\t{seq_of(rid)}\t{tid}\t{seq_of(tid)}\t{label}\t{split[k]}\n')

    with open(os.path.join(args.out_dir, 'prep_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    log('done')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

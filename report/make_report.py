'''Builds report/Mimosa_reproduction_report.pdf. Needs reportlab (not part of the mimosa env).'''
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
pdfmetrics.registerFont(TTFont('DV', f'{FONT_DIR}/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB', f'{FONT_DIR}/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DVM', f'{FONT_DIR}/DejaVuSansMono.ttf'))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB', italic='DV', boldItalic='DVB')

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mimosa_reproduction_report.pdf')
INK, MUTED, RULE, HEAD_BG = colors.HexColor('#1f2328'), colors.HexColor('#57606a'), colors.HexColor('#d0d7de'), colors.HexColor('#eef1f4')

S = {
    'title': ParagraphStyle('title', fontName='DVB', fontSize=16, leading=19, textColor=INK, spaceAfter=3),
    'sub': ParagraphStyle('sub', fontName='DV', fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=6),
    'h1': ParagraphStyle('h1', keepWithNext=1, fontName='DVB', fontSize=11.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3),
    'h2': ParagraphStyle('h2', keepWithNext=1, fontName='DVB', fontSize=9.5, leading=12, textColor=INK, spaceBefore=4, spaceAfter=2),
    'p': ParagraphStyle('p', allowWidows=0, fontName='DV', fontSize=8.4, leading=11, textColor=INK, spaceAfter=3, alignment=TA_LEFT),
    'b': ParagraphStyle('b', fontName='DV', fontSize=8.4, leading=11, textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=1),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=7.3, leading=8.9, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='DVB', fontSize=7.3, leading=8.9, textColor=INK),
    'note': ParagraphStyle('note', fontName='DV', fontSize=7.3, leading=9.2, textColor=MUTED, spaceAfter=3),
    'code': ParagraphStyle('code', fontName='DVM', fontSize=7.2, leading=9.4, textColor=INK, leftIndent=6,
                           backColor=colors.HexColor('#f6f8fa'), borderPadding=4, spaceBefore=2, spaceAfter=6),
}
story = []


def h1(t): story.append(Paragraph(t, S['h1']))
def h2(t): story.append(Paragraph(t, S['h2']))
def p(t): story.append(Paragraph(t, S['p']))
def note(t): story.append(Paragraph(t, S['note']))
def bullets(items):
    for t in items:
        story.append(Paragraph(t, S['b'], bulletText='•'))
    story.append(Spacer(1, 3))
def code(t): story.append(Paragraph(t.replace('\n', '<br/>').replace(' ', '&nbsp;'), S['code']))
def c(t): return f'<font face="DVM" size="7.4">{t}</font>'


def table(rows, widths, header=True, bold_rows=(), title=None):
    data = [[Paragraph(str(x), S['cellb'] if (header and i == 0) or i in bold_rows else S['cell']) for x in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, RULE), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
          ('TOPPADDING', (0, 0), (-1, -1), 1.1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.1)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), HEAD_BG))
    t.setStyle(TableStyle(st))
    if title:
        story.append(KeepTogether([Paragraph(title, S['h2']), t, Spacer(1, 4)]))
    else:
        story.append(t)
        story.append(Spacer(1, 4))


# ------------------------------------------------------------------------------------------
story.append(Paragraph('Mimosa Reproduction', S['title']))
story.append(Paragraph('Zeynel Yıldırım and Bora Kafadar · September 2026', S['sub']))

h1('1. Reproducing the paper')
p(f'Mimosa (Bi et al., Nucleic Acids Research 52:11455, 2024). Released model {c("training/model_mimosa.pth")} on the '
  f'paper\'s test sets {c("data/miRAW_Test0–9.txt")} (548 positive + 548 negative pairs each; the 548 negatives are '
  f'identical in all 10 sets), step size 5.')
table([
    ['', 'Accuracy', 'PPV', 'Recall', 'Specificity', 'F1', 'NPV'],
    ['Paper, Figure 3B (mean of 10 sets)', '0.7568', '0.6771', '0.9332', '0.5803', '0.7932', '0.8971'],
    ['Our run (mean of 10 sets)', '0.7567', '0.6897 (AP 0.6771)', '0.9332', '0.5803', '0.7932', '0.8972'],
], [50, 18, 32, 18, 22, 18, 18])
note('The original code computes "PPV" with average_precision_score (AP) instead of precision. The paper\'s 0.6771 '
     'equals the AP implied by its own recall and specificity; the true precision is 0.690. We report both from here on.')

h1('2. Changes to the original code')
p(f'All experiments keep Mimosa\'s 40-nt architecture. Only {c("training/Mimosa.py")} was modified:')
table([
    ['Change', 'Why', 'Effect on the method'],
    ['Command-line interface (--mode, --data, --stepsize, --threads), progress output, model loaded next to the '
     'script on CPU', 'Running the scripts on the cluster', 'None'],
    ['PPV = precision_score; AP printed as "paper PPV"', 'Original PPV was AP (Section 1)', 'Reported metric only'],
    ['Authors\' commented-out torch.save line enabled', 'Without it training saves no model; the paper selects '
     'the minimum-validation-loss model', 'None (restores the documented behaviour)'],
    ['--epochs argument, default 40', 'Needed for the 120-epoch run', 'None at the default'],
], [72, 62, 44])
p('Not changed: architecture (16 encoder blocks, 8 heads, dim 64), input encoding, base-pairing features '
  '(Smith–Waterman of miRNA nt 1–10 against the 30-nt core), loss (CrossEntropy on softmax output, as in the '
  'original), Adam lr 1e-4 / weight decay 1e-5, batch 256, dropout 0.1, test procedure.')
table(title='New files', rows=[
    ['Path', 'Purpose'],
    [c('training/prepare_dataset.py'), 'Builds our dataset files (Section 3)'],
    [c('training/predict_site40.py'), 'Gene-level test on our files by calling the authors\' get_cts / kmers_predict / '
     'decision_for_whole'],
    [c('scripts/run_prepare.slurm, run_train_site40.slurm, run_predict_site40.slurm, run_mimosa_all.slurm'), 'Slurm jobs'],
    [c('experiments/cross_eval, window_analysis, composition'), 'Diagnostic experiments (Section 6)'],
], widths=[78, 100])

# ------------------------------------------------------------------------------------------
h1('3. Data: sources, preparation and use')
h2('3.1 Raw inputs (outside the repository)')
table([
    ['Input', 'Content', 'Used for'],
    [c('~/training_chunks/'), '346 JSONL chunks, 34,560,002 rows: 17,280,000 RNA–protein, 11,987,208 miRNA–mRNA, the rest '
     'other RNA–RNA types. A resampled training stream built from data_with_negatives final_train (every miRNA–mRNA row '
     'is in final_train; each pair repeated 4.3× on average, up to 200×)', 'Training pairs (miRNA–mRNA rows only)'],
    [c('~/data_with_negatives/rna_rna/miRNA_mRNA/'), 'final_test_unseen_{pair,source,target}.jsonl', 'Test splits'],
    [c('~/fasta_files.zip'), 'rna.fa (1.85 GB; read directly from the zip, T→U)', 'All sequences'],
], [46, 97, 35])
p('Split semantics, checked against final_train positives: <b>unseen_source</b> = miRNAs never seen in training '
  '(targets 86 % seen); <b>unseen_target</b> = targets never seen (miRNAs 92 % seen); <b>unseen_pair</b> = '
  'neither seen. No test pair occurs in training. Negatives are made by swapping the miRNA or the target of a positive '
  'pair, so the same miRNAs and targets appear with both labels.')

table(title='3.2 Preparation (training/prepare_dataset.py)', rows=[
    ['Step', 'Rows'],
    ['miRNA–mRNA rows in training_chunks', '11,987,208'],
    ['Unique (miRNA, target, label)', '2,761,271'],
    ['Dropped: miRNA sequence not in rna.fa (mostly mirXplain isomiR names)', '78,345'],
    ['Dropped: target not in rna.fa (mostly mirXplain chromosome coordinates, some ENSG ids)', '51,556'],
    ['Dropped: miRNA longer than 30 nt (precursors; Mimosa pads miRNAs to 30 nt)', '30,147'],
    ['Dropped: target with non-ACGU characters / shorter than 40 nt', '1,751 / 333'],
    ['Dropped: label conflicts / pairs also in a test split', '0 / 0'],
    ['Kept', '2,599,139'],
    ['<b>Used for training</b>: pairs whose target is already a 40-nt site (given as a sequence in the data, mostly '
     'competitor_deeptargetpro), split randomly into train / validation (7 %, seed 42, stratified by label)',
     '<b>39,770</b> = 36,986 (20,647 pos / 16,339 neg) / 2,784 (1,554 / 1,230)'],
], widths=[128, 50])
p('Mimosa\'s input is a 40-nt site, so only these rows can be used for training without changing the method. They have '
  'the same layout as the paper\'s training sites (30-nt core + 5 nt flanks, 5\'→3\'; the seed match sits at positions '
  '25–29 in both). Test splits are written in full, in their original row order, with a '
  f'{c("mimosa_status")} column marking rows Mimosa cannot process (sequence missing, miRNA over 30 nt, non-ACGU):')
table([
    ['Split', 'Rows', 'Processable', '%'],
    ['test_unseen_pair', '44,580', '42,476', '95.3'],
    ['test_unseen_source', '398,828', '379,727', '95.2'],
    ['test_unseen_target', '415,348', '394,110', '94.9'],
], [70, 42, 42, 24])
table(title='3.3 Where each prepared file is used (data/custom/)', rows=[
    ['File', 'Used in'],
    [c('site40_Train_Validation.txt'), 'Training, both runs (same format as the paper\'s miRAW_Train_Validation.txt)'],
    [c('pairs_test_unseen_{pair,source,target}.tsv'), 'Gene-level tests (Section 5) and experiment A'],
    [c('sequences.fa'), 'Sequences for all of the above (168,646 sequences)'],
    [c('data/miRAW_*'), 'Paper reproduction, experiment B and the analyses in Section 6'],
], widths=[70, 108])

# ------------------------------------------------------------------------------------------
h1('4. Training')
p('Original perform_train, unchanged, on site40_Train_Validation.txt (the paper trains on 58,793 miRAW sites). '
  'Validation metrics are site-level. <b>The 120-epoch model was used for all tests.</b>')
table([
    ['Run', 'Best epoch', 'Val loss', 'Acc', 'PPV', 'Recall', 'Spec', 'F1', 'NPV'],
    ['40 epochs (paper)', '40', '0.467', '0.843', '0.844', '0.882', '0.793', '0.862', '0.841'],
    ['120 epochs', '113', '0.454', '0.855', '0.845', '0.907', '0.790', '0.875', '0.870'],
], [34, 20, 18, 16, 16, 18, 16, 16, 16])

h1('5. Gene-level test')
p('Procedure exactly as in the paper: target reversed to 3\'→5\', cut into 40-nt windows with step 5 (get_cts), every '
  'window scored with the authors\' features (kmers_predict), pair positive if any window scores above 0.5 '
  '(decision_for_whole).')
note('kmers_predict is called inside torch.no_grad() (saves memory only; predictions are unchanged).')
table([
    ['Split', 'Rows', 'n', 'Acc', 'PPV', 'AP', 'Recall', 'Spec', 'F1', 'NPV'],
    ['test_unseen_pair', 'processable', '42,476', '0.514', '0.511', '0.510', '0.881', '0.141', '0.647', '0.537'],
    ['', 'all rows, unprocessable = negative', '44,580', '0.518', '0.511', '0.509', '0.847', '0.189', '0.637', '0.552'],
    ['test_unseen_source', 'processable', '379,727', '0.516', '0.512', '0.511', '0.881', '0.143', '0.648', '0.540'],
    ['', 'all rows, unprocessable = negative', '398,828', '0.520', '0.512', '0.510', '0.847', '0.193', '0.638', '0.558'],
    ['test_unseen_target', 'processable', '394,110', '0.512', '0.509', '0.508', '0.882', '0.136', '0.645', '0.530'],
    ['', 'all rows, unprocessable = negative', '415,348', '0.515', '0.509', '0.508', '0.843', '0.186', '0.635', '0.543'],
    ['Paper, miRAW', '10 × 1,096', '', '0.757', '0.690', '0.677', '0.933', '0.580', '0.793', '0.897'],
], [32, 50, 16, 11, 11, 11, 12, 11, 11, 11], bold_rows=(7,))

# ------------------------------------------------------------------------------------------
h1('6. Diagnostic experiments')
p('All use the original test procedure. Code and inputs in experiments/, outputs in results/experiments/.')

table(title='6.1 Effect of target length (test_unseen_pair, 120-epoch model)', rows=[
    ['Target length (kb)', '&lt;0.5', '0.5–1', '1–1.5', '1.5–2', '2–2.5', '2.5–3', '3–3.5', '3.5–4', '≥4'],
    ['Negatives called positive', '0.25', '0.77', '0.84', '0.87', '0.86', '0.88', '0.88', '0.89', '0.91'],
], widths=[40, 15, 15, 15, 15, 15, 15, 15, 15, 15])
p('False positives rise with the number of windows. Length alone does not explain the gap to the paper, though: '
  'our targets have 505 windows on average (median 2,612 nt), the miRAW test targets 408 (median 1,409 nt).')

table(title='6.2 Cross-evaluation', rows=[
    ['Test data', 'Model', 'Acc', 'PPV', 'Recall', 'Spec', 'F1', 'NPV'],
    ['A: test_unseen_pair, 4,000 random pairs (2,000 per label)', 'Authors\' model', '0.499', '0.499', '0.907', '0.090', '0.644', '0.493'],
    ['', 'Our model', '0.521', '0.512', '0.889', '0.152', '0.650', '0.580'],
    ['B: the paper\'s 10 miRAW test sets (mean)', 'Authors\' model', '0.757', '0.690', '0.933', '0.580', '0.793', '0.897'],
    ['', 'Our model', '0.513', '0.507', '0.894', '0.131', '0.647', '0.556'],
], widths=[66, 24, 13, 13, 14, 13, 13, 13])
p('Both problems are present: the authors\' model fails on our data, and our model fails on the paper\'s data.')

table(title='6.3 Window-level behaviour (100 negative + 100 positive pairs per dataset)', rows=[
    ['Data', 'Pairs', 'Model', 'Windows > 0.5', 'Positive windows per transcript (median)', 'Called positive'],
    ['miRAW', 'negative', 'Authors\'', '13 %', '0', '39 %'],
    ['', '', 'Ours', '18 %', '30', '89 %'],
    ['', 'positive', 'Authors\'', '63 %', '154', '95 %'],
    ['', '', 'Ours', '18 %', '22', '91 %'],
    ['Ours', 'negative', 'Authors\' / ours', '60 % / 11 %', '294 / 7', '88 % / 85 %'],
    ['', 'positive', 'Authors\' / ours', '59 % / 10 %', '282 / 11', '89 % / 93 %'],
], widths=[18, 18, 26, 24, 56, 36])
p('Our model fires on a steady ~18 % of windows whatever the pair, so nearly every transcript gets at least one '
  'positive window. The authors\' model fires on 63 % of all windows of miRAW positives, far more than the few binding '
  'sites a 3\'UTR contains, i.e. it reacts to something other than individual sites.')

table(title='6.4 What predicts the miRAW label without the model?', rows=[
    ['Feature (5-fold logistic regression or direct score)', 'miRAW ROC-AUC', 'Ours ROC-AUC'],
    ['Target length', '0.48', '0.51'],
    ['Target length + GC content', '0.55', '0.51'],
    ['Target length + mono- and dinucleotide frequencies', '0.53', '0.50'],
    ['<b>miRNA alone</b>: its positive rate in the paper\'s training data', '<b>0.86</b>', '–'],
], widths=[110, 34, 34])
p('The target transcript carries no label signal, but the miRNA does: miRNAs of miRAW test positives were positive '
  'in 84 % of their training examples, miRNAs of test negatives in 54 % (761 vs 116 distinct miRNAs; 5,379 vs 548 pairs).')

table(title='6.5 miRNA swap on miRAW (150 targets per group)', rows=[
    ['Target', 'miRNA', 'Authors\' model: called positive', 'Our model'],
    ['From a negative pair', 'its own', '0.413', '0.873'],
    ['From a negative pair', '<b>taken from a positive pair</b>', '<b>0.947</b>', '0.933'],
    ['From a positive pair', 'its own', '0.933', '0.873'],
    ['From a positive pair', '<b>taken from a negative pair</b>', '<b>0.387</b>', '0.873'],
], widths=[42, 52, 50, 34])
p('The authors\' model\'s decision follows the miRNA: a negative target becomes positive at the rate of real positives '
  'once it gets a "positive" miRNA, and vice versa. The published performance therefore rests largely on a miRNA-identity '
  'bias of the miRAW benchmark, which our benchmark does not have.')

h1('7. Deviations from the paper')
bullets([
    'Training data: 39,770 40-nt sites from our dataset instead of 58,793 miRAW sites.',
    'Epochs: the tested model was trained for 120 epochs (the 40-epoch model was also trained; Section 4).',
    'Validation split: random 7 % of our training rows (the paper follows TargetNet\'s split).',
    'Test data: our unseen splits (several species, targets are transcript 3\' ends up to 4,076 nt, swap negatives) '
    'instead of miRAW (human 3\'UTRs). About 5 % of test pairs cannot be processed and are reported both ways.',
])

h1('8. Conclusions')
bullets([
    'Mimosa trained on our data with the paper\'s method does not generalise to gene-level prediction '
    '(accuracy ~0.52 on all three splits, 0.51 even on the paper\'s own test sets).',
    'The published model is also at chance on our benchmark (accuracy 0.499).',
    'On miRAW the published model\'s decisions are driven by miRNA identity, which the miRAW benchmark leaks '
    '(ROC-AUC 0.86 from the miRNA alone). Our benchmark, where each miRNA and target appears with both labels, '
    'removes this shortcut; the published results do not transfer to it.',
])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DV', 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, 'Mimosa Reproduction')
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f'{doc.page}')
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm,
                        bottomMargin=14 * mm, title='Mimosa Reproduction',
                        author='Zeynel Yıldırım and Bora Kafadar')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', OUT)

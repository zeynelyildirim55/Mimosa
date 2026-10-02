'''Builds report/Mimosa_MIL_report.pdf. Needs reportlab (not part of the mimosa env).'''
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mimosa_MIL_report.pdf')
INK, MUTED, RULE, HEAD_BG = colors.HexColor('#1f2328'), colors.HexColor('#57606a'), colors.HexColor('#d0d7de'), colors.HexColor('#eef1f4')

S = {
    'title': ParagraphStyle('title', fontName='DVB', fontSize=16, leading=19, textColor=INK, spaceAfter=3),
    'sub': ParagraphStyle('sub', fontName='DV', fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=6),
    'h1': ParagraphStyle('h1', keepWithNext=1, fontName='DVB', fontSize=11.5, leading=14, textColor=INK, spaceBefore=7, spaceAfter=3),
    'h2': ParagraphStyle('h2', keepWithNext=1, fontName='DVB', fontSize=9.5, leading=12, textColor=INK, spaceBefore=4, spaceAfter=2),
    'p': ParagraphStyle('p', fontName='DV', fontSize=8.4, leading=11, textColor=INK, spaceAfter=3, alignment=TA_LEFT),
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


def table(rows, widths, header=True, bold_rows=(), title=None, title_style='h2'):
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
        story.append(KeepTogether([Paragraph(title, S[title_style]), t, Spacer(1, 4)]))
    else:
        story.append(t)
        story.append(Spacer(1, 4))


# ------------------------------------------------------------------------------------------
story.append(Paragraph('Mimosa MIL Experiment', S['title']))
story.append(Paragraph('Zeynel Yıldırım · September 2026', S['sub']))

h1('1. Motivation')
p('In the reproduction (first report), Mimosa trained with the paper\'s method on the 39,770 40-nt sites of our '
  'dataset was at chance at gene level (accuracy ~0.52, specificity ~0.14): trained only on binding sites, it fires '
  'on ordinary windows of a transcript. It also used only 1.5 % of our miRNA–mRNA pairs. This experiment keeps '
  'Mimosa\'s architecture and test rule and changes only the training procedure, so that all 2.4M gene-level pairs '
  'can be used.')

h1('2. Method: multiple-instance training')
p('Mimosa calls a pair positive if at least one 40-nt window scores above 0.5. Training is made consistent with that '
  'rule: each pair is a bag of windows, the model picks the window it scores highest, and that window is trained with '
  'the pair\'s label. For a positive pair, at least one window must score high; for a negative pair, even its '
  'highest-scoring window must score low, so the windows that cause false positives are exactly the ones trained '
  'against. Each round has three steps:')
bullets([
    '<b>Selection</b>: every window of every training pair is scored with the current model (no gradients); each '
    'pair keeps its best window.',
    '<b>Training</b>: one epoch (three in rounds 7–9) over the selected windows with the original training loop.',
    '<b>Validation</b>: gene-level metrics on a validation subset with the updated model.',
])
p('The model is trained <b>from scratch</b>: the initial model is the one the original perform_train builds (same '
  'architecture, same random seed). In round 1 the best window of an untrained model is arbitrary, so a random window '
  'per pair is used instead. Nine rounds were run: rounds 1–6 with one epoch on the selection, rounds 7–9 with three '
  'epochs (optimizer state carried over), added because the training loss was still falling.')
table([
    ['Unchanged from Mimosa', 'Changed'],
    ['Architecture: 16 encoder blocks, 8 heads, dim 64, 40-nt input, Linear(40, 12)',
     'Training data: 2,417,199 gene-level pairs instead of 40-nt sites'],
    ['Input encoding and base-pairing features (the authors\' get_cts, get_interaction_map_for_test)',
     'Training procedure: the training window of each pair is re-selected by the model before every epoch'],
    ['Loss (CrossEntropy on the softmax output), Adam lr 1e-4, weight decay 1e-5, batch 256',
     'Nine rounds: one epoch each in rounds 1–6, three in rounds 7–9'],
    ['Test procedure: step 5, pair positive if any window > 0.5 (decision_for_whole)', ''],
], [89, 89])

h1('3. Code')
p(f'{c("training/Mimosa.py")} is not modified; the experiment only adds files, which call the authors\' functions:')
table([
    ['Path', 'Purpose'],
    [c('training/mil_features.py'), 'Computes the authors\' window features for every pair once (1.2 billion training '
     'windows) and stores them compactly (13 bytes per window)'],
    [c('training/mil_select.py'), 'Scores all windows on GPU and keeps each pair\'s best window; also used for '
     'validation and testing (pair positive if best window > 0.5, i.e. decision_for_whole)'],
    [c('training/mil_train.py'), 'Epochs on the selected windows with the original training loop'],
    [c('training/mil_common.py, mil_metrics.py'), 'Feature storage helpers; gene-level metrics'],
    [c('scripts/run_mil_*.slurm, submit_mil_scratch.sh, submit_mil_test.sh'), 'Slurm jobs; the rounds are chained with job '
     'dependencies'],
    [c('experiments/composition/mil_shuffle.py'), 'miRNA swap test (Section 7)'],
], [72, 106])
p('Checks: the stored and rebuilt features are identical to the inputs kmers_predict builds (20/20 pairs, 7,996 '
  'windows), and the selection\'s gene-level decisions are identical to kmers_predict (20/20 pairs). The 120-epoch site '
  'model from the first report, run through the new path on the validation subset, gives metrics consistent with the '
  'original path on test_unseen_pair (accuracy 0.515 vs 0.514, specificity 0.148 vs 0.141).')

h1('4. Data')
table([
    ['Use', 'Rows', 'Pairs', 'Windows'],
    ['Training', f'{c("pairs_train_val.tsv")}, split = train (2,380,213 gene-level + 36,986 40-nt sites; 50.6 % positive)',
     '2,417,199', '1,203.1 M (median 510 per pair)'],
    ['Validation', 'every 9th row of split = val (the random 7 % validation split)', '20,216', '10.0 M'],
    ['Test', 'test_unseen_pair / source / target, processable rows', '42,476 / 379,727 / 394,110', '21.5 M / 192.0 M / 199.4 M'],
], [22, 92, 32, 32])
p(f'Same prepared files as in the first report ({c("data/custom/")}). Features: {c("data/custom/mil_features/")} '
  '(11 GB on disk).')

table(title='5. Training rounds', title_style='h1', rows=[
    ['Round', 'Training window', 'Epochs', 'Train loss', 'Acc', 'PPV', 'Recall', 'Spec', 'F1', 'NPV'],
    ['0', '— (untrained model)', '—', '—', '0.502', '0.502', '1.000', '0.000', '0.668', '0.000'],
    ['1', 'random', '1', '0.693', '0.503', '0.502', '0.999', '0.003', '0.669', '0.757'],
    ['2', 'best (model of round 1)', '1', '0.675', '0.508', '0.505', '0.990', '0.023', '0.669', '0.693'],
    ['3', 'best (round 2)', '1', '0.666', '0.589', '0.555', '0.918', '0.258', '0.692', '0.758'],
    ['4', 'best (round 3)', '1', '0.588', '0.733', '0.732', '0.736', '0.729', '0.734', '0.733'],
    ['5', 'best (round 4)', '1', '0.565', '0.744', '0.751', '0.734', '0.754', '0.742', '0.738'],
    ['6', 'best (round 5)', '1', '0.561', '0.746', '0.747', '0.746', '0.746', '0.747', '0.745'],
    ['7', 'best (round 6)', '3', '0.558 → 0.554', '0.747', '0.736', '0.775', '0.719', '0.755', '0.761'],
    ['8', 'best (round 7)', '3', '0.549 → 0.547', '0.756', '0.753', '0.764', '0.747', '0.759', '0.759'],
    ['9', 'best (round 8)', '3', '0.552 → 0.550', '0.757', '0.757', '0.759', '0.755', '0.758', '0.757'],
], widths=[12, 36, 13, 26, 14, 14, 15, 14, 14, 14], bold_rows=(10,))
note('Validation metrics are gene-level (20,216 pairs). Selection took ~3.4 h per round on 6 GTX 1080 Ti + 2 RTX 4090 '
     '(rounds 2–6), ~8.5 h on the 1080 Ti alone (rounds 7–8) and ~6.4 h in round 9, when an RTX 4090 became free. '
     'One epoch takes 9 min on an RTX 4090 and 23 min on a 1080 Ti. Feature precomputation: ~1,000 CPU-hours, once. '
     'Loss values are not comparable across rounds, because the training windows change every round.')
p('Round 1 learns nothing: a random window of a positive pair is almost never its binding site. From round 2 the '
  'model\'s own selections carry signal; rounds 3–4 show a sharp jump and rounds 4–6 plateau near 0.745. Rounds 7–9 '
  'gain about one point, and rounds 8 and 9 are equal within noise. <b>The round-9 model was chosen</b> (highest '
  'validation accuracy, recall ≈ specificity) <b>and tested</b>; the round-6 model had been tested earlier.')
note('The training loss stays high because the original code applies softmax twice (inside the model and again in '
     'CrossEntropyLoss), which puts a floor of 0.313 on it even for a perfect model. It was kept for fidelity.')

h1('6. Gene-level test')
p('Same test procedure and same pairs as in the first report.')
table([
    ['Split', 'Model', 'Rows', 'n', 'Acc', 'PPV', 'AP', 'Recall', 'Spec', 'F1', 'NPV'],
    ['test_unseen_pair', 'MIL, round 9', 'processable', '42,476', '0.767', '0.759', '0.705', '0.789', '0.745', '0.774', '0.776'],
    ['', 'MIL, round 9', 'all rows', '44,580', '0.759', '0.759', '0.697', '0.759', '0.759', '0.759', '0.759'],
    ['', 'MIL, round 6', 'processable', '42,476', '0.752', '0.745', '0.690', '0.774', '0.730', '0.759', '0.760'],
    ['', 'Site model', 'processable', '42,476', '0.514', '0.511', '0.510', '0.881', '0.141', '0.647', '0.537'],
    ['test_unseen_source', 'MIL, round 9', 'processable', '379,727', '0.771', '0.762', '0.710', '0.796', '0.746', '0.779', '0.782'],
    ['', 'MIL, round 9', 'all rows', '398,828', '0.763', '0.762', '0.701', '0.766', '0.761', '0.764', '0.764'],
    ['', 'MIL, round 6', 'processable', '379,727', '0.756', '0.748', '0.695', '0.781', '0.731', '0.764', '0.766'],
    ['', 'Site model', 'processable', '379,727', '0.516', '0.512', '0.511', '0.881', '0.143', '0.648', '0.540'],
    ['test_unseen_target', 'MIL, round 9', 'processable', '394,110', '0.763', '0.759', '0.702', '0.777', '0.749', '0.768', '0.768'],
    ['', 'MIL, round 9', 'all rows', '415,348', '0.753', '0.759', '0.692', '0.744', '0.763', '0.751', '0.748'],
    ['', 'MIL, round 6', 'processable', '394,110', '0.749', '0.745', '0.688', '0.764', '0.734', '0.754', '0.754'],
    ['', 'Site model', 'processable', '394,110', '0.512', '0.509', '0.508', '0.882', '0.136', '0.645', '0.530'],
], [36, 24, 25, 17, 11, 11, 11, 12, 11, 11, 11], bold_rows=(1, 5, 9))
note('"Site model": the 120-epoch model of the first report. "all rows": unprocessable pairs counted as negative '
     f'predictions. Results: {c("runs/mil_scratch/test_round9/&lt;split&gt;/metrics.json")} (round 9) and '
     f'{c("runs/mil_scratch/test/&lt;split&gt;/metrics.json")} (round 6).')
p('The MIL model is balanced (recall ≈ specificity), and its results are the same for the three kinds of '
  'generalisation (new miRNA, new target, both new) and match the validation subset. Round 9 is 1.4–1.5 points more '
  'accurate than round 6 on every split, as on validation (+1.1). Its accuracy being close to the paper\'s 0.757 on '
  'miRAW is a coincidence: the benchmarks differ.')

h1('7. Does the decision depend on the pair? (miRNA swap)')
p('Same test as in the first report: targets are kept and miRNAs are swapped between positive and negative pairs '
  '(150 targets per group). On our data the test uses test_unseen_pair, whose miRNAs and targets were never seen in '
  'training.')
table([
    ['Target', 'miRNA', 'MIL (round 9), our data', 'MIL (round 9), miRAW', 'Authors\' model, miRAW (first report)'],
    ['From a negative pair', 'its own', '0.293', '0.180', '0.413'],
    ['From a negative pair', 'taken from a positive pair', '0.220', '0.280', '<b>0.947</b>'],
    ['From a positive pair', 'its own', '0.840', '0.453', '0.933'],
    ['From a positive pair', 'taken from a negative pair', '0.193', '0.180', '0.387'],
], [34, 42, 34, 32, 36])
note('Share of pairs called positive. Swapped pairs have no known label; random miRNA–target combinations are '
     'expected to be mostly non-interacting.')
p('On our data a positive target keeps its high call only with its own miRNA: with another miRNA it falls to the rate '
  'of negatives (0.840 → 0.193), and a miRNA taken from a positive pair does not raise a negative target (0.293 → '
  '0.220). The decision depends on the miRNA–target pair, not on either one alone, which is the opposite of the '
  'authors\' model on miRAW. On miRAW the MIL model behaves the same way apart from a small miRNA effect (0.180 → '
  '0.280); it is more conservative there (positives called at 0.453), as expected from a model trained on a '
  'different benchmark. The round-6 model gave the same picture on our data (0.807 → 0.233 and 0.267 → 0.227).')

h1('8. Limitations')
bullets([
    'The training procedure differs from the paper (gene-level data, window re-selection between epochs): these are '
    'results for Mimosa\'s architecture with multiple-instance training, not a reproduction.',
    'Single run; the variance across random seeds is unknown.',
    'Round 1 uses a random window. Rounds 7–9 (three epochs each) were added because the loss was still falling '
    'after round 6. Two MIL models (rounds 6 and 9) were tested; round 9 was chosen on validation before testing.',
    'The swap test uses a sample of 150 targets per group, not a full evaluation.',
])

h1('9. Conclusions')
bullets([
    'With the same architecture, features and test rule, multiple-instance training on all gene-level pairs gives a '
    'balanced model: accuracy 0.76–0.77, recall 0.78–0.80 and specificity ≈0.75 on all three test splits, against '
    'accuracy 0.51–0.52 and specificity 0.14 with the paper\'s training method.',
    'Its decisions depend on the miRNA–target pair, not on miRNA identity, also for miRNAs and targets never seen in '
    'training.',
    'Since the architecture is unchanged, the gap between the two reports comes from the training data and procedure.',
])
story.pop()  # drop the list's trailing spacer, which alone would spill onto an empty page


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DV', 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, 'Mimosa MIL Experiment')
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f'{doc.page}')
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm,
                        bottomMargin=14 * mm, title='Mimosa MIL Experiment',
                        author='Zeynel Yıldırım')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', OUT)

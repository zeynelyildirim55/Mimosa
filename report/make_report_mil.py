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
    'p': ParagraphStyle('p', allowWidows=0, fontName='DV', fontSize=8.4, leading=11, textColor=INK, spaceAfter=3, alignment=TA_LEFT),
    'b': ParagraphStyle('b', fontName='DV', fontSize=8.4, leading=11, textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=1),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=7.3, leading=8.9, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='DVB', fontSize=7.3, leading=8.9, textColor=INK),
    'note': ParagraphStyle('note', fontName='DV', fontSize=7.3, leading=9.2, textColor=MUTED, spaceAfter=3),
    'code': ParagraphStyle('code', fontName='DVM', fontSize=7.2, leading=9.4, textColor=INK, leftIndent=6,
                           backColor=colors.HexColor('#f6f8fa'), borderPadding=4, spaceBefore=2, spaceAfter=6),
}
S['pk'] = ParagraphStyle('pk', parent=S['p'], keepWithNext=1)
story = []


def h1(t): story.append(Paragraph(t, S['h1']))
def h2(t): story.append(Paragraph(t, S['h2']))
def p(t): story.append(Paragraph(t, S['p']))
def pk(t): story.append(Paragraph(t, S['pk']))
def note(t): story.append(Paragraph(t, S['note']))
def bullets(items):
    for t in items:
        story.append(Paragraph(t, S['b'], bulletText='•'))
    story.append(Spacer(1, 3))
def code(t): story.append(Paragraph(t.replace('\n', '<br/>').replace(' ', '&nbsp;'), S['code']))
def c(t): return f'<font face="DVM" size="7.4">{t}</font>'


def table(rows, widths, header=True, bold_rows=(), title=None, title_style='h2', keep=False, intro=None):
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
        head = [Paragraph(title, S[title_style])] + ([Paragraph(intro, S['p'])] if intro else [])
        story.append(KeepTogether(head + [t, Spacer(1, 4)]))
    elif keep:
        story.append(KeepTogether([t, Spacer(1, 4)]))
    else:
        story.append(t)
        story.append(Spacer(1, 4))


# ------------------------------------------------------------------------------------------
story.append(Paragraph('Mimosa MIL Experiment', S['title']))
story.append(Paragraph('Zeynel Yıldırım and Bora Kafadar · September 2026', S['sub']))

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
    ['9', 'best (round 8)', '3', '0.552 → 0.550', '0.757', '0.757', '0.759', '0.755', '0.758', '0.757'],
], widths=[12, 36, 13, 26, 14, 14, 15, 14, 14, 14])
note('Validation metrics are gene-level (20,216 pairs).')

table(title='6. Gene-level test', title_style='h1', intro='Same test procedure and same pairs as in the first report.', rows=[
    ['Split', 'Rows', 'n', 'Acc', 'PPV', 'AP', 'Recall', 'Spec', 'F1', 'NPV'],
    ['test_unseen_pair', 'processable', '42,476', '0.767', '0.759', '0.705', '0.789', '0.745', '0.774', '0.776'],
    ['', 'all rows, unprocessable = negative', '44,580', '0.759', '0.759', '0.697', '0.759', '0.759', '0.759', '0.759'],
    ['test_unseen_source', 'processable', '379,727', '0.771', '0.762', '0.710', '0.796', '0.746', '0.779', '0.782'],
    ['', 'all rows, unprocessable = negative', '398,828', '0.763', '0.762', '0.701', '0.766', '0.761', '0.764', '0.764'],
    ['test_unseen_target', 'processable', '394,110', '0.763', '0.759', '0.702', '0.777', '0.749', '0.768', '0.768'],
    ['', 'all rows, unprocessable = negative', '415,348', '0.753', '0.759', '0.692', '0.744', '0.763', '0.751', '0.748'],
], widths=[32, 50, 16, 11, 11, 11, 12, 11, 11, 11])

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
  'different benchmark.')

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
                        author='Zeynel Yıldırım and Bora Kafadar')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', OUT)

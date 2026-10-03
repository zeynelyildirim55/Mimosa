'''Builds report/Mimosa_report.pdf. Needs reportlab (not part of the mimosa env).'''
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Mimosa_report.pdf')
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
story.append(Paragraph('Zeynel Yıldırım and Bora Kafadar · October 2026', S['sub']))

h1('1. Checking the code on the paper\'s data')
p(f'Mimosa (Bi et al., Nucleic Acids Research 52:11455, 2024). Released model {c("training/model_mimosa.pth")} on the '
  f'paper\'s test sets {c("data/miRAW_Test0–9.txt")} (548 positive + 548 negative pairs each), step size 5.')
table([
    ['', 'Accuracy', 'PPV', 'Recall', 'Specificity', 'F1', 'NPV'],
    ['Paper, Figure 3B (mean of 10 sets)', '0.7568', '0.6771', '0.9332', '0.5803', '0.7932', '0.8971'],
    ['Released model, our run (mean of 10 sets)', '0.7567', '0.6897 (AP 0.6771)', '0.9332', '0.5803', '0.7932', '0.8972'],
], [50, 18, 32, 18, 22, 18, 18])
note('The original code computes "PPV" with average_precision_score (AP) instead of precision. The paper\'s 0.6771 '
     'equals the AP implied by its own recall and specificity; the true precision is 0.690. We report both.')

h1('2. Changes to the original code')
p(f'The 40-nt architecture, the training procedure and the test procedure are the authors\'. Only {c("training/Mimosa.py")} '
  'was modified:')
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
  'original), Adam lr 1e-4 / weight decay 1e-5, batch 256, dropout 0.1, seed 1234, test procedure. Training runs on CPU, '
  'as in the original code.')
table(title='New files', rows=[
    ['Path', 'Purpose'],
    [c('training/prepare_dataset.py'), 'Builds our dataset files (Section 3)'],
    [c('training/make_site40_valsplits.py'), 'Training file: our 40-nt training sites + the 40-nt sites of the validation splits'],
    [c('training/predict_site40.py'), 'Gene-level test on CPU by calling the authors\' get_cts / kmers_predict / decision_for_whole'],
    ['GPU scoring scripts in ' + c('training/') + ' (listed in WALKTHROUGH.md)', 'The same gene-level test on GPU from '
     'precomputed window features (Section 5)'],
    [c('training/metrics_by_kind.py'), 'Metrics by target type (Section 6)'],
    [c('slurm/*.slurm, experiments/cpu_gpu_check/'), 'Slurm jobs; CPU/GPU agreement check'],
    [c('WALKTHROUGH.md'), 'Every decision and command, in order'],
], widths=[78, 100])

# ------------------------------------------------------------------------------------------
h1('3. Data')
h2('3.1 Raw inputs (outside the repository)')
table([
    ['Input', 'Content', 'Used for'],
    [c('~/training_chunks/'), '346 JSONL chunks, 34,560,002 rows, of which 11,987,208 miRNA–mRNA. A resampled training '
     'stream built from data_with_negatives final_train (each pair repeated 4.3× on average)', 'Training'],
    [c('~/data_with_negatives/rna_rna/miRNA_mRNA/'), 'final_valid_unseen_{pair,source,target}.jsonl', 'Model selection'],
    ['', 'final_test_unseen_{pair,source,target}.jsonl', 'Test (used once)'],
    [c('~/fasta_files/rna.fa'), 'All RNA sequences (T→U)', 'Sequences'],
], [46, 97, 35])
p('<b>unseen_source</b> = miRNAs never seen in training; <b>unseen_target</b> = targets never seen; <b>unseen_pair</b> = '
  'neither seen. Negatives are made by swapping the miRNA or the target of a positive pair, so the same miRNAs and targets '
  'appear with both labels. The train split of data_with_negatives is not used.')

table(title='3.2 Training rows (training/prepare_dataset.py)', rows=[
    ['Step', 'Rows'],
    ['miRNA–mRNA rows in training_chunks', '11,987,208'],
    ['Unique (miRNA, target, label)', '2,761,271'],
    ['Dropped: miRNA / target sequence not in rna.fa', '78,345 / 51,556'],
    ['Dropped: miRNA longer than 30 nt (Mimosa pads miRNAs to 30 nt)', '30,147'],
    ['Dropped: target with non-ACGU characters / shorter than 40 nt', '1,751 / 333'],
    ['Dropped: label conflicts / pairs also in a validation or test split', '0 / 0'],
    ['Kept', '2,599,139'],
    ['<b>Used for training</b>: pairs whose target is already a 40-nt site (Mimosa\'s training input)',
     '<b>39,770</b> (22,201 pos / 17,569 neg)'],
], widths=[128, 50])
p('These sites have the same layout as the paper\'s training sites (30-nt core + 5-nt flanks, 5\'→3\'). Validation and '
  f'test splits are kept in full, in their original order, without deduplication or resampling; a {c("mimosa_status")} '
  'column marks rows Mimosa cannot process (sequence missing from rna.fa, miRNA outside 10–30 nt, non-ACGU).')
table([
    ['Split', 'Rows', 'Processable', '%', '40-nt-site rows (processable)'],
    ['valid_unseen_pair', '12,200', '11,281', '92.5', '180'],
    ['valid_unseen_source', '206,056', '191,156', '92.8', '3,615'],
    ['valid_unseen_target', '189,364', '179,660', '94.9', '2,579'],
    ['test_unseen_pair', '44,580', '42,476', '95.3', '607'],
    ['test_unseen_source', '398,828', '379,727', '95.2', '5,905'],
    ['test_unseen_target', '415,348', '394,110', '94.9', '5,057'],
], [46, 30, 30, 18, 54])

# ------------------------------------------------------------------------------------------
h1('4. Training and model selection')
p('Original perform_train, unchanged: 39,770 training sites (the paper uses 58,793 miRAW sites), validated each epoch on '
  'the 6,374 processable 40-nt-site rows of the three validation splits (3,419 pos / 2,955 neg). As in the paper, the model '
  'with the lowest validation loss is selected. 120 epochs (8 h 18 min on 24 CPU cores).')
table([
    ['Epoch', 'Val loss', 'Acc', 'PPV', 'Recall', 'Spec', 'F1', 'NPV'],
    ['36 (best of the first 40)', '0.4826', '0.823', '0.840', '0.828', '0.818', '0.834', '0.804'],
    ['<b>118 (selected)</b>', '<b>0.4784</b>', '0.830', '0.849', '0.831', '0.829', '0.840', '0.809'],
], [44, 20, 18, 18, 20, 18, 18, 18])
note('Validation loss is flat after about epoch 36; the selected model differs little from a 40-epoch model.')

h1('5. Gene-level test')
p('Procedure as in the paper: target reversed to 3\'→5\', cut into 40-nt windows with step 5 (get_cts), every window '
  'scored with the authors\' features (kmers_predict), pair positive if any window scores above 0.5 (decision_for_whole). '
  'The test was run once, with the selected model.')
note('For speed the windows were scored on GPU from precomputed features. On valid_unseen_pair (11,281 pairs) this gives '
     'the same decision as the original CPU code for every pair.')
table([
    ['Split', 'Rows', 'n', 'Acc', 'PPV', 'AP', 'Recall', 'Spec', 'F1', 'NPV'],
    ['test_unseen_pair', 'all, unprocessable = negative', '44,580', '0.517', '0.510', '0.509', '0.866', '0.168', '0.642', '0.556'],
    ['', 'processable', '42,476', '0.513', '0.510', '0.509', '0.900', '0.118', '0.651', '0.539'],
    ['test_unseen_source', 'all, unprocessable = negative', '398,828', '0.518', '0.511', '0.509', '0.867', '0.169', '0.643', '0.560'],
    ['', 'processable', '379,727', '0.514', '0.511', '0.510', '0.901', '0.118', '0.652', '0.540'],
    ['test_unseen_target', 'all, unprocessable = negative', '415,348', '0.514', '0.508', '0.507', '0.864', '0.165', '0.640', '0.548'],
    ['', 'processable', '394,110', '0.511', '0.508', '0.508', '0.903', '0.112', '0.651', '0.534'],
    ['Paper, miRAW', '10 × 1,096', '', '0.757', '0.690', '0.677', '0.933', '0.580', '0.793', '0.897'],
], [32, 46, 16, 11, 11, 11, 12, 11, 11, 11], bold_rows=(7,))

h1('6. 40-nt sites versus transcripts')
p('The same test predictions split by target type (decided before the test was run). A 40-nt target gives a single '
  'window, so its prediction is a site-level prediction.')
table([
    ['Split', 'Target', 'n', 'Acc', 'PPV', 'Recall', 'Spec', 'F1', 'NPV'],
    ['test_unseen_pair', '40-nt site', '625', '0.818', '0.855', '0.782', '0.857', '0.817', '0.784'],
    ['', 'transcript', '43,953', '0.512', '0.507', '0.867', '0.158', '0.640', '0.544'],
    ['test_unseen_source', '40-nt site', '6,083', '0.834', '0.867', '0.795', '0.874', '0.830', '0.805'],
    ['', 'transcript', '392,707', '0.513', '0.508', '0.868', '0.158', '0.641', '0.546'],
    ['test_unseen_target', '40-nt site', '5,593', '0.864', '0.840', '0.903', '0.826', '0.870', '0.893'],
    ['', 'transcript', '409,711', '0.510', '0.506', '0.864', '0.156', '0.638', '0.533'],
], [34, 24, 18, 15, 15, 16, 15, 15, 15])
note('All rows, unprocessable = negative. 84 targets that are raw sequences of other lengths are left out of this table.')
table(title='Negatives called positive by target length (valid_unseen_pair, selected model)', rows=[
    ['Target length (kb)', '&lt;0.5', '0.5–1', '1–1.5', '1.5–2', '2–2.5', '2.5–3', '3–3.5', '3.5–4', '≥4'],
    ['Negatives', '88', '194', '429', '787', '922', '989', '1,008', '972', '137'],
    ['Called positive', '0.20', '0.78', '0.88', '0.91', '0.93', '0.90', '0.93', '0.92', '0.91'],
], widths=[40, 15, 15, 15, 15, 15, 15, 15, 15, 15])
p('On unseen pairs the model separates 40-nt sites well (accuracy 0.82–0.86, specificity 0.83–0.87). On transcripts it '
  'calls almost every pair positive: our targets have about 510 windows on average, and with that many windows at least '
  'one exceeds 0.5 for about 90 % of negatives.')

h1('7. Deviations from the paper')
bullets([
    'Training data: 39,770 40-nt sites from our dataset instead of 58,793 miRAW sites.',
    'Validation: the 40-nt-site rows of our validation splits (the paper follows TargetNet\'s split). Epochs: 120 instead '
    'of 40; the selected epoch (118) is chosen by validation loss as in the paper.',
    'Test data: our unseen splits (several species, transcript targets up to 4,076 nt, swap negatives) instead of miRAW '
    '(human 3\'UTRs). About 5 % of test rows cannot be processed and are reported both ways.',
    'Test windows were scored on GPU from precomputed features; decisions are identical to the original CPU code on the '
    'validation pairs checked.',
])

h1('8. Conclusions')
bullets([
    'Mimosa trained on our data with the paper\'s method is at chance at gene level: accuracy 0.51–0.52 on all three '
    'test splits, with most negatives called positive.',
    'The model does learn to recognise binding sites: on 40-nt targets of unseen pairs, miRNAs and targets its accuracy '
    'is 0.82–0.86.',
    'The gap is consistent with applying a site classifier to whole transcripts with the any-window-above-0.5 rule: '
    'false positives grow with target length and reach about 90 % above 1 kb.',
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

#!/usr/bin/env python
"""Build docs/RECAP_STEAM_RLToken.pptx (python-pptx). Style follows docs/publication (DL Papers look).
Usage: python scripts/build_slides_recap_steam_rlt.py docs/RECAP_STEAM_RLToken.pptx   (needs python-pptx; edit the R dict for results)
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = sys.argv[1] if len(sys.argv) > 1 else "RECAP_STEAM_RLToken.pptx"

# ---------------------------------------------------------------- results (edit here)
R = dict(
    pi05_sft=48.8, pi05_recap=66.5,
    base=40.0, recap=50.0, steam=48.0, rlt=68.0,
)

FONT = "Yu Gothic"
BLUE = RGBColor(0x0B, 0x66, 0xC3)
NAVY = RGBColor(0x14, 0x24, 0x3B)
LIGHT = RGBColor(0xE8, 0xF1, 0xFB)
PALE = RGBColor(0xF3, 0xF7, 0xFC)
GRAY = RGBColor(0x59, 0x59, 0x59)
LGRAY = RGBColor(0xBF, 0xBF, 0xBF)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xD9, 0x6B, 0x2B)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
SLATE = RGBColor(0x6B, 0x8B, 0xB3)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
_page = [1]


# ---------------------------------------------------------------- helpers
def _font(run, size, bold=False, color=BLACK, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.name = FONT
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)


TOKEN = re.compile(r"(\*\*.+?\*\*|\^\^.+?\^\^)")


def _runs(p, text, size, bold=False, color=BLACK, accent=BLUE, font=None):
    """**bold**  ^^accent bold^^"""
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run()
            r.text = part[2:-2]
            _font(r, size, True, color)
        elif part.startswith("^^"):
            r = p.add_run()
            r.text = part[2:-2]
            _font(r, size, True, accent)
        else:
            r = p.add_run()
            r.text = part
            _font(r, size, bold, color)


SUBSUP = re.compile(r"([_^])\{([^}]*)\}")


def eq_runs(p, text, size, color=BLACK, bold=False, font="Cambria Math"):
    """text with _{sub} and ^{sup}"""
    pos = 0
    for m in SUBSUP.finditer(text):
        if m.start() > pos:
            r = p.add_run()
            r.text = text[pos:m.start()]
            _font(r, size, bold, color)
            r.font.name = font
        r = p.add_run()
        r.text = m.group(2)
        _font(r, size, bold, color)
        r.font.name = font
        r._r.get_or_add_rPr().set("baseline", "-25000" if m.group(1) == "_" else "30000")
        pos = m.end()
    if pos < len(text):
        r = p.add_run()
        r.text = text[pos:]
        _font(r, size, bold, color)
        r.font.name = font
    for r in p.runs:
        rPr = r._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):
            el = rPr.find(qn(tag))
            if el is not None:
                el.set("typeface", "Yu Gothic")


def _bullet(p, level, char):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.30 + 0.34 * level))))
    pPr.set("indent", str(int(-Inches(0.28))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", char)


def textbox(slide, x, y, w, h, items, size=16, color=BLACK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            name=None, space=6, margin=0.05):
    """items: list of str | (level, str) | dict(text, level, size, bold, color, bullet)"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for it in items:
        if isinstance(it, str):
            it = dict(text=it, level=None)
        elif isinstance(it, tuple):
            it = dict(level=it[0], text=it[1])
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        lvl = it.get("level")
        sz = it.get("size", size - (2 if lvl == 1 else 0) if lvl is not None else size)
        p.alignment = it.get("align", align)
        p.space_after = Pt(it.get("space", space))
        _runs(p, it["text"], sz, it.get("bold", False), it.get("color", color))
        if lvl is not None:
            _bullet(p, lvl, "•" if lvl == 0 else "–")
    return tb


def header(slide, title):
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.85))
    band.name = "title band"
    band.fill.solid()
    band.fill.fore_color.rgb = BLUE
    band.line.fill.background()
    tf = band.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    _font(r, 28, True, WHITE)


def page_number(slide):
    _page[0] += 1
    textbox(slide, 12.2, 7.05, 0.9, 0.35, [dict(text=str(_page[0]), align=PP_ALIGN.RIGHT, size=12, color=GRAY)],
            name="page number")


def new_slide(title, notes=None):
    s = prs.slides.add_slide(BLANK)
    header(s, title)
    page_number(s)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def key_box(slide, text, y=6.25, h=0.7):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(y), Inches(12.1), Inches(h))
    b.name = "key point"
    b.adjustments[0] = 0.18
    b.fill.solid()
    b.fill.fore_color.rgb = LIGHT
    b.line.color.rgb = BLUE
    b.line.width = Pt(1.25)
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.25)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    _runs(p, "→ " + text, 16, True, BLUE)


def box(slide, x, y, w, h, text, fill=LIGHT, line=BLUE, size=14, color=BLACK, bold=False, name=None,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, lw=1.0,
        accent=BLUE):
    b = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        b.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = 0.12
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line
        b.line.width = Pt(lw)
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    lines = text if isinstance(text, list) else [text]
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if isinstance(t, dict):
            _runs(p, t["text"], t.get("size", size), t.get("bold", bold), t.get("color", color), accent)
        else:
            _runs(p, t, size, bold, color, accent)
    return b


def arrow(slide, x1, y1, x2, y2, color=BLUE, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def table(slide, x, y, w, col_w, rows, size=14, row_h=0.45, header_fill=BLUE, bold_rows=(), hl_rows=(), hl_fill=LIGHT,
          align_cols=None, header_h=None):
    nrows, ncols = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    tbl = gs.table
    # plain look: remove default style banding
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(header_h if (i == 0 and header_h) else row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if align_cols and j in align_cols:
                p.alignment = align_cols[j]
            hdr = i == 0
            _runs(p, str(val), size, bold=hdr or i in bold_rows, color=WHITE if hdr else BLACK)
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if hdr else (hl_fill if i in hl_rows else WHITE if i % 2 else PALE)
    return gs


def eq_block(slide, y, num, desc, formula, h_desc=0.62, h_eq=0.6, size=18, name=None):
    box(slide, 0.6, y + 0.04, 0.55, 0.55, str(num), fill=BLUE, line=None, size=18, color=WHITE, bold=True,
        shape=MSO_SHAPE.OVAL, name=f"num {num}")
    textbox(slide, 1.35, y - 0.04, 11.4, h_desc, [dict(text=desc, size=15, space=0)], anchor=MSO_ANCHOR.MIDDLE,
            name=f"desc {num}")
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.35), Inches(y + h_desc + 0.04), Inches(11.35), Inches(h_eq))
    b.name = f"equation {num}"
    b.adjustments[0] = 0.12
    b.fill.solid()
    b.fill.fore_color.rgb = PALE
    b.line.color.rgb = LGRAY
    b.line.width = Pt(0.75)
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.2)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    eq_runs(p, formula, size, BLACK)
    return b


def style_chart(chart, size=12):
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = BLACK


# ================================================================ 1. title
s = prs.slides.add_slide(BLANK)
bg = s.background.fill
bg.solid()
bg.fore_color.rgb = NAVY
textbox(s, 0.9, 1.55, 11.5, 1.9,
        [dict(text="VLA のための強化学習:", size=40, bold=True, color=WHITE, space=2),
         dict(text="RECAP・STEAM・RL Token", size=44, bold=True, color=WHITE, space=2)], name="title")
textbox(s, 0.9, 3.6, 11.5, 0.9,
        [dict(text="smolVLA と LIBERO-10 Task 0 で、3 つの手法を実装して比べる", size=22, color=RGBColor(0xCA, 0xDC, 0xFC))],
        name="subtitle")
for i, (t, sub) in enumerate([("RECAP", "価値関数で advantage を作り、方策に条件付ける"),
                              ("STEAM", "時間オフセットの自己教師ありアンサンブル"),
                              ("RL Token", "凍結 VLA の上で、軽い actor-critic を学習")]):
    box(s, 0.9 + i * 4.0, 4.75, 3.75, 1.2, [dict(text=t, size=20, bold=True, color=WHITE), dict(text=sub, size=14)],
        fill=RGBColor(0x1E, 0x35, 0x56), line=RGBColor(0x5B, 0x9B, 0xE6), size=14, color=RGBColor(0xCA, 0xDC, 0xFC),
        name=f"chip {t}")
textbox(s, 0.9, 6.5, 11.5, 0.5,
        [dict(text="libero_RL プロジェクト  |  2026/10/04", size=16, color=RGBColor(0xCA, 0xDC, 0xFC))], name="date")
s.notes_slide.notes_text_frame.text = (
    "RECAP、STEAM、RL Token の 3 手法を紹介し、smolVLA (LIBERO-10 Task 0) での実験結果を比較します。")

# ================================================================ 2. 概要
s = new_slide("概要", "発表全体の要約。結果の数値は 15 枚目の表を参照。")
rows = [
    ("目的", "実環境の経験（ロールアウト）から VLA を改善する 3 つの強化学習手法を理解し、同じ設定（smolVLA、LIBERO-10 Task 0）で比較する"),
    ("背景", "模倣学習（SFT）だけでは、精度・速度・失敗からの回復に限界がある。10 億パラメータ級の VLA 全体を RL で更新するのは高コスト"),
    ("手法", "**RECAP**: 価値関数から advantage を作り、方策に条件付けて学習　**STEAM**: 時間オフセットの自己教師ありアンサンブルで advantage を推定　"
            "**RL Token**: 凍結した VLA の読み出しトークンの上で、軽量な actor-critic をオンライン学習"),
    ("実験", "smolVLA（lerobot/smolvla_libero）で、Task 0（2 つの物体をカゴへ）の success_once を、固定の初期状態 50 個で測る。実装は RLinf"),
    ("結果", f"ベース {R['base']/100:.2f} に対し、^^RL Token が {R['rlt']/100:.2f}^^、RECAP が {R['recap']/100:.2f}、STEAM が {R['steam']/100:.2f}（いずれも最良値）。"
            "RECAP と STEAM は学習の途中段階で、差は評価の標準誤差（±0.07）の範囲内"),
    ("結論", "3 手法は、RL が作用する場所が違う（方策の学習目標 / advantage の推定器 / 行動の洗練）。成功で打ち切る収集設計が、価値の信号に重要だった"),
]
y = 1.15
for label, text in rows:
    box(s, 0.6, y, 1.5, 0.78, label, fill=BLUE, line=None, size=18, color=WHITE, bold=True, name=f"label {label}")
    textbox(s, 2.3, y - 0.02, 10.4, 0.85, [dict(text=text, size=15)], anchor=MSO_ANCHOR.MIDDLE, name=f"text {label}")
    y += 0.9

# ================================================================ 3. 書誌情報
s = new_slide("書誌情報", "3 論文の書誌。本資料の日本語訳は docs/ にある（機械翻訳）。")
rows = [
    ["手法", "論文", "著者・所属", "arXiv"],
    ["RECAP", "π*0.6: a VLA That Learns From Experience", "Physical Intelligence", "2511.14759\n（2025/11）"],
    ["STEAM", "STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning",
     "Zhihao Liu, Qiuyi Gu ほか（中国科学院自動化研究所、清華大学 ほか）", "2606.29834"],
    ["RL Token", "RL Token: Bootstrapping Online RL with Vision-Language-Action Models",
     "Charles Xu, Jost Tobias Springenberg, Sergey Levine, Liyiming Ke ほか（Physical Intelligence）", "2604.23073\n（2026/04）"],
]
table(s, 0.6, 1.2, 12.1, [1.6, 5.0, 3.9, 1.6], rows, size=14, row_h=1.0, header_h=0.5)
textbox(s, 0.6, 4.95, 12.1, 1.1, [
    (0, "実装基盤: **RLinf**（強化学習の学習基盤。RECAP / STEAM のパイプラインと、RL Token 用の actor-critic を使用）"),
    (0, "ベースモデル: **smolVLA**（lerobot/smolvla_libero、約 4.5 億パラメータ）。参考値として、RLinf のチュートリアルにある π₀.₅ の RECAP 結果も使う"),
], size=15)
key_box(s, "論文の内容（引用）と、私たちの実験（smolVLA）を、スライド上で区別して示す")

# ================================================================ 4. 背景
s = new_slide("背景: なぜ VLA に強化学習か", "3 手法は『RL をどこに入れるか』が違う、という整理が主題。")
textbox(s, 0.6, 1.1, 5.6, 4.9, [
    (0, "**VLA は、模倣学習（SFT）で多様な技能を学べる**"),
    (1, "ただし、実タスクが要求する精度・速度・失敗からの回復には届かない"),
    (0, "**強化学習（RL）で、自分の経験から改善したい**"),
    (1, "10 億パラメータ級のモデル全体を、オンラインで更新するのは、計算コストもサンプル効率も悪い"),
    (1, "flow matching の行動生成は、対数尤度が扱いにくく、方策勾配法（PPO など）を適用しにくい"),
    (0, "**3 手法は、RL を入れる場所が違う**"),
], size=19)
cards = [
    ("RECAP", "方策の学習目標", "価値関数から advantage を作り、「良い行動」の指標を方策に入力して学習する（advantage 条件付き SFT）"),
    ("STEAM", "advantage の推定器", "時間オフセットを自己教師ありで学習した critic のアンサンブルで、フレームごとの advantage を作る"),
    ("RL Token", "行動の洗練", "VLA は凍結。読み出しトークンの上の小さな actor-critic が、VLA の行動チャンクを改善する"),
]
for i, (t, where, desc) in enumerate(cards):
    y = 1.1 + i * 1.7
    box(s, 6.5, y, 2.0, 1.5, [f"^^{t}^^"], fill=PALE, line=BLUE, size=20, name=f"card {t}")
    box(s, 8.6, y, 4.1, 1.5, [f"**RL が作用する場所: {where}**", desc], fill=WHITE, line=LGRAY, size=14,
        align=PP_ALIGN.LEFT, name=f"desc {t}")
key_box(s, "RECAP は「方策の学習目標」、STEAM は「評価者」、RL Token は「行動の洗練」に RL を入れる")

# ================================================================ 5. 実験設定
s = new_slide("実験設定", "評価は 50 エピソード。標準誤差が大きいことを最初に断っておく。")
textbox(s, 0.6, 1.1, 7.7, 5.0, [
    (0, "**タスク**: LIBERO-10 の Task 0「put both the alphabet soup and the tomato sauce in the basket」（2 つの物体をカゴに入れる）"),
    (0, "**モデル**: smolVLA（lerobot/smolvla_libero）。50 ステップの行動チャンクのうち 10 ステップを実行"),
    (0, "**評価**: 固定の初期状態 50 個を 1 回ずつ、最大 520 ステップ。指標は success_once（1 度でも成功）"),
    (0, "**ロールアウトの収集**: 自律ロールアウト 4,122 本（成功 41.0%）。成功した時点で打ち切る設定で再収集"),
    (1, "旧データは、成功後も 520 ステップ続けたため、return が「時間」と「成否」だけで決まっていた"),
    (0, "**人のデモ**: Task 0 の 3 本（836 フレーム）"),
    (0, "**実装**: RLinf（RECAP / STEAM のパイプライン、RL Token の actor-critic）"),
], size=17)
stats = [("50", "初期状態（固定）"), ("0.40", "ベースの success_once（20/50）"), ("4,122", "ロールアウト（成功 41.0%）")]
for i, (n, label) in enumerate(stats):
    y = 1.15 + i * 1.65
    box(s, 8.7, y, 4.0, 1.45, [dict(text=n, size=40, bold=True, color=BLUE), dict(text=label, size=15)], fill=PALE, line=None, size=15, name=f"stat {n}")
key_box(s, "評価は 50 エピソード。標準誤差は約 ±0.07 なので、0.05 程度の差は有意とは言えない")

# ================================================================ 6. RECAP 提案
s = new_slide("RECAP: 提案（論文: π*0.6）", "RECAP は、価値関数の出力を『改善指標』として方策に条件付けるのが核心。")
textbox(s, 0.6, 1.1, 6.6, 5.0, [
    (0, "**RECAP**: RL with Experience and Corrections via Advantage-conditioned Policies"),
    (0, "次の 3 ステップを繰り返す"),
    (1, "① データ収集: ロールアウトを動かし、結果ラベルを付ける（必要なら人の介入も）"),
    (1, "② 価値関数の学習: 全データで、分布型の価値関数 V を学習する"),
    (1, "③ advantage 条件付き学習: advantage から作る「改善指標」を方策の入力に入れて学習する"),
    (0, "**利点**"),
    (1, "方策勾配が不要で、flow matching の VLA にもスケールする"),
    (1, "デモ・自律ロールアウト・人の介入を、同じ枠組みで使える"),
    (0, "**論文の結果**: 洗濯物たたみ・エスプレッソ作り・箱の組み立てで、成功率とスループットが改善。難しいタスクでは、失敗率が 1/2 以下"),
], size=15)
# loop diagram
nodes = [("① データ収集\nロールアウト", 7.7, 1.2), ("② 価値関数 V の学習", 7.7, 2.75), ("③ advantage 条件付き\n方策の学習", 7.7, 4.3)]
for t, x, y in nodes:
    box(s, x, y, 3.6, 1.0, t.split("\n"), fill=LIGHT, line=BLUE, size=15, bold=True, name=t.split("\n")[0])
arrow(s, 9.5, 2.2, 9.5, 2.75)
arrow(s, 9.5, 3.75, 9.5, 4.3)
# loop back
arrow(s, 11.3, 4.8, 12.3, 4.8)
arrow(s, 12.3, 4.8, 12.3, 1.7)
arrow(s, 12.3, 1.7, 11.3, 1.7)
textbox(s, 11.4, 5.35, 1.6, 0.6, [dict(text="新しい方策で\n再収集", size=12, color=GRAY, align=PP_ALIGN.CENTER)])
key_box(s, "価値関数で作った「改善指標」を方策に入力することで、最適でないデータから、良い行動を選んで学習する")

# ================================================================ 7. RECAP 式
s = new_slide("RECAP: 手法（学習の式）", "式(1)(3) の記号は論文のもの。")
eq_block(s, 1.0, 1, "**分布型の価値関数**: 観測 o と言語 ℓ から、B 個の価値ビンの分布を予測する（交差エントロピー H、R^B はビンに離散化した return）",
         "min_{φ}  E_{τ}[ Σ_{t} H( R^{B}_{t}(τ), p_{φ}(V | o_{t}, ℓ) ) ]　…(1)")
eq_block(s, 2.27, 2, "**advantage**: 価値関数から、行動 a_t の良さを求める（N ステップ先まで）",
         "A(o_{t}, a_{t}) = r_{t:t+N} + V(o_{t+N}) − V(o_{t})")
eq_block(s, 3.54, 3, "**改善指標と方策の学習**: タスクごとの閾値 ε_ℓ で 2 値の指標 I を作り、方策に条件付ける（ドロップアウトで、指標なしの方策も同時に学習）",
         "min_{θ}  E[ − log π_{θ}(a_{t} | o_{t}, ℓ) − α log π_{θ}(a_{t} | I_{t}, o_{t}, ℓ) ] ,　I_{t} = 1[ A > ε_{ℓ} ]　…(3)")
eq_block(s, 4.81, 4, "**推論**: 指標を「良い（I = 1）」に固定して行動を生成する。CFG の重み β > 1 で、さらに強調できる",
         "π̂(a | o, ℓ) ∝ π(a | o, ℓ) · ( π(a | I, o, ℓ) / π(a | o, ℓ) )^{β}　…(2)")
key_box(s, "方策は「指標あり」と「指標なし」の両方を表現できるので、改善確率を明示的に推定しなくてよい", y=6.3)

# ================================================================ 8. RECAP smolVLA
s = new_slide("RECAP: smolVLA での実装（私たちの実験）", "Step1〜4 と評価（Step9）。名称は私たちのパイプラインの番号。")
steps = ["Step1\nreturn の計算", "Step2\n価値モデルの学習", "Step3\nadvantage の計算", "Step4\n条件付き SFT", "Step9\nシミュレータで評価"]
for i, t in enumerate(steps):
    x = 0.6 + i * 2.5
    box(s, x, 1.1, 2.2, 0.95, t.split("\n"), fill=BLUE if i < 4 else GREEN, line=None, size=14, color=WHITE, bold=True,
        name=f"step {i}")
    if i < 4:
        arrow(s, x + 2.2, 1.57, x + 2.5, 1.57, color=GRAY)
textbox(s, 0.6, 2.3, 12.1, 3.9, [
    (0, "**Step1**: 報酬は 1 ステップごとに −1、失敗は終端で −300。return を [−819, 0] に正規化"),
    (0, "**Step2**: 価値モデル（SigLIP2 + Gemma3-270M）が、201 ビンの分布を予測。**成功した時点で打ち切る設定で再収集した 4,122 本**で学習"),
    (1, "旧データでは、成功後も 520 ステップ続けていたため、価値が「フレーム番号と成否」でほぼ決まっていた"),
    (0, "**Step3**: A = r_{t:t+10} + V(o_{t+10}) − V(o_t)。上位 30% を positive（人のデモは全て positive）。閾値は全データセットを結合して 1 つに決める"),
    (0, "**Step4**: プロンプトに \"Advantage: positive\" を付けて SFT（ドロップアウト 0.1）。評価も positive プロンプトで行う"),
    (0, "**状況**: Step4 は途中段階（2,000 step 前後。計画は 30,000 step）。step 2,000 の評価は 0.44〜0.50（環境数・乱数・プロンプトで変動）"),
], size=17)
key_box(s, "価値モデルの信号は、データの作り方（成功で打ち切るか）に強く依存した")

# ================================================================ 9. STEAM 提案
s = new_slide("STEAM: 提案（論文）", "報酬ラベル不要で、専門家デモだけから advantage を作る点が特徴。")
textbox(s, 0.6, 1.1, 6.9, 5.0, [
    (0, "**STEAM**: Self-supervised Temporal Ensemble Advantage Modeling"),
    (0, "**自己教師あり**: 専門家デモのフレームペア (f_i, f_j) の、軌跡長で正規化した時間オフセット (j − i)·L_max / L を、N ビンの分布で予測する"),
    (1, "軌跡を逆順にすると、負のオフセット（後退・失敗の疑似データ）も学べる"),
    (0, "**advantage**: H ステップ先のフレームとのペアで予測した時間オフセットを、基準と比べてスカラーにする"),
    (0, "**保守的なアンサンブル**: M 個の予測器の最小値 A_STEAM = min_m A(θ_m) で、分布外のロールアウトでの過大評価を抑える"),
    (0, "**方策学習**: 分位点 δ_q で 2 値ラベルを作り、CFGRL の条件に使う"),
], size=17)
flow = ["フレームペア\n(f_i, f_i+H)", "時間オフセット\n予測器 × M", "最小値を取る\nmin_m A(θ_m)", "A_STEAM", "2 値ラベル\nA ≥ δ_q", "CFGRL で\n方策学習"]
for i, t in enumerate(flow):
    y = 1.1 + i * 0.82
    box(s, 8.2, y, 3.3, 0.62, t.split("\n"), fill=LIGHT if i not in (3,) else BLUE, line=BLUE, size=13,
        color=WHITE if i == 3 else BLACK, bold=i == 3, name=f"flow {i}")
    if i < 5:
        arrow(s, 9.85, y + 0.62, 9.85, y + 0.82)
key_box(s, "報酬を作らず、デモの時間構造から、「前進」と「停滞・後退」を区別する advantage を学ぶ")

# ================================================================ 10. STEAM 論文結果
s = new_slide("STEAM: 論文の結果（実機）", "表は論文 Table 1 の成功率（Succ.）のみ。RECAP は同じ CFGRL の骨格で 1 反復。")
rows = [
    ["手法（成功率 %）", "タオル折り", "チップのレジ会計", "コーラ補充", "ピック＆プレイス"],
    ["Behavior Cloning", "33.3", "39.5", "52", "63.8"],
    ["HG-DAgger", "40", "53.3", "58.3", "—"],
    ["RECAP（1 反復）", "55.6", "53.3", "52.9", "53.8"],
    ["**STEAM**", "**92.3**（↑59）", "**93.8**（↑54.3）", "**75**（↑23）", "**80**（↑16.2）"],
]
table(s, 0.6, 1.2, 12.1, [3.3, 2.2, 2.4, 2.1, 2.1], rows, size=16, row_h=0.72, hl_rows=(4,),
      align_cols={1: PP_ALIGN.CENTER, 2: PP_ALIGN.CENTER, 3: PP_ALIGN.CENTER, 4: PP_ALIGN.CENTER})
textbox(s, 0.6, 5.0, 12.1, 1.1, [
    (0, "↑ はベースライン（Behavior Cloning）に対する改善（pt）。双腕のタオル折り、チップのレジ会計、コーラ補充、単腕のピック＆プレイスの 4 タスク"),
    (0, "STEAM は、専門家・人の修正・自律ロールアウトのすべてを使い、全タスクで最高の成功率"),
], size=14, color=GRAY)
key_box(s, "論文では、STEAM が RECAP（1 反復）を全タスクで上回る。タオル折りでは +36.7 pt")

# ================================================================ 11. STEAM smolVLA
s = new_slide("STEAM: smolVLA での実装（私たちの実験）", "グラフは学習中の指標（学習データ上）。検証データの値は左の箇条書き。")
textbox(s, 0.6, 1.1, 6.0, 5.0, [
    (0, "**critic**: SigLIP so400m + Gemma3-270M。k = 32 ステップ先との時間オフセットを、32 ビンの分布で予測"),
    (0, "**学習**: 人のデモ 3 本（重み 4）と、成功ロールアウト。2,000 step"),
    (0, "**advantage**: ロールアウトは上位 30%、デモは上位 80% を positive（アンサンブルのメンバー数は 1）"),
    (0, "**学習データ上の精度**（step 500 → 2,000）"),
    (1, "accuracy 0.277 → 0.353"),
    (1, "隣接ビンを含む accuracy 0.375 → 0.569"),
    (0, "**未知のロールアウト**（eval 64 本、step 600）: accuracy 0.106、隣接 0.189。失敗エピソードを含むため、学習時より低い"),
], size=16)
cd = CategoryChartData()
cd.categories = ["500-600", "600-700", "700-800", "800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000"]
cd.add_series("accuracy", (0.277, 0.295, 0.295, 0.297, 0.316, 0.333, 0.348, 0.345, 0.353))
cd.add_series("accuracy（隣接ビンを含む）", (0.375, 0.405, 0.416, 0.432, 0.471, 0.504, 0.531, 0.539, 0.569))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.8), Inches(1.1), Inches(6.0), Inches(4.9), cd)
ch = gf.chart
style_chart(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "critic の学習中の精度（学習データ上）"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
va = ch.value_axis
va.minimum_scale = 0.2
va.maximum_scale = 0.65
va.major_unit = 0.1
va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
va.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(11)
for ser, col in zip(ch.plots[0].series, (SLATE, BLUE)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.5)
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.smooth = False
key_box(s, "critic は学習データ上では改善したが、未知のロールアウトでの精度は低い（今後の課題）")

# ================================================================ 12. RL Token 提案
s = new_slide("RL Token: 提案（論文）", "VLA を凍結して、RL token という小さな状態表現の上で RL をする。")
textbox(s, 0.6, 1.1, 6.6, 5.0, [
    (0, "**課題**: 巨大な VLA 全体のオンライン RL は、高コストで、サンプル効率が悪い"),
    (0, "**RL token**: VLA に追加した小さな encoder-decoder transformer が、VLA の埋め込み z_{1:M} を 1 本の読み出しトークン z_rl に圧縮する（再構成損失で学習）"),
    (0, "**オンライン RL**: VLA と RL token を凍結し、軽量な actor π_θ と critic Q_ψ だけを学習"),
    (1, "入力は x = (z_rl, 固有受容感覚 s^p)。VLA の行動チャンク ã も actor に入力する"),
    (1, "ã に近づける正則化と、ã のドロップアウトで、コピーを防ぐ"),
    (0, "**論文の結果**: 実機で数時間の練習で、重要な局面の成功率と速度が向上。HIL-SERL、PLD などの他のオンライン RL より速く学習"),
], size=16)
box(s, 7.5, 1.2, 2.2, 0.9, ["VLA（凍結）", "VLM + action expert"], fill=PALE, line=GRAY, size=13, name="vla")
box(s, 7.5, 2.65, 2.2, 0.9, ["RL token", "z_rl（凍結）"], fill=LIGHT, line=BLUE, size=13, bold=True, name="rltoken")
box(s, 10.4, 2.65, 2.3, 0.9, ["actor π_θ", "critic Q_ψ"], fill=BLUE, line=None, size=14, color=WHITE, bold=True, name="actor critic")
box(s, 10.4, 4.4, 2.3, 0.9, ["改善した", "行動チャンク a_1:C"], fill=GREEN, line=None, size=13, color=WHITE, bold=True, name="out")
arrow(s, 8.6, 2.1, 8.6, 2.65)
arrow(s, 9.7, 3.1, 10.4, 3.1)
arrow(s, 11.55, 3.55, 11.55, 4.4)
arrow(s, 9.7, 1.65, 11.55, 1.65, color=GRAY)
arrow(s, 11.55, 1.65, 11.55, 2.65, color=GRAY)
textbox(s, 9.8, 1.2, 2.9, 0.45, [dict(text="参照行動 ã（VLA の提案）", size=12, color=GRAY)])
textbox(s, 7.5, 3.7, 2.8, 0.8, [dict(text="埋め込み z_{1:M} →\nencoder-decoder で圧縮", size=12, color=GRAY)])
key_box(s, "RL は小さな actor-critic だけが担当し、VLA の知識を壊さずに、数時間で行動を洗練できる")

# ================================================================ 13. RL Token 式
s = new_slide("RL Token: 手法（学習の式）", "式の番号は論文のもの。")
eq_block(s, 1.0, 1, "**RL token の抽出**: 小さな transformer g_φ が、VLA の埋め込み z と学習可能なトークン e_rl から、読み出し z_rl を作る",
         "z_{rl} = g_{φ}( [ z_{1:M}, e_{rl} ] )_{M+1}　…(1)")
eq_block(s, 2.27, 2, "**再構成損失**: デコーダ d_φ が、z_rl から元の埋め込みを自己回帰的に再構成する（VLA は stop-gradient）",
         "L_{ro} = E[ Σ_{i} ‖ h_{φ}( d_{φ}([ z_{rl}, z̄_{1:i−1} ]) )_{i} − z̄_{i} ‖² ]　…(2)")
eq_block(s, 3.54, 3, "**critic**: 行動チャンク（長さ C）の TD 学習",
         "L_{Q} = E[ ( Σ_{t'=1}^{C} γ^{t'−1} r_{t'} + γ^{C} E_{a' ~ πθ}[ Q_{ψ'}(x', a') ] − Q_{ψ}(x, a_{1:C}) )² ]　…(3)")
eq_block(s, 4.81, 4, "**actor**: Q を最大化しつつ、VLA の提案 ã に近づける（ã はドロップアウトする）",
         "L_{π} = E[ − Q_{ψ}(x, a_{1:C}) + β ‖ a_{1:C} − ã_{1:C} ‖² ] ,　ã ~ π_{vla}(· | s, ℓ)　…(5)")
key_box(s, "actor は、VLA の提案 ã を条件に、ゼロからではなく、強い初期案を改善する", y=6.3)

# ================================================================ 14. RL Token smolVLA
s = new_slide("RL Token: smolVLA での実装（私たちの実験）", "グラフの 0.46 は、この RL プロトコルでの VLA のみの評価（50 エピソード）。")
textbox(s, 0.6, 1.1, 6.2, 5.0, [
    (0, "**RL token**: 凍結した smolVLA の prefix の隠れ状態（241 × 960）から、RL token を作る"),
    (0, "**RL の対象**: タスクの局面ごとの actor-critic（1 つ目の把持 → 2 つ目の把持の「2 局面ゲート」）。それ以外は VLA が実行"),
    (0, "**オンライン学習**: 3 環境で、更新/データ比 5、探索ノイズ σ = 0.05。約 5 時間（400 エポック）"),
    (0, "**失敗の分析**: 1 つ目の把持は確立。残る失敗は、2 つ目の把持が最大"),
    (1, "→ 2 つ目専用の actor と、空振り（何も挟まずに閉じる）への罰 −0.2"),
    (0, "**actor の重み**: 約 5 MB と、軽量"),
], size=16)
cd = CategoryChartData()
cd.categories = ["VLA のみ", "1 つ目を\nRL", "2 局面\nゲート", "+ 2 つ目専用\n+ 空振り罰"]
cd.add_series("success_once", (0.46, 0.62, 0.66, 0.68))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.0), Inches(1.1), Inches(5.8), Inches(4.9), cd)
ch = gf.chart
style_chart(ch, 12)
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "RL Token（smolVLA）の success_once"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
va = ch.value_axis
va.minimum_scale = 0
va.maximum_scale = 0.8
va.major_unit = 0.2
va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
va.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(12)
pl = ch.plots[0]
pl.gap_width = 60
pl.has_data_labels = True
pl.data_labels.show_value = True
pl.data_labels.number_format = "0.00"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(14)
pl.data_labels.font.bold = True
ser = pl.series[0]
for i, col in enumerate((LGRAY, SLATE, BLUE, GREEN)):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
key_box(s, "凍結した VLA の上の小さな actor-critic だけで、success_once が 0.46 → 0.68 に向上した")

# ================================================================ 15. 実験結果
s = new_slide("実験結果", "π₀.₅ は RLinf のチュートリアルの値。smolVLA は私たちの実験（各 50 エピソード）。")
pts = lambda v: f"{v:.1f}%"
rows = [
    ["モデル", "手法", "成功率", "ベース比"],
    ["π₀.₅", "SFT（few-shot）", pts(R["pi05_sft"]), "—"],
    ["π₀.₅", "RECAP（1 iteration）", pts(R["pi05_recap"]), f"+{R['pi05_recap'] - R['pi05_sft']:.1f} pt"],
    ["smolVLA", "ベース", pts(R["base"]), "—"],
    ["smolVLA", "RECAP（最良）", pts(R["recap"]), f"+{R['recap'] - R['base']:.1f} pt"],
    ["smolVLA", "STEAM（最良）", pts(R["steam"]), f"+{R['steam'] - R['base']:.1f} pt"],
    ["smolVLA", "RL Token（最良）", pts(R["rlt"]), f"+{R['rlt'] - R['base']:.1f} pt"],
]
table(s, 0.6, 1.1, 7.2, [1.5, 2.7, 1.5, 1.5], rows, size=15, row_h=0.56, hl_rows=(6,), bold_rows=(6,),
      align_cols={2: PP_ALIGN.CENTER, 3: PP_ALIGN.CENTER})
cd = CategoryChartData()
cd.categories = ["π₀.₅\nSFT", "π₀.₅\nRECAP", "smolVLA\nベース", "smolVLA\nRECAP", "smolVLA\nSTEAM", "smolVLA\nRL Token"]
cd.add_series("成功率 (%)", (R["pi05_sft"], R["pi05_recap"], R["base"], R["recap"], R["steam"], R["rlt"]))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(8.0), Inches(1.0), Inches(4.9), Inches(3.9), cd)
ch = gf.chart
style_chart(ch, 11)
ch.has_legend = False
ch.has_title = False
va = ch.value_axis
va.minimum_scale = 0
va.maximum_scale = 80
va.major_unit = 20
va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
va.tick_labels.font.size = Pt(11)
ch.category_axis.tick_labels.font.size = Pt(10)
pl = ch.plots[0]
pl.gap_width = 50
pl.has_data_labels = True
pl.data_labels.show_value = True
pl.data_labels.number_format = "0.0"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
pl.data_labels.font.bold = True
for i, col in enumerate((LGRAY, SLATE, LGRAY, SLATE, SLATE, GREEN)):
    pt = pl.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
textbox(s, 0.6, 5.15, 12.1, 1.1, [
    (0, f"smolVLA: 各 50 エピソード（標準誤差 ±0.07）。RECAP・STEAM は、Step4 の途中の checkpoint（step 2,000 前後）の最良値（advantage プロンプトなしの評価を含む。プロンプトありは、RECAP 0.48、STEAM 0.42）"),
    (0, "π₀.₅ は RLinf チュートリアルの記載値で、モデル・設定・評価数が異なり、smolVLA との直接比較はできない"),
], size=12, color=GRAY, space=3)
key_box(s, f"smolVLA では RL Token が最良（{R['base']:.0f}% → {R['rlt']:.0f}%）。RECAP・STEAM は誤差の範囲内で、途中段階")

# ================================================================ 16. 考察
s = new_slide("考察: 3 手法の比較", "表は定性的な比較。smolVLA での経験に基づく。")
rows = [
    ["", "RL が作用する場所", "必要なもの", "長所", "今回の経験での課題"],
    ["RECAP", "方策の学習目標（advantage 条件付き SFT）", "価値関数、ロールアウト",
     "勾配不要。flow matching VLA に使える。異種データを使える", "価値の信号がデータの作り方に依存。SFT が重い（30,000 step）"],
    ["STEAM", "advantage の推定器（critic）", "デモ（自己教師あり）、critic のアンサンブル",
     "報酬ラベル不要。保守的なアンサンブルで過大評価を抑える", "未知データでの critic の精度が低い。advantage の計算に約 2.5 時間/15.5 万フレーム"],
    ["RL Token", "行動の洗練（凍結 VLA の出力）", "オンラインの環境と報酬",
     "軽量（約 5 MB）で、約 5 時間で学習。success_once 0.46 → 0.68", "局面の設計（ゲート）が必要。シミュレータ（実機）が必要"],
]
table(s, 0.6, 1.1, 12.1, [1.3, 2.6, 2.2, 3.0, 3.0], rows, size=14, row_h=1.3, header_h=0.55, bold_rows=(), align_cols=None)
key_box(s, "RECAP・STEAM は「良い行動」を学び、RL Token は「行動の直し方」を学ぶ。目的は補完的")

# ================================================================ 17. まとめ
s = new_slide("まとめと今後", "次の作業: Step4 の本学習、critic の精度、評価の統計的な信頼性。")
box(s, 0.6, 1.1, 5.9, 0.6, "まとめ", fill=BLUE, line=None, size=20, color=WHITE, bold=True, name="summary head")
textbox(s, 0.6, 1.85, 5.9, 3.0, [
    (0, "3 手法は、VLA の経験からの改善を、異なる場所で実現する"),
    (0, f"smolVLA では **RL Token が最良**（{R['base']/100:.2f} → {R['rlt']/100:.2f}）"),
    (0, f"RECAP・STEAM は、Step4 の途中段階の最良値で {R['steam']/100:.2f}〜{R['recap']/100:.2f}。差は誤差の範囲内"),
    (0, "成功で打ち切る収集設計が、価値・critic の信号に重要"),
], size=17)
box(s, 6.8, 1.1, 5.9, 0.6, "今後", fill=GREEN, line=None, size=20, color=WHITE, bold=True, name="future head")
textbox(s, 6.8, 1.85, 5.9, 3.0, [
    (0, "RECAP・STEAM の Step4 を、本学習（30,000 step）まで進める"),
    (0, "STEAM の critic のアンサンブルを増やし、未知データでの精度を上げる"),
    (0, "評価を 100 エピソード以上にして、標準誤差を縮める"),
    (0, "RL Token を、LIBERO-10 の他のタスクに広げる"),
], size=17)
strip = [("ベース", R["base"], LGRAY, BLACK), ("STEAM", R["steam"], SLATE, WHITE), ("RECAP", R["recap"], SLATE, WHITE), ("RL Token", R["rlt"], GREEN, WHITE)]
for i, (n, v, col, tc) in enumerate(strip):
    box(s, 0.6 + i * 3.05, 4.85, 2.85, 1.2, [dict(text=f"{v/100:.2f}", size=32, bold=True, color=tc), dict(text=n, size=15, color=tc)],
        fill=col, line=None, size=15, name=f"result {n}")
key_box(s, "最良は RL Token（0.68）。RECAP・STEAM は、学習を進めて、同じ条件で再評価する")

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))

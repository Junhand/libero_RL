#!/usr/bin/env python
"""Build docs/RECAP_STEAM_RLToken.pptx (python-pptx). Style follows docs/publication (DL Papers look).
Usage: python scripts/build_slides_recap_steam_rlt.py docs/RECAP_STEAM_RLToken.pptx   (needs python-pptx; edit the R dict for results; figures and videos in scripts/assets)
"""
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
    base=40.0, recap=50.0, steam=52.0, rlt=74.0,
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



# ---------------------------------------------------------------- diagram helpers (beginner-friendly figures)
FROZEN_FILL = RGBColor(0xE4, 0xE4, 0xE4)
LOSS_FILL = RGBColor(0xFF, 0xF2, 0xCC)
LOSS_LINE = RGBColor(0xE0, 0xA8, 0x00)
KINDS = dict(
    data=(LIGHT, BLUE, BLACK),
    train=(ORANGE, None, WHITE),
    frozen=(FROZEN_FILL, LGRAY, GRAY),
    out=(GREEN, None, WHITE),
    plain=(WHITE, LGRAY, BLACK),
    loss=(LOSS_FILL, LOSS_LINE, BLACK),
)


def node(slide, x, y, w, h, title, sub=None, kind="data", size=14, name=None):
    fill, line, col = KINDS[kind]
    lines = [dict(text=title, size=size, bold=True, color=col)]
    if sub:
        for t in (sub if isinstance(sub, list) else [sub]):
            lines.append(dict(text=t, size=max(size - 2, 11), color=col))
    return box(slide, x, y, w, h, lines, fill=fill, line=line, size=size, color=col, name=name)


def legend(slide, x, y):
    items = (("更新する", "train"), ("凍結（更新しない）", "frozen"), ("入力・出力", "data"), ("損失", "loss"))
    cx = x
    for t, k in items:
        w = 1.2 if len(t) < 6 else 2.0
        node(slide, cx, y, w, 0.34, t, kind=k, size=11)
        cx += w + 0.12


def arrow_label(slide, x1, y1, x2, y2, text, tx, ty, tw=1.8, color=BLUE, size=12, th=0.5, align=PP_ALIGN.CENTER):
    arrow(slide, x1, y1, x2, y2, color=color)
    if text:
        textbox(slide, tx, ty, tw, th, [dict(text=text, size=size, color=GRAY, align=align)])


def spec_table(slide, x, y, w, rows, size=13, row_h=0.46, label_w=1.5):
    data = [["項目", "内容"]] + rows
    return table(slide, x, y, w, [label_w, w - label_w], data, size=size, row_h=row_h, header_h=0.38)


def io_diagram(slide, in_t, model_t, out_t, res_t, teacher_t, loss_t, y=1.4):
    legend(slide, 0.6, 0.95)
    node(slide, 0.6, y, 2.5, 1.2, in_t[0], in_t[1], kind="data", size=14, name="in")
    node(slide, 3.7, y, 3.1, 1.2, model_t[0], model_t[1], kind="train", size=15, name="model")
    node(slide, 7.4, y, 2.6, 1.2, out_t[0], out_t[1], kind="data", size=14, name="out")
    node(slide, 10.6, y, 2.1, 1.2, res_t[0], res_t[1], kind="out", size=14, name="result")
    arrow(slide, 3.1, y + 0.6, 3.7, y + 0.6)
    arrow(slide, 6.8, y + 0.6, 7.4, y + 0.6)
    arrow(slide, 10.0, y + 0.6, 10.6, y + 0.6)
    y2 = y + 1.7
    node(slide, 0.6, y2, 2.8, 0.9, teacher_t[0], teacher_t[1], kind="data", size=13, name="teacher")
    node(slide, 3.7, y2, 3.1, 0.9, loss_t[0], loss_t[1], kind="loss", size=14, name="loss")
    arrow(slide, 3.4, y2 + 0.45, 3.7, y2 + 0.45)
    arrow(slide, 8.7, y + 1.2, 8.7, y2 + 0.45)
    arrow(slide, 8.7, y2 + 0.45, 6.8, y2 + 0.45)
    arrow(slide, 5.25, y2, 5.25, y + 1.2, color=ORANGE)
    textbox(slide, 5.4, y + 1.22, 2.2, 0.45, [dict(text="損失が小さくなるよう更新", size=12, color=ORANGE, bold=True)])


import os as _os
FIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets")
if not _os.path.isdir(FIG_DIR):
    FIG_DIR = "/workspace/libero_RL/scripts/assets"


def fig_pic(slide, name, x, y, max_w, max_h, caption=None):
    from PIL import Image as _Image
    path = _os.path.join(FIG_DIR, name)
    w, h = _Image.open(path).size
    sc = min(max_w / w, max_h / h)
    pw, ph = w * sc, h * sc
    px = x + (max_w - pw) / 2
    pic = slide.shapes.add_picture(path, Inches(px), Inches(y), Inches(pw), Inches(ph))
    pic.name = name
    if caption:
        textbox(slide, x, y + ph + 0.03, max_w, 0.3, [dict(text=caption, size=11, color=GRAY, align=PP_ALIGN.CENTER)])
    return pic, ph


def side_notes(slide, x, y, w, items, h=0.9, gap=0.08, size=13):
    for i, (title, text) in enumerate(items):
        yy = y + i * (h + gap)
        box(slide, x, yy, 0.5, h, str(i + 1), fill=BLUE, line=None, size=16, color=WHITE, bold=True, name=f"no {i+1}")
        box(slide, x + 0.55, yy, w - 0.55, h, [dict(text=title, size=size, bold=True), dict(text=text, size=size - 1)],
            fill=PALE, line=LGRAY, size=size, align=PP_ALIGN.LEFT, name=f"note {i+1}")



def video_panel(slide, x, name, vid, poster, outcome, col, desc, so, pw=2.65, y_label=1.7):
    box(slide, x, y_label, pw, 0.45, name, fill=BLUE, line=None, size=14, color=WHITE, bold=True, name=f"label {name}")
    mv = slide.shapes.add_movie(_os.path.join(FIG_DIR, vid), Inches(x), Inches(y_label + 0.5), Inches(pw), Inches(pw),
                                poster_frame_image=_os.path.join(FIG_DIR, poster), mime_type="video/mp4")
    mv.name = f"video {name}"
    box(slide, x, y_label + 3.23, pw, 0.8, [dict(text=outcome, size=17, bold=True, color=WHITE), dict(text=desc, size=11, color=WHITE)],
        fill=col, line=None, size=14, name=f"result {name}")
    textbox(slide, x - 0.2, y_label + 4.06, pw + 0.4, 0.3, [dict(text=so, size=13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)])


def instruction_box(slide):
    box(slide, 0.6, 1.0, 12.1, 0.62, [dict(text="指示文（全手法で共通）:  \"put both the alphabet soup and the tomato sauce in the basket\"", size=15, bold=True, color=BLUE),
                                      dict(text="（アルファベットスープの缶とトマトソースの、両方をカゴに入れる。Task 0）", size=12, color=GRAY)],
        fill=LIGHT, line=BLUE, size=14, name="instruction")


# ================ old 1
import os
ASSET_BG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "title_bg_blue_mesh.jpg")
if not os.path.exists(ASSET_BG):
    ASSET_BG = "/workspace/libero_RL/scripts/assets/title_bg_blue_mesh.jpg"


def _plain_run(p, text, size, color, font, bold=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.name = font
    f.color.rgb = color
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", font)
    return r


s = prs.slides.add_slide(BLANK)
bgpic = s.shapes.add_picture(ASSET_BG, 0, 0, prs.slide_width, prs.slide_height)
bgpic.name = "background (blue mesh)"
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(2.39))
band.name = "title band"
band.fill.solid()
band.fill.fore_color.rgb = RGBColor(0x33, 0x33, 0x33)
band.line.fill.background()

# series logo (text, Impact, per-letter colours) at the left of the band
logo = s.shapes.add_textbox(Inches(0.2), Inches(0.55), Inches(3.02), Inches(1.04))
logo.name = "series logo"
tf = logo.text_frame
tf.word_wrap = False
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
for t, c in (("D", RGBColor(0xE9, 0x4C, 0x3F)), ("EEP ", WHITE), ("L", RGBColor(0xC8, 0xCC, 0x41)),
             ("EARNING ", WHITE), ("JP", RGBColor(0x48, 0x97, 0xF6))):
    _plain_run(p, t, 28, c, "Impact")
p2 = tf.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
_plain_run(p2, "[DL Papers]", 28, WHITE, "Impact")

# title: white Arial, centred, two lines
tt = s.shapes.add_textbox(Inches(3.22), Inches(0.17), Inches(10.0), Inches(1.0))
tt.name = "title"
tf = tt.text_frame
tf.word_wrap = True
for i, line_ in enumerate(("RECAP, STEAM, and RL Token:", "Reinforcement Learning for Vision-Language-Action Models")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    _plain_run(p, line_, 26, WHITE, "Arial")

# presenter / affiliation block (3 lines, left aligned)
au = s.shapes.add_textbox(Inches(4.27), Inches(1.15), Inches(8.6), Inches(1.1))
au.name = "presenter block"
tf = au.text_frame
tf.word_wrap = True
for i, line_ in enumerate(("smolVLA × LIBERO-10 Task 0 での実験", "RLinf / libero_RL プロジェクト", "2026/10/04")):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    _plain_run(p, line_, 20, WHITE, "Yu Gothic")

# URL at the bottom right
url = s.shapes.add_textbox(Inches(10.43), Inches(6.62), Inches(2.91), Inches(0.4))
url.name = "url"
tf = url.text_frame
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
_plain_run(p, "http://deeplearning.jp/", 16, WHITE, "Yu Gothic")
s.notes_slide.notes_text_frame.text = (
    "RECAP、STEAM、RL Token の 3 手法を紹介し、smolVLA (LIBERO-10 Task 0) での実験結果を比較します。")

# ================ old 2
s = new_slide("概要", "発表全体の要約。結果の数値は、「実験結果」のスライドの表を参照。")
rows = [
    ("目的", "実環境の経験（ロールアウト）から VLA を改善する 3 つの強化学習手法を理解し、同じ設定（smolVLA、LIBERO-10 Task 0）で比較する"),
    ("背景", "模倣学習（SFT）だけでは、精度・速度・失敗からの回復に限界がある。10 億パラメータ級の VLA 全体を RL で更新するのは高コスト"),
    ("手法", "**RECAP**: 価値関数から advantage を作り、方策に条件付けて学習　**STEAM**: 時間オフセットの自己教師ありアンサンブルで advantage を推定　"
            "**RL Token**: 凍結した VLA の読み出しトークンの上で、軽量な actor-critic をオンライン学習"),
    ("実験", "smolVLA（lerobot/smolvla_libero）で、Task 0（2 つの物体をカゴへ）の success_once を、固定の初期状態 50 個で測る。実装は RLinf"),
    ("結果", f"ベース {R['base']/100:.2f} に対し、^^RL Token が {R['rlt']/100:.2f}^^、RECAP が {R['recap']/100:.2f}、STEAM が {R['steam']/100:.2f}（いずれも最良値）。"
            "RECAP と STEAM は学習の途中段階で、ベースとの差は、50 エピソードの評価では有意とは言えない"),
    ("結論", "RECAP と STEAM は、どちらも advantage から「良い」の札を作って方策を学習し、評価者（advantage の作り方）だけが違う。RL Token は、オンライン RL で行動を洗練する。成功で打ち切る収集設計が、価値の信号に重要だった"),
]
y = 1.15
for label, text in rows:
    box(s, 0.6, y, 1.5, 0.78, label, fill=BLUE, line=None, size=18, color=WHITE, bold=True, name=f"label {label}")
    textbox(s, 2.3, y - 0.02, 10.4, 0.85, [dict(text=text, size=15)], anchor=MSO_ANCHOR.MIDDLE, name=f"text {label}")
    y += 0.9

# ================ old 3
s = new_slide("書誌情報", "3 論文の書誌。本資料の日本語訳は docs/ にある（機械翻訳）。")
rows = [
    ["手法", "論文", "所属", "主な著者", "arXiv"],
    ["RECAP", "π*0.6: a VLA That Learns From Experience", "Physical Intelligence", "Ali Amin, Sergey Levine ほか", "2511.14759\n（2025/11）"],
    ["STEAM", "STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning",
     "中国科学院自動化研究所、清華大学 ほか", "Zhihao Liu, Qiuyi Gu ほか", "2606.29834"],
    ["RL Token", "RL Token: Bootstrapping Online RL with Vision-Language-Action Models",
     "Physical Intelligence", "Charles Xu, Jost Tobias Springenberg, Sergey Levine, Liyiming Ke ほか", "2604.23073\n（2026/04）"],
]
table(s, 0.6, 1.2, 12.1, [1.3, 3.8, 2.4, 2.9, 1.7], rows, size=14, row_h=1.0, header_h=0.5)
textbox(s, 0.6, 4.95, 12.1, 1.1, [
    (0, "実装基盤: **RLinf**（強化学習の学習基盤。RECAP / STEAM のパイプラインと、RL Token 用の actor-critic を使用）"),
    (0, "ベースモデル: **smolVLA**（lerobot/smolvla_libero、約 4.5 億パラメータ）。参考値として、RLinf のチュートリアルにある π₀.₅ の RECAP 結果も使う"),
], size=15)
key_box(s, "論文の内容（引用）と、私たちの実験（smolVLA）を、スライド上で区別して示す")

# ================ new basics
s = new_slide("強化学習の基本用語", "強化学習を知らない人向けの前提。『やってみて、結果から、良い行動を増やす』が一言。")
node(s, 0.6, 1.2, 2.4, 1.2, "エージェント", ["方策 π（= VLA）が", "行動を決める"], kind="data", size=15, name="agent")
node(s, 4.3, 1.2, 2.2, 1.2, "環境", ["シミュレータ / 実世界"], kind="plain", size=15, name="env")
arrow(s, 3.0, 1.5, 4.3, 1.5)
textbox(s, 3.0, 1.1, 1.3, 0.35, [dict(text="行動 a", size=13, color=BLUE, bold=True, align=PP_ALIGN.CENTER)])
arrow(s, 4.3, 2.1, 3.0, 2.1)
textbox(s, 3.0, 2.15, 1.3, 0.7, [dict(text="観測 o\n報酬 r", size=13, color=BLUE, bold=True, align=PP_ALIGN.CENTER)])
textbox(s, 0.6, 3.0, 5.9, 0.75, [dict(text="観測 o = カメラ画像と関節の状態。これを何百ステップも繰り返す = 1 エピソード", size=13, color=GRAY)])
node(s, 0.6, 3.9, 2.85, 1.55, "模倣学習（SFT）", ["お手本の行動を", "そのまま真似る"], kind="plain", size=15, name="sft")
node(s, 3.65, 3.9, 2.85, 1.55, "強化学習（RL）", ["結果（報酬）を見て、", "良い行動を増やす"], kind="data", size=15, name="rl")
terms = [
    ("方策 π", "観測から行動を決める関数。ここでは VLA（画像と指示文から、ロボットの動きを出力するモデル）"),
    ("報酬 r", "行動の結果の良さを表す数値（例: 成功で +1、毎ステップ −1）"),
    ("return", "これから先に得られる報酬の合計"),
    ("価値 V", "いまの状態から、将来得られる return の予想"),
    ("advantage A", "その行動で、予想よりどれだけ良くなったか（実際 − 予想）"),
    ("評価者（critic）", "状態や行動の良さを、数値で答えるモデル。RECAP は価値関数 V、STEAM は時間差を当てる critic"),
]
for i, (t, d) in enumerate(terms):
    y = 1.1 + i * 0.84
    box(s, 6.9, y, 1.9, 0.74, t, fill=BLUE, line=None, size=14, color=WHITE, bold=True, name=f"term {t}")
    textbox(s, 8.9, y - 0.02, 3.9, 0.78, [dict(text=d, size=13)], anchor=MSO_ANCHOR.MIDDLE, name=f"def {t}")
key_box(s, "RL は、結果から良い行動を増やす学習。RECAP・STEAM は、評価者（critic）で advantage を作る")

# ================ old 4
s = new_slide("背景: なぜ VLA に強化学習か", "RECAP と STEAM は評価者（advantage の作り方）だけが違う。RL Token は、オンラインで行動を洗練する、という整理が主題。")
textbox(s, 0.6, 1.1, 5.6, 4.9, [
    (0, "**VLA は、模倣学習（SFT）で多様な技能を学べる**"),
    (1, "ただし、実タスクが要求する精度・速度・失敗からの回復には届かない"),
    (0, "**強化学習（RL）で、自分の経験から改善したい**"),
    (1, "10 億パラメータ級のモデル全体を、オンラインで更新するのは、計算コストもサンプル効率も悪い"),
    (1, "flow matching の行動生成は、対数尤度が扱いにくく、方策勾配法（PPO など）を適用しにくい"),
    (0, "**3 手法の位置づけ**: RECAP と STEAM は評価者だけが違う。RL Token は、行動を洗練する"),
], size=17)
cards = [
    ("RECAP", "方策の学習目標", "評価者は価値関数。成功への近さ V から advantage を作り、「良い」の札を付けて方策を学習する"),
    ("STEAM", "方策の学習目標", "評価者は、時間オフセットを当てる critic のアンサンブル。advantage を作り、同じ札付きの学習で方策を学習する"),
    ("RL Token", "行動の洗練", "VLA は凍結。読み出しトークンの上の小さな actor-critic が、オンライン RL で VLA の行動を改善する"),
]
for i, (t, where, desc) in enumerate(cards):
    y = 1.1 + i * 1.7
    box(s, 6.5, y, 2.0, 1.5, [f"^^{t}^^"], fill=PALE, line=BLUE, size=20, name=f"card {t}")
    box(s, 8.6, y, 4.1, 1.5, [f"**{where}**", desc], fill=WHITE, line=LGRAY, size=14,
        align=PP_ALIGN.LEFT, name=f"desc {t}")
key_box(s, "RECAP と STEAM は評価者だけが違う。RL Token は、オンラインで行動を洗練する")

# ================ old 5
s = new_slide("実験設定", "評価は 50 エピソード。標準誤差が大きいことを最初に断っておく。")
textbox(s, 0.6, 1.1, 7.7, 5.0, [
    (0, "**タスク**: LIBERO-10 の Task 0「put both the alphabet soup and the tomato sauce in the basket」（2 つの物体をカゴに入れる）"),
    (0, "**モデル**: smolVLA（lerobot/smolvla_libero）。50 ステップの行動チャンクのうち 10 ステップを実行"),
    (0, "**評価**: 固定の初期状態 50 個を 1 回ずつ、最大 520 ステップ。指標は success_once（1 度でも成功）"),
    (0, "**ロールアウトの収集**: 自律ロールアウト 4,122 本（成功 41.0%）。成功した時点で打ち切る設定で再収集"),
    (1, "旧データは、成功後も 520 ステップ続けたため、return が「時間」と「成否」だけで決まっていた"),
    (0, "**人のデモ**: Task 0 の 3 本（836 フレーム）"),
    (0, "**実装**: RLinf（RECAP / STEAM のパイプライン、RL Token の actor-critic）"),
    (0, "**GPU**: RECAP は **RTX 6000 Ada（48GB）**、それ以外は **RTX A5000（24GB）**"),
], size=17)
stats = [("50", "初期状態（固定）"), ("0.40", "ベースの success_once（20/50）"), ("4,122", "ロールアウト（成功 41.0%）")]
for i, (n, label) in enumerate(stats):
    y = 1.15 + i * 1.65
    box(s, 8.7, y, 4.0, 1.45, [dict(text=n, size=40, bold=True, color=BLUE), dict(text=label, size=15)], fill=PALE, line=None, size=15, name=f"stat {n}")
key_box(s, "評価は 50 エピソード。2 つの評価の差の誤差は約 ±0.1 なので、0.1 程度の差は有意とは言えない")

# ================ new recap_prop
s = new_slide("RECAP: 提案（論文: π*0.6）", "RECAP は、価値関数の出力を『良い/悪い』の札として方策に条件付けるのが核心。")
textbox(s, 0.6, 1.1, 6.6, 5.0, [
    (0, "**考え方**: うまくいった行動に「良い」の札を付けて学習し、実行時は「良い」の札を付けて動かす"),
    (0, "次の 3 ステップを繰り返す"),
    (1, "① **集める**: ロボットを動かし、結果（成功/失敗）を記録する"),
    (1, "② **価値関数を学習する**: 「あとどれくらいで成功か」を予想するモデル"),
    (1, "③ **方策を学習する**: 良い行動に「良い」の札を付けて、VLA を学習する"),
    (0, "**利点**"),
    (1, "方策勾配が不要で、flow matching の VLA にもスケールする"),
    (1, "デモ・自律ロールアウト・人の介入を、同じ枠組みで使える"),
    (0, "**論文の結果**: 洗濯物たたみ・エスプレッソ作り・箱の組み立てで、成功率とスループットが改善。難しいタスクでは、失敗率が 1/2 以下"),
], size=15)
nodes = [("① 集める\n（ロールアウト）", 7.7, 1.2), ("② 価値関数 V を学習\n（成功への近さを予想）", 7.7, 2.75), ("③ 「良い」札付きで\n方策を学習", 7.7, 4.3)]
for t, x, y in nodes:
    box(s, x, y, 3.6, 1.0, t.split("\n"), fill=LIGHT, line=BLUE, size=15, bold=True, name=t.split("\n")[0])
arrow(s, 9.5, 2.2, 9.5, 2.75)
arrow(s, 9.5, 3.75, 9.5, 4.3)
arrow(s, 11.3, 4.8, 12.3, 4.8)
arrow(s, 12.3, 4.8, 12.3, 1.7)
arrow(s, 12.3, 1.7, 11.3, 1.7)
textbox(s, 11.4, 5.35, 1.6, 0.6, [dict(text="新しい方策で\n再収集", size=12, color=GRAY, align=PP_ALIGN.CENTER)])
key_box(s, "「良い行動」を選んで学習するための札を、価値関数から作る。それが RECAP の核心")

# ================ new fig_recap
s = new_slide("RECAP: アーキテクチャ（論文の図）", "論文 π*0.6 の図 3 より。下が価値関数、上が VLA。")
fig_pic(s, "fig_recap.png", 0.6, 1.05, 6.6, 4.85, "図: π*0.6 論文（Physical Intelligence）図 3 より")
side_notes(s, 7.4, 1.1, 5.3, [
    ("価値関数（図の下）", "画像と指示文から、価値 V を出す（SigLIP + Gemma 270M に、value head を付けたもの）"),
    ("advantage", "A = r + V(o_t+N) − V(o_t)。行動の前後で、成功に近づいたか"),
    ("2 値化", "A > ε なら「良い」。この札（metadata）を、VLA の入力に加える"),
    ("VLA（図の上）", "画像・指示文・札から、行動を出す。離散化した行動と、action expert（860M）が出す連続の行動"),
], h=1.1, size=13)
key_box(s, "価値関数が作った「良い/悪い」の札を、VLA の入力に加えて学習する")

# ================ new recap_return
s = new_slide("RECAP: return の計算方法", "Step1。実データ 368 エピソード（r1_m0）で、この式と一致することを確認済み。")
textbox(s, 0.6, 1.0, 5.9, 0.4, [dict(text="① 1 ステップごとの報酬", size=15, bold=True, color=BLUE)])
vals_s = ["−1", "−1", "−1", "…", "−1", "0"]
vals_f = ["−1", "−1", "−1", "…", "−1", "−300"]
box(s, 0.6, 1.5, 0.9, 0.5, "成功", fill=GREEN, line=None, size=13, color=WHITE, bold=True, name="succ label")
box(s, 0.6, 2.1, 0.9, 0.5, "失敗", fill=ORANGE, line=None, size=13, color=WHITE, bold=True, name="fail label")
for i, (a, b) in enumerate(zip(vals_s, vals_f)):
    last = i == len(vals_s) - 1
    box(s, 1.6 + i * 0.8, 1.5, 0.72, 0.5, a, fill=GREEN if last else LIGHT, line=BLUE, size=13, color=WHITE if last else BLACK, bold=last, name=f"s{i}")
    box(s, 1.6 + i * 0.8, 2.1, 0.72, 0.5, b, fill=ORANGE if last else LIGHT, line=BLUE, size=13, color=WHITE if last else BLACK, bold=last, name=f"f{i}")
textbox(s, 1.6, 2.62, 4.9, 0.3, [dict(text="フレーム 0 → 最後のフレーム（最後だけ、結果で値が違う）", size=11, color=GRAY)])
textbox(s, 0.6, 3.0, 5.9, 0.4, [dict(text="② return = そのフレーム以降の報酬の合計", size=15, bold=True, color=BLUE)])
textbox(s, 0.6, 3.4, 5.9, 0.9, [
    (0, "return_t = r_t + r_{t+1} + … + r_最後（割引なし）"),
    (0, "早く成功するほど、return は 0 に近い"),
], size=13, space=3)
textbox(s, 0.6, 4.25, 5.9, 0.4, [dict(text="③ 正規化（価値モデルの正解にするため）", size=15, bold=True, color=BLUE)])
textbox(s, 0.6, 4.65, 5.9, 0.6, [(0, "最大 520 フレームの失敗の最小値 −819 で割り、[−1, 0] にそろえる（0 が成功、−1 が最悪）")], size=13)
node(s, 0.6, 5.3, 2.85, 0.85, "成功（260 フレーム）", ["100 フレーム目: −159 → −0.19"], kind="out", size=12, name="ex succ")
node(s, 3.65, 5.3, 2.85, 0.85, "失敗（520 フレーム）", ["100 フレーム目: −719 → −0.88"], kind="train", size=12, name="ex fail")
from pptx.chart.data import XyChartData
cd = XyChartData()
a = cd.add_series("成功（260 フレームで終了）")
a.add_data_point(0, -259)
a.add_data_point(259, 0)
b = cd.add_series("失敗（520 フレーム、最後に −300）")
b.add_data_point(0, -819)
b.add_data_point(519, -300)
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(6.8), Inches(1.0), Inches(5.9), Inches(4.5), cd)
ch = gf.chart
style_chart(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "フレームごとの return（例）"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
va = ch.value_axis
va.minimum_scale = -900
va.maximum_scale = 0
va.major_unit = 300
va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
va.tick_labels.font.size = Pt(12)
ca = ch.category_axis
ca.minimum_scale = 0
ca.maximum_scale = 540
ca.major_unit = 100
ca.tick_labels.font.size = Pt(12)
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
for ser, col in zip(ch.plots[0].series, (GREEN, ORANGE)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(3)
    ser.smooth = False
textbox(s, 6.8, 5.5, 5.9, 0.5, [dict(text="横軸: フレーム番号。縦軸: return（正規化前）。傾きは 1 ステップあたり +1", size=12, color=GRAY)])
key_box(s, "return は、「そのフレームから先の報酬の合計」。成功に近いほど 0、失敗は −300 の罰の分だけ低くなる")

# ================ new recap_value
s = new_slide("RECAP ①: 価値関数の学習（入力・出力・更新）", "Step2。価値モデルだけを更新する。VLA はまだ触らない。")
io_diagram(s, ("入力", ["カメラ画像", "指示文"]), ("価値モデル", ["SigLIP2 + Gemma3-270M", "更新する"]),
           ("出力", ["201 個のビンの確率", "（成功への近さの分布）"]), ("V（価値）", ["確率の期待値", "例: −0.12"]),
           ("正解", ["その後の実際の return を", "1 つのビンにしたもの"]), ("損失", ["予想の分布と", "正解のズレ"]))
spec_table(s, 0.6, 4.15, 12.1, [
    ["入力", "ロールアウトの 1 フレーム（カメラ画像、指示文）。成功・失敗の両方のデータを使う"],
    ["出力", "201 個のビンの確率。確率の期待値が、価値 V（−1 〜 0。0 が成功）"],
    ["正解", "そのフレームから先の return（1 ステップ −1、失敗は終端で −300）を正規化し、ビンにしたもの"],
    ["損失 / 更新", "交差エントロピーを小さくするよう、価値モデルのパラメータを更新（VLA は更新しない）"],
], size=13, row_h=0.42, label_w=1.6)
key_box(s, "価値モデルは、画像を見て「成功まであとどれくらいか」を当てる問題として学習する", y=6.35, h=0.6)

# ================ new recap_adv
s = new_slide("RECAP ②: advantage の計算と「良い/悪い」の札", "Step3。モデルの更新はなく、データに札を付ける処理。")
node(s, 0.6, 1.15, 3.6, 0.8, "いまの価値 V(o_t)", kind="data", size=14, name="v0")
node(s, 0.6, 2.1, 3.6, 0.8, "10 ステップ後の価値 V(o_t+10)", kind="data", size=14, name="v1")
node(s, 0.6, 3.05, 3.6, 0.8, "その間の報酬の合計 r", kind="data", size=14, name="r")
node(s, 5.0, 1.65, 3.2, 1.7, "advantage", ["A = r + V(o_t+10) − V(o_t)", "（行動の前後の変化）"], kind="loss", size=15, name="adv")
node(s, 9.0, 1.15, 3.7, 1.15, "上位 30%", ["→ positive（良い）"], kind="out", size=15, name="pos")
node(s, 9.0, 2.7, 3.7, 1.15, "それ以外", ["→ negative（札なし）"], kind="frozen", size=15, name="neg")
for y in (1.55, 2.5, 3.45):
    arrow(s, 4.2, y, 5.0, 2.5)
arrow(s, 8.2, 2.5, 9.0, 1.7)
arrow(s, 8.2, 2.5, 9.0, 3.2)
textbox(s, 0.6, 4.1, 12.1, 0.4, [dict(text="例（数値は説明用）", size=14, bold=True, color=BLUE)])
ex = [("V(o_t) = −0.50", "data"), ("V(o_t+10) = −0.45", "data"), ("r = −0.01", "data"), ("A = −0.01 − 0.45 + 0.50 = +0.04", "loss"), ("閾値 0.02 より大きい → positive", "out")]
xs = [0.6, 2.95, 5.4, 7.55, 10.55]
ws = [2.2, 2.3, 1.95, 2.85, 2.15]
for (t, k), x, w in zip(ex, xs, ws):
    node(s, x, 4.5, w, 0.9, t, kind=k, size=13, name=f"ex {t[:6]}")
textbox(s, 0.6, 5.55, 12.1, 0.55, [dict(text="閾値は、全データの advantage を並べて、上位 30% に入る値。人のデモは、全て positive にする", size=13, color=GRAY)])
key_box(s, "行動の前後で、成功に近づいたか（V が上がったか）で、「良い」の札を付けるフレームを決める")

# ================ new recap_policy
s = new_slide("RECAP ③: 方策の学習（入力・出力・更新）", "Step4。札を付けたデータで VLA を学習する。実行時は『良い』の札を付ける。")
textbox(s, 0.6, 1.0, 2.0, 0.4, [dict(text="学習時", size=16, bold=True, color=BLUE)])
node(s, 0.6, 1.4, 3.3, 1.3, "入力", ["カメラ画像・関節状態・指示文", "+ 札「Advantage: positive」"], kind="data", size=14, name="in")
node(s, 4.5, 1.4, 2.6, 1.3, "VLA（smolVLA）", ["更新する", "（損失を小さく）"], kind="train", size=15, name="vla")
node(s, 7.7, 1.4, 2.2, 1.3, "予測した", ["行動チャンク"], kind="data", size=14, name="pred")
node(s, 10.5, 1.4, 2.2, 1.3, "損失", ["予測と実際の", "行動のズレ"], kind="loss", size=14, name="loss")
arrow(s, 3.9, 2.05, 4.5, 2.05)
arrow(s, 7.1, 2.05, 7.7, 2.05)
arrow(s, 9.9, 2.05, 10.5, 2.05)
node(s, 7.7, 3.05, 5.0, 0.55, "正解: データの実際の行動チャンク", kind="data", size=13, name="teacher")
arrow(s, 11.6, 3.05, 11.6, 2.7)
textbox(s, 0.6, 2.95, 6.8, 0.6, [dict(text="positive のフレームには札を付け、10% の確率で札を外す（札なしの方策も一緒に学習）", size=12, color=GRAY)])
textbox(s, 0.6, 3.85, 2.0, 0.4, [dict(text="実行時（評価）", size=16, bold=True, color=GREEN)])
node(s, 0.6, 4.25, 3.3, 1.3, "入力", ["カメラ画像・関節状態・指示文", "+ 札「Advantage: positive」"], kind="data", size=14, name="in2")
node(s, 4.5, 4.25, 2.6, 1.3, "VLA（学習済み）", ["更新しない"], kind="frozen", size=15, name="vla2")
node(s, 7.7, 4.25, 5.0, 1.3, "出力: 行動チャンク", ["「良い」の札を付けたので、良い行動が出やすい"], kind="out", size=15, name="out2")
arrow(s, 3.9, 4.9, 4.5, 4.9)
arrow(s, 7.1, 4.9, 7.7, 4.9)
legend(s, 0.6, 5.7)
key_box(s, "学習では「良い札付き」のデータを多く見せ、実行時に「良い」の札を付けて、良い行動を引き出す", y=6.3, h=0.65)

# ================ new recap_eq
s = new_slide("RECAP: 式の補足（論文の式）", "式(1)(3) の記号は論文のもの。式の意味は、直前の 3 枚の図と対応。")
eq_block(s, 1.0, 1, "**価値関数**: 「これから先の return が、どのビンに入るか」を、確率で答える。正解のビンとのズレ（交差エントロピー H）を小さくする",
         "min_{φ}  E_{τ}[ Σ_{t} H( R^{B}_{t}(τ), p_{φ}(V | o_{t}, ℓ) ) ]　…(1)")
eq_block(s, 2.27, 2, "**advantage**: 行動の前後で、成功への近さがどれだけ変わったか（N ステップ先まで見る）",
         "A(o_{t}, a_{t}) = r_{t:t+N} + V(o_{t+N}) − V(o_{t})")
eq_block(s, 3.54, 3, "**方策の学習**: 「良い」の札付きと、札なしの両方で、データの行動を再現する（I = 1 が「良い」。ε_ℓ は閾値）",
         "min_{θ}  E[ − log π_{θ}(a_{t} | o_{t}, ℓ) − α log π_{θ}(a_{t} | I_{t}, o_{t}, ℓ) ] ,　I_{t} = 1[ A > ε_{ℓ} ]　…(3)")
eq_block(s, 4.81, 4, "**実行時**: 札を「良い」に固定して行動を生成する。β > 1 にすると、「良い」の方向をさらに強調できる",
         "π̂(a | o, ℓ) ∝ π(a | o, ℓ) · ( π(a | I, o, ℓ) / π(a | o, ℓ) )^{β}　…(2)")
key_box(s, "方策は「札あり」と「札なし」の両方を表現できるので、改善確率を明示的に推定しなくてよい", y=6.3)

# ================ old 8
s = new_slide("RECAP: smolVLA での実装（私たちの実験）", "Step1〜4 と評価（Step5）。名称は私たちのパイプラインの番号。【発表者向け】0.50 は、advantage プロンプトなしで評価した最良値。プロンプトあり（Advantage: positive）では 0.48。Step4 は途中段階（2,000 step）。")
steps = ["Step1\nreturn の計算", "Step2\n価値モデルの学習", "Step3\nadvantage の計算", "Step4\n条件付き SFT", "Step5\nシミュレータで評価"]
for i, t in enumerate(steps):
    x = 0.6 + i * 2.5
    box(s, x, 1.1, 2.2, 0.95, t.split("\n"), fill=BLUE if i < 4 else GREEN, line=None, size=14, color=WHITE, bold=True,
        name=f"step {i}")
    if i < 4:
        arrow(s, x + 2.2, 1.57, x + 2.5, 1.57, color=GRAY)
textbox(s, 0.6, 2.3, 12.1, 3.9, [
    (0, "**Step1**: 報酬は 1 ステップごとに −1、失敗は終端で −300。return を [−819, 0] に正規化"),
    (0, "**Step2**: 価値モデル（SigLIP2 + Gemma3-270M）が、201 ビンの分布を予測。**成功した時点で打ち切る設定で再収集した 4,122 本**で学習（**4,000 step、20.6 時間**、RTX 6000 Ada 48GB）"),
    (1, "旧データでは、成功後も 520 ステップ続けていたため、価値が「フレーム番号と成否」でほぼ決まっていた"),
    (0, "**Step3**: A = r_{t:t+10} + V(o_{t+10}) − V(o_t)。上位 30% を positive（人のデモは全て positive）。閾値は全データセットを結合して 1 つに決める（**62.3 万サンプル、4.2 時間**、RTX 6000 Ada 48GB）"),
    (0, "**Step4**: プロンプトに \"Advantage: positive\" を付けて SFT（ドロップアウト 0.1）。評価も positive プロンプトで行う（**2,000 step、20.3 時間**、RTX 6000 Ada 48GB）"),
    (0, "**Step5**: success_once は **0.50**（最良。ベース 0.40）。Step4 は途中段階（2,000 step 前後、計画は 30,000 step）で、評価は 0.44〜0.50 の範囲で変動"),
], size=17)
key_box(s, "価値モデルの信号は、データの作り方（成功で打ち切るか）に強く依存した")

# ================ new steam_prop
s = new_slide("STEAM: 提案（論文）", "報酬ラベル不要で、専門家デモだけから advantage を作る点が特徴。")
textbox(s, 0.6, 1.1, 6.4, 5.0, [
    (0, "**STEAM**: Self-supervised Temporal Ensemble Advantage Modeling"),
    (0, "**考え方**: デモ動画の 2 つのフレームを見せて、「何ステップ離れているか」を当てる critic を学習する"),
    (1, "答えは、デモの時間から自動で分かる（人のラベルや報酬が不要 = 自己教師あり）"),
    (0, "**前進 = 良い行動**: 時間が進む方向に動く行動に、高いスコアを付ける"),
    (1, "動画を逆再生すると、「後退」の例も作れる"),
    (0, "**アンサンブル**: 複数の critic のうち、最も厳しい評価（最小値）を使い、過大評価を防ぐ"),
    (0, "**方策学習**: スコアの高いフレームに札を付けて学習（RECAP と同じ「条件付け」）"),
], size=15)
tl = ["f1", "f2", "f3", "f4", "f5", "f6"]
for i, t in enumerate(tl):
    node(s, 7.3 + i * 0.9, 1.4, 0.75, 0.75, t, kind="data" if i not in (1, 4) else "loss", size=14, name=f"frame {t}")
textbox(s, 7.3, 1.0, 5.4, 0.4, [dict(text="デモ動画のフレーム（時間の順）", size=13, color=GRAY)])
arrow(s, 8.55, 2.3, 11.15, 2.3, color=GREEN)
textbox(s, 7.3, 2.35, 5.4, 0.7, [dict(text="f2 → f5: +3 ステップ先（前進）\n正解の時間差 = +3", size=13, color=GREEN, bold=True, align=PP_ALIGN.CENTER)])
arrow(s, 11.15, 3.25, 8.55, 3.25, color=ORANGE)
textbox(s, 7.3, 3.3, 5.4, 0.7, [dict(text="逆再生: f5 → f2\n正解の時間差 = −3（後退）", size=13, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)])
box(s, 7.3, 4.2, 5.4, 1.2, ["**正規化**", "エピソードの長さで時間差を割り、長さの違うデモを、同じ尺度でそろえる"], fill=PALE, line=LGRAY, size=13, align=PP_ALIGN.LEFT, name="norm")
key_box(s, "報酬を作らず、デモの時間構造から、「前進」と「停滞・後退」を区別する advantage を学ぶ")

# ================ new fig_steam
s = new_slide("STEAM: アーキテクチャ（論文の図）", "論文 STEAM の図 1 より。左から、ラベル付け、予測器の学習、方策の改善。")
fig_pic(s, "fig_steam.png", 0.6, 1.1, 12.1, 2.6, "図: STEAM 論文図 1 より")
for i, (t, d) in enumerate([
    ("① フレームペアのラベル付け", "専門家デモの 2 つのフレーム（前進 forward / 後退 backward）の時間差を、エピソード長で正規化して、正解にする"),
    ("② 時間オフセット予測器の学習", "2 つのフレームと指示文から、時間差の分布を出す。M 個の予測器を学習して、アンサンブルにする"),
    ("③ 方策の改善", "専門家・ロールアウト・人の修正のデータに、アンサンブルで A_STEAM を付け、CFGRL（札付き学習）で VLA を学習する"),
]):
    box(s, 0.6 + i * 4.1, 4.15, 3.9, 1.9, [dict(text=t, size=16, bold=True, color=BLUE), dict(text=d, size=14)],
        fill=PALE, line=LGRAY, size=14, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=f"panel {i+1}")
key_box(s, "報酬を使わず、デモの時間構造から advantage を作り、VLA を学習する")

# ================ new steam_critic
s = new_slide("STEAM ①: critic の学習（入力・出力・更新）", "デモだけで学習する。報酬は使わない。")
io_diagram(s, ("入力", ["フレーム f_i と f_j", "（2 台のカメラ）+ 指示文"]), ("critic", ["SigLIP + Gemma3-270M", "更新する"]),
           ("出力", ["時間差の分布", "（N = 32 個のビン）"]), ("予想", ["時間差の期待値"]),
           ("正解", ["(j − i) を正規化して", "1 つのビンにしたもの"]), ("損失", ["予想の分布と", "正解のズレ"]))
spec_table(s, 0.6, 4.15, 12.1, [
    ["入力", "同じデモの中の、2 つのフレーム（f_i, f_j）。カメラ 2 台の画像と指示文"],
    ["出力", "「f_j は f_i の何ステップ先か」を、N 個のビンの確率で答える（プラスは前進、マイナスは後退）"],
    ["正解", "デモの実際の時間差 (j − i) × L_max / L を、1 つのビンにしたもの。逆再生のペアは、負の時間差"],
    ["損失 / 更新", "交差エントロピーを小さくするよう、critic のパラメータを更新（報酬ラベルは使わない）"],
], size=13, row_h=0.42, label_w=1.6)
key_box(s, "critic は、「2 枚のフレームが、時間でどれだけ離れているか」を当てる問題として学習する", y=6.35, h=0.6)

# ================ new steam_adv
s = new_slide("STEAM ②: advantage の計算（アンサンブルと最小値）", "モデルの更新はなく、データに札を付ける処理。smolVLA の実験では、critic は 1 個（M = 1）。")
node(s, 0.6, 2.0, 2.6, 1.5, "フレームのペア", ["f_i と、H ステップ先の", "f_i+H"], kind="data", size=14, name="pair")
for j, t in enumerate(("critic 1", "critic 2", "critic M")):
    node(s, 4.0, 1.15 + j * 1.15, 2.2, 0.85, t, ["A_" + str(j + 1 if j < 2 else "M")], kind="frozen", size=14, name=f"c{j}")
    arrow(s, 3.2, 2.75, 4.0, 1.58 + j * 1.15)
    arrow(s, 6.2, 1.58 + j * 1.15, 7.0, 2.75)
node(s, 7.0, 2.2, 2.0, 1.1, "最小値", ["min_m A_m"], kind="loss", size=15, name="min")
arrow(s, 9.0, 2.75, 9.6, 2.75)
node(s, 9.6, 1.9, 3.1, 1.7, "A_STEAM", ["閾値 δ_q 以上なら 1（良い）", "それ以外は 0"], kind="out", size=14, name="label")
box(s, 0.6, 4.75, 5.9, 1.2, ["**なぜ最小値か**", "全ての critic が「前進」と言うときだけ高いスコア。知らない状態での、過大評価を防ぐ"], fill=PALE, line=LGRAY, size=13, align=PP_ALIGN.LEFT, name="why")
box(s, 6.8, 4.75, 5.9, 1.2, ["**スコアの意味**", "予想した時間差が、基準の時間差より大きいほど、「効率よく前進した」行動"], fill=PALE, line=LGRAY, size=13, align=PP_ALIGN.LEFT, name="meaning")
key_box(s, "最も厳しい critic の評価で「前進」と判断したフレームにだけ、「良い」の札を付ける", y=6.2)

# ================ old 10
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

# ================ old 11
s = new_slide("STEAM: smolVLA での実装（私たちの実験）", "Step1〜4 と評価（Step5）。RECAP の実装スライドと同じ形式。Step 名は私たちのパイプラインの番号。【発表者向け】0.52 は、最初の 1,000 step は札付き SFT、続く 450 step は positive のフレームだけで学習したモデルを、advantage プロンプトなしで評価した値。札付き SFT（プロンプトあり）の評価は 0.42、両側タグ（negative にも札を付ける）は 0.36。")
steps = ["Step1\nフレームペアの作成", "Step2\ncritic の学習", "Step3\nadvantage の計算", "Step4\n方策の学習（SFT）", "Step5\nシミュレータで評価"]
for i, t in enumerate(steps):
    x = 0.6 + i * 2.5
    box(s, x, 1.1, 2.2, 0.95, t.split("\n"), fill=BLUE if i < 4 else GREEN, line=None, size=14, color=WHITE, bold=True,
        name=f"step {i}")
    if i < 4:
        arrow(s, x + 2.2, 1.57, x + 2.5, 1.57, color=GRAY)
textbox(s, 0.6, 2.3, 12.1, 3.9, [
    (0, "**Step1**: 人のデモ 3 本（重み 4）と、成功ロールアウトから、k = 32 ステップ先とのフレームペアを作る。逆再生のペアは、負の時間差。時間差は軌跡長で正規化し、32 ビンに分ける"),
    (0, "**Step2**: critic（SigLIP so400m + Gemma3-270M）が、時間差の 32 ビンの分布を予測（**2,000 step、バッチ 64、8.3 時間**、RTX A5000 24GB）"),
    (1, "学習データ上の accuracy は 0.277 → 0.353。未知のロールアウト（step 600）では 0.106 と低い"),
    (0, "**Step3**: ペアの予想から advantage を作る（アンサンブルのメンバーは 1 個）。ロールアウトの上位 30%、デモの上位 80% を positive（**1 データセット約 2.4 時間**、RTX A5000）"),
    (0, "**Step4**: 札付き SFT で 1,000 step、続けて positive のフレームだけで 450 step（**合計 1,450 step、16.5 時間**、RTX A5000）"),
    (0, "**Step5**: success_once は **0.52**（最良。ベース 0.40）"),
], size=16)
key_box(s, "critic は学習データ上では改善したが、未知のロールアウトでの精度は低い（今後の課題）")

# ================ new rlt_prop
s = new_slide("RL Token: 提案（論文）", "VLA を凍結して、RL token という小さな状態表現の上で RL をする。")
textbox(s, 0.6, 1.1, 6.6, 5.0, [
    (0, "**課題**: 巨大な VLA 全体を、オンラインで RL するのは、重くて、データも多く要る"),
    (0, "**発想**: VLA は凍結して「行動の提案役」にし、小さな追加ネットワークが、提案を微調整する"),
    (0, "**RL token**: VLA の内部情報を、1 本の短いベクトルに圧縮したもの。RL にとっての「状態」になる"),
    (0, "**学習するもの**: 小さな actor（行動を出す）と critic（行動の良さを予想する）だけ"),
    (0, "**論文の結果**: 実機で数時間の練習で、重要な局面の成功率と速度が向上。HIL-SERL、PLD などの他のオンライン RL より速く学習"),
], size=15)
node(s, 7.6, 1.2, 5.0, 1.3, "先生: VLA（凍結）", ["状況を見て、行動の「下書き」を提案する"], kind="frozen", size=15, name="teacher")
node(s, 7.6, 3.1, 5.0, 1.3, "生徒: 小さな actor（更新）", ["下書きを見て、「清書」（最終の行動）を出す"], kind="train", size=15, name="student")
node(s, 7.6, 5.0, 5.0, 0.8, "最終の行動", kind="out", size=15, name="final")
arrow(s, 10.1, 2.5, 10.1, 3.1)
arrow(s, 10.1, 4.4, 10.1, 5.0)
textbox(s, 10.25, 2.55, 2.4, 0.5, [dict(text="下書き ã", size=12, color=GRAY)])
key_box(s, "RL は小さな actor-critic だけが担当し、VLA の知識を壊さずに、数時間で行動を洗練できる")

# ================ new fig_rlt
s = new_slide("RL Token: アーキテクチャ（論文の図）", "論文 RL Token の図 2 より。VLA の内部の埋め込みを、1 本の RL token に圧縮する。")
fig_pic(s, "fig_rl_token.png", 0.6, 1.05, 5.4, 4.85, "図: RL Token 論文（Physical Intelligence）図 2 より")
side_notes(s, 6.3, 1.1, 6.4, [
    ("入力（図の下）", "3 台のカメラ画像と指示文を、VLA（SigLIP + Gemma）に入れる"),
    ("画像の埋め込み", "VLA の内部で作られる、画像の埋め込み（N × 2048）"),
    ("encoder → RL token", "埋め込みと、学習可能なトークン e_rl から、1 本の RL token（1 × 2048）に圧縮する"),
    ("decoder", "RL token から、元の埋め込みを再構成する。これを損失にして、RL token に情報を詰める"),
    ("行動チャンク（図の右）", "VLA の action expert が、ノイズから行動チャンクを出す。RL では、これを「参照行動」として使う"),
], h=0.92, size=13)
key_box(s, "VLA は凍結したまま、RL token が、RL にとっての小さな状態表現になる")

# ================ new rlt_step
s = new_slide("RL Token ①: 1 ステップの流れ（入力 → 出力）", "実行時と学習データの収集で、同じ流れ。灰色は凍結、オレンジは更新する部品。")
node(s, 0.5, 1.2, 2.3, 1.6, "入力", ["カメラ画像", "指示文", "関節状態"], kind="data", size=14, name="in")
node(s, 3.3, 1.2, 2.0, 1.6, "VLA", ["凍結"], kind="frozen", size=15, name="vla")
node(s, 5.9, 1.1, 2.4, 0.85, "RL token z_rl", ["凍結して抽出"], kind="frozen", size=13, name="tok")
node(s, 5.9, 2.2, 2.4, 0.85, "参照行動 ã", ["VLA の提案"], kind="data", size=13, name="ref")
node(s, 9.0, 1.2, 2.2, 1.6, "actor π_θ", ["入力: z_rl, 関節状態, ã", "更新する"], kind="train", size=14, name="actor")
node(s, 11.7, 1.2, 1.2, 1.6, "行動 a", ["チャンク"], kind="out", size=13, name="act")
arrow(s, 2.8, 2.0, 3.3, 2.0)
arrow(s, 5.3, 1.65, 5.9, 1.55)
arrow(s, 5.3, 2.35, 5.9, 2.6)
arrow(s, 8.3, 1.55, 9.0, 1.7)
arrow(s, 8.3, 2.6, 9.0, 2.4)
arrow(s, 11.2, 2.0, 11.7, 2.0)
node(s, 9.0, 3.6, 3.9, 0.9, "環境で行動を実行", ["次の観測 o' と報酬 r が返る"], kind="plain", size=14, name="env")
arrow(s, 12.3, 2.8, 12.3, 3.6)
node(s, 3.3, 3.6, 4.6, 0.9, "リプレイバッファに保存", ["(状態 x, 行動 a, 報酬 r, 次の状態 x')"], kind="data", size=14, name="buf")
arrow(s, 9.0, 4.05, 7.9, 4.05)
legend(s, 0.5, 4.75)
spec_table(s, 0.5, 5.2, 12.4, [
    ["入力 / 出力", "入力は、RL token（VLA の内部情報を圧縮したもの）、関節状態、VLA の提案 ã。出力は、行動チャンク（数ステップ分の行動のまとまり）"],
], size=13, row_h=0.5, label_w=1.7)
key_box(s, "actor は、VLA の提案 ã を見て、それを少し直した行動を出す。保存したデータが、次の更新に使われる", y=6.3, h=0.65)

# ================ new rlt_loop
s = new_slide("RL Token ②: 学習のループとパラメータ更新", "更新されるのは、小さな actor と critic だけ（actor の重みは約 5 MB）。")
node(s, 0.6, 1.15, 3.2, 1.45, "① 実行して集める", ["actor で行動し、結果を", "リプレイバッファに保存"], kind="plain", size=14, name="s1")
node(s, 4.4, 1.15, 3.2, 1.45, "② バッファから取り出す", ["過去のデータを、", "ランダムに何度も使う"], kind="data", size=14, name="s2")
node(s, 4.4, 3.3, 3.2, 1.65, "③ critic を更新", ["Q(x, a) を、r + γ Q(x', a')", "に近づける（TD 学習）"], kind="train", size=14, name="s3")
node(s, 0.6, 3.3, 3.2, 1.65, "④ actor を更新", ["Q が大きくなる行動を出す。", "ただし提案 ã から離れすぎない"], kind="train", size=14, name="s4")
arrow(s, 3.8, 1.88, 4.4, 1.88)
arrow(s, 6.0, 2.6, 6.0, 3.3)
arrow(s, 4.4, 4.12, 3.8, 4.12)
arrow(s, 2.2, 3.3, 2.2, 2.6)
textbox(s, 0.6, 5.05, 7.0, 0.7, [dict(text="更新後の actor で、また ① に戻る。critic の更新回数は、集めたデータ 1 件につき、複数回（UTD = 5）", size=12, color=GRAY)])
rows = [["部品", "更新", "役割"],
        ["VLA", "しない（凍結）", "行動の下書き ã を出す"],
        ["RL token の抽出", "しない（凍結）", "VLA の情報を、短いベクトルに圧縮"],
        ["actor π_θ", "する", "下書きを直して、最終の行動を出す"],
        ["critic Q_ψ", "する", "行動の良さを予想する"]]
table(s, 7.9, 1.15, 4.8, [1.35, 1.65, 1.8], rows, size=12, row_h=0.72, header_h=0.45, hl_rows=(3, 4))
box(s, 7.9, 4.9, 4.8, 0.95, ["**参照行動のドロップアウト**", "ã を、ときどき隠して、actor が「コピー」だけにならないようにする"], fill=PALE, line=LGRAY, size=12, align=PP_ALIGN.LEFT, name="dropout")
key_box(s, "「集める → critic を更新 → actor を更新」を繰り返し、小さな actor が VLA の提案を改善する", y=6.2)

# ================ new rlt_eq
s = new_slide("RL Token: 式の補足（論文の式）", "式の番号は論文のもの。式の意味は、直前の 2 枚の図と対応。")
eq_block(s, 1.0, 1, "**RL token の抽出**: 小さな transformer g_φ が、VLA の内部情報 z と、学習可能なトークン e_rl から、読み出し z_rl（RL token）を作る",
         "z_{rl} = g_{φ}( [ z_{1:M}, e_{rl} ] )_{M+1}　…(1)")
eq_block(s, 2.27, 2, "**再構成損失**: z_rl から、元の内部情報を復元できるように学習する（= z_rl に情報が詰まる。VLA は凍結）",
         "L_{ro} = E[ Σ_{i} ‖ h_{φ}( d_{φ}([ z_{rl}, z̄_{1:i−1} ]) )_{i} − z̄_{i} ‖² ]　…(2)")
eq_block(s, 3.54, 3, "**critic**: Q(x, a) を、「報酬 + 次の状態の Q」に近づける（C は行動チャンクの長さ、γ は割引率）",
         "L_{Q} = E[ ( Σ_{t'=1}^{C} γ^{t'−1} r_{t'} + γ^{C} E_{a' ~ πθ}[ Q_{ψ'}(x', a') ] − Q_{ψ}(x, a_{1:C}) )² ]　…(3)")
eq_block(s, 4.81, 4, "**actor**: Q が大きい行動を出す。ただし、VLA の提案 ã から離れすぎないよう、距離に罰を付ける（β が強さ）",
         "L_{π} = E[ − Q_{ψ}(x, a_{1:C}) + β ‖ a_{1:C} − ã_{1:C} ‖² ] ,　ã ~ π_{vla}(· | s, ℓ)　…(5)")
key_box(s, "actor は、VLA の提案 ã を条件に、ゼロからではなく、強い初期案を改善する", y=6.3)

# ================ old 14
s = new_slide("RL Token: smolVLA での実装（私たちの実験）", "Step1〜4 と評価（Step5）。RECAP・STEAM の実装スライドと同じ形式。Step 名は私たちのパイプラインの番号。")
steps = ["Step1\n特徴量の抽出", "Step2\nRL token の学習", "Step3\nactor-critic の学習", "Step4\n2 つ目専用の actor", "Step5\nシミュレータで評価"]
for i, t in enumerate(steps):
    x = 0.6 + i * 2.5
    box(s, x, 1.1, 2.2, 0.95, t.split("\n"), fill=BLUE if i < 4 else GREEN, line=None, size=14, color=WHITE, bold=True,
        name=f"step {i}")
    if i < 4:
        arrow(s, x + 2.2, 1.57, x + 2.5, 1.57, color=GRAY)
textbox(s, 0.6, 2.3, 12.1, 3.9, [
    (0, "**Step1**: 凍結した smolVLA に、デモとロールアウトの動画フレームを通し、画像トークン（128 × 960）を保存（把持の前後を中心に抽出。**1 データセット約 17 分**、RTX A5000）"),
    (0, "**Step2**: 小さな encoder-decoder が、保存した特徴から RL token を作る（再構成損失で学習。**6,000 step、バッチ 64、約 25 分**、RTX A5000）。再構成の val MSE は 0.073（平均を出す基準は 0.269）"),
    (0, "**Step3**: 1・2 つ目の把持の actor-critic を、オンライン RL で学習（3 環境、更新/データ比 5、探索ノイズ σ = 0.05。**750 エポック、バッチ 512、約 8.2 時間**、RTX A5000 24GB）。それ以外は VLA が実行"),
    (0, "**Step4**: 残る失敗は 2 つ目の把持が最大 → 2 つ目専用の actor を、初期値なしで学習（1 つ目は Step3 の actor を固定。**400 エポック、バッチ 512、約 5.5 時間**、RTX A5000）。空振りへの罰 −0.2 を加えると 0.68 で、効果は確認できなかった"),
    (0, "**Step5**: success_once は **0.74**（最良。3 つの評価シードの平均は 0.71）。VLA のみ 0.46、1 つ目だけ RL 0.62、2 つ目を VLA に任せる 0.68。actor の重みは約 5 MB"),
], size=16)
key_box(s, "凍結した VLA の上の小さな actor-critic だけで、success_once が 0.46 → 0.74 に向上した")

# ================ old 15
s = new_slide("実験結果", "π₀.₅ は RLinf のチュートリアルの値。smolVLA は私たちの実験（各 50 エピソード）。【発表者向け】RECAP 0.50 と STEAM 0.52 は、advantage プロンプトなしで評価した最良値。プロンプトあり（Advantage: positive）では RECAP 0.48、STEAM 0.42。")
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
    (0, f"smolVLA: 各 50 エピソード（標準誤差 ±0.07）。RECAP・STEAM は、Step4 の途中の checkpoint の最良値。評価条件の詳細は、口頭で補足"),
    (0, "π₀.₅ は RLinf チュートリアルの記載値で、モデル・設定・評価数が異なり、smolVLA との直接比較はできない"),
], size=12, color=GRAY, space=3)
key_box(s, f"smolVLA では RL Token が最良（{R['base']:.0f}% → {R['rlt']:.0f}%）。RECAP・STEAM は途中段階で、差は有意とは言えない")

# ================ new videos
s = new_slide("動画で見る手法の効果（初期状態 14）", "初期状態 14 での比較。ベースと RL Token の動画は wandb: https://wandb.ai/junhand/libero_RL/runs/w3vkksd0（run 名 demo-init14-base-fail-vs-rlt-success）。RECAP の動画は wandb: https://wandb.ai/junhand/libero_RL/runs/30cagwmg（run 名 recap-noprompt-vs-base-init14）。RL Token の動画は H（1 つ目は G の actor、2 つ目は専用 actor）。RECAP の動画は global_step_2000（advantage プロンプトなしで評価、success_once 0.50）。")
instruction_box(s)
video_panel(s, 1.0, "ベース（smolVLA）", "video_init14_base.mp4", "poster_init14_base.png", "失敗", ORANGE, "1 つ目は運ぶが、2 つ目をカゴに入れられない", "success_once 0.40")
video_panel(s, 5.1, "RECAP", "video_init14_recap.mp4", "poster_init14_recap.png", "成功", GREEN, "2 つともカゴに入る", "success_once 0.50")
video_panel(s, 9.2, "RL Token（最良）", "video_init14_rlt.mp4", "poster_init14_rlt.png", "成功", GREEN, "2 つともカゴに入る", "success_once 0.74")
textbox(s, 0.6, 6.0, 12.1, 0.3, [dict(text="2 環境のうち左側を 512 × 512 に拡大。クリックで再生（約 52 秒）。success_once は 50 エピソードの値。3 つとも同じ初期状態 14", size=11, color=GRAY)])
key_box(s, "同じ初期状態で、ベースは 2 つ目をカゴに入れられず失敗し、RECAP と RL Token は成功した（1 例）", y=6.4, h=0.55)

# ================ new videos8
s = new_slide("動画で見る手法の効果（初期状態 8）", "初期状態 8 での比較（STEAM）。wandb: https://wandb.ai/junhand/libero_RL/runs/kydvjtsn（run 名 demo-init8-base-fail-vs-steamP-success）。STEAM の動画は実験 P（positive のフレームだけで学習、advantage プロンプトなしで評価、success_once 0.52）。画面の reward / termination の文字は、評価の画面表示。")
instruction_box(s)
video_panel(s, 3.0, "ベース（smolVLA）", "video_init8_base.mp4", "poster_init8_base.png", "失敗", ORANGE, "缶がカゴの外のテーブルに残る", "success_once 0.40")
video_panel(s, 7.7, "STEAM", "video_init8_steam.mp4", "poster_init8_steam.png", "成功", GREEN, "缶がカゴの中に入る", "success_once 0.52")
textbox(s, 0.6, 6.0, 12.1, 0.3, [dict(text="2 環境のうち左側を 512 × 512 に拡大。クリックで再生（約 52 秒）。success_once は 50 エピソードの値。初期状態 8（前のページの初期状態 14 とは別）", size=11, color=GRAY)])
key_box(s, "同じ初期状態 8 で、ベースは失敗し、STEAM は成功した（1 例）", y=6.4, h=0.55)

# ================ new cost
s = new_slide("学習コストの比較（学習ステップ数・時間・GPU）", "実測値（TensorBoard と実行ログ）。GPU は、RECAP が RTX 6000 Ada、それ以外が RTX A5000。")
A5 = "RTX A5000（24GB）"
ADA = "RTX 6000 Ada（48GB）"
rows = [
    ["手法", "段階", "学習ステップ数", "バッチ", "所要時間", "GPU"],
    ["RECAP", "Step2: 価値モデル", "4,000 step", "256", "20.6 時間（18.6 秒/step）", ADA],
    ["RECAP", "Step3: advantage 計算", "62.3 万サンプル（推論）", "—", "4.2 時間（41 サンプル/秒）", ADA],
    ["RECAP", "Step4: 方策（SFT）", "2,000 step", "256", "20.3 時間（約 36 秒/step）", ADA],
    ["STEAM", "critic の学習", "2,000 step", "64", "8.3 時間（14.9 秒/step）", A5],
    ["STEAM", "advantage 計算", "15.5 万フレーム/データセット", "—", "約 2.4 時間/データセット", A5],
    ["STEAM", "Step4: 方策（SFT）", "1,450 step", "256", "16.5 時間（約 41 秒/step）", A5],
    ["RL Token", "Stage 1: RL token", "6,000 step", "64", "約 25 分", A5],
    ["RL Token", "Stage 2: 1・2 つ目の把持", "750 エポック", "512", "約 8.2 時間", A5],
    ["RL Token", "2 つ目専用の actor", "400 エポック", "512", "約 5.5 時間（約 49 秒/エポック）", A5],
]
table(s, 0.6, 1.1, 12.1, [1.0, 2.3, 2.6, 0.8, 3.2, 2.2], rows, size=12, row_h=0.46, header_h=0.42,
      hl_rows=(4, 5, 6, 8, 9), align_cols={3: PP_ALIGN.CENTER})
textbox(s, 0.6, 5.72, 12.1, 0.55, [dict(text="バッチは global batch（RL Token の Stage 2 は micro 128）。RL Token の 1 エポック = 3 環境のロールアウトと更新。RECAP の advantage は、merged（175.8 万フレーム）のうち 62.3 万サンプル分", size=11, color=GRAY)])
key_box(s, "合計: RECAP 約 45 時間、STEAM 約 25 時間（advantage 計算を除く）、RL Token 約 14 時間", y=6.35, h=0.6)

# ================ old 16
s = new_slide("考察: 3 手法の比較", "表は定性的な比較。smolVLA での経験に基づく。")
rows = [
    ["", "RL が作用する場所", "評価者 / 必要なもの", "長所", "今回の経験での課題"],
    ["RECAP", "方策の学習目標（advantage 条件付き SFT）", "評価者: 価値関数（return を予測）。ロールアウト",
     "勾配不要。flow matching VLA に使える。異種データを使える", "価値の信号がデータの作り方に依存。SFT が重い（30,000 step）"],
    ["STEAM", "方策の学習目標（RECAP と同じ条件付き SFT）", "評価者: 時間オフセットの critic のアンサンブル。デモ（自己教師あり）",
     "報酬ラベル不要。保守的なアンサンブルで過大評価を抑える", "未知データでの critic の精度が低い。advantage の計算に約 2.5 時間/15.5 万フレーム"],
    ["RL Token", "行動の洗練（凍結 VLA の出力）", "オンラインの環境と報酬。小さな actor-critic",
     "軽量（約 5 MB）で、約 5 時間で学習。success_once 0.46 → 0.74", "局面の設計（ゲート）が必要。シミュレータ（実機）が必要"],
]
table(s, 0.6, 1.1, 12.1, [1.2, 2.6, 2.7, 2.8, 2.8], rows, size=14, row_h=1.3, header_h=0.55, bold_rows=(), align_cols=None)
key_box(s, "RECAP と STEAM は評価者だけが違い、RL Token は行動を直す。目的は補完的")

# ================ old 17
s = new_slide("まとめと今後", "次の作業: Step4 の本学習、critic の精度、評価の統計的な信頼性。")
box(s, 0.6, 1.1, 5.9, 0.6, "まとめ", fill=BLUE, line=None, size=20, color=WHITE, bold=True, name="summary head")
textbox(s, 0.6, 1.85, 5.9, 3.0, [
    (0, "RECAP と STEAM は評価者が違う（advantage 条件付き学習）。RL Token は、オンライン RL で行動を洗練する"),
    (0, f"smolVLA では **RL Token が最良**（{R['base']/100:.2f} → {R['rlt']/100:.2f}）"),
    (0, f"RECAP・STEAM は、Step4 の途中段階の最良値で {min(R['steam'], R['recap'])/100:.2f}〜{max(R['steam'], R['recap'])/100:.2f}。ベースとの差は、有意とは言えない"),
    (0, "成功で打ち切る収集設計が、価値・critic の信号に重要"),
], size=16)
box(s, 6.8, 1.1, 5.9, 0.6, "今後", fill=GREEN, line=None, size=20, color=WHITE, bold=True, name="future head")
textbox(s, 6.8, 1.85, 5.9, 3.0, [
    (0, "RECAP・STEAM の Step4 を、本学習（30,000 step）まで進める"),
    (0, "STEAM の critic のアンサンブルを増やし、未知データでの精度を上げる"),
    (0, "評価を 100 エピソード以上にして、標準誤差を縮める"),
    (0, "RL Token を、LIBERO-10 の他のタスクに広げる"),
], size=16)
strip = [("ベース", R["base"], LGRAY, BLACK), ("RECAP", R["recap"], SLATE, WHITE), ("STEAM", R["steam"], SLATE, WHITE), ("RL Token", R["rlt"], GREEN, WHITE)]
for i, (n, v, col, tc) in enumerate(strip):
    box(s, 0.6 + i * 3.05, 4.85, 2.85, 1.2, [dict(text=f"{v/100:.2f}", size=32, bold=True, color=tc), dict(text=n, size=15, color=tc)],
        fill=col, line=None, size=15, name=f"result {n}")
key_box(s, "最良は RL Token（0.74）。RECAP・STEAM は、学習を進めて、同じ条件で再評価する")

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))

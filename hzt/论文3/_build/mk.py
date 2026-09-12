# -*- coding: utf-8 -*-
"""把 _build/src/*.md（自定义标记）同时渲染为 论文3.docx 与 论文3.tex。

标记：
  @@H1 / @@H2 / @@H3 标题
  @@P   正文段落（$...$ 为行内公式）
  @@EQ  latex | (1)     行间公式，右侧编号
  @@TBL ... @@ENDTBL     表格
        caption: 表1 xxx
        cols: a | b | c
        row:  ... | ... | ...
        note: 注：xxx
  @@FIG  文件名 | 图1 xxx
  @@PB   分页
  @@ABS  摘要标题行（居中加粗）
"""
import os, re, sys, shutil, glob
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement
from lxml import etree
import latex2mathml.converter as l2m

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
OUT = os.path.dirname(HERE)                      # hzt/
FIGDIR = os.path.join(HERE, 'fig')

XSL = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_transform = etree.XSLT(etree.parse(XSL))
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

WARN = []


def latex_to_omml(latex):
    mathml = l2m.convert(latex)
    dom = etree.fromstring(mathml)
    return _transform(dom).getroot()


# ---------------------------------------------------------------- 解析
def parse():
    blocks = []
    buf = []
    for fn in sorted(glob.glob(os.path.join(SRC, '*.md'))):
        with open(fn, encoding='utf-8') as f:
            buf.append(f.read())
    text = '\n'.join(buf)
    # @@INCLUDE 展开：从 src/_tables/ 内联表格片段
    tbdir = os.path.join(SRC, '_tables')
    def _inc(m):
        fp = os.path.join(tbdir, m.group(1).strip())
        with open(fp, encoding='utf-8') as fh:
            return fh.read().rstrip('\n')
    text = re.sub(r'^@@INCLUDE\s+(\S+)\s*$', _inc, text, flags=re.M)
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith('@@TBL'):
            i += 1
            cap, cols, rows, note = '', [], [], ''
            while i < len(lines) and not lines[i].startswith('@@ENDTBL'):
                s = lines[i].strip()
                if s.startswith('caption:'):
                    cap = s[8:].strip()
                elif s.startswith('note:'):
                    note = s[5:].strip()
                elif s.startswith('cols:'):
                    cols = [c.strip() for c in s[5:].split('|')]
                elif s.startswith('row:'):
                    rows.append([c.strip() for c in s[4:].split('|')])
                i += 1
            i += 1
            blocks.append(('tbl', dict(caption=cap, cols=cols, rows=rows, note=note)))
            continue
        m = re.match(r'@@(H1|H2|H3|P|EQ|FIG|PB|ABS)\s*(.*)$', ln)
        if m:
            tag, rest = m.group(1), m.group(2).strip()
            if tag == 'EQ':
                if '|' in rest:
                    tex, num = rest.rsplit('|', 1)
                else:
                    tex, num = rest, ''
                blocks.append(('eq', tex.strip(), num.strip()))
            elif tag == 'FIG':
                f, c = [x.strip() for x in rest.split('|', 1)]
                blocks.append(('fig', f, c))
            elif tag == 'PB':
                blocks.append(('pb',))
            elif tag == 'ABS':
                blocks.append(('abs', rest))
            else:
                blocks.append((tag.lower(), rest))
        elif ln.strip():
            blocks.append(('p', ln.strip()))
        i += 1
    return blocks


INLINE = re.compile(r'\$(.+?)\$', re.S)


# ---------------------------------------------------------------- Word
def set_font(run, cn='宋体', en='Times New Roman', size=12, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), cn)


BOLD = re.compile(r'\*\*(.+?)\*\*', re.S)


def add_runs(p, text, size=12, cn='宋体', bold=False):
    """把含 $...$ 与 **粗体** 的文本写进段落，公式走 OMML。"""
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            _plain(p, text[pos:m.start()], size, cn, bold)
        tex = m.group(1)
        try:
            p._p.append(latex_to_omml(tex))
        except Exception as e:
            WARN.append('行内公式失败: %s (%s)' % (tex, e))
            r = p.add_run(tex)
            set_font(r, cn=cn, size=size, bold=bold)
        pos = m.end()
    if pos < len(text):
        _plain(p, text[pos:], size, cn, bold)


def _plain(p, seg, size, cn, bold):
    """写不含行内公式的片段，处理 **粗体**。"""
    pos = 0
    for m in BOLD.finditer(seg):
        if m.start() > pos:
            r = p.add_run(seg[pos:m.start()])
            set_font(r, cn=cn, size=size, bold=bold)
        r = p.add_run(m.group(1))
        set_font(r, cn=cn, size=size, bold=True)
        pos = m.end()
    if pos < len(seg):
        r = p.add_run(seg[pos:])
        set_font(r, cn=cn, size=size, bold=bold)


def body_para(doc, text, size=12, indent=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(0)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    add_runs(p, text, size=size)
    return p


def build_docx(blocks, path, figmap):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)

    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    eqno = 0
    for b in blocks:
        k = b[0]
        if k == 'h1':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(b[1])
            set_font(r, cn='黑体', size=15, bold=True)
        elif k == 'h2':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            r = p.add_run(b[1])
            set_font(r, cn='黑体', size=13.5, bold=True)
        elif k == 'h3':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(b[1])
            set_font(r, cn='黑体', size=12, bold=True)
        elif k == 'abs':
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(b[1])
            set_font(r, cn='黑体', size=16, bold=True)
        elif k == 'p':
            body_para(doc, b[1])
        elif k == 'eq':
            tex, num = b[1], b[2]
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            # 制表位：公式居中，编号右对齐
            pPr = p._p.get_or_add_pPr()
            tabs = OxmlElement('w:tabs')
            for pos, al in ((Cm(8.0), 'center'), (Cm(16.0), 'right')):
                t = OxmlElement('w:tab')
                t.set(qn('w:val'), al)
                t.set(qn('w:pos'), str(int(pos.twips)))
                tabs.append(t)
            pPr.append(tabs)
            r = p.add_run('\t')
            set_font(r)
            try:
                p._p.append(latex_to_omml(tex))
            except Exception as e:
                WARN.append('行间公式失败: %s (%s)' % (tex, e))
                r = p.add_run(tex)
                set_font(r)
            if num:
                r = p.add_run('\t' + num)
                set_font(r)
        elif k == 'tbl':
            t = b[1]
            if t['caption']:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(3)
                r = p.add_run(t['caption'])
                set_font(r, cn='黑体', size=10.5, bold=True)
            ncol = len(t['cols'])
            tab = doc.add_table(rows=0, cols=ncol)
            tab.style = 'Table Grid'
            tab.alignment = WD_TABLE_ALIGNMENT.CENTER
            for row in [t['cols']] + t['rows']:
                cells = tab.add_row().cells
                for ci, txt in enumerate(row[:ncol]):
                    cp = cells[ci].paragraphs[0]
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cp.paragraph_format.line_spacing = 1.0
                    cp.paragraph_format.space_after = Pt(0)
                    add_runs(cp, txt, size=9)
            if t['note']:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.line_spacing = 1.0
                add_runs(p, t['note'], size=9)
        elif k == 'fig':
            f, cap = b[1], b[2]
            src = figmap.get(f)
            if src and os.path.exists(src):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                p.add_run().add_picture(src, width=Cm(14.5))
            else:
                WARN.append('缺图: %s' % f)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run(cap)
            set_font(r, size=10.5)
        elif k == 'pb':
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.save(path)
    print('wrote', path)


# ---------------------------------------------------------------- LaTeX
TEX_PRE = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{geometry}
\usepackage{caption}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\captionsetup[figure]{labelsep=space,font=small}
\captionsetup[table]{labelsep=space,font=small}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\setlength{\parindent}{2em}
\linespread{1.4}
\renewcommand{\arraystretch}{1.2}
\title{论文3}
\date{}
\begin{document}
"""

TEX_POST = "\n\\end{document}\n"


SPECIAL = {'%': r'\%', '&': r'\&', '#': r'\#', '_': r'\_',
           '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
           '∶': '$:$', '⁻': '$^{-}$', '⁺': '$^{+}$',
           '≤': r'$\le$', '≥': r'$\ge$', '×': r'$\times$',
           '≈': r'$\approx$', '±': r'$\pm$',
           '“': '``', '”': "''"}


def _esc_plain(seg):
    """转义非公式片段中的 LaTeX 特殊字符，并把 **粗体** 转成 \\textbf。"""
    out = []
    pos = 0
    for m in BOLD.finditer(seg):
        out.append(_esc_raw(seg[pos:m.start()]))
        out.append('\\textbf{%s}' % _esc_raw(m.group(1)))
        pos = m.end()
    out.append(_esc_raw(seg[pos:]))
    return ''.join(out)


def _esc_raw(s):
    return ''.join(SPECIAL.get(ch, ch) for ch in s)


def esc(s):
    """把 $...$ 之外的部分转义，$...$ 内保持原样。"""
    out, pos = [], 0
    for m in INLINE.finditer(s):
        out.append(_esc_plain(s[pos:m.start()]))
        out.append('$%s$' % m.group(1))
        pos = m.end()
    out.append(_esc_plain(s[pos:]))
    return ''.join(out)


def build_tex(blocks, path, figmap):
    out = [TEX_PRE]
    for b in blocks:
        k = b[0]
        if k == 'h1':
            out.append('\n\\section*{%s}\n' % esc(b[1]))
        elif k == 'h2':
            out.append('\n\\subsection*{%s}\n' % esc(b[1]))
        elif k == 'h3':
            out.append('\n\\subsubsection*{%s}\n' % esc(b[1]))
        elif k == 'abs':
            out.append('\n\\begin{center}\\Large\\bfseries %s\\end{center}\n' % esc(b[1]))
        elif k == 'p':
            out.append('\n%s\n' % esc(b[1]))
        elif k == 'eq':
            tex, num = b[1], b[2]
            if num:
                m = re.match(r'^\((\d+)\)$', num)
                if m:
                    out.append('\n\\begin{equation}\n%s\n\\end{equation}\n' % tex)
                    continue
            out.append('\n\\[ %s \\]\n' % tex)
        elif k == 'tbl':
            t = b[1]
            ncol = len(t['cols'])
            # 超长 3 列表（如符号说明）高过一页，用 longtable 允许跨页
            if ncol == 3 and len(t['rows']) >= 20:
                spec = 'C{0.24\\textwidth}C{0.38\\textwidth}C{0.38\\textwidth}'
                if t['caption']:
                    out.append('\n\\begin{longtable}{%s}\n' % spec)
                    out.append('\\caption*{%s}\\\\\n' % esc(t['caption']))
                else:
                    out.append('\n\\begin{longtable}{%s}\n' % spec)
                out.append('\\toprule\n')
                out.append(' & '.join(esc(c) for c in t['cols']) + ' \\\\\n\\midrule\n\\endfirsthead\n')
                out.append('\\toprule\n')
                out.append(' & '.join(esc(c) for c in t['cols']) + ' \\\\\n\\midrule\n\\endhead\n')
                for row in t['rows']:
                    cells = (row + [''] * ncol)[:ncol]
                    out.append(' & '.join(esc(c) for c in cells) + ' \\\\\n')
                out.append('\\bottomrule\n\\end{longtable}\n')
                if t['note']:
                    out.append('\\par\\vspace{2pt}\\footnotesize %s\n' % esc(t['note']))
                continue
            out.append('\n\\begin{table}[H]\n\\centering\n')
            if t['caption']:
                out.append('\\caption*{%s}\n' % esc(t['caption']))
            wide = ncol >= 4
            if ncol == 3:
                spec = 'C{0.24\\textwidth}C{0.38\\textwidth}C{0.38\\textwidth}'
            elif wide:
                spec = 'c' * ncol
                out.append('\\resizebox{\\textwidth}{!}{%\n')
            else:
                spec = 'c' * ncol
            out.append('\\begin{tabular}{%s}\n\\toprule\n' % spec)
            out.append(' & '.join(esc(c) for c in t['cols']) + ' \\\\\n\\midrule\n')
            for row in t['rows']:
                cells = (row + [''] * ncol)[:ncol]
                out.append(' & '.join(esc(c) for c in cells) + ' \\\\\n')
            out.append('\\bottomrule\n\\end{tabular}\n')
            if wide:
                out.append('}\n')
            if t['note']:
                out.append('\\par\\vspace{2pt}\\footnotesize %s\n' % esc(t['note']))
            out.append('\\end{table}\n')
        elif k == 'fig':
            f, cap = b[1], b[2]
            out.append('\n\\begin{figure}[H]\n\\centering\n'
                       '\\includegraphics[width=0.86\\textwidth]{%s}\n'
                       '\\caption*{%s}\n\\end{figure}\n' % (f, esc(cap)))
        elif k == 'pb':
            out.append('\n\\clearpage\n')
    out.append(TEX_POST)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(out))
    print('wrote', path)


def main():
    blocks = parse()
    figmap = {}
    # 图件映射：src 中写的名字 -> 实际文件
    for pat, tgt in [
        (os.path.join(FIGDIR, '*'), None),
    ]:
        pass
    for f in glob.glob(os.path.join(FIGDIR, '*')):
        figmap[os.path.basename(f)] = f
    for f in glob.glob(os.path.join(os.path.dirname(HERE), '论文3_LaTeX', 'figures', '*')):
        figmap[os.path.basename(f)] = f
    build_tex(blocks, os.path.join(OUT, '论文3_LaTeX', '论文3.tex'), figmap)
    try:
        build_docx(blocks, os.path.join(OUT, '论文3.docx'), figmap)
    except PermissionError:
        print('!! 论文3.docx 被占用（可能在 Word 中打开），本轮未更新 docx；'
              '关闭 Word 后重跑 mk.py 即可。tex 已正常更新。')
    if WARN:
        print('--- WARNINGS ---')
        for w in WARN:
            print(' ', w)


if __name__ == '__main__':
    main()

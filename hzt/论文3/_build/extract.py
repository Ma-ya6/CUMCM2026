# -*- coding: utf-8 -*-
"""提取 docx 正文：段落文本 + OMML公式 + 表格，按文档顺序输出。"""
import sys, json, re
from docx import Document
from docx.oxml.ns import qn

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'


def omml_to_text(el, depth=0):
    """把 OMML 元素粗略转换为可读文本/类LaTeX。"""
    tag = el.tag
    if tag == M + 't':
        return el.text or ''
    if tag == M + 'f':  # fraction
        num = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'num') or [])
        den = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'den') or [])
        return '(%s)/(%s)' % (num, den)
    if tag == M + 'sSub':
        base = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        sub = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sub') or [])
        return '%s_{%s}' % (base, sub)
    if tag == M + 'sSup':
        base = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        sup = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sup') or [])
        return '%s^{%s}' % (base, sup)
    if tag == M + 'sSubSup':
        base = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        sub = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sub') or [])
        sup = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sup') or [])
        return '%s_{%s}^{%s}' % (base, sub, sup)
    if tag == M + 'nary':
        chr_el = el.find(M + 'naryPr/' + M + 'chr')
        op = chr_el.get(M + 'val') if chr_el is not None else '∫'
        sub = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sub') or [])
        sup = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'sup') or [])
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        return '%s_{%s}^{%s} %s' % (op, sub, sup, e)
    if tag == M + 'd':  # delimiter
        inner = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        pr = el.find(M + 'dPr')
        beg, end = '(', ')'
        if pr is not None:
            b = pr.find(M+'begChr'); e_ = pr.find(M+'endChr')
            if b is not None: beg = b.get(M+'val', '(')
            if e_ is not None: end = e_.get(M+'val', ')')
        return '%s%s%s' % (beg, inner, end)
    if tag == M + 'rad':
        deg = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'deg') or [])
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        return 'sqrt[%s]{%s}' % (deg, e) if deg else 'sqrt(%s)' % e
    if tag == M + 'func':
        fname = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'fName') or [])
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        return '%s(%s)' % (fname, e)
    if tag == M + 'acc':
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        return 'acc(%s)' % e
    if tag == M + 'm':  # matrix
        rows = []
        for mr in el.findall(M + 'mr'):
            rows.append(' & '.join(''.join(omml_to_text(c, depth+1) for c in e)
                                   for e in mr.findall(M+'e')))
        return 'matrix[' + '; '.join(rows) + ']'
    if tag == M + 'eqArr':
        return ' ; '.join(''.join(omml_to_text(c, depth+1) for c in e)
                          for e in el.findall(M+'e'))
    if tag == M + 'box':
        return ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
    if tag == M + 'groupChr':
        return ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
    if tag == M + 'limLow':
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        lim = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'lim') or [])
        return '%s_{%s}' % (e, lim)
    if tag == M + 'limUpp':
        e = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'e') or [])
        lim = ''.join(omml_to_text(c, depth+1) for c in el.find(M+'lim') or [])
        return '%s^{%s}' % (e, lim)
    # 默认：拼接子元素
    return ''.join(omml_to_text(c, depth+1) for c in el)


def para_content(p_el):
    """按顺序处理段落内的 run / oMath，保留行内公式位置。"""
    parts = []
    for child in p_el.iter():
        pass
    # 直接遍历直接子元素（含 hyperlink 需展开）
    def walk(node):
        for c in node:
            if c.tag == M + 'oMath':
                parts.append(('math', omml_to_text(c)))
            elif c.tag == M + 'oMathPara':
                parts.append(('math', omml_to_text(c)))
            elif c.tag == W + 'r':
                txt = ''.join(t.text or '' for t in c.findall(W + 't'))
                # 处理 tab / br
                if not txt:
                    for br in c.findall(W + 'br'):
                        txt += '\n'
                    for tb in c.findall(W + 'tab'):
                        txt += '\t'
                if txt:
                    parts.append(('text', txt))
            elif c.tag in (W + 'hyperlink', W + 'smartTag', W + 'sdt', W + 'sdtContent',
                           W + 'ins', W + 'del'):
                walk(c)
            elif c.tag == W + 'bookmarkStart' or c.tag == W + 'bookmarkEnd':
                pass
    walk(p_el)
    return parts


def get_style(p):
    try:
        return p.style.name
    except Exception:
        return ''


def dump(path, out):
    doc = Document(path)
    body = doc.element.body
    lines = []
    idx = 0
    for child in body:
        if child.tag == W + 'p':
            from docx.text.paragraph import Paragraph
            p = Paragraph(child, doc)
            parts = para_content(child)
            text = ''.join(t for k, t in parts)
            is_math = all(k == 'math' for k, _ in parts) and parts
            style = get_style(p)
            # 检测图片
            has_img = bool(child.findall('.//' + W.replace('wordprocessingml/2006/main', 'drawingml/2006/main'))) if False else ('<w:drawing' in child.xml or '<w:pict' in child.xml)
            tag = 'MATH' if is_math else 'P'
            lines.append('[%04d][%s][%s]%s %s' % (idx, tag, style, ' [IMG]' if has_img else '', text))
            idx += 1
        elif child.tag == W + 'tbl':
            from docx.table import Table
            t = Table(child, doc)
            lines.append('[%04d][TABLE] rows=%d cols=%d' % (idx, len(t.rows), len(t.columns)))
            for ri, row in enumerate(t.rows):
                cells = []
                for c in row.cells:
                    ct = ' '.join(''.join(x for _, x in para_content(pp._p)) for pp in c.paragraphs)
                    cells.append(ct.strip())
                lines.append('   R%02d | %s' % (ri, ' | '.join(cells)))
            idx += 1
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('wrote', out, len(lines), 'blocks')


if __name__ == '__main__':
    dump(sys.argv[1], sys.argv[2])

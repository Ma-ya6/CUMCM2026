# -*- coding: utf-8 -*-
"""按文档顺序导出 docx 中嵌入的图片，并给出每张图所在的段落上下文。"""
import sys, os, zipfile, shutil
from docx import Document
from docx.oxml.ns import qn

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'


def main(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    doc = Document(src)
    body = doc.element.body
    idx = 0
    lines = []
    for child in body:
        if child.tag != W + 'p':
            continue
        blips = child.findall('.//' + A + 'blip')
        if not blips:
            continue
        rels = []
        for b in blips:
            rid = b.get(R + 'embed')
            part = doc.part.rels[rid].target_part
            ext = os.path.splitext(part.partname)[1]
            idx += 1
            name = 'fig%02d%s' % (idx, ext)
            with open(os.path.join(outdir, name), 'wb') as f:
                f.write(part.blob)
            rels.append(name)
        txt = ''.join(t.text or '' for t in child.findall('.//' + W + 't'))
        lines.append('%s  <- 段内文字: %s' % (','.join(rels), txt.strip()))
    with open(os.path.join(outdir, '_order.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

# -*- coding: utf-8 -*-
"""编译日志体检：页数、缺字、溢出、报错。"""
import re, io, sys

p = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\36852\.claude\progress\tex3.log'
s = io.open(p, encoding='utf-8', errors='replace').read()
print('页数     :', re.findall(r'\((\d+) pages', s))
print('缺字     :', re.findall(r'There is no (\S+) in font', s))
ov = [float(m.group(1)) for m in re.finditer(r'Overfull \\hbox \(([\d.]+)pt', s)]
print('Overfull : %d 处，最大 %.1fpt' % (len(ov), max(ov) if ov else 0.0))
print('报错     :', re.findall(r'^! .*', s, re.M)[:8])
print('未定义引用/图:', re.findall(r'(?:Reference|File) `[^\']+\' .*', s)[:8])

# -*- coding: utf-8 -*-
"""扫描表格行里未用 $...$ 包裹的数学符号（LaTeX 会报 Missing $）。"""
import io, glob, re, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
BAD = set('ηΔλ×≤≥⁺⁻√∑∏αβγθκσ₀₁₂₃₄₅₆₇₈₉')


def main():
    hits = 0
    files = sorted(glob.glob(os.path.join(SRC, '*.md'))) + \
            sorted(glob.glob(os.path.join(SRC, '_tables', '*.md')))
    for fp in files:
        for i, ln in enumerate(io.open(fp, encoding='utf-8'), 1):
            s = ln.rstrip('\n')
            if not s.startswith(('cols:', 'row:', 'note:')):
                continue
            body = s.split(':', 1)[1]
            stripped = re.sub(r'\$[^$]*\$', '', body)
            if any(ch in stripped for ch in BAD) or re.search(r'[A-Za-z]_[A-Za-z0-9]', stripped):
                print('%s:%d  %s' % (os.path.relpath(fp, SRC), i, s.strip()[:160]))
                hits += 1
    print('--- 命中 %d 行 ---' % hits)


RISK = ['%', '&', '#', '_', '\\']


def lint():
    """扫描正文/表格中会把 LaTeX 打挂的裸字符（已包在 $...$ 内的除外）。"""
    n = 0
    files = sorted(glob.glob(os.path.join(SRC, '*.md'))) + \
            sorted(glob.glob(os.path.join(SRC, '_tables', '*.md')))
    for fp in files:
        for i, ln in enumerate(io.open(fp, encoding='utf-8'), 1):
            s = ln.rstrip('\n')
            if not s or s.startswith(('@@TBL', '@@ENDTBL', '@@EQ')):
                continue
            if s.startswith(('caption:', 'cols:', 'row:', 'note:')):
                body = s.split(':', 1)[1]
            elif s.startswith('@@'):
                body = s[2:].split(' ', 1)[1] if ' ' in s else ''
            else:
                body = s
            st = re.sub(r'\$[^$]*\$', '', body)
            for ch in RISK:
                if ch in st:
                    print('%s:%d [%s] %s' % (os.path.relpath(fp, SRC), i, ch, st.strip()[:130]))
                    n += 1
                    break
    print('--- 风险 %d 行 ---' % n)


if __name__ == '__main__':
    main()
    lint()

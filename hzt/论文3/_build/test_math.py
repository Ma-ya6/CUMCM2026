# -*- coding: utf-8 -*-
"""测试 LaTeX -> MathML -> OMML 管线，产出可编辑的 Word 公式。"""
import latex2mathml.converter as l2m
from lxml import etree

XSL = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_transform = etree.XSLT(etree.parse(XSL))

NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def latex_to_omml(latex: str):
    """返回 oMath 元素（可直接 append 到 w:p）。"""
    mathml = l2m.convert(latex)
    dom = etree.fromstring(mathml)
    omml = _transform(dom)
    root = omml.getroot()
    return root


if __name__ == "__main__":
    tests = [
        r"\min C=\sum_{t=1}^{T} c_t x_t \Delta t",
        r"\Delta F_{d,j}^{(k)}=c_j\left[\min(m_{d,j},z_{d,j})-m_{d,j}\right]\Delta t+0.5c_j(\Delta^-)\Delta t",
        r"E_{d,i}=E_{d,i-1}+\eta_c u_{d,i}\Delta t-\frac{v_{d,i}\Delta t}{\eta_d}",
        r"q^{*}=\frac{4c}{4c+c}=0.80",
        r"\pi(d,j)=\begin{cases}(d,j+1), & j\le 143\\ (d+1,1), & j=144\end{cases}",
        r"\lambda_E=\eta_d\bar{c}",
        r"\xi_{d,i}=N^{\mathrm{act}}_{d,i}-\widehat{N}_{d,i\mid 0}",
    ]
    for t in tests:
        try:
            el = latex_to_omml(t)
            print("OK  ", t[:50], "->", etree.QName(el).localname)
        except Exception as e:
            print("FAIL", t[:50], "->", type(e).__name__, e)

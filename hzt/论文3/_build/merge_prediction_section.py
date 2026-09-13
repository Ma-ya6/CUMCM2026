"""仅替换Word的§5.3，保留其他章节及其原有格式。"""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
import hashlib
from docx import Document
import mk

p=Path(__file__).resolve().parent.parent
generated=p/'_build/论文3_预测修订生成稿.docx'
figmap={x.name:str(x) for folder in [p/'_build/fig',p/'论文3_LaTeX/figures'] for x in folder.glob('*') if x.is_file()}
mk.build_docx(mk.parse(),str(generated),figmap)
old=Document(p/'论文3.docx');new=Document(generated)
def span(doc):
    nodes=list(doc._element.body)
    texts=[''.join(n.xpath('.//w:t/text()')) for n in nodes]
    a=next(i for i,t in enumerate(texts) if t.startswith('5.3 问题三'))
    b=next(i for i,t in enumerate(texts) if t.startswith('5.4 问题四'))
    return nodes,a,b
before,a,b=span(old);incoming,c,d=span(new)
rels={}
for node in incoming[c:d]:
    for e in node.iter():
        for attr,value in e.attrib.items():
            if attr.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'):
                if value not in rels:
                    r=new.part.rels[value]
                    assert r.reltype.endswith('/image'),r.reltype
                    rels[value]=old.part.get_or_add_image(BytesIO(r.target_part.blob))[0]
copies=[deepcopy(n) for n in incoming[c:d]]
for node in copies:
    for e in node.iter():
        for attr,value in list(e.attrib.items()):
            if attr.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'):
                e.set(attr,rels[value])
for node in before[a:b]:old._element.body.remove(node)
for i,node in enumerate(copies):old._element.body.insert(a+i,node)
after,x,y=span(old)
assert [n.xml for n in before[:a]]==[n.xml for n in after[:x]]
assert [n.xml for n in before[b:]]==[n.xml for n in after[y:]]
staged=p/'_build/论文3_待更新.docx';old.save(staged)
print('staged',staged,'other chapter XML unchanged')

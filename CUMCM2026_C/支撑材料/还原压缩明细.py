"""只还原无损压缩的主模型明细，不执行预测或优化。"""
from pathlib import Path
import hashlib, json, lzma
root=Path(__file__).resolve().parent
records=json.loads((root/'压缩明细还原清单.json').read_text(encoding='utf-8'))
for row in records:
    target=(root/row['path']).resolve()
    if not target.is_relative_to(root): raise ValueError('明细路径超出交付目录')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError(f'已存在的明细与交付版本不同，未覆盖：{target}')
        print('已还原:',target.relative_to(root))
        continue
    data=lzma.decompress(Path(str(target)+'.xz').read_bytes())
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise RuntimeError(f'明细校验失败：{target}')
    target.write_bytes(data)
    print('还原完成:',target.relative_to(root))
print('四份主模型明细完整；未执行模型计算。')

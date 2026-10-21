#!/usr/bin/env python3
"""Verify the published artifact manifest and project structure (stdlib only)."""
import hashlib,json,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
count=0
for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split('  ',1);path=ROOT/name
    if not path.is_file():raise SystemExit('缺少文件：'+name)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise SystemExit('校验不一致：'+name)
    count+=1
items=json.loads((ROOT/'catalog.json').read_text())
assert len(items)==29 and [x['number'] for x in items]==list(range(1,30))
assert sum(x['problems'] for x in items)==950
assert sum(x['pages'] for x in items)==1255
for item in items:
    for key in ['source','pdf','tikz_source','tikz_pdf']:assert (ROOT/item[key]).is_file(),item[key]
    for key in ['pdf','tikz_pdf']:assert (ROOT/item[key]).read_bytes().startswith(b'%PDF-'),item[key]
with (ROOT/'audit/950题图与解答复审覆盖表.csv').open(encoding='utf-8-sig',newline='') as f:
    assert len(list(csv.DictReader(f)))==950
for sub in ['src/lectures','src/tikz']:assert len(list((ROOT/sub).glob('*.tex')))==29
for sub in ['pdf/lectures','pdf/tikz']:assert len(list((ROOT/sub).glob('*.pdf')))==29
print(f'PASS: {count} 个文件校验一致；29 讲、950 题、1255 页，正文与 TikZ 源码齐全。')

#!/usr/bin/env python3
"""Compile lecture or TikZ sources with Tectonic, leaving published PDFs intact."""
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
g=p.add_mutually_exclusive_group(required=True)
g.add_argument('--lecture',type=int,choices=range(1,30))
g.add_argument('--all',action='store_true')
p.add_argument('--diagrams',action='store_true')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
items=json.loads((ROOT/'catalog.json').read_text())
key='tikz_source' if a.diagrams else 'source'
out=ROOT/'build'/('tikz' if a.diagrams else 'lectures')
if not a.dry_run:
    if not shutil.which('tectonic'):p.error('请先安装 Tectonic，并确认 tectonic 在 PATH 中。')
    out.mkdir(parents=True,exist_ok=True)
for item in items:
    if a.lecture and item['number']!=a.lecture:continue
    cmd=['tectonic','--keep-logs','--outdir',str(out),str(ROOT/item[key])]
    print('编译：',item[key],flush=True)
    if a.dry_run:print(cmd)
    else:subprocess.run(cmd,cwd=ROOT,check=True)

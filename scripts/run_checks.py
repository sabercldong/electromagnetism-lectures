#!/usr/bin/env python3
"""Run preserved independent calculation checks sequentially."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--focused',action='store_true',help='仅运行二轮新增的 46 项图解核验')
a=p.parse_args()
base=ROOT/'checks'
for name in ['检查','记录']:(base/name).mkdir(exist_ok=True)
files=[base/'新增图解关系复核.py'] if a.focused else sorted(base.glob('*.py'))
for script in files:
    print('运行：',script.name,flush=True)
    subprocess.run([sys.executable,str(script)],cwd=base,check=True)
print('全部通过：',len(files),'份脚本')

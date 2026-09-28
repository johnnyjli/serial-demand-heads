import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, re, json, collections, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,str(paths.CAPTURE))
import e2lib, serial_tasks as st
idx_e=e2lib.index('eager'); idx_s=e2lib.index('sdpa')
def get(model,cell):
    attn='sdpa' if model=='Qwen2.5-7B-Instruct' else 'eager'
    return e2lib.load(model,cell,attn)
# 1. shapes per model
for m in e2lib.ORDER:
    d=get(m,'pointer/w0')
    print(m, d['L'], d['H'], d['contribution'].shape, d['meta']['attn_impl'], d['meta'].get('proj_name'))

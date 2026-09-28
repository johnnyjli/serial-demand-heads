import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, re, json, collections, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,str(paths.CAPTURE))
import e2lib, serial_tasks as st
def get(model,cell):
    attn='sdpa' if model=='Qwen2.5-7B-Instruct' else 'eager'
    return e2lib.load(model,cell,attn)
ref=json.load(open('texts_llama1b.json'))
for M in e2lib.ORDER:
    diff=0; ntokvar=[]; app=collections.Counter(); mh=collections.defaultdict(collections.Counter); checks=None
    for cell in e2lib.CELLS:
        d=get(M,cell)
        for it in d['items']:
            if ref.get(it['id'])!=it['text']: diff+=1
            app[tuple(it.get('appended',[]))]+=1
            if cell.startswith('fixed_digit/w0'): mh[it['k']][it['min_hops']]+=1
        nt=collections.defaultdict(set)
        for it in d['items']: nt[it['k']].add(it['n_tokens'])
        allnt=set().union(*nt.values())
        if len(allnt)>1: ntokvar.append((cell,{k:sorted(v) for k,v in nt.items()}))
        if cell=='pointer/w0': checks=d['meta']['checks']
    print(M,'textdiff',diff,'ntok varies in',ntokvar,'appended',dict(app))
    if M=='Llama-3.2-1B': print(' fixed min_hops by k',{k:dict(v) for k,v in sorted(mh.items())}); print(json.dumps(checks)[:1500])

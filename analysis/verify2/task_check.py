import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, re, json, collections, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,str(paths.CAPTURE))
import e2lib, serial_tasks as st
def get(model,cell):
    attn='sdpa' if model=='Qwen2.5-7B-Instruct' else 'eager'
    return e2lib.load(model,cell,attn)
M='Llama-3.2-1B'
tot_prompts=0; tot_ex=0; mism=0
texts={}
for cell in e2lib.CELLS:
    d=get(M,cell); items=d['items']; fam,w=cell.split('/')
    ks=collections.Counter(it['k'] for it in items)
    splits=collections.Counter(it['split'] for it in items)
    asz=collections.Counter(len(it['answer_set']) for it in items)
    print('==',cell,'n',len(items),'k',dict(ks),'split',dict(splits),'|A|',dict(asz), 'chance', d['meta']['chance'], 'setting', d['meta']['setting'])
    # gold balance per k
    for k in sorted(ks):
        g=collections.Counter(it['gold'] for it in items if it['k']==k)
        print('  k',k,'gold counts',dict(sorted(g.items())), 'ntok', sorted(set(it['n_tokens'] for it in items if it['k']==k)))
    # examples: count, depths, answers equal gold?
    eqg=0; nex=collections.Counter(); exk=collections.Counter()
    for it in items:
        blocks=it['text'].split('\n\n'); nex[len(blocks)-1]+=1
        exans=[re.search(r' (\d)\.$',b).group(1) for b in blocks[:-1]]
        if it['gold'] in exans: eqg+=1
        # solve
        ans,info=st.solve_text(fam,w,it['text'])
        if ans!=it['gold'] or info['depth']!=it['k']: mism+=1
        efam='pointer' if fam=='fixed_digit' else fam
        for b in blocks[:-1]:
            mm=re.search(r' (\d)\.$',b); a,inf=st.solve_text(efam,w,b[:mm.start()])
            exk[inf['depth']]+=1
            if a!=mm.group(1): mism+=1
            tot_ex+=1
        tot_prompts+=1
        texts[it['id']]=it['text']
    print('  n examples per item',dict(nex),'example depths',dict(exk),'items where an example answer == gold', eqg, '/', len(items))
    print('  last block sample:', repr(items[0]['text'].split('\n\n')[-1][:400]))
print('total prompts',tot_prompts,'examples',tot_ex,'mismatches',mism)
json.dump(texts,open('texts_llama1b.json','w'))

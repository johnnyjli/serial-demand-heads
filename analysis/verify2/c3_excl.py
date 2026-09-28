import sys, json, numpy as np, os
os.environ.pop('E2_CLEAN', None)
sys.path.insert(0,'.')
import e2lib
T=['pointer','arith','binding','relational']
C={r['model']:r for r in json.load(open('consist.json')) if r['stat']=='q10'}
for m in e2lib.ORDER:
    attn='sdpa' if m=='Qwen2.5-7B-Instruct' else 'eager'
    row=[]
    for t in T:
        keys={}
        for w in ('w0','w1'):
            d=e2lib.load(m,f'{t}/{w}',attn); S=d['contribution'].astype(float)
            bad=((S.max(-1)/np.maximum(np.median(S,-1),1e-9))>50).any(1)
            keys[w]={tuple(it['id'].split('/')[-2:]):b for it,b in zip(d['items'],bad)}
        common=set(keys['w0'])&set(keys['w1'])
        drop=sum(keys['w0'][x] or keys['w1'][x] for x in common)
        nw0=sum(keys['w0'].values()); nw1=sum(keys['w1'].values())
        row.append(f"{t}: drop {drop}/{len(common)} (w0 {nw0}, w1 {nw1}; consist n={C[m]['families'][t]['n']})")
    print(m, ' | '.join(row))

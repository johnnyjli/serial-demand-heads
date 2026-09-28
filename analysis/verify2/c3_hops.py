import sys, re, json, numpy as np, os
os.environ.pop('E2_CLEAN', None)
sys.path.insert(0,'.')
import e2lib
def hop(fam,w,text,pred):
    b=text.split('\n\n')[-1]
    if pred is None or not str(pred).isdigit(): return None
    pred=int(pred)
    if fam=='pointer':
        tbl=dict((int(a),int(c)) for a,c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s=int(re.search(r"(?:Start at|From) (\d)", b).group(1)); path=[s]
        for _ in range(9): path.append(tbl[path[-1]])
        return path.index(pred) if pred in path else None
    if w=='w0': d,mod,s=map(int,re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)",b).groups())
    else: s,d,mod=map(int,re.search(r"x = (\d)\. Repeat \d+ times: x = \(x \+ (\d)\) mod (\d+)",b).groups())
    if pred>=mod: return None
    return ((pred-s)*pow(d,-1,mod))%mod
# hops.json reference
H=json.load(open('hops.json'))
for fam in ('pointer','arith'):
    v=[h for m in e2lib.ORDER for h in H[m][fam]['2']]
    print('hops.json',fam,'k=2 n',len(v),'hop1 %.4f hop2 %.4f'%(np.mean([h==1 for h in v]),np.mean([h==2 for h in v])))
for q7 in ('eager','sdpa'):
    for predkey in ('pred_constrained','pred_first_digit'):
        for fam in ('pointer','arith'):
            h1=[];h2=[];strict=[];permodel=[]
            for m in e2lib.ORDER:
                attn=q7 if m=='Qwen2.5-7B-Instruct' else 'eager'
                mh=[]
                for w in ('w0','w1'):
                    d=e2lib.load(m,f'{fam}/{w}',attn)
                    for it,cs in zip(d['items'],d['correct_strict']):
                        if it['k']!=2: continue
                        h=hop(fam,w,it['text'],it.get(predkey)); h1.append(h==1); h2.append(h==2); strict.append(bool(cs)); mh.append(h==1)
                permodel.append(np.mean(mh))
            print(f'Q7I={q7:5s} {predkey:17s} {fam:7s} k=2: one-step/once {100*np.mean(h1):.2f}%  correct(hop2) {100*np.mean(h2):.2f}%  strict acc {100*np.mean(strict):.2f}%  n={len(h1)}')

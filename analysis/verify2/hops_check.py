import sys, json, re, numpy as np
sys.path.insert(0,'..')
import e2lib
H=json.load(open('../hops.json'))
print('models in hops.json',len(H))
for fam in ('pointer','arith'):
    for k in ('1','2','3'):
        per=[]; pooled=[]
        for m in e2lib.ORDER:
            v=np.array([np.nan if x is None else x for x in H[m][fam][k]],float)
            per.append([np.mean(v==j) for j in range(0,4)]); pooled+=list(v)
        per=np.array(per); pooled=np.array(pooled)
        print(fam,'k',k,'model-mean frac hops=0..3',np.round(per.mean(0),3),'pooled',np.round([np.mean(pooled==j) for j in range(4)],3))
# recompute with SDPA for Qwen7B-Instruct
def hops_done(fam, w, t):
    b = t["text"].split("\n\n")[-1]; pred = int(t["pred_constrained"])
    if fam == "pointer":
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); path = [s]
        for _ in range(9): path.append(tbl[path[-1]])
        return path.index(pred) if pred in path else None
    if w == "w0": d, mod, s = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)", b).groups())
    else: s, d, mod = map(int, re.search(r"x = (\d)\. Repeat \d+ times: x = \(x \+ (\d)\) mod (\d+)", b).groups())
    if pred >= mod: return None
    return ((pred - s) * pow(d, -1, mod)) % mod
def hops_first(fam,w,t):
    # using first generated digit (strict)
    t2=dict(t); 
    if t.get('pred_first_digit') is None: return None
    t2['pred_constrained']=t['pred_first_digit']; return hops_done(fam,w,t2)
for src in ('constrained','firstdigit'):
  for fam in ('pointer','arith'):
    per=[];percorr=[]
    for m in e2lib.ORDER:
        attn='sdpa' if m=='Qwen2.5-7B-Instruct' else 'eager'
        v=[]
        for w in ('w0','w1'):
            d=e2lib.load(m,f'{fam}/{w}',attn)
            for t in d['items']:
                if t['k']==2: v.append(hops_done(fam,w,t) if src=='constrained' else hops_first(fam,w,t))
        v=np.array([np.nan if x is None else x for x in v],float)
        per.append([np.mean(v==j) for j in range(4)])
    per=np.array(per)
    print('SDPA-Q7I',src,fam,'k=2 model-mean frac hops=0..3',np.round(per.mean(0),4))

import sys, re, collections, numpy as np
sys.path.insert(0,'.')
import e2lib
M='Llama-3.2-1B'
tot=0; own_q=0
for cell in e2lib.CELLS:
    d=e2lib.load(M,cell); items=d['items']; fam,w=cell.split('/')
    n=len(items); eqg=0; nblocks=collections.Counter(); depths=collections.Counter()
    ntok=collections.Counter(); owns=0; exeq_by_k=collections.Counter(); nk=collections.Counter()
    for it in items:
        blocks=it['text'].split('\n\n'); nblocks[len(blocks)]+=1
        final=blocks[-1]
        exs=blocks[:-1]; ans=[re.search(r' (\d)\.$',b).group(1) for b in exs]
        stripped=[re.sub(r' \d\.$','',b) for b in exs]
        if final in stripped: owns+=1
        if it['gold'] in ans: eqg+=1; exeq_by_k[it['k']]+=1
        nk[it['k']]+=1
        ntok[it['n_tokens']]+=1
    tot+=n; own_q+=owns
    extra=''
    if fam in ('binding','relational'):
        # group by body (suite_index across k?) check examples shared & text diff
        byb=collections.defaultdict(list)
        for it in items:
            byb[it['id'].split('/')[-1]].append(it)
        nb=len(byb); shared=0; onechar=0; nstmt=collections.Counter()
        for b,its in byb.items():
            pres=set('\n\n'.join(i['text'].split('\n\n')[:-1]) for i in its)
            if len(pres)==1 and len(its)==6: shared+=1
            ts=[i['text'] for i in its]
            if all(len(t)==len(ts[0]) for t in ts):
                diffs=[sum(a!=c for a,c in zip(ts[0],t)) for t in ts[1:]]
                if all(x==1 for x in diffs): onechar+=1
            fin=its[0]['text'].split('\n\n')[-1]
            nstmt[len(re.findall(r'\.\s', fin))]+=1
        extra=f' bodies {nb} shared-examples-across-k {shared} one-char-diff {onechar} final-block sentence-count {dict(nstmt)}'
    print(f'{cell:16s} n={n} blocks={dict(nblocks)} ntok={dict(ntok)} own-query-as-example={owns} example-ans==gold {eqg}/{n} by k {dict(sorted(exeq_by_k.items()))}{extra}')
print('total',tot,'own',own_q)

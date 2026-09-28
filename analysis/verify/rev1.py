import json, numpy as np
A=json.load(open('align_null.json'))
A=[a for a in A if a['stat']=='q25']
print('n models',len(A))
print('spread',sum(a['n_spread'] for a in A),'shift null',sum(a['shift_spread_mean'] for a in A),'perm null',sum(a['perm_spread_mean'] for a in A))
print('conc',sum(a['n_conc'] for a in A),'shift null',sum(a['shift_conc_mean'] for a in A))
print('sigfrac spread',np.mean([a['sig_frac_spread'] for a in A]),'conc',np.mean([a['sig_frac_conc'] for a in A]))
print('profile r',np.mean([a['profile_r'] for a in A]),'n p<.05',sum(a['profile_p']<.05 for a in A))
C=json.load(open('consist.json'))
C=[c for c in C if c['stat']=='q25']
T=["pointer","arith","binding","relational"]
zf=[];zb=[]
for c in C:
    L=c['L'];x=np.arange(L)/(L-1)
    Z=np.mean([np.array(c['families'][f]['z']) for f in T],0)
    zf.append(Z[x<.5].mean()); zb.append(Z[x>=.5].mean())
    # back-half consistent count
zf=np.array(zf);zb=np.array(zb)
print('attn',[c['attn'] for c in C])
for c,a,b in zip(C,zf,zb): print(f"{c['model']:30s} zf={a:+.2f} zb={b:+.2f}")
print('zf>0',(zf>0).sum(),'zb<0',(zb<0).sum(),'zb<zf',(zb<zf).sum())
print('mean zf',zf.mean(),'mean zb',zb.mean())
# per family model-level
for f in T:
    a=[];b=[]
    for c in C:
        L=c['L'];x=np.arange(L)/(L-1);Z=np.array(c['families'][f]['z'])
        a.append(Z[x<.5].mean());b.append(Z[x>=.5].mean())
    a=np.array(a);b=np.array(b)
    print(f, 'models zb<0',(b<0).sum(),'zb<zf',(b<a).sum(),'zf>0',(a>0).sum(), 'mean zf %.2f zb %.2f'%(a.mean(),b.mean()))
print('---- per model count of families with zb<0')
for c in C:
    L=c['L'];x=np.arange(L)/(L-1)
    n=sum(np.array(c['families'][f]['z'])[x>=.5].mean()<0 for f in T)
    print(c['model'],n)

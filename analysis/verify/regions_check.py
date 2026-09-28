import json, numpy as np
A=[a for a in json.load(open('align_null.json')) if a['stat']=='q25']
print('n models',len(A))
print('mean sig_frac_spread', np.mean([a['sig_frac_spread'] for a in A]), 'mean sig_frac_conc', np.mean([a['sig_frac_conc'] for a in A]))
print('sum n_spread', sum(a['n_spread'] for a in A), 'sum shift_spread_mean', sum(a['shift_spread_mean'] for a in A), 'sum perm_spread_mean', sum(a['perm_spread_mean'] for a in A))
print('sum n_conc', sum(a['n_conc'] for a in A), 'sum shift_conc_mean', sum(a['shift_conc_mean'] for a in A), 'perm', sum(a['perm_conc_mean'] for a in A))
print('mean profile_r', np.mean([a['profile_r'] for a in A]), 'median', np.median([a['profile_r'] for a in A]))
print('profile_p<.05', sum(a['profile_p']<.05 for a in A))
print('shift_p_spread<.05', sum(a['shift_p_spread']<.05 for a in A))
C=[c for c in json.load(open('consist.json')) if c['stat']=='q25']
print('consist n', len(C), set(c.get('attn') for c in C))
tot=0; back=0; totc=0; backc=0
for c in C:
    L=c['L']; x=np.arange(L)/(L-1); cons=np.array(c['consistent'])
    tot+=(cons<0).sum(); back+=((cons<0)&(x>=.5)).sum(); totc+=(cons>0).sum(); backc+=((cons>0)&(x>=.5)).sum()
print('consist spread tot',tot,'back',back,'conc tot',totc,'back',backc)
print('consist null spread sum', sum(c['null_spread_mean'] for c in C), 'null conc sum', sum(c['null_conc_mean'] for c in C))
# per-model/per-task fraction
fs=[];fc=[]
for c in C:
    L=c['L']
    for f,v in c['families'].items():
        fs.append(v['n_sig_spread']/L); fc.append(v['n_sig_conc']/L)
print('mean frac per model x task spread', np.mean(fs), 'conc', np.mean(fc), len(fs))
# pooled over all layers
Ltot=sum(c['L']*4 for c in C)
print('pooled spread', sum(v['n_sig_spread'] for c in C for v in c['families'].values())/Ltot, 'pooled conc', sum(v['n_sig_conc'] for c in C for v in c['families'].values())/Ltot)
# compare consist vs align_null counts
for c,a in zip(C,A): 
    assert c['model']==a['model']
    if c['n_spread']!=a['n_spread'] or c['n_conc']!=a['n_conc']: print('mismatch',c['model'])

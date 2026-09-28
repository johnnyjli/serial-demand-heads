import sys, json, numpy as np
sys.path.insert(0,'..')
import model_diff as md
R=json.load(open('../vet3/cells.json'))
def Zfrom(variant_olmo):
    D={}
    for r in R:
        if r['attn']=='eager' and r['fam'] in md.T:
            if r['variant']=='all' and not (r['model'].startswith('OLMo') and variant_olmo=='noclean'): D[(r['model'],r['fam'],r['w'])]=np.array(r['z'],float)
            if r['variant']=='noclean' and variant_olmo=='noclean': D[(r['model'],r['fam'],r['w'])]=np.array(r['z'],float)
    return np.array([[md.dec_prof(0.5*(D[(m,t,'w0')]+D[(m,t,'w1')])) for t in md.T] for m in md.ORDER])
Zc=Zfrom('clean'); Zn=Zfrom('noclean')
for nm,Z in (('clean',Zc),('noclean',Zn)):
    S=md.ss(Z); tot=sum(S.values()); ns=tot-S['shared']
    a=md.decompose(Z)[1]; Ra=np.corrcoef(a)
    lin=np.mean([Ra[i,j] for i in range(17) for j in range(i+1,17) if md.LIN[md.ORDER[i]]==md.LIN[md.ORDER[j]]])
    print(nm,'model %.3f task %.3f'%(S['model']/ns,S['task']/ns),'lineage r %.3f'%lin)
ac=md.decompose(Zc)[1]; an=md.decompose(Zn)[1]
print('r(a_clean,a_noclean)',np.corrcoef(ac.ravel(),an.ravel())[0,1])
for i,m in enumerate(md.ORDER):
    if m.startswith('OLMo'): print(m, 'per-model r', np.corrcoef(ac[i],an[i])[0,1])

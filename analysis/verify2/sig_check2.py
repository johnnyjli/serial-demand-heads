import json, numpy as np
d=json.load(open('model_diff.json'))
pm=d['per_model']
for m in ["Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-7B-Instruct","Qwen2.5-14B"]:
    print(m, 'back', round(pm[m]['back'],3), 'dev per task', np.round(pm[m]['sig_back_dev_per_task'],2), 'back per task', np.round(pm[m]['back_per_task'],2))
# any other model with all-positive back deviation
for m,v in pm.items():
    if all(x>0 for x in v['sig_back_dev_per_task']): print('all+ dev:', m)
c=json.load(open('conc_q10.json'))
for r in c['per_model']: print(r['model'], {k:round(v,2) for k,v in r.items() if k!='model'})
print('PC3 loadings', np.round(d['pca']['loadings'][2],2))

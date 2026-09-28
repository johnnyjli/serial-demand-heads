import json, sys
for fn in ["../pairs_q25.json","../pairs.json","../pairs_clean.json"]:
    R=json.load(open(fn))
    for key in ["mean_all","mean_k2plus","median_all","nosink_all"]:
        sp=sum(1 for r in R if key in r and r[key][1]<.05 and r[key][0]<0); cc=sum(1 for r in R if key in r and r[key][1]<.05 and r[key][0]>0)
        R4=[r for r in R if not r["cell"].startswith("fixed_digit")]
        sp4=sum(1 for r in R4 if key in r and r[key][1]<.05 and r[key][0]<0); cc4=sum(1 for r in R4 if key in r and r[key][1]<.05 and r[key][0]>0)
        print(fn, key, "n",len(R), "spread",sp,"conc",cc, "| serial only n",len(R4),"spread",sp4,"conc",cc4)
    # per pair
R=json.load(open("../pairs_q25.json"))
from collections import defaultdict
pp=defaultdict(lambda:[0,0,0,0])
for r in R:
    s,p=r["mean_all"]; key=(r["base"],r["post"])
    if p<.05 and s<0: pp[key][0]+=1
    if p<.05 and s>0: pp[key][1]+=1
    if not r["cell"].startswith("fixed"):
        if p<.05 and s<0: pp[key][2]+=1
        if p<.05 and s>0: pp[key][3]+=1
for k,v in pp.items(): print(k, "all10 spread/conc", v[0],v[1], " serial8 spread/conc", v[2], v[3])
print(sorted(set(r["cell"] for r in R)))

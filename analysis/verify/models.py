import json, numpy as np, sys
sys.path.insert(0,"..")
C=json.load(open("../consist.json"))
print(C[0].keys())
T=["pointer","arith","binding","relational"]
rows=[]
for r in C:
    if r["stat"]!="q25": continue
    L=r["L"]; x=np.arange(L)/(L-1); z=np.mean([np.array(r["families"][f]["z"]) for f in T],0)
    rows.append((r["model"], z[x<.5].mean(), z[x>=.5].mean(), r.get("attn")))
rows.sort(key=lambda t:t[2])
for i,(m,f,b,a) in enumerate(rows): print(i+1, f"{m:30s} front {f:+.2f} back {b:+.2f} contrast {b-f:+.2f} attn {a}")

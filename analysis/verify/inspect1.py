import sys, json, numpy as np
sys.path.insert(0, "..")
import e2lib
idx = e2lib.index("eager")
f = idx[("Llama-3.2-1B","pointer/w0")]
z = np.load(f, allow_pickle=False)
print(z.files)
for n in z.files:
    if n!="meta": print(n, z[n].shape, z[n].dtype)
m = json.loads(str(z["meta"]))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m.items()})
print(json.dumps(m["items"][0], indent=1)[:4000])

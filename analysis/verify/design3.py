import sys, json, re, collections, numpy as np
sys.path.insert(0, "..")
import e2lib
idx = e2lib.index("eager")
ap = collections.defaultdict(set)
for (model, cell), f in idx.items():
    m = json.loads(str(np.load(f)["meta"])); ap[model].add(json.dumps(m["appended_forms"])+" "+json.dumps(m["measured_tokens"])[:60])
for k,v in sorted(ap.items()): print(k, v)
def items(model, cell):
    return json.loads(str(np.load(idx[(model, cell)])["meta"]))["items"]
M="Llama-3.2-1B"
for cell, pat in [("pointer/w0", "0 goes to 5."), ("arith/w0", "add 2 and"), ("relational/w0","The top boss above Z is"), ("binding/w0","Therefore z =")]:
    its = items(M, cell)
    hits = [it for it in its if pat in it["text"].split("\n\n")[-1]]
    print(cell, "final blocks containing", repr(pat), len(hits))
    print("  first item final:", its[0]["text"].split("\n\n")[-1][-160:])
p = [it for it in items(M,"pointer/w0") if "0 goes to 5." in it["text"].split("\n\n")[-1] and "Start at 4." in it["text"].split("\n\n")[-1]]
print("pointer finals with '0 goes to 5' and 'Start at 4':", len(p))

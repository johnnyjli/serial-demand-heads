import sys, json, re, collections, numpy as np
sys.path.insert(0, "..")
import e2lib, solve
idx = e2lib.index("eager")
tot_prompts = collections.Counter(); tot_ex = collections.Counter(); mism = collections.Counter()
texts_by_model = {}
for (model, cell), f in sorted(idx.items()):
    z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
    fam, w = cell.split("/")
    for it in m["items"]:
        bl = solve.blocks(it["text"])
        ans, shown = solve.solve_block(fam, w, bl[-1])
        tot_prompts[model] += 1
        if str(ans) != it["gold"] or shown is not None: mism[model] += 1
        for b in bl[:-1]:
            a, s = solve.solve_block(fam, w, b)
            tot_ex[model] += 1
            if s is None or a != s: mism[model] += 1
    texts_by_model.setdefault(model, {})[cell] = [it["text"] for it in m["items"]]
print("prompts", set(tot_prompts.values()), "examples", set(tot_ex.values()), "mismatch", dict(mism))
# are prompts identical across models?
ref = texts_by_model["Llama-3.2-1B"]
same = {mdl: all(texts_by_model[mdl][c] == ref[c] for c in ref) for mdl in texts_by_model}
print("identical prompts across models:", same)

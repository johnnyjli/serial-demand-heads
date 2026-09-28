"""Relational/binding stepwise: share of traces whose FIRST step is a reverse lookup (q -> the entity whose boss/value is q), by k."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, re, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.REPLICATION))
import e2b
E2B_ROOT = str(paths.E2B)
runs = e2b.load_results(E2B_ROOT)
items = json.load(open(os.path.join(E2B_ROOT, "items.json")))
ITEM = {it["id"]: it for f in items for it in items[f]}
MODELS = ["Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
          "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
R = {(r["model"].split("/")[-1], r["mode"], r["family"]): r for r in runs}
out = {}
for fam, sep, pat in (("relational", "->", r"The boss of (\w) is (\w)\."), ("binding", "=", r"\b(\w) = (\w)\.")):
    for m in MODELS:
        res = {}
        for k in range(1, 7):
            rr = []
            for x in R[(m, "stepwise", fam)]["items"]:
                if x["k"] != k: continue
                b = ITEM[x["id"]]["text_stepwise"].split("\n\n")[-1].split("Question:")[0]
                st = dict(re.findall(pat, b)); rev = collections.defaultdict(set)
                for a, c in st.items(): rev[c].add(a)
                body = x["gen"].split(".")[0] if "." in x["gen"] else x["gen"]
                el = [e.strip() for e in body.split(sep)]
                first_rev = len(el) > 1 and el[1] in rev.get(el[0], set()) and st.get(el[0]) != el[1]
                rr.append((first_rev, x["correct"]))
            res[k] = {"first_step_reverse": float(np.mean([a for a, _ in rr])), "acc": float(np.mean([c for _, c in rr]))}
        out.setdefault(fam, {})[m] = res
json.dump(out, open(os.path.join(HERE, "e2b_relational_reverse_by_k.json"), "w"), indent=1)
for fam in out:
    print("==", fam, "first step reverse, by k (acc)")
    for m, res in out[fam].items():
        print(f"{m:28s}", " ".join(f"{v['first_step_reverse']:.2f}({v['acc']:.2f})" for v in res.values()))

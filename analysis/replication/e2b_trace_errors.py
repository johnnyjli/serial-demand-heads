"""E2b stepwise: classify each transition in the model's own trace (binding / relational / pointer / arith).
forward = a correct lookup; reverse = follows a statement backwards (b's value is a); other.
Output e2b_trace_errors.json.
"""
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
SEP = {"pointer": "->", "arith": "->", "binding": "=", "relational": "->"}


def table(fam, text):
    b = text.split("\n\n")[-1]
    if fam == "binding":
        return dict(re.findall(r"\b(\w) = (\w)\.", b.split("Question:")[0]))
    if fam == "relational":
        return dict(re.findall(r"The boss of (\w) is (\w)\.", b.split("Question:")[0]))
    if fam == "pointer":
        return dict(re.findall(r"(\d) goes to (\d)\.", b))
    d, mod, s = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)", b).groups())
    return {str(x): str((x + d) % mod) for x in range(mod)}


out = {}
first_err = {}
for fam in ("binding", "relational", "pointer", "arith"):
    for m in MODELS:
        cnt = collections.Counter()
        k1_over = collections.Counter()
        fe = collections.Counter()
        for x in R[(m, "stepwise", fam)]["items"]:
            it = ITEM[x["id"]]
            st = table(fam, it["text_stepwise"])
            rev = collections.defaultdict(set)
            for a, b in st.items(): rev[b].add(a)
            body = x["gen"].split(".")[0] if "." in x["gen"] else x["gen"]
            el = [e.strip() for e in body.split(SEP[fam])]
            first = None
            for a, b in zip(el[:-1], el[1:]):
                if st.get(a) == b: t = "forward"
                elif a in rev and b in rev[a]: t = "reverse"
                elif a.isdigit() and fam in ("binding", "relational"): t = "past_root"
                else: t = "other"
                cnt[t] += 1
                if first is None and t != "forward": first = t
            gold = it["gold_trace"].split(SEP[fam]); gold = [g.strip() for g in gold]
            # a trace that reproduces the gold trace and then keeps going
            if len(el) > len(gold) and el[:len(gold)] == gold: first = "continues_past_answer"
            if first: fe[first] += 1
            if x["k"] == 1 and len(el) > 2:
                k1_over[first or "none"] += 1
        tot = sum(cnt.values())
        out.setdefault(fam, {})[m] = {"transitions": dict(cnt), "share": {k: v / tot for k, v in cnt.items()},
                                      "first_error_type": dict(fe), "k1_overshoot_first_error": dict(k1_over)}
json.dump(out, open(os.path.join(HERE, "e2b_trace_errors.json"), "w"), indent=1)
for fam in out:
    print("==", fam)
    for m in out[fam]:
        d = out[fam][m]
        print(f"{m:28s}", {k: round(v, 3) for k, v in d["share"].items()}, "first err", d["first_error_type"], "k1 over", d["k1_overshoot_first_error"])

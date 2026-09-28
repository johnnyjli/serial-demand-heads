"""Verify every design fact of the E2b replication used in the appendix paragraph.

Reads items.json, e2b.py and the raw direct-mode run files; writes verify_design.json.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, re, collections, os
import numpy as np
sys.path.insert(0, str(paths.REPLICATION))
import e2b

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = str(paths.E2B)
items = json.load(open(f"{ROOT}/items.json"))
res = {}

# 1. items.json equals a fresh make_all(n_per=30, shots=8, seed=0)
fresh = e2b.make_all(n_per=30, shots=8, seed=0)
res["items_json_equals_make_all_30_8_0"] = all(
    [json.loads(json.dumps(x)) for x in fresh[f]] == items[f] for f in e2b.FAMILIES)

# 2. counts, depths, shots
per = {}
for f, its in items.items():
    ks = collections.Counter(x["k"] for x in its)
    shots = collections.Counter(len(x["shot_ks"]) for x in its)
    lo, hi = e2b.K_RANGE[f]
    cover = all(set(x["shot_ks"]) == set(range(lo, hi + 1)) for x in its)
    gold_in_shots = sum(x["gold"] in [str(g) for g in x["shot_golds"]] for x in its)
    blocks = collections.Counter(len(x["text_direct"].split("\n\n")) for x in its)
    # answers balanced within depth: counts differ by at most one over the admissible set
    bal = []
    for k in range(lo, hi + 1):
        c = collections.Counter(x["gold"] for x in its if x["k"] == k)
        adm = [str(g) for g in e2b._admissible(f, k)]
        cnt = [c.get(a, 0) for a in adm]
        bal.append(max(cnt) - min(cnt) <= 1 and set(c) <= set(adm))
    echo_ok = all(x["echo"] is None or x["gold"] != x["echo"] for x in its)
    # every block has its own table
    if f == "pointer":
        tab = all(len(re.findall(r"Rules:", b)) == 1 for x in its for b in x["text_direct"].split("\n\n"))
    elif f == "arith":
        tab = all(len(re.findall(r"Rule:", b)) == 1 for x in its for b in x["text_direct"].split("\n\n"))
    elif f == "binding":
        tab = all(len(re.findall(r"Assignments:", b)) == 1 for x in its for b in x["text_direct"].split("\n\n"))
    else:
        tab = all(len(re.findall(r"Chart:", b)) == 1 for x in its for b in x["text_direct"].split("\n\n"))
    d = {"n_items": len(its), "per_k": dict(sorted(ks.items())), "n_shots": dict(shots),
         "shots_cover_every_depth": cover, "items_with_gold_among_shot_answers": gold_in_shots,
         "blocks_per_prompt": dict(blocks), "answers_balanced_within_depth": all(bal),
         "gold_never_equals_written_k": echo_ok, "each_block_has_own_table": tab}
    if f in ("binding", "relational"):
        q = [x["text_direct"].split("\n\n")[-1] for x in its]
        if f == "binding":
            nst = [len(re.findall(r"\b\w = \w\.", b.split("Question:")[0])) for b in q]
            qre = r"Question: what is (\w)\? Therefore \1 =$"
        else:
            nst = [len(re.findall(r"The boss of \w is \w\.", b.split("Question:")[0])) for b in q]
            qre = r"Question: who is the top boss above (\w)\? The top boss above \1 is$"
        bodies = [b.split(" Question:")[0] for b in q]
        d["statements_per_query_table"] = sorted(set(nst))
        d["query_matches_question_line"] = all(re.search(qre, b) for b in q)
        d["distinct_bodies"] = len(set(bodies))
        # chain length: follow from each root
        d["example_query_tail"] = q[0][-80:]
        # k=6 query is the chain leaf: occurrences of the queried name in the body
        occ = collections.defaultdict(list)
        for x, b, body in zip(its, q, bodies):
            v = re.search(qre, b).group(1)
            occ[x["k"]].append(len(re.findall(rf"\b{v}\b", body)))
        d["query_name_occurrences_in_body_by_k"] = {k: sorted(set(v)) for k, v in sorted(occ.items())}
    else:
        q = [x["text_direct"].split("\n\n")[-1] for x in its]
        d["example_query"] = q[0][-160:]
    per[f] = d
res["families"] = per

# 3. independent re-solve of every item and worked example (e2b.verify raises on any mismatch)
for f in e2b.FAMILIES:
    e2b.verify(items[f])
res["e2b_verify_all_items_and_shots"] = True
res["n_prompts"] = sum(len(v) for v in items.values())
res["n_worked_examples"] = sum(len(x["shot_golds"]) for v in items.values() for x in v)

# 4. raw run files: direct mode, same texts for every model, split error, OLMo exclusion
runs = [r for r in e2b.load_results(ROOT) if r["mode"] == "direct"]
res["direct_runs"] = len(runs)
res["models"] = sorted({r["model"] for r in runs})
allruns = e2b.load_results(ROOT)
res["max_split_err_direct"] = max(r["checks"]["max_split_err"] for r in runs)
res["max_split_err_all_modes"] = max(r["checks"]["max_split_err"] for r in allruns)
res["n_runs_all_modes"] = len(allruns)
res["split_warn_items_all_modes"] = int(sum(r["checks"]["split_warn"] for r in allruns))
# same prompts in every model: compare ids and texts in the saved meta
texts_ok = True
for r in runs:
    z = np.load(e2b.out_path(ROOT, r["model"], "direct", r["family"]), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    want = {x["id"]: x["text_direct"] for x in items[r["family"]]}
    texts_ok &= meta["texts"] == want
res["run_texts_equal_items_json"] = bool(texts_ok)
excl = {}
for r in runs:
    C = r["C"]
    flag = ((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
    if flag.any():
        excl[f"{r['model'].split('/')[-1]}|{r['family']}"] = f"{int(flag.sum())}/{len(flag)}"
res["flagged_items_direct"] = excl
res["LH"] = {r["model"].split("/")[-1]: [r["L"], r["H"]] for r in runs}
json.dump(res, open(f"{OUT}/verify_design.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))

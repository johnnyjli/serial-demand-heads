"""Section 1: prompt/answer verification with the independent sentence parser (v_parse.py)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import collections, json, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v_parse import parse

D = pickle.load(open(os.path.join(HERE, "v_items.pkl"), "rb"))
runs = pickle.load(open(os.path.join(HERE, "v_runs.pkl"), "rb"))
D["cell"] = D.fam + "/" + D.w
R = {}

# prompts: one text per (cell, id) across all runs
g = D.groupby(["cell", "iid"])
ntext = g.text.nunique(); R["max_distinct_texts_per_prompt"] = int(ntext.max())
R["runs_per_prompt"] = sorted(set(g.size().tolist()))
R["n_runs"] = len(runs); R["models_per_prompt"] = sorted(set(g.model.nunique().tolist()))
first = g.first().reset_index()
R["n_prompts"] = len(first)
R["prompts_by_cell"] = first.cell.value_counts().sort_index().to_dict()
R["prompts_by_fam"] = first.fam.value_counts().sort_index().to_dict()
R["splits"] = D.split.value_counts().to_dict()
R["items_per_cell_k"] = sorted(set(first.groupby(["cell", "k"]).size().tolist()))

cnt = collections.Counter(); shotdepth = collections.Counter(); shots_per = collections.Counter()
shot_ans = {}; bad = collections.defaultdict(list)
for _, r in first.iterrows():
    bl = r.text.split("\n\n"); q = parse(r.fam, r.w, bl[-1]); shots = [parse(r.fam, r.w, b) for b in bl[:-1]]
    shots_per[len(shots)] += 1
    if q["shown"] is not None: bad["query_has_shown"].append(r.iid)
    if q["answer"] != r.gold: bad["gold"].append(r.iid)
    if r.fam in ("pointer", "arith"):
        kq = q["steps"]
    elif r.fam == "fixed_digit":
        kq = len(q["cycle"]) - 1
        if q["steps"] != 7: bad["fixed_written_not7"].append(r.iid)
        # table: one cycle through start, all other states fixed points
        if any(q["edges"][x] != x for x in range(7) if x not in q["cycle"]): bad["fixed_not_fixed_points"].append(r.iid)
    else:
        kq = q["hops"]
    if kq != r.k: bad["k"].append(r.iid)
    if r.fam == "pointer":
        # single 10-cycle => fewest lookups = min(k, 10-k)
        if len(q["cycle"]) != 10: bad["pointer_not_10cycle"].append(r.iid)
        if min(r.k, 10 - r.k) != r.k: bad["pointer_shortcut"].append(r.iid)
        if r.min_hops != r.k: bad["pointer_minhops_meta"].append(r.iid)
    if r.fam in ("binding", "relational"):
        if len(q["stm"]) != 21 or len(q["roots"]) != 3 or tuple(q["roots"]) != tuple(sorted(r.aset)):
            bad["br_structure"].append(r.iid)
    sa = []
    for s in shots:
        cnt["shots"] += 1
        if s["shown"] is None or s["shown"] != s["answer"]: bad["shot_answer"].append(r.iid)
        if r.fam in ("pointer", "arith", "fixed_digit"):
            dep = s["steps"]
        else:
            dep = s["hops"]
        shotdepth[(r.fam, dep)] += 1
        # same question as the query?
        if r.fam in ("pointer", "fixed_digit"):
            same = (s["edges"] == q["edges"] and s["start"] == q["start"] and s["steps"] == q["steps"])
        elif r.fam == "arith":
            same = (s["start"], s["inc"], s["steps"]) == (q["start"], q["inc"], q["steps"])
        else:
            same = (s["stm"] == q["stm"] and s["qvar"] == q["qvar"])
        if same: bad["shot_is_query"].append(r.iid)
        # shares the query's body (binding/relational) or table (pointer)?
        if r.fam in ("binding", "relational") and set(s["stm"]) == set(q["stm"]): cnt["br_shot_same_body"] += 1
        if r.fam == "pointer" and s["edges"] == q["edges"]: cnt["ptr_shot_same_table"] += 1
        sa.append(s["shown"])
    shot_ans[(r.cell, r.iid)] = sa
    if r.fam in ("pointer", "arith") and r.gold == r.echo: bad["gold_eq_echo"].append(r.iid)
    if r.fam in ("pointer", "arith") and r.echo != r.k: bad["echo_ne_k"].append(r.iid)

R["shots_total"] = cnt["shots"]; R["shots_per_prompt"] = dict(shots_per)
R["shot_depths"] = {f"{f}:{d}": n for (f, d), n in sorted(shotdepth.items())}
R["br_shot_same_body_as_query"] = cnt["br_shot_same_body"]; R["ptr_shot_same_table"] = cnt["ptr_shot_same_table"]
R["mismatch_counts"] = {k: len(v) for k, v in bad.items()}
for k in ["query_has_shown", "gold", "k", "shot_answer", "shot_is_query", "gold_eq_echo", "echo_ne_k",
          "pointer_not_10cycle", "pointer_shortcut", "pointer_minhops_meta", "br_structure",
          "fixed_written_not7", "fixed_not_fixed_points"]:
    R["mismatch_counts"].setdefault(k, 0)

# answer balance per cell x k
bal = first.groupby(["cell", "k"]).gold.apply(lambda s: (s.nunique(), s.value_counts().min(), s.value_counts().max()))
R["gold_balance_range"] = {c: sorted(set(v for (cc, k), v in bal.items() if cc == c)) for c in sorted(first.cell.unique())}

# prompt length (tokens) per run: distinct values over all items, and whether it varies with k
var_k, var_any = 0, 0
for (m, kern, cell), T in D.groupby(["model", "kernel", "cell"]):
    per_k = T.groupby("k").ntok.agg(lambda s: tuple(sorted(set(s))))
    if len(set(per_k.values)) > 1: var_k += 1
    if T.ntok.nunique() > 1: var_any += 1
R["runs_prompt_length_differs_across_k"] = var_k; R["runs_with_any_length_variation"] = var_any

# solve.py circularity claim
sys.path.insert(0, str(paths.ANALYSIS))
from solve import solve_block
b = ("Assignments: u = e. w = g. k = j. z = 3. r = y. j = s. q = z. b = k. s = u. c = 2. p = m. x = l. g = r. "
     "t = p. m = x. n = v. l = 6. y = q. e = c. v = t. d = w. Therefore y = 9.")
R["solvepy_on_corrupted_binding_w0"] = list(solve_block("binding", "w0", b))
R["my_parser_on_corrupted_binding_w0"] = [parse("binding", "w0", b)["answer"], parse("binding", "w0", b)["shown"]]
# how many real binding-w0 worked examples would solve.py fail to catch?  corrupt every shown answer.
circ = 0; tot = 0
for _, r in first[first.cell == "binding/w0"].iterrows():
    for sb in r.text.split("\n\n")[:-1]:
        head, tail = sb.rstrip(".").rsplit(" ", 1); wrong = (int(tail) + 1) % 10
        a, s = solve_block("binding", "w0", f"{head} {wrong}."); tot += 1; circ += (a == s)
R["solvepy_binding_w0_corrupted_shots_passing"] = [circ, tot]

pickle.dump(shot_ans, open(os.path.join(HERE, "v_shot_answers.pkl"), "wb"))
json.dump(R, open(os.path.join(HERE, "v1_prompts.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

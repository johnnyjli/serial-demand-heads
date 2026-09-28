"""Completeness critic: design checks on E2b not covered by the analysts.
(1) per-head split error vs the paper's stated bound; (2) elimination cue created by
'worked-example answers never equal the query's answer'; (3) whether models' direct predictions
avoid/copy worked-example answers; (4) E2 comparison of the same cue if E2 items are available."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, numpy as np, collections
sys.path.insert(0, str(paths.REPLICATION))
import e2b
ROOT = str(paths.E2B)
items = json.load(open(f"{ROOT}/items.json"))
if isinstance(items, dict): allit = [it for f in items for it in items[f]]
else: allit = items
IT = {it["id"]: it for it in allit}
out = {}
# (2) elimination cue
cue = {}
for fam in e2b.FAMILIES:
    res = collections.defaultdict(list)
    for it in allit:
        if it["family"] != fam: continue
        aset = set(it["answer_set"]); shown = set(str(g) for g in it["shot_golds"])
        cand = aset - shown
        if e2b.WRITES_K[fam]: cand -= {str(it["k"])}
        adm = aset - ({str(it["k"])} if e2b.WRITES_K[fam] else set())
        assert it["gold"] in cand
        res["elim_acc"].append(1 / len(cand)); res["chance_adm"].append(1 / len(adm))
        res["n_cand"].append(len(cand)); res["unique"].append(len(cand) == 1)
    cue[fam] = {k: float(np.mean(v)) for k, v in res.items()}
out["elimination_cue"] = cue
# (1) split error, (3) prediction behaviour
runs = e2b.load_results(ROOT)
se = {}; beh = collections.defaultdict(dict)
for r in runs:
    m = r["model"].split("/")[-1]; key = f"{m}|{r['mode']}|{r['family']}"
    se[key] = r["checks"]
    if r["mode"] != "direct": continue
    fam = r["family"]; inshot = []; exp_inshot = []; incand = []; exp_incand = []; acc = []
    for x in r["items"]:
        it = IT[x["id"]]; p = x.get("pred_constrained")
        if p is None: continue
        aset = set(it["answer_set"]); shown = set(str(g) for g in it["shot_golds"]) & aset
        adm = aset - ({str(it["k"])} if e2b.WRITES_K[fam] else set())
        cand = adm - shown
        inshot.append(p in shown); exp_inshot.append(len(shown & adm) / len(adm) if adm else np.nan)
        incand.append(p in cand); exp_incand.append(len(cand) / len(adm))
        acc.append(x["correct_constrained"])
    beh[m][fam] = {"pred_in_shot_answers": float(np.mean(inshot)), "uniform_expect": float(np.mean(exp_inshot)),
                   "pred_in_unshown": float(np.mean(incand)), "uniform_expect_unshown": float(np.mean(exp_incand))}
out["split_checks"] = {"max_err_over_runs": max(v["max_split_err"] for v in se.values()),
                       "n_runs_with_warn": sum(v["split_warn"] > 0 for v in se.values()),
                       "total_warn_items": sum(v["split_warn"] for v in se.values()),
                       "runs_with_warn": {k: v for k, v in se.items() if v["split_warn"] > 0}}
out["direct_prediction_vs_shot_answers"] = beh
# pooled
for fam in e2b.FAMILIES:
    a = [beh[m][fam]["pred_in_shot_answers"] for m in beh]; b = [beh[m][fam]["uniform_expect"] for m in beh]
    out.setdefault("pooled", {})[fam] = {"pred_in_shot_mean_over_models": float(np.mean(a)), "uniform_expect": float(np.mean(b)),
                                         "n_models_below_uniform": int(sum(x < y for x, y in zip(a, b)))}
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ["elimination_cue", "split_checks", "pooled"]}, indent=1))

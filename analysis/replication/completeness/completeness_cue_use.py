"""Do models use the E2b elimination cue (worked-example answers never equal the gold)?
Accuracy (constrained argmax) on items whose gold is the unique admissible digit not shown as a
worked-example answer vs. other items, by family and depth band; cue strength vs k; arith k=1 copying."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, numpy as np, collections
from scipy.stats import fisher_exact, spearmanr
sys.path.insert(0, str(paths.REPLICATION))
import e2b
ROOT = str(paths.E2B)
items = json.load(open(f"{ROOT}/items.json"))
allit = [it for f in items for it in items[f]] if isinstance(items, dict) else items
IT = {it["id"]: it for it in allit}
def cand(it):
    fam = it["family"]; aset = set(it["answer_set"]); shown = set(str(g) for g in it["shot_golds"])
    c = aset - shown
    if e2b.WRITES_K[fam]: c -= {str(it["k"])}
    return c
out = {"cue_by_k": {}, "use": {}, "arith_k1": {}}
for fam in e2b.FAMILIES:
    ks = sorted({it["k"] for it in allit if it["family"] == fam})
    out["cue_by_k"][fam] = {k: float(np.mean([1 / len(cand(it)) for it in allit if it["family"] == fam and it["k"] == k])) for k in ks}
runs = [r for r in e2b.load_results(ROOT) if r["mode"] == "direct"]
pool = collections.defaultdict(lambda: collections.defaultdict(list))
for r in runs:
    m = r["model"].split("/")[-1]; fam = r["family"]
    for x in r["items"]:
        it = IT[x["id"]]; c = cand(it); uniq = len(c) == 1
        band = "k1" if x["k"] == 1 else ("k2" if x["k"] == 2 else "k>=3")
        pool[(fam, band)]["uniq" if uniq else "multi"].append(bool(x["correct_constrained"]))
        pool[(fam, band, m)]["uniq" if uniq else "multi"].append(bool(x["correct_constrained"]))
        # prediction inside the elimination candidate set (vs expected |c|/|adm| if uniform)
        p = x.get("pred_constrained")
        if p is not None and not x["correct_constrained"]:
            adm = set(it["answer_set"]) - ({str(it["k"])} if e2b.WRITES_K[fam] else set())
            wrong_adm = adm - {it["gold"]}
            pool[(fam, band)]["wrong_in_cand"].append(p in c)
            pool[(fam, band)]["wrong_in_cand_exp"].append(len(c - {it["gold"]}) / len(wrong_adm))
            if fam == "arith" and x["k"] == 1:
                shown = set(str(g) for g in it["shot_golds"])
                out["arith_k1"].setdefault(m, []).append(p in shown)
for key, v in pool.items():
    if len(key) != 2: continue
    fam, band = key; u, mu = v["uniq"], v["multi"]
    tab = [[sum(u), len(u) - sum(u)], [sum(mu), len(mu) - sum(mu)]]
    p = fisher_exact(tab)[1] if len(u) and len(mu) else None
    out["use"][f"{fam}|{band}"] = {"acc_unique_cand": float(np.mean(u)) if u else None, "n_unique": len(u),
                                   "acc_multi_cand": float(np.mean(mu)) if mu else None, "n_multi": len(mu), "fisher_p": p,
                                   "wrong_pred_in_unshown": float(np.mean(v["wrong_in_cand"])) if v["wrong_in_cand"] else None,
                                   "wrong_pred_in_unshown_uniform": float(np.mean(v["wrong_in_cand_exp"])) if v["wrong_in_cand_exp"] else None,
                                   "n_wrong": len(v["wrong_in_cand"])}
    # models where unique > multi
    ms = sorted({k[2] for k in pool if len(k) == 3 and k[0] == fam and k[1] == band})
    diffs = [np.mean(pool[(fam, band, m)]["uniq"]) - np.mean(pool[(fam, band, m)]["multi"]) for m in ms
             if pool[(fam, band, m)]["uniq"] and pool[(fam, band, m)]["multi"]]
    out["use"][f"{fam}|{band}"]["n_models_unique_higher"] = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
out["arith_k1"] = {m: {"wrong_pred_is_a_shot_answer": float(np.mean(v)), "n_wrong": len(v)} for m, v in out["arith_k1"].items()}
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
print(json.dumps(out, indent=1))

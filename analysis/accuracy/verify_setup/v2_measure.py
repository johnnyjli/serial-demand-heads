"""Section 2: per-run measurement checks (meta) and digit mass at the answer position."""
import collections, json, os, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(HERE, "v_items.pkl"), "rb"))
runs = pickle.load(open(os.path.join(HERE, "v_runs.pkl"), "rb"))
R = {}
for kern in ("eager", "sdpa"):
    rr = [r for r in runs if r["kernel"] == kern]
    C = pd.DataFrame([dict(model=r["model"], cell=f'{r["fam"]}/{r["w"]}', **{k: v for k, v in r["checks"].items()
                                                                              if not isinstance(v, (dict, list))}) for r in rr])
    i = C.split_err.idxmax()
    warn = C[C.split_err > 1e-2]
    R[kern] = dict(n_runs=len(rr), n_items=int(sum(r["n"] for r in rr)), n_models=C.model.nunique(),
                   max_split_err=float(C.split_err.max()), where=f"{C.model[i]} {C.cell[i]}",
                   split_tol=sorted(set(C.split_tol)),
                   n_runs_split_err_gt_1e2=int(len(warn)), n_runs_split_warn_flag=int((C.split_warn > 0).sum()),
                   models_split_err_gt_1e2=sorted(set(warn.model)),
                   max_split_exact_err=float(C.split_exact_err.max()),
                   max_post_norm_err=float(C.post_norm_err.max()), max_attn_row_err=float(C.attn_row_err.max()),
                   all_verified=bool(C.verified.all()),
                   precision=sorted(set(r["precision"] for r in rr)))
    # appended token per run
    af = collections.Counter()
    for r in rr:
        af[(r["model"], tuple(tuple(a) for a in r["appended_forms"]))] += 1
    R[kern]["appended_forms_by_model"] = {f"{m}": [list(a) for a in forms] for (m, forms) in af}
T = D.copy()
R["items_appended_len"] = T.appended.map(len).value_counts().to_dict()
R["appended_forms_items"] = T.appended.map(lambda a: "|".join(a)).value_counts().to_dict()
R["appended_form_by_model"] = T.groupby("model").appended.agg(lambda s: sorted(set("|".join(a) for a in s))).to_dict()
R["measured_token_eq_appended"] = bool((T.mtok == T.appended.map(lambda a: a[-1] if a else None)).all())

# main table: 17 models, eager except Qwen2.5-7B-Instruct (SDPA)
M = T[((T.kernel == "eager") & (T.model != "Qwen2.5-7B-Instruct")) | ((T.model == "Qwen2.5-7B-Instruct") & (T.kernel == "sdpa"))]
assert len(M) == 17 * 2436
pm = M.groupby("model").dmass.mean()
R["digit_mass_main"] = dict(mean=float(M.dmass.mean()), median=float(M.dmass.median()),
                            frac_below_0p5=float((M.dmass < .5).mean()),
                            lowest_model_mean=[pm.idxmin(), float(pm.min())],
                            second_lowest=[pm.sort_values().index[1], float(pm.sort_values().iloc[1])])
# digit mass restricted to the answer set is not stored; digit mass from logits sanity (softmax over 10 digits is not total prob)
json.dump(R, open(os.path.join(HERE, "v2_measure.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

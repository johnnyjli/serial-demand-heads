"""E2b stepwise: the context-length confound.  In stepwise mode the measured position follows the model's own
trace, so its context length n_tokens_measured grows with k.  Prompt text has constant length within a family
(checked: text_stepwise has one character length per family), so all length variation is trace length.

Per stepwise cell (measured items after the OLMo-2 exclusion; also correct items only):
  - collinearity: Pearson r(k, n_tok), tokens per step, pooled within-k SD of n_tok, share of items whose
    length differs from the modal length at their k, share of k's variance NOT explained by n_tok (linear),
    and share of k's variance that lies within exact-length strata (what a stratified analysis can use);
  - k effect net of length, three ways: (a) Freedman-Lane partial slope of each layer on k given n_tok (linear);
    (b) pooled within-stratum slope on k within exact n_tok strata (k permuted within strata); (c) same within
    3-token-wide strata;  back-half mean z with a joint permutation test (B=4000);
  - length effect at fixed k: pooled within-k slope of each layer on n_tok (n_tok permuted within k).
Primary statistic top-10% share; also top-25% share and effective number of heads."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json
import numpy as np
from scipy import stats as sst
sys.path.insert(0, str(paths.ANALYSIS / "replication"))
import stepwise_vs_direct_lib as E

OUT = str(paths.ANALYSIS / "replication" / "stepwise_vs_direct_length.json")
B = 4000
D = E.load(exclude=True)


def collin(k, t):
    r = float(np.corrcoef(k, t)[0, 1]) if t.std() > 0 else float("nan")
    tps = float(np.polyfit(k, t, 1)[0])
    ks = np.unique(k); res = np.concatenate([t[k == v] - t[k == v].mean() for v in ks])
    wsd = float(np.sqrt((res ** 2).sum() / max(len(t) - len(ks), 1)))
    nonmodal = 0; modal = {}
    for v in ks:
        vals, cnt = np.unique(t[k == v], return_counts=True); mo = vals[cnt.argmax()]; modal[int(v)] = float(mo)
        nonmodal += int((t[k == v] != mo).sum())
    X = np.column_stack([np.ones(len(t)), t]); kr = k - X @ np.linalg.lstsq(X, k, rcond=None)[0]
    kc = k - k.mean()
    frac_lin = float((kr @ kr) / (kc @ kc))
    _, kw, n_inf = E.within_centre(np.zeros((len(k), 1)), k, t)
    frac_strat = float((kw @ kw) / (kc @ kc))
    return {"r_k_len": r, "tokens_per_step": tps, "within_k_sd": wsd, "n_nonmodal": nonmodal, "frac_nonmodal": nonmodal / len(t),
            "modal_len": modal, "frac_k_var_unexpl_linear": frac_lin, "frac_k_var_within_exact_strata": frac_strat,
            "n_items_informative_exact": n_inf, "n": int(len(t))}


def strat_test(Y, k, g, L, rng):
    Yw, kw, n_inf = E.within_centre(Y, k, g)
    if n_inf < 11 or (kw @ kw) < 1e-9: return {"n_inf": n_inf, "ok": False}
    r = E.cell_test(Yw, kw, L, rng, B, body=g, within=True)
    return {"n_inf": n_inf, "ok": True, "back": r["back"], "p_back_lo": r["p_back_lo"], "p_back_hi": r["p_back_hi"],
            "back_slope_pp": r["back_slope_pp"], "dec": r["dec"].tolist(), "n_sig_back": int(((r["p"] < .05) & (np.arange(L) / (L - 1) >= .5)).sum())}


RES = {}
for stat in E.STATS:
    for m in E.MODELS:
        for f in E.FAMS:
            d = D[(m, "stepwise", f)]
            for sub in ("all", "correct"):
                i = np.arange(len(d["k"])) if sub == "all" else np.nonzero(d["correct"])[0]
                if len(i) < 15: continue
                k, t, Y, L = d["k"][i], d["ntok"][i], d[stat][i], d["L"]
                rng = np.random.default_rng(E.seed(stat, m, f, sub, "len"))
                o = {"collinearity": collin(k, t)}
                base = E.cell_test(Y, k, L, rng, B)
                o["unadjusted"] = {"back": base["back"], "p_back_lo": base["p_back_lo"], "p_back_hi": base["p_back_hi"], "back_slope_pp": base["back_slope_pp"]}
                fl = E.partial_k_given_len(Y, k, t, L, rng, B)
                o["partial_k_given_len"] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in fl.items() if kk != "z"}
                o["strat_exact"] = strat_test(Y, k, t, L, rng)
                o["strat_3tok"] = strat_test(Y, k, np.floor(t / 3.0), L, rng)
                # length at fixed k (n_tok permuted within k)
                o["len_within_k"] = strat_test(Y, t, k, L, rng)
                # length at fixed k, per extra token (slope in pp per token), sign convention as z
                RES[f"{stat}|{m}|{f}|{sub}"] = o
print("done cells")


def agg(stat, sub, models, part):
    rows = [(m, f, RES.get(f"{stat}|{m}|{f}|{sub}")) for m in models for f in E.FAMS]
    rows = [(m, f, r) for m, f, r in rows if r is not None]
    ok = [(m, f, r[part]) for m, f, r in rows if r[part].get("ok", "back" in r[part])]
    if not ok: return {"n_cells": 0}
    back = np.array([x["back"] for _, _, x in ok])
    return {"n_cells": len(ok), "n_back_neg": int((back < 0).sum()), "n_sig_spread": int(sum(x["p_back_lo"] < .05 for _, _, x in ok)),
            "n_sig_conc": int(sum(x["p_back_hi"] < .05 for _, _, x in ok)), "median_back": float(np.median(back)), "mean_back": float(back.mean()),
            "cells": [f"{m}|{f}" for m, f, _ in ok]}


SUMM = {}
for stat in E.STATS:
    for sub in ("all", "correct"):
        for sname, ms in {"all8": E.MODELS, "no_qwen7": [m for m in E.MODELS if m not in E.QWEN7]}.items():
            for part in ("unadjusted", "partial_k_given_len", "strat_exact", "strat_3tok", "len_within_k"):
                SUMM[f"{stat}|{sub}|{sname}|{part}"] = agg(stat, sub, ms, part)
json.dump({"cells": RES, "summary": SUMM}, open(OUT, "w"), indent=1)
print("saved", OUT)

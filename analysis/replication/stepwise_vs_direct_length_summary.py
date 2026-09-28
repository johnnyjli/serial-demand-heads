"""Compact summary of the stepwise length confound (reads stepwise_vs_direct_length.json); writes
stepwise_vs_direct_length_summary.json."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
SPD = str(paths.ANALYSIS / "replication")
J = json.load(open(f"{SPD}/stepwise_vs_direct_length.json"))["cells"]
M = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B", "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
F = ["pointer", "arith", "binding", "relational"]
out = {}
for sub in ("all", "correct"):
    rows = [(m, f, J.get(f"q10|{m}|{f}|{sub}")) for m in M for f in F]
    rows = [(m, f, o) for m, f, o in rows if o]
    r = np.array([o["collinearity"]["r_k_len"] for _, _, o in rows])
    fu = np.array([o["collinearity"]["frac_k_var_unexpl_linear"] for _, _, o in rows])
    wsd = np.array([o["collinearity"]["within_k_sd"] for _, _, o in rows])
    fnm = np.array([o["collinearity"]["frac_nonmodal"] for _, _, o in rows])
    tps = {f: float(np.median([o["collinearity"]["tokens_per_step"] for _, ff, o in rows if ff == f])) for f in F}
    base = {f: float(np.median([o["collinearity"]["modal_len"]["1"] for _, ff, o in rows if ff == f])) for f in F}
    s = {"n_cells": len(rows), "median_r": float(np.median(r)), "n_r_ge_0.98": int((r >= .98).sum()), "n_r_ge_0.99": int((r >= .99).sum()),
         "min_r": float(np.nanmin(r)), "n_zero_within_k_var": int((wsd == 0).sum()), "cells_zero_within_k_var": [f"{m}|{f}" for (m, f, o), w in zip(rows, wsd) if w == 0],
         "median_frac_k_unexpl": float(np.median(fu)), "n_frac_k_unexpl_ge_0.10": int((fu >= .10).sum()),
         "median_frac_nonmodal": float(np.median(fnm)), "tokens_per_step_median": tps, "modal_len_k1_median": base,
         "per_model_median_r": {m: float(np.median([o["collinearity"]["r_k_len"] for mm, _, o in rows if mm == m])) for m in M},
         "per_model_frac_nonmodal": {m: float(np.mean([o["collinearity"]["frac_nonmodal"] for mm, _, o in rows if mm == m])) for m in M}}
    for part in ("partial_k_given_len", "strat_exact", "len_within_k"):
        ok = [(m, f, o) for m, f, o in rows if ("back" in o[part] and o[part].get("ok", True))]
        un = [o["unadjusted"] for _, _, o in ok]; ad = [o[part] for _, _, o in ok]
        def cnt(L): return {"n_neg": int(sum(x["back"] < 0 for x in L)), "sig_spread": int(sum(x["p_back_lo"] < .05 for x in L)),
                            "sig_conc": int(sum(x["p_back_hi"] < .05 for x in L)), "median": float(np.median([x["back"] for x in L]))}
        keep_dir = sum((u["p_back_lo"] < .05 and a["p_back_lo"] < .05) or (u["p_back_hi"] < .05 and a["p_back_hi"] < .05) for u, a in zip(un, ad))
        s[part] = {"n_cells": len(ok), "adjusted": cnt(ad), "unadjusted_same_cells": cnt(un), "n_unadj_sig": int(sum(u["p_back_lo"] < .05 or u["p_back_hi"] < .05 for u in un)),
                   "n_sig_retained_same_direction": int(keep_dir),
                   "r_adjusted_vs_unadjusted_back": float(np.corrcoef([u["back"] for u in un], [a["back"] for a in ad])[0, 1])}
        # leverage >= 10% subset for the partial analysis
        if part == "partial_k_given_len":
            lv = [(u, a) for (m, f, o), u, a in zip(ok, un, ad) if o["collinearity"]["frac_k_var_unexpl_linear"] >= .10]
            s[part]["leverage_ge_0.10"] = {"n_cells": len(lv), "adjusted": cnt([a for _, a in lv]), "unadjusted_same_cells": cnt([u for u, _ in lv]),
                                           "n_sig_retained_same_direction": int(sum((u["p_back_lo"] < .05 and a["p_back_lo"] < .05) or (u["p_back_hi"] < .05 and a["p_back_hi"] < .05) for u, a in lv))}
    out[sub] = s
json.dump(out, open(f"{SPD}/stepwise_vs_direct_length_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

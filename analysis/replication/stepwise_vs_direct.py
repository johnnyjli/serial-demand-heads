"""E2b: stepwise mode against direct mode with the paper's statistic (top-10% head share; top-25% share and
effective number of heads as robustness).  Writes stepwise_vs_direct.json.

Per model x family x mode x subset x statistic: per-layer z (v2lib.zprof, k shuffled across items), per-layer
two-sided permutation p, back-half mean z with a joint one-sided permutation test (B=4000, same shuffle for
all layers, as in the paper's per-cell test), decile profile (model_diff.dec_prof), back-half mean slope in
percentage points per step.  Subsets: all measured items; correct (written answer) only; for direct also the
items measured in stepwise ('matched').  Model sets: all 8, and without Qwen2.5-7B and 7B-Instruct."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json
import numpy as np
from scipy import stats as sst
sys.path.insert(0, str(paths.ANALYSIS / "replication"))
import stepwise_vs_direct_lib as E

OUT = str(paths.ANALYSIS / "replication" / "stepwise_vs_direct.json")
B = 4000
D = E.load(exclude=True)
DN = E.load(exclude=False)
SETS = {"all8": E.MODELS, "no_qwen7": [m for m in E.MODELS if m not in E.QWEN7]}


def subset_idx(d, sub, other=None):
    if sub == "all": return np.arange(len(d["k"]))
    if sub == "correct": return np.nonzero(d["correct"])[0]
    if sub == "matched":            # direct items that were also measured (and kept) in stepwise
        ids = set(other["ids"]); return np.array([i for i, x in enumerate(d["ids"]) if x in ids], int)
    if sub == "trace_exact": return np.nonzero(d["trace_exact"])[0]
    raise ValueError(sub)


def run_cell(d, idx, stat, tag):
    k = d["k"][idx]
    npk = {int(v): int((k == v).sum()) for v in np.unique(d["k_all"])}
    if len(idx) < 11 or len(np.unique(k)) < 2:
        return {"n": int(len(idx)), "n_per_k": npk, "ok": False}
    rng = np.random.default_rng(E.seed(*tag))
    r = E.cell_test(d[stat][idx], k, d["L"], rng, B)
    r.update({"n": int(len(idx)), "n_per_k": npk, "ok": True})
    return r


# ---------------------------------------------------------------- per-cell results
R = {}
for stat in E.STATS:
    for m in E.MODELS:
        for f in E.FAMS:
            dd, ds = D[(m, "direct", f)], D[(m, "stepwise", f)]
            for sub in ("all", "correct"):
                R[(stat, "direct", sub, m, f)] = run_cell(dd, subset_idx(dd, sub), stat, (stat, m, f, "direct", sub))
                R[(stat, "stepwise", sub, m, f)] = run_cell(ds, subset_idx(ds, sub), stat, (stat, m, f, "stepwise", sub))
            R[(stat, "direct", "matched", m, f)] = run_cell(dd, subset_idx(dd, "matched", ds), stat, (stat, m, f, "direct", "matched"))
            R[(stat, "stepwise", "trace_exact", m, f)] = run_cell(ds, subset_idx(ds, "trace_exact"), stat, (stat, m, f, "stepwise", "te"))
            # direct items that the model answered correctly in stepwise vs matched stepwise items (same ids)
            if m.startswith("OLMo"):
                for mode in E.MODES:
                    dn = DN[(m, mode, f)]
                    R[(stat, mode, "noexcl", m, f)] = run_cell(dn, subset_idx(dn, "all"), stat, (stat, m, f, mode, "noexcl"))
print("cells done")


def lineage_mean(vals, models):
    lin = {}
    for m, v in zip(models, vals): lin.setdefault(E.LINEAGE[m], []).append(v)
    return np.array([np.mean(v, 0) for v in lin.values()])


def tci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return [float("nan")] * 2
    h = sst.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean() - h), float(x.mean() + h)]


def boot_models(per_model, nb=10000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(per_model, float)
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(nb)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summary(stat, mode, sub, models):
    cells = [(m, f) for m in models for f in E.FAMS]
    rr = [R.get((stat, mode, sub, m, f)) for m, f in cells]
    ok = [(c, r) for c, r in zip(cells, rr) if r is not None and r.get("ok")]
    if not ok: return None
    back = np.array([r["back"] for _, r in ok])
    s = {"n_cells": len(ok), "n_back_neg": int((back < 0).sum()),
         "n_back_sig_spread": int(sum(r["p_back_lo"] < .05 for _, r in ok)),
         "n_back_sig_conc": int(sum(r["p_back_hi"] < .05 for _, r in ok)),
         "median_back_z": float(np.median(back)), "mean_back_z": float(back.mean()),
         "median_back_slope_pp": float(np.median([r["back_slope_pp"] for _, r in ok])),
         "mean_back_slope_pp": float(np.mean([r["back_slope_pp"] for _, r in ok])),
         "frac_back_layers_sig_spread": float(sum(r["n_sig_spread_back"] for _, r in ok) / sum(r["n_back"] for _, r in ok)),
         "frac_back_layers_sig_conc": float(sum(r["n_sig_conc_back"] for _, r in ok) / sum(r["n_back"] for _, r in ok)),
         "per_family_mean_back_z": {f: float(np.mean([r["back"] for (m, ff), r in ok if ff == f])) for f in E.FAMS},
         "per_family_n_back_neg": {f: int(sum(r["back"] < 0 for (m, ff), r in ok if ff == f)) for f in E.FAMS},
         "per_family_n_sig_spread": {f: int(sum(r["p_back_lo"] < .05 for (m, ff), r in ok if ff == f)) for f in E.FAMS}}
    # shared decile profile: model = mean over its families that are ok; mean over models
    mprof, mnames = [], []
    for m in models:
        ps = [r["dec"] for (mm, f), r in ok if mm == m]
        if ps: mprof.append(np.nanmean(ps, 0)); mnames.append(m)
    mprof = np.array(mprof)
    s["decile_mean_over_models"] = mprof.mean(0).tolist()
    s["decile_ci_models"] = [tci(mprof[:, j]) for j in range(10)]
    lp = lineage_mean(mprof, mnames)
    s["decile_mean_over_lineages"] = lp.mean(0).tolist()
    s["decile_ci_lineages"] = [tci(lp[:, j]) for j in range(10)]
    s["n_models_dec4_gt_dec5"] = int((mprof[:, 4] > mprof[:, 5]).sum())
    s["n_models_dec4_pos"] = int((mprof[:, 4] > 0).sum())
    s["n_models_50_80_neg"] = int((mprof[:, 5:8].mean(1) < 0).sum())
    s["mean_50_80"] = float(mprof[:, 5:8].mean())
    s["ci_50_80_models"] = tci(mprof[:, 5:8].mean(1))
    s["per_model_back_z"] = {m: float(np.mean([r["back"] for (mm, f), r in ok if mm == m])) for m in mnames}
    s["per_model_decile"] = {m: p.tolist() for m, p in zip(mnames, mprof)}
    return s


SUM = {}
for stat in E.STATS:
    for mode, sub in [("direct", "all"), ("stepwise", "all"), ("direct", "correct"), ("stepwise", "correct"),
                      ("direct", "matched"), ("stepwise", "trace_exact")]:
        for sname, ms in SETS.items():
            SUM[(stat, mode, sub, sname)] = summary(stat, mode, sub, ms)


# ---------------------------------------------------------------- paired stepwise - direct
def paired(stat, sub_d, sub_s, models, key="back"):
    rows = []
    for m in models:
        for f in E.FAMS:
            a, b = R[(stat, "direct", sub_d, m, f)], R[(stat, "stepwise", sub_s, m, f)]
            if a.get("ok") and b.get("ok"): rows.append((m, f, b[key] - a[key]))
    if not rows: return None
    dlt = np.array([r[2] for r in rows]); npos = int((dlt > 0).sum()); nneg = int((dlt < 0).sum())
    pm = {m: float(np.mean([r[2] for r in rows if r[0] == m])) for m in models if any(r[0] == m for r in rows)}
    return {"n_cells": len(rows), "n_step_lower": nneg, "n_step_higher": npos,
            "sign_test_p": float(sst.binomtest(nneg, nneg + npos, .5).pvalue),
            "mean": float(dlt.mean()), "median": float(np.median(dlt)),
            "mean_over_models": float(np.mean(list(pm.values()))), "boot_ci_models": boot_models(list(pm.values())),
            "per_model": pm, "per_cell": {f"{m}|{f}": float(v) for m, f, v in rows}}


PAIR = {}
for stat in E.STATS:
    for sname, ms in SETS.items():
        for key in ("back", "back_slope_pp"):
            PAIR[(stat, "all-all", key, sname)] = paired(stat, "all", "all", ms, key)
            PAIR[(stat, "matched-all", key, sname)] = paired(stat, "matched", "all", ms, key)
            PAIR[(stat, "correct-correct", key, sname)] = paired(stat, "correct", "correct", ms, key)

# model-level agreement between modes: full profile r and signature r (deviation from the 8-model mean)
AGREE = {}
for stat in E.STATS:
    for sname, ms in SETS.items():
        Pd = np.array([SUM[(stat, "direct", "all", sname)]["per_model_decile"][m] for m in ms])
        Ps = np.array([SUM[(stat, "stepwise", "all", sname)]["per_model_decile"][m] for m in ms])
        full = [float(np.corrcoef(Pd[i], Ps[i])[0, 1]) for i in range(len(ms))]
        Ad, As = Pd - Pd.mean(0), Ps - Ps.mean(0)
        sig = [float(np.corrcoef(Ad[i], As[i])[0, 1]) for i in range(len(ms))]
        AGREE[(stat, sname)] = {"full_profile_r": dict(zip(ms, full)), "mean_full_r": float(np.mean(full)),
                                "signature_r": dict(zip(ms, sig)), "mean_signature_r": float(np.mean(sig)),
                                "pooled_signature_r": float(np.corrcoef(Ad.ravel(), As.ravel())[0, 1]),
                                "shared_profile_r": float(np.corrcoef(Pd.mean(0), Ps.mean(0))[0, 1])}
        # per-cell decile-profile correlation direct vs stepwise
        cr = [float(np.corrcoef(R[(stat, "direct", "all", m, f)]["dec"], R[(stat, "stepwise", "all", m, f)]["dec"])[0, 1]) for m in ms for f in E.FAMS]
        AGREE[(stat, sname)]["cell_profile_r_mean"] = float(np.mean(cr))

# ---------------------------------------------------------------- answer uncertainty at the measured position
ENT = {}
for m in E.MODELS:
    for f in E.FAMS:
        for mode in E.MODES:
            d = D[(m, mode, f)]
            for sub in ("all", "correct"):
                i = subset_idx(d, sub)
                if len(i) < 5: continue
                k = d["k"][i]
                sl = float(np.polyfit(k, d["ent"][i], 1)[0]) if len(np.unique(k)) > 1 else float("nan")
                ENT[f"{m}|{f}|{mode}|{sub}"] = {"mean_entropy": float(d["ent"][i].mean()), "mean_pmax": float(d["pmax"][i].mean()),
                                               "entropy_slope_per_k": sl}


def key(t): return "|".join(t)


out = {"cells": {key(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in R.items()},
       "summary": {key(k): v for k, v in SUM.items()}, "paired": {key(k): v for k, v in PAIR.items()},
       "mode_agreement": {key(k): v for k, v in AGREE.items()}, "answer_uncertainty": ENT,
       "n": {f"{m}|{mode}|{f}": {"n_total": D[(m, mode, f)]["n_total"], "n_measured": D[(m, mode, f)]["n_measured"],
                                  "n_flagged_measured": D[(m, mode, f)]["n_flagged_measured"], "n_kept": int(len(D[(m, mode, f)]["k"])),
                                  "n_correct_kept": int(D[(m, mode, f)]["correct"].sum())}
             for m in E.MODELS for mode in E.MODES for f in E.FAMS}}
json.dump(out, open(OUT, "w"), indent=1)
print("saved", OUT)

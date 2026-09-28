"""E7 (top-10% head ablation, 17 models): robustness of the within-task association between the
top-10% share and the dependence on the top heads, per-band results, and the Llama / Qwen comparison.
Recomputes every row from the npz files with ablation_window.summarise (same definitions as the report).
Run from the repository root: python3 analysis/ablation/e7_analysis.py
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
HERE = Path(__file__).resolve().parent; RUNS = paths.E7_ABLATION / "bf16_sdpa"
sys.path.insert(0, str(paths.CAPTURE))
import ablation_window as aw
rng = np.random.default_rng(0); NP = 10000
FILES = sorted(glob.glob(str(RUNS / "*__abwin.npz")))
rows = [aw.summarise(f) for f in FILES]
df = pd.DataFrame(rows)
df["family"] = df.model.str.split("-").str[0].str.replace("Qwen2.5", "Qwen")
df["base"] = ~df.model.str.contains("Instruct|SFT|DPO")
df["floor"] = df.q_heads / df.H
df["share_excess"] = df.share_win - df.floor
TASKS = ["pointer", "arith", "binding", "relational"]

def within_task(d, x, y, n_perm=NP, resid_on=None):
    """mean over tasks of Spearman(x, y) across models; permutation shuffles y within task."""
    per, null = {}, np.zeros(n_perm)
    for t in TASKS:
        e = d[(d.task == t) & np.isfinite(d[x]) & np.isfinite(d[y])]
        if len(e) < 5: continue
        a, b = rankdata(e[x]), rankdata(e[y])
        if resid_on is not None:                       # partial: residualise both ranks on covariates
            Z = np.column_stack([np.ones(len(e))] + [rankdata(e[c]) for c in resid_on])
            a = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]; b = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
        per[t] = (float(np.corrcoef(a, b)[0, 1]), len(e))
        null += np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    obs = np.mean([v[0] for v in per.values()]); null /= len(per)
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1)), per

def show(label, d, x="share_win", y="excess_win", **kw):
    o, p, per = within_task(d, x, y, **kw)
    print(f"  {label:58s} mean rho {o:+.2f} (p={p:.4f})  " + "  ".join(f"{t[:4]} {v[0]:+.2f}/{v[1]}" for t, v in per.items()))
    return o, p

print("== within task, across models: x = top-10% share of the 40-80% window, y = excess drop (top-q minus random) ==")
R = {}
R["primary"] = show("primary (report)", df)
R["floor"] = show("share minus floor q/H", df, x="share_excess")
R["partial_H"] = show("partial on heads per layer", df, resid_on=["H"])
R["H>=24"] = show("models with H >= 24 (drops Qwen2.5 0.5B/1.5B/3B)", df[df.H >= 24])
R["bases"] = show("11 base models", df[df.base])
R["n15"] = show("cells with >= 15 correct items", df[df.n_correct >= 15])
R["no_qwen"] = show("without Qwen2.5 (11 models)", df[df.family != "Qwen"])
R["no_olmo"] = show("without OLMo-2 (13 models)", df[df.family != "OLMo"])
R["drop_only"] = show("y = top-q drop (no random subtracted)", df, y="drop_win_topq_mean")
R["zero"] = show("y = top-q zero-ablation drop", df, y="drop_win_topq_zero")
R["abs"] = show("y = excess drop in logits", df, y="excess_win_abs")
R["rand"] = show("y = random-head drop (should not follow share)", df, y="drop_win_randq")
# within family: demean x and y by family within task
d2 = df.copy()
for c in ("share_win", "excess_win"):
    d2[c + "_fd"] = d2[c] - d2.groupby(["task", "family"])[c].transform("mean")
R["within_family"] = show("within family (family means removed per task)", d2, x="share_win_fd", y="excess_win_fd")

print("\n== across tasks, within a model: Spearman over its tasks (>= 3 tasks with data) ==")
pm = {}
for m, d in df.groupby("model"):
    e = d[np.isfinite(d.excess_win)]
    if len(e) >= 3: pm[m] = spearmanr(e.share_win, e.excess_win)[0]
print("  mean rho", round(float(np.nanmean(list(pm.values()))), 2), "over", len(pm), "models; positive in",
      sum(v > 0 for v in pm.values()), "; spread of the share across a model's tasks (mean range):",
      round(float(df.groupby('model').share_win.agg(lambda s: s.max() - s.min()).mean()), 3),
      "vs across models within a task:", round(float(df.groupby('task').share_win.agg(lambda s: s.max() - s.min()).mean()), 3))

print("\n== per depth band: within-task rho(share of the band, excess drop of the band) ==")
for b in ["b0", "b1", "b2", "b3", "b4"]:
    show(f"band {b}", df, x=f"share_{b}", y=f"excess_{b}", n_perm=2000)

print("\n== Llama vs Qwen2.5, per task (share, top-q drop, random drop, excess) ==")
cols = ["model", "task", "H", "n_correct", "share_win", "drop_win_topq_mean", "drop_win_randq", "excess_win", "drop_win_topq_zero", "drop_win_top3_mean"]
lq = df[df.family.isin(["Llama", "Qwen"])].sort_values(["task", "family", "model"])
print(lq[cols].round(3).to_string(index=False))
fam = df.groupby(["family", "task"])[["share_win", "excess_win", "drop_win_randq"]].median().round(3)
print("\nmedian per family and task:\n", fam.unstack("task").to_string())
# matched sizes: Llama-3.1-8B vs Qwen2.5-7B; Llama-3.2-3B vs Qwen2.5-3B; instruct pairs
pairs = [("Llama-3.1-8B", "Qwen2.5-7B"), ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"), ("Llama-3.2-3B", "Qwen2.5-3B"),
         ("Llama-3.2-1B", "Qwen2.5-1.5B")]
print("\nmatched pairs: share and excess (Llama | Qwen) per task")
for a, b in pairs:
    for t in TASKS:
        ra, rb = df[(df.model == a) & (df.task == t)].iloc[0], df[(df.model == b) & (df.task == t)].iloc[0]
        print(f"  {a:22s} vs {b:20s} {t:10s} share {ra.share_win:.3f} | {rb.share_win:.3f}   excess {ra.excess_win:+.3f} | {rb.excess_win:+.3f}")
df.to_csv(HERE / "e7_rows_full.csv", index=False)
json.dump({k: list(v) for k, v in R.items()}, open(HERE / "e7_within_task.json", "w"), indent=1)

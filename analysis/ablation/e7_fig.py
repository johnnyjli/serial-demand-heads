"""Appendix figure for E7: does a more concentrated model depend more on its most active heads?
Reads e7_rows_full.csv (e7_analysis.py).  Style as the paper's figures (model_figs.py).
Run from the repository root: python3 analysis/ablation/e7_fig.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "e7_rows_full.csv")
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FC = {"Llama": "#2a78d6", "Qwen": "#eb6834", "OLMo": "#1baf7a", "Mistral": "#eda100"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 6})
TASKS = [("pointer", "Pointer chasing"), ("arith", "Modular arithmetic"), ("binding", "Variable binding"), ("relational", "Relational lookup")]
short = lambda m: (m.replace("Llama-3.2-", "L").replace("Llama-3.1-", "L").replace("Qwen2.5-", "Q").replace("OLMo-2-1124-7B", "O7B")
                    .replace("Mistral-Small-24B-Base-2501", "M24B").replace("Mistral-7B-v0.1", "M7B").replace("-Instruct", "-I"))
post = df.model.str.contains("Instruct|SFT|DPO")

fig = plt.figure(figsize=(5.5, 4.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.28, left=0.115, right=0.985, top=0.89, bottom=0.14)
# ── row 1: share vs excess drop, per task ──
for j, (t, title) in enumerate(TASKS):
    ax = fig.add_subplot(gs[0, j]); d = df[(df.task == t) & np.isfinite(df.excess_win)]
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    for i, r in d.iterrows():
        c = FC[r.family]
        ax.scatter(r.share_win, r.excess_win, s=13, color=SURF if post[i] else c, edgecolor=c, lw=0.8, zorder=3)
        if r.model == "Qwen2.5-14B":
            ax.annotate("Q14B", (r.share_win, r.excess_win), (-10, 9), textcoords="offset points", ha="right", va="center", fontsize=5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2.5))
    rho = spearmanr(d.share_win, d.excess_win)[0]
    ax.set_title(title, pad=3)
    ax.text(0.97, 0.05, f"ρ = {rho:+.2f}\nn = {len(d)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK2)
    ax.set_xlim(0.19, 0.45); ax.set_ylim(-0.55, 0.85); ax.set_xticks([0.2, 0.3, 0.4])
    if j == 0: ax.set_ylabel("excess drop\n(top 10% minus random heads)")
    else: ax.set_yticklabels([])
    ax.set_xlabel("top-10% share", labelpad=1)
# ── row 2 left: top-q vs random drop per model (mean over tasks), ordered by share ──
ax = fig.add_subplot(gs[1, :3])
m = df.groupby("model").agg(share=("share_win", "mean"), top=("drop_win_topq_mean", "mean"),
                           rnd=("drop_win_randq", "mean"), fam=("family", "first")).sort_values("share")
x = np.arange(len(m)); ax.axhline(0, color=AXIS, lw=0.6)
for i, (name, r) in enumerate(m.iterrows()):
    c = FC[r.fam]
    ax.plot([i, i], [r.rnd, r.top], color=c, lw=1.0, alpha=0.6, zorder=2)
    ax.scatter(i, r.top, s=16, color=c, zorder=3); ax.scatter(i, r.rnd, s=16, color=SURF, edgecolor=c, lw=0.8, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([short(n) for n in m.index], rotation=60, ha="right", fontsize=5.6)
ax.set_xlim(-0.6, len(m) - 0.4)
ax.set_ylabel("drop in answer margin\n(fraction of clean, mean over tasks)")
ax.set_title("Models ordered by top-10% share (low to high)", pad=3)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=4, label="top 10% of heads"),
                   Line2D([], [], marker="o", ls="", color=INK2, mfc=SURF, ms=4, label="random heads")],
          loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8)
# ── row 2 right: within-task rho by depth band ──
ax = fig.add_subplot(gs[1, 3])
bands = ["b0", "b1", "b2", "b3", "b4"]; labs = ["0–20", "20–40", "40–60", "60–80", "80–100"]
per = np.array([[spearmanr(*df[(df.task == t) & np.isfinite(df[f"excess_{b}"])][[f"share_{b}", f"excess_{b}"]].to_numpy().T)[0]
                 for t, _ in TASKS] for b in bands])
ax.axhline(0, color=AXIS, lw=0.6)
for k in range(4):
    ax.scatter(np.arange(5) + (k - 1.5) * 0.08, per[:, k], s=5, color=MUTED, lw=0, zorder=2)
ax.plot(np.arange(5), per.mean(1), color=INK, lw=1.1, marker="o", ms=3, zorder=3)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=5.6)
ax.set_ylim(-0.8, 0.9); ax.set_xlabel("depth band (%)", labelpad=1)
ax.set_title("ρ by depth band", pad=3)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=4, label=f if f != "Qwen" else "Qwen2.5") for f, c in FC.items()]
           + [Line2D([], [], marker="o", ls="", color=INK2, mfc=SURF, ms=4, label="post-trained (row 1)")],
           loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.0)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"figE7_ablation.{ext}", dpi=300)
print("mean band rho:", dict(zip(labs, per.mean(1).round(2))))

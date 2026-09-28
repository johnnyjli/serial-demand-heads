"""Appendix figure: replication signatures (rows) against main-run signatures (columns), eight models.
Style (rcParams, palette, diverging map, lineage outlines, family separators) copied from model_figs.py (figM2).
Reads replication_numbers.json; writes figR_replication_signatures.pdf and .png next to this script."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 6})
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a92b2b"])

R = json.load(open(f"{OUT}/replication_numbers.json"))
M8 = R["models"]; X = np.array(R["vs_main_averaged"]["X"])
def short(m):
    return (m.replace("OLMo-2-1124-7B", "OLMo-2-7B").replace("-Base-2501", "").replace("-v0.1", "").replace("Mistral-Small-24B", "Mistral-24B")
             .replace("-Instruct", "-Inst"))
ab = lambda m: short(m).replace("OLMo-2-7B", "OLMo").replace("Llama-", "L").replace("Qwen2.5-", "Q").replace("Mistral-", "M")
labels = [ab(m) for m in M8]

fig, ax = plt.subplots(figsize=(3.0, 2.6))
fig.subplots_adjust(left=0.24, right=0.88, top=0.97, bottom=0.25)
im = ax.imshow(X, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="equal")
ax.grid(False)
ax.set_yticks(range(8)); ax.set_yticklabels(labels, fontsize=5.8)
ax.set_xticks(range(8)); ax.set_xticklabels(labels, fontsize=5.8, rotation=90)
for b in [1.5, 4.5, 6.5]:                                   # family boundaries: Llama | Qwen | OLMo | Mistral
    ax.axhline(b, color=SURF, lw=1.5); ax.axvline(b, color=SURF, lw=1.5)
for i, j in [(0, 1), (2, 3), (5, 6)]:                         # lineages: base model and its Instruct model
    ax.add_patch(plt.Rectangle((i - .5, i - .5), j - i + 1, j - i + 1, fill=False, ec=INK, lw=0.9, zorder=5))
for i in range(8):
    v = X[i, i]
    ax.text(i, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color=SURF if abs(v) > 0.55 else INK, zorder=6)
ax.set_ylabel("replication (fresh items)", labelpad=2)
ax.set_xlabel("main run", labelpad=2)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[-1, 0, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
cb.set_label("correlation of model signatures", fontsize=6, color=INK2)
fig.savefig(f"{OUT}/figR_replication_signatures.pdf")
fig.savefig(f"{OUT}/figR_replication_signatures.png", dpi=300)
plt.close(fig)
print("diagonal:", np.round(np.diag(X), 3).tolist(), " off-diagonal mean:", round(float(X[~np.eye(8, dtype=bool)].mean()), 3))

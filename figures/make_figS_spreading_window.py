"""Appendix figure: fraction of layers that spread activity as k grows, in a sliding window over depth
(after panel (b) of Figure 1 in the earlier serial_depth_layers draft).

For each model, task and layer l: z = OLS slope of the layer's top-10% head share on serial depth k divided by
the s.d. of its permutation null (the paper's primary statistic).  At each relative depth c, the curve gives the
fraction of layers with z < 0 among the layers of the four serial tasks whose relative depth l/(L-1) lies within
+-0.10 of c (layers pooled over tasks; windows are truncated at 0 and 1).

Data: 17-model main run: fig1_data/consist_q10.json (as Figure 1 and Figure 2); Llama-3.1-70B and Qwen2.5-72B:
figS_data/large_pair_layers.json (large_pair_layers.py; the captures and statistic of Appendix D).

    python3 make_figS_spreading_window.py [pairs|all]  -> figS_spreading_window.pdf (+ figS_data/*.png, values json)
"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/figS_data"
MODE = sys.argv[1] if len(sys.argv) > 1 else "pairs"
C = {r["model"]: r for r in json.load(open(f"{HERE}/fig1_data/consist_q10.json")) if r["stat"] == "q10"}
LP = json.load(open(f"{OUT}/large_pair_layers.json"))
TASKS = ["pointer", "arith", "binding", "relational"]; W = 0.10
cs = np.round(np.linspace(0, 1, 201), 4)

def zlayers(m, t):
    return np.array(LP[m][t]["z"]) if m in LP else np.array(C[m]["families"][t]["z"], float)
def curve(m):
    X = np.concatenate([np.arange(len(zlayers(m, t))) / (len(zlayers(m, t)) - 1) for t in TASKS])
    Z = np.concatenate([zlayers(m, t) for t in TASKS])
    return np.array([np.mean(Z[np.abs(X - c) <= W + 1e-9] < 0) for c in cs])

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.4, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2, "ytick.major.size": 2, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 6.5})

if MODE == "pairs":   # (model, label, colour, dash, marker)
    SHOW = [("Llama-3.1-70B", "Llama-3.1-70B", BLUE, "-", "o"), ("Llama-3.1-8B", "Llama-3.1-8B", "#6aa2e6", (0, (4, 1.6)), "s"),
            ("Qwen2.5-72B", "Qwen2.5-72B", ORANGE, "-", "D"), ("Qwen2.5-7B", "Qwen2.5-7B", "#f0915f", (0, (4, 1.6)), "^")]
else:
    LL = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.1-70B"]; QQ = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-72B"]
    bl = ["#b9d4f5", "#86b4ec", "#4f90de", "#1d5fb0"]; og = ["#f9d2bf", "#f5b08f", "#ef8b5d", "#e2632d", "#a8431a"]
    SHOW = [(m, m, c, "-", None) for m, c in zip(LL, bl)] + [(m, m, c, "-", None) for m, c in zip(QQ, og)]

fig, ax = plt.subplots(figsize=(5.5, 2.35))
fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.17)
ax.axvspan(50, 100, color="#eef3f9", zorder=0, lw=0)
ax.text(75, 1.035, "back half", ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
vals = {}
for m, lab, col, ls, mk in SHOW:
    f = curve(m); vals[m] = f.round(4).tolist()
    ax.plot(100 * cs, f, color=col, lw=1.4 if MODE == "pairs" else 1.1, ls=ls, zorder=3, label=lab,
            marker=mk, markevery=list(range(6, len(cs), 16)) if mk else None, ms=3.2, mew=0)
ax.set_xlim(0, 100); ax.set_ylim(0, 1.0); ax.set_xticks(range(0, 101, 20)); ax.set_yticks([0, .25, .5, .75, 1])
ax.set_xlabel("relative depth (%)", labelpad=1.5); ax.set_ylabel("fraction of layers spreading\n(window ±10% of depth)", linespacing=1.1)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], color=INK2, lw=0.8, ls=(0, (4, 2.5)))); l.append("half of the layers")
ax.legend(h, l, loc="lower left", ncol=3, bbox_to_anchor=(0.005, 0.0), handlelength=2.6, columnspacing=1.4)
suffix = "" if MODE == "pairs" else "_all"
fig.savefig(f"{HERE}/figS_spreading_window{suffix}.pdf"); fig.savefig(f"{OUT}/figS_spreading_window{suffix}.png", dpi=200)
back = cs >= 0.5
summ = {m: {"mean_fraction_back": float(np.mean(np.array(v)[back])), "share_of_back_positions_above_half": float(np.mean(np.array(v)[back] > 0.5)),
            "min_back": float(np.min(np.array(v)[back]))} for m, v in vals.items()}
json.dump({"window": W, "centers": cs.tolist(), "fraction_spreading": vals, "summary": summ}, open(f"{OUT}/figS_values{suffix}.json", "w"), indent=1)
for m, s in summ.items(): print(f"{m:16s} back: mean {s['mean_fraction_back']:.2f}, above half at {100*s['share_of_back_positions_above_half']:.0f}% of positions, min {s['min_back']:.2f}")

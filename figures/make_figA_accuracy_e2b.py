"""Accuracy by serial depth k in the replication on fresh items (E2b), for the eight rerun models.

Top row: the model answers directly, as in the main run.  Bottom row: the worked examples write out their
intermediate states and the model writes its own steps before answering.  Accuracy counts the answer the
model writes (the `correct` field of each item).  Colour: family; dashed: post-trained models; grey: chance
(one over the number of admissible answers).

    python3 make_figA_accuracy_e2b.py   -> figA_accuracy_e2b.pdf, fig_acc_data/accuracy_e2b.json (+ PNG preview)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, str(paths.REPLICATION)); import e2b
DATA = str(paths.E2B)
OUT = os.path.join(HERE, "fig_acc_data"); os.makedirs(OUT, exist_ok=True)

TASKS = [("pointer", "pointer chasing"), ("arith", "arithmetic (mod 7)"), ("binding", "variable binding"),
         ("relational", "relational lookup")]
MODES = [("direct", "direct answer"), ("stepwise", "steps written out")]
fam = lambda m: m.split("-")[0].replace("Qwen2.5", "Qwen")
post = lambda m: any(s in m for s in ("Instruct", "SFT", "DPO"))

runs = e2b.load_results(DATA)
acc, chance = {}, {}
for r in runs:
    m = r["model"].split("/")[-1]; ks = sorted({x["k"] for x in r["items"]})
    acc[(m, r["mode"], r["family"])] = (ks, [float(np.mean([x["correct"] for x in r["items"] if x["k"] == k])) for k in ks])
    chance[r["family"]] = float(np.mean([x["chance"] for x in r["items"]]))
MODELS = sorted({k[0] for k in acc}, key=lambda m: (["Llama", "Qwen", "OLMo", "Mistral"].index(fam(m)), m))

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FC = {"Llama": "#2a78d6", "Qwen": "#eb6834", "OLMo": "#1baf7a", "Mistral": "#eda100"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 6})

fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.75), sharey=True)
fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.2, wspace=0.12, hspace=0.42)
for i, (mode, mlab) in enumerate(MODES):
    for j, (t, tlab) in enumerate(TASKS):
        ax = axes[i, j]
        for m in MODELS:
            ks, a = acc[(m, mode, t)]
            ax.plot(ks, a, color=FC[fam(m)], lw=1.0, ls=(0, (3, 1.5)) if post(m) else "-", zorder=3)
        ax.axhline(chance[t], color=MUTED, lw=0.8, zorder=2)
        ax.set_ylim(0, 1.03); ax.set_xticks(ks)
        if i == 0: ax.set_title(tlab, pad=3)
        if i == 1: ax.set_xlabel("k", labelpad=1)
        if j == 0: ax.set_ylabel(f"accuracy\n({mlab})", linespacing=1.0)
handles = [Line2D([], [], color=c, lw=1.2, label=f) for f, c in FC.items()] + \
          [Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="post-trained"),
           Line2D([], [], color=MUTED, lw=0.8, label="chance")]
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.54, 0.0), handlelength=2.2, columnspacing=1.4)
fig.savefig(os.path.join(HERE, "figA_accuracy_e2b.pdf")); fig.savefig(os.path.join(OUT, "figA_accuracy_e2b.png"), dpi=200)

summary = {"models": MODELS, "chance": chance,
           "accuracy": {f"{m}|{mode}|{t}": {"k": acc[(m, mode, t)][0], "acc": acc[(m, mode, t)][1]}
                        for m in MODELS for mode, _ in MODES for t, _ in TASKS}}
for mode, _ in MODES:
    ge3 = np.mean([np.mean([a for k, a in zip(*acc[(m, mode, t)]) if k >= 3]) for m in MODELS for t, _ in TASKS])
    k1 = np.mean([acc[(m, mode, t)][1][0] for m in MODELS for t, _ in TASKS])
    summary[f"mean_{mode}"] = {"k1": float(k1), "k_ge3": float(ge3)}
    print(mode, "mean k=1 %.3f  k>=3 %.3f" % (k1, ge3))
json.dump(summary, open(os.path.join(OUT, "accuracy_e2b.json"), "w"), indent=1)

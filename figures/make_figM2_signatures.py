"""Figure 2: how models differ (left) and which models resemble each other (right).

Left:  each model's task-averaged depth profile (mean z of the top-10% head share over the four tasks, by
       relative-depth decile; dots: deciles where all four tasks have the same sign), followed by a four-cell
       strip, one cell per task (P pointer, A arithmetic, B binding, R relational), coloured by the model's
       back-half response minus the mean over the 17 models in that task (blue: spreads more than the average
       model; red: less).  A marker right of the strip flags models whose deviation has the same sign in all
       four tasks.
Right: correlation between model signatures (unchanged from the earlier version of this figure).

Data: fig2_data/model_diff_primary.json (decile profiles Z[model][task] and signature similarity R, copied
from the analysis pipeline's model_diff.json; Z equals the q10 profiles of consist.json).

    python3 make_figM2_signatures.py        -> figM2_signatures.pdf (+ fig2_data/figM2_signatures.png)
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
MD = json.load(open(f"{HERE}/fig2_data/model_diff_primary.json"))
T = ["pointer", "arith", "binding", "relational"]; TL = ["P", "A", "B", "R"]
ORDER = list(MD["primary"]["Z"])
Z = np.array([[MD["primary"]["Z"][m][t] for t in T] for m in ORDER])          # [17, 4, 10]
R = np.array(MD["similarity"]["R"])

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a92b2b"])

def short(m):
    return (m.replace("OLMo-2-1124-7B", "OLMo-2-7B").replace("-Base-2501", "").replace("-v0.1", "")
             .replace("Mistral-Small-24B", "Mistral-24B").replace("-Instruct", "-Inst"))
ab = lambda m: short(m).replace("OLMo-2-7B", "OLMo").replace("Llama-", "L").replace("Qwen2.5-", "Q").replace("Mistral-", "M")
GROUPS = [4.5, 10.5, 14.5]                                               # family boundaries in ORDER

# back-half response per task and its deviation from the all-model mean of that task
B = Z[:, :, 5:].mean(-1)                                                  # [17, 4]
D = B - B.mean(0, keepdims=True)
same = (np.sign(D) == np.sign(D[:, :1])).all(1)                           # same sign in all four tasks

W, H = 5.5, 2.75
fig = plt.figure(figsize=(W, H))
fx = lambda x: x / W; fy = lambda y: y / H
BOT, TOP = 0.56, 0.30
hA = H - BOT - TOP
a1 = fig.add_axes([fx(0.96), fy(BOT), fx(1.62), fy(hA)])                  # task-averaged profile
a1s = fig.add_axes([fx(2.64), fy(BOT), fx(0.40), fy(hA)])                 # per-task back-half deviation strip
a2 = fig.add_axes([fx(3.62), fy(BOT), fx(1.40), fy(hA)])                  # signature similarity

# left: task-averaged profile
mz = Z.mean(1); agree = (np.sign(Z) == np.sign(mz)[:, None]).all(1)
im = a1.imshow(mz, cmap=cmap, norm=TwoSlopeNorm(0, -5, 5), aspect="auto")
yy, xx = np.nonzero(agree); a1.scatter(xx, yy, s=3, color=INK, lw=0)
a1.set_yticks([]); a1.set_xticks([-0.5, 4.5, 9.5])
a1.set_xticklabels(["0", "50", "100"]); a1.set_xlabel("relative depth (%)", labelpad=1.5)
for i, m in enumerate(ORDER):                                             # bold: same sign in all four tasks
    a1.text(-0.75, i, short(m), ha="right", va="center", fontsize=6.5, clip_on=False,
            color=INK if same[i] else INK2, fontweight="bold" if same[i] else "normal",
            fontfamily="Arial")                                            # Arial has a true bold face
a1.set_title("mean over tasks", fontsize=6.3, color=INK2, pad=2.5)
for b in GROUPS: a1.axhline(b, color=SURF, lw=1.5)
for sp in a1.spines.values(): sp.set_visible(False)

# strip: back-half deviation from the average model, per task
DMAX = 3
a1s.imshow(D, cmap=cmap, norm=TwoSlopeNorm(0, -DMAX, DMAX), aspect="auto")
a1s.set_xticks(range(4)); a1s.set_xticklabels(TL, fontsize=6); a1s.tick_params(axis="x", length=0, pad=1.5)
a1s.set_yticks([])
for x in (0.5, 1.5, 2.5): a1s.axvline(x, color=SURF, lw=0.6)
for b in GROUPS: a1s.axhline(b, color=SURF, lw=1.5)
for sp in a1s.spines.values(): sp.set_visible(False)
a1s.set_title("back half vs\naverage model", fontsize=6.3, color=INK2, pad=2.5, linespacing=0.95)
for i in np.nonzero(same)[0]:                                             # flag rows with the same sign in all tasks
    a1s.plot([3.85], [i], marker="<", ms=3.2, color=INK, mew=0, clip_on=False, ls="")

# colour keys under the left panels
k1 = fig.add_axes([fx(0.96), fy(0.20), fx(1.44), fy(0.055)])
k1.imshow(np.linspace(-5, 5, 256)[None], cmap=cmap, norm=TwoSlopeNorm(0, -5, 5), aspect="auto", extent=(-5, 5, 0, 1))
k1.set_yticks([]); k1.set_xticks([-5, 0, 5]); k1.tick_params(axis="x", labelsize=5.5, length=1.5, pad=1)
for sp in k1.spines.values(): sp.set_visible(False)
fig.text(fx(0.96 + 0.72), fy(0.02), "mean z (blue: spreads; red: concentrates)", fontsize=5.8, color=INK2, ha="center")
k2 = fig.add_axes([fx(2.64), fy(0.20), fx(0.40), fy(0.055)])
k2.imshow(np.linspace(-DMAX, DMAX, 256)[None], cmap=cmap, norm=TwoSlopeNorm(0, -DMAX, DMAX), aspect="auto",
          extent=(-DMAX, DMAX, 0, 1))
k2.set_yticks([]); k2.set_xticks([-3, 3]); k2.tick_params(axis="x", labelsize=5.5, length=1.5, pad=1)
for sp in k2.spines.values(): sp.set_visible(False)
fig.text(fx(2.64 + 0.20), fy(0.02), "difference", fontsize=5.8, color=INK2, ha="center")

# right: similarity of model signatures (as before)
im2 = a2.imshow(R, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="equal")
a2.set_yticks(range(17)); a2.set_yticklabels([ab(m) for m in ORDER], fontsize=5.8); a2.set_xticks([])
LIN = {}
for m in ORDER:
    base = m
    for s_ in ("-Instruct", "-SFT", "-DPO"): base = base.replace(s_, "")
    LIN[m] = base
for b in GROUPS: a2.axhline(b, color=SURF, lw=1.5); a2.axvline(b, color=SURF, lw=1.5)
i = 0
while i < 17:
    j = i
    while j + 1 < 17 and LIN[ORDER[j + 1]] == LIN[ORDER[i]]: j += 1
    if j > i: a2.add_patch(plt.Rectangle((i - .5, i - .5), j - i + 1, j - i + 1, fill=False, ec=INK, lw=0.9, zorder=5))
    i = j + 1
for f, (lo, hi) in {"Llama": (0, 4), "Qwen": (5, 10), "OLMo": (11, 14), "Mistral": (15, 16)}.items():
    a2.text((lo + hi) / 2, 17.1, f, ha="center", va="top", fontsize=5.8, color=INK2)
for sp in a2.spines.values(): sp.set_visible(False)
a2.set_title("similarity of signatures", fontsize=6.3, color=INK2, pad=2.5)
fig.canvas.draw(); bb = a2.get_position()                                  # colour bar as tall as the matrix
cax = fig.add_axes([bb.x1 + fx(0.08), bb.y0, fx(0.07), bb.height])
cb = fig.colorbar(im2, cax=cax, ticks=[-1, 0, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
cb.set_label("correlation", fontsize=6, color=INK2, labelpad=1)

fig.savefig(f"{HERE}/figM2_signatures.pdf"); fig.savefig(f"{HERE}/fig2_data/figM2_signatures.png", dpi=200)
json.dump({"models": ORDER, "tasks": T, "back_half_z": B.round(4).tolist(), "deviation_from_task_mean": D.round(4).tolist(),
           "same_sign_all_tasks": [ORDER[i] for i in np.nonzero(same)[0]]},
          open(f"{HERE}/fig2_data/strip_values.json", "w"), indent=1)
print("same sign in all four tasks:", [ORDER[i] for i in np.nonzero(same)[0]])

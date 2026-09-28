"""Paper figures for model differences (top-10% head share).  python3 model_figs.py <scratchpad> <out_dir>
figM1_model_profiles.pdf  per-model depth profiles, one line per task + model mean + all-model mean
figM2_signatures.pdf      model x depth map of the task-averaged profile; similarity of model signatures
figM3_axes.pdf            PCA of signatures; back-half response vs accuracy at k=1"""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
SP, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FC = {"Llama": "#2a78d6", "Qwen": "#eb6834", "OLMo": "#1baf7a", "Mistral": "#eda100"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "legend.fontsize": 6})
MD = json.load(open(f"{SP}/model_diff.json")); M2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
P = MD["primary"]; T = ["pointer", "arith", "binding", "relational"]
ORDER = list(P["Z"]); fam = lambda m: m.split("-")[0].replace("Qwen2.5", "Qwen")
POST = lambda m: any(s in m for s in ("Instruct", "SFT", "DPO"))
def short(m):
    return (m.replace("OLMo-2-1124-7B", "OLMo-2-7B").replace("-Base-2501", "").replace("-v0.1", "").replace("Mistral-Small-24B", "Mistral-24B")
             .replace("-Instruct", "-Inst"))
Z = np.array([[P["Z"][m][t] for t in T] for m in ORDER]); mu = np.array(P["mu"]); X = np.arange(10) * 10 + 5
# ── M1 ──────────────────────────────────────────────────────────────────────
rows = [[m for m in ORDER if fam(m) == f] for f in ["Llama", "Qwen", "OLMo", "Mistral"]]
fig, axes = plt.subplots(4, 6, figsize=(5.5, 4.3), sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.12, wspace=0.12, hspace=0.42)
for ax in axes.flat: ax.set_visible(False)
for i, row in enumerate(rows):
    for j, m in enumerate(row):
        ax = axes[i, j]; ax.set_visible(True); k = ORDER.index(m)
        ax.axhline(0, color=AXIS, lw=0.6); ax.axvline(50, color=GRID, lw=0.6)
        for t in range(4): ax.plot(X, Z[k, t], color="#b9b8b2", lw=0.6, zorder=2)
        ax.plot(X, mu, color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=3)
        ax.plot(X, Z[k].mean(0), color=FC[fam(m)], lw=1.5, zorder=4)
        ax.set_title(short(m), pad=2, fontsize=6.5)
        ax.text(0.03, 0.04, f"r={P['within_model_cross_task_r'][m]:.2f}", transform=ax.transAxes, fontsize=6, color=INK2)
        ax.set_ylim(-9.5, 9); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_yticks([-5, 0, 5])
        ax.tick_params(labelbottom=True); tl = ax.set_xticklabels(["0", "50", "100"]); tl[0].set_ha("left"); tl[-1].set_ha("right")
for i in range(4): axes[i, 0].set_ylabel("z", labelpad=1)
fig.text(0.53, 0.035, "relative depth (%)", ha="center", color=INK2)
fig.legend(handles=[Line2D([], [], color=INK, lw=1.5, label="model mean over the four tasks (colour = family)"),
                    Line2D([], [], color="#b9b8b2", lw=0.8, label="each task"),
                    Line2D([], [], color=INK2, lw=0.8, ls=(0, (2, 1.5)), label="mean of all 17 models")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.53, -0.02), handlelength=2.2, columnspacing=1.5)
fig.savefig(f"{OUT}/figM1_model_profiles.pdf"); plt.close(fig)
# ── M2 ──────────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a92b2b"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.75), gridspec_kw={"width_ratios": [1.0, 1.0]})
fig.subplots_adjust(left=0.19, right=0.92, top=0.95, bottom=0.17, wspace=0.62)
mz = Z.mean(1); agree = (np.sign(Z) == np.sign(mz)[:, None]).all(1)
im = a1.imshow(mz, cmap=cmap, norm=TwoSlopeNorm(0, -5, 5), aspect="auto")
yy, xx = np.nonzero(agree); a1.scatter(xx, yy, s=3, color=INK, lw=0)
a1.set_yticks(range(17)); a1.set_yticklabels([short(m) for m in ORDER]); a1.set_xticks([-0.5, 4.5, 9.5]); a1.set_xticklabels(["0", "50", "100"])
a1.grid(False); a1.set_xlabel("relative depth (%)")
for b in [4.5, 10.5, 14.5]: a1.axhline(b, color=SURF, lw=1.5)
cb = fig.colorbar(im, ax=a1, fraction=0.05, pad=0.02, ticks=[-5, 0, 5]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
cb.set_label("mean z (blue: spreads, red: concentrates)", fontsize=6, color=INK2)
R = np.array(MD["similarity"]["R"])
im2 = a2.imshow(R, cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect="equal")
ab = lambda m: short(m).replace("OLMo-2-7B", "OLMo").replace("Llama-", "L").replace("Qwen2.5-", "Q").replace("Mistral-", "M")
a2.set_yticks(range(17)); a2.set_yticklabels([ab(m) for m in ORDER], fontsize=5.8); a2.set_xticks([]); a2.grid(False)
LIN = {}; 
for m in ORDER:
    base = m
    for s_ in ("-Instruct", "-SFT", "-DPO"): base = base.replace(s_, "")
    LIN[m] = base
for b in [4.5, 10.5, 14.5]: a2.axhline(b, color=SURF, lw=1.5); a2.axvline(b, color=SURF, lw=1.5)
blocks = []; i = 0
while i < 17:
    j = i
    while j + 1 < 17 and LIN[ORDER[j + 1]] == LIN[ORDER[i]]: j += 1
    if j > i: a2.add_patch(plt.Rectangle((i - .5, i - .5), j - i + 1, j - i + 1, fill=False, ec=INK, lw=0.9, zorder=5))
    i = j + 1
for f, (lo, hi) in {"Llama": (0, 4), "Qwen": (5, 10), "OLMo": (11, 14), "Mistral": (15, 16)}.items():
    a2.text((lo + hi) / 2, 17.1, f, ha="center", va="top", fontsize=5.8, color=INK2)
cb2 = fig.colorbar(im2, ax=a2, fraction=0.05, pad=0.02, ticks=[-1, 0, 1]); cb2.outline.set_visible(False); cb2.ax.tick_params(labelsize=5.5)
cb2.set_label("correlation of model signatures", fontsize=6, color=INK2)
fig.savefig(f"{OUT}/figM2_signatures.pdf"); plt.close(fig)
# ── M3 ──────────────────────────────────────────────────────────────────────
S = MD["pca"]["scores"]; ev = MD["pca"]["explained"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(5.5, 2.7))
fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.17, wspace=0.28)
back = {m: MD["per_model"][m]["back"] for m in ORDER}; acc1 = M2["acc_k1"]
OFF1 = {"Qwen2.5-7B-Instruct": (4, -6), "OLMo-2-1124-7B": (-22, -3), "OLMo-2-1124-7B-DPO": (3, 3), "OLMo-2-1124-7B-Instruct": (3, -7), "OLMo-2-1124-7B-SFT": (-30, 5),
        "Llama-3.2-3B-Instruct": (-40, -7), "Llama-3.2-3B": (4, 2), "Qwen2.5-3B": (3, 3), "Qwen2.5-1.5B": (3, -6), "Qwen2.5-7B": (3, 2),
        "Mistral-Small-24B-Base-2501": (4, -3)}
OFF2 = {"OLMo-2-1124-7B": (-22, -8), "OLMo-2-1124-7B-SFT": (-10, -11), "OLMo-2-1124-7B-DPO": (5, -4), "OLMo-2-1124-7B-Instruct": (-12, -11),
        "Llama-3.2-3B-Instruct": (5, -3), "Mistral-7B-v0.1": (-14, 5), "Qwen2.5-7B-Instruct": (4, 4), "Qwen2.5-14B": (-4, -10),
        "Llama-3.1-8B-Instruct": (-52, -3), "Mistral-Small-24B-Base-2501": (-10, 5), "Qwen2.5-1.5B": (-22, -6), "Qwen2.5-3B": (4, 3), "Qwen2.5-7B": (4, -3)}
for ax, get, OFF in [(b1, lambda m: (S[m][0], S[m][1]), OFF1), (b2, lambda m: (acc1[m], back[m]), OFF2)]:
    for m in ORDER:
        if POST(m):
            base = LIN[m]; x0, y0 = get(base); x1, y1 = get(m)
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=FC[fam(m)], lw=0.6, shrinkA=3, shrinkB=3, mutation_scale=5, alpha=0.8))
    for m in ORDER:
        x, y = get(m); c = FC[fam(m)]
        ax.scatter(x, y, s=16, facecolor=SURF if POST(m) else c, edgecolor=c, lw=1.0, zorder=4)
        ax.annotate(short(m).replace("OLMo-2-7B", "OLMo").replace("Llama-", "L").replace("Qwen2.5-", "Q").replace("Mistral-", "M"),
                    (x, y), xytext=OFF.get(m, (3, 2)), textcoords="offset points", fontsize=6, color=INK2, zorder=5)
b1.axhline(0, color=AXIS, lw=0.6); b1.axvline(0, color=AXIS, lw=0.6)
b1.set_xlabel(f"PC1 ({100*ev[0]:.0f}%): stronger spreading at 50–70% depth (higher = more)"); b1.set_ylabel(f"PC2 ({100*ev[1]:.0f}%): spreading later, at 80–100%")
b2.set_xlim(0.3, 1.0); b2.axhline(0, color=AXIS, lw=0.6); b2.set_xlabel("accuracy at k = 1 (mean of the four tasks)"); b2.set_ylabel("back-half mean z (negative: spreads)")
b2.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=c, label=f) for f, c in FC.items()] +
                   [Line2D([], [], ls="", marker="o", ms=4, mfc=SURF, mec=INK2, label="post-trained")], loc="lower left", ncol=1, fontsize=6, handletextpad=0.2)
fig.savefig(f"{OUT}/figM3_axes.pdf"); plt.close(fig)
# ── accuracy by k (all items; Qwen2.5-7B-Instruct under SDPA, as in the accuracy table) ──
import os
os.environ["VPRIM"] = "y10"; sys.path.insert(0, f"{SP}/vet3")
from v2lib import load as vload
FT = ["pointer", "arith", "binding", "relational", "fixed_digit"]
FN = {"pointer": "pointer chasing", "arith": "arithmetic (mod 7)", "binding": "variable binding", "relational": "relational lookup", "fixed_digit": "numeral control"}
CH = {"pointer": .1, "arith": 1 / 7, "binding": 1 / 3, "relational": 1 / 3, "fixed_digit": 1 / 7}
fig, axs = plt.subplots(1, 5, figsize=(5.5, 1.9), sharey=True)
fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.36, wspace=0.12)
for ax, t in zip(axs, FT):
    for m in ORDER:
        acc = {}
        for w in ("w0", "w1"):
            d = vload(m, t, w, "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", False)
            for k in np.unique(d["k"]): acc.setdefault(int(k), []).append(d["correct"][d["k"] == k].mean())
        ks = sorted(acc); ax.plot(ks, [np.mean(acc[k]) for k in ks], color=FC[fam(m)], lw=0.9, ls=(0, (2.5, 1.2)) if POST(m) else "-", alpha=0.9)
    ax.axhline(CH[t], color=MUTED, lw=0.8); ax.set_title(FN[t], fontsize=6.5, pad=2); ax.set_xticks(ks); ax.set_ylim(0, 1.02); ax.set_xlabel("k", labelpad=1)
axs[0].set_ylabel("strict accuracy")
fig.legend(handles=[Line2D([], [], color=c, lw=1.2, label=f) for f, c in FC.items()] + [Line2D([], [], color=INK2, lw=1, ls=(0, (2.5, 1.2)), label="post-trained"),
           Line2D([], [], color=MUTED, lw=0.8, label="chance")], loc="lower center", ncol=6, bbox_to_anchor=(0.53, -0.01), fontsize=6, handlelength=2)
fig.savefig(f"{OUT}/figA_accuracy_by_k.pdf"); plt.close(fig)

# ── post-trained vs base: back-half mean z in the serial tasks and in the numeral control ──
PT = json.load(open(f"{SP}/trends/posttrain_trend.json"))
PM = PT["s1_model_level"]["per_model"]; NC = PT["s5_confounds"]["numeral_control_model_level"]["per_model"]
ser = {m: float(np.mean([PM[m][t]["back_z"] for t in T])) for m in PM}
ctl = {m: NC[m]["back"] for m in NC}
LINS = [("Llama-3.2-3B", ["Llama-3.2-3B-Instruct"]), ("Llama-3.1-8B", ["Llama-3.1-8B-Instruct"]),
        ("Qwen2.5-7B", ["Qwen2.5-7B-Instruct"]), ("OLMo-2-1124-7B", ["OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"])]
LL = {"Llama-3.2-3B": "L3.2-3B", "Llama-3.1-8B": "L3.1-8B", "Qwen2.5-7B": "Q2.5-7B", "OLMo-2-1124-7B": "OLMo-2-7B"}
SUF = {"Llama-3.2-3B-Instruct": "Inst", "Llama-3.1-8B-Instruct": "Inst", "Qwen2.5-7B-Instruct": "Inst",
       "OLMo-2-1124-7B-SFT": "SFT", "OLMo-2-1124-7B-DPO": "DPO", "OLMo-2-1124-7B-Instruct": "Inst"}
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.2, wspace=0.08)
for ax, val, ttl in [(axs[0], ser, "serial tasks (mean of four)"), (axs[1], ctl, "numeral control")]:
    ax.axhline(0, color=AXIS, lw=0.7)
    for i, (b, posts) in enumerate(LINS):
        c = FC[fam(b)]; ax.scatter(i, val[b], s=22, color=c, zorder=4)
        for j, q in enumerate(posts):
            x = i + 0.2 + 0.2 * j if len(posts) > 1 else i + 0.3
            ax.annotate("", (x, val[q]), (i, val[b]), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6, shrinkA=3, shrinkB=3,
                        ls=(0, (2, 1.5)) if b.startswith("Qwen") else "-"))
            ax.scatter(x, val[q], s=22, facecolor=SURF, edgecolor=c, lw=1.0, zorder=5)
    ax.set_xticks(range(len(LINS))); ax.set_xticklabels([LL[b] for b, _ in LINS]); ax.set_xlim(-0.4, len(LINS) - 0.15)
    ax.set_title(ttl, fontsize=7, pad=3); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("back-half mean z\n(negative: spreads)")
fig.savefig(f"{OUT}/figP_posttraining.pdf"); plt.close(fig)
print("ok")

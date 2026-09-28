"""E2 figures 16-17: behaviour that is consistent across serial tasks.  python3 plots4.py <scratchpad> <out_dir>"""
import sys, os, json, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
SP, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, SP)
from e2lib import ORDER
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.5})
T = ["pointer", "arith", "binding", "relational"]
TN = {"pointer": "pointer chase", "arith": "arithmetic (mod 7)", "binding": "variable binding", "relational": "relational lookup"}
TC = {"pointer": BLUE, "arith": ORANGE, "binding": AQUA, "relational": YELLOW}
def label(m): return m.replace("OLMo-2-1124-7B", "OLMo-2-7B").replace("-Base-2501", "").replace("-v0.1", "")
POST = {m for m in ORDER if any(t in m for t in ("Instruct", "SFT", "DPO"))}
STAT = os.environ.get("E2_STAT", "q25"); QL = {"q10": "top-10%", "q25": "top-25%", "q50": "top-50%"}[STAT]
NOTE = (f"Statistic: {QL} head share (summed share of the strongest {QL[4:]} of heads in a layer); each item averaged over its two "
        "wordings. Per-layer slope on k tested against a permutation null (k shuffled across items, or within prompt body for binding and "
        "relational). Eager attention, bf16 (Qwen2.5-7B-Instruct: SDPA, see reliability notes). OLMo-2 massive-output items excluded.")
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == STAT}
CA = {(r["model"], r["stat"]): r for r in json.load(open(f"{SP}/consist.json"))}
BH = json.load(open(f"{SP}/backhalf.json" if STAT == "q25" else f"{SP}/backhalf_{STAT}.json"))
figs = []
PAPER = bool(os.environ.get("PAPER"))
def finish(fig, name, title, sub):
    if PAPER:                                  # paper version: no embedded title, caption carries the text
        fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); figs.append(fig); return
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(NOTE, int(fig.get_figwidth() * 21), break_on_hyphens=False)),
             ha="left", va="bottom", fontsize=6.2, color=MUTED, linespacing=1.3)
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=10, color=INK, fontweight="bold")
    fig.text(0.01, 0.962, "\n".join(textwrap.wrap(sub, int(fig.get_figwidth() * 19), break_on_hyphens=False)),
             ha="left", va="top", fontsize=7.5, color=INK2, linespacing=1.4)
    fig.savefig(f"{OUT}/{name}.png", dpi=200); figs.append(fig)
def sigmap(r, f):
    return np.sign(r["families"][f]["rel"]) * (np.array(r["families"][f]["p"]) < .05)

# ── 16: the shared depth pattern ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1.15, 1.35, 1.1]})
fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.27, wspace=0.42)
ax = axes[0]; ax.axhline(0, color=AXIS, lw=0.9); ax.axvspan(50, 100, color=GRID, alpha=0.35, lw=0)
x = np.arange(10) * 10 + 5; allD = []
for f in T:
    D = []
    for m in ORDER:
        r = C[m]; L = r["L"]; xx = np.arange(L) / (L - 1); dec = np.minimum((xx * 10).astype(int), 9); M = sigmap(r, f)
        D.append([(M[dec == d] < 0).mean() - (M[dec == d] > 0).mean() for d in range(10)])
    D = np.array(D); allD.append(D)
    ax.plot(x, D.mean(0), color=TC[f], marker="o", ms=3.2, label=TN[f], zorder=3)
ax.plot(x, np.mean(allD, (0, 1)), color=INK, lw=2.2, label="mean of the four", zorder=4)
ax.set_xlabel("relative depth (%)"); ax.set_ylabel("share of layers spreading − concentrating")
ax.set_title("depth profile per task\n(mean over 17 models)", loc="left"); ax.legend(loc="upper left", fontsize=6.5)
ax.text(52, ax.get_ylim()[0] + 0.02, "back half", color=MUTED, fontsize=6.5)
# (b) front vs back per model and task
ax = axes[1]; ax.axvline(0, color=AXIS, lw=0.9)
offs = {"pointer": -0.27, "arith": -0.09, "binding": 0.09, "relational": 0.27}
for i, m in enumerate(ORDER):
    r = C[m]; L = r["L"]; xx = np.arange(L) / (L - 1)
    for f in T:
        z = np.array(r["families"][f]["z"]); fr, bk = z[xx < .5].mean(), z[xx >= .5].mean(); y = i + offs[f]
        ax.plot([fr, bk], [y, y], color=TC[f], lw=0.9, alpha=0.6, zorder=2)
        ax.scatter([fr], [y], s=12, facecolors=SURF, edgecolors=TC[f], linewidths=0.9, zorder=3)
        ax.scatter([bk], [y], s=13, color=TC[f], zorder=4, linewidths=0)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([label(m) for m in ORDER], fontsize=6.6); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
for b in (4.5, 10.5, 14.5): ax.axhline(b, color=GRID, lw=0.8)
ax.set_xlabel("mean per-layer z (< 0 spreads, > 0 concentrates)")
ax.set_title("front half (open) vs back half (filled), each task", loc="left")
ax.legend(handles=[Line2D([], [], color=TC[f], marker="o", lw=0, label=TN[f]) for f in T], loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.4, ncol=2)
# (c) robustness
def pct(rs, key):
    return 100 * np.mean([(r["back"] < 0) if key == "back" else (r["back"] < r["front"]) for r in rs])
prim = [r for r in BH if r["variant"] == "all" and ((r["attn"] == "eager" and r["model"] != "Qwen2.5-7B-Instruct") or (r["attn"] == "sdpa" and r["model"] == "Qwen2.5-7B-Instruct"))]
rows = [("all 136 cells", prim)] + [(f"  {TN[f]}", [r for r in prim if r["cell"].startswith(f)]) for f in T] + \
       [("  wording w0", [r for r in prim if r["cell"].endswith("w0")]), ("  wording w1", [r for r in prim if r["cell"].endswith("w1")]),
        ("SDPA attention (115 cells)", [r for r in BH if r["attn"] == "sdpa" and r["variant"] == "all"]),
        ("sink heads removed", [r for r in BH if r["variant"] == "nosink"])]
ax = axes[2]; y = np.arange(len(rows))
a = [pct(rs, "back") for _, rs in rows]; b = [pct(rs, "contrast") for _, rs in rows]
ax.barh(y - 0.19, a, height=0.36, color=BLUE, label="back half spreads")
ax.barh(y + 0.19, b, height=0.36, color="#86b6ef", label="back spreads more than front")
ax.axvline(50, color=INK2, lw=0.9); ax.text(51, -0.75, "50%", color=INK2, fontsize=6.5)
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("% of model × task × wording cells"); ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.4, ncol=1); ax.set_title("robustness", loc="left")
for yy, v in zip(y, a): ax.text(v + 1, yy - 0.19, f"{v:.0f}", va="center", fontsize=6, color=INK2)
finish(fig, "fig16_consistent_depth_pattern", "16 · Consistent across serial tasks: the back half of the network spreads activity as depth grows",
       f"Left: for each task, the share of layers whose {QL} head share falls significantly with k minus the share where it rises, by depth "
       "decile, averaged over 17 models. Middle: mean standardised per-layer slope in the front half (0–50% depth) and the back half "
       "(50–100%) for every model and task. Right: how often the back half spreads, and spreads more than the front, per model × task × "
       "wording cell, overall and under each check.")

# ── 17: layer x task maps for every model ───────────────────────────────────
if PAPER:                                      # print width (5.5 in), fonts >= 5.5 pt
    fig, axes = plt.subplots(6, 3, figsize=(5.5, 8.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.975, bottom=0.03, hspace=0.75, wspace=0.42)
    FS = dict(row=5.5, tick=5.5, title=6.2, tri=7, leg=6)
else:
    fig, axes = plt.subplots(6, 3, figsize=(12, 10.5))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.075, hspace=0.95, wspace=0.18)
    FS = dict(row=5.8, tick=6, title=7.4, tri=14, leg=7)
cm = ListedColormap([BLUE, "#ecebe6", RED]); nm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
for i, m in enumerate(ORDER):
    ax = axes.flat[i]; r = C[m]; L = r["L"]
    M = np.stack([sigmap(r, f) for f in T]); cons = np.array(r["consistent"])
    ax.imshow(M, cmap=cm, norm=nm, aspect="auto", extent=(-0.5, L - 0.5, 3.5, -0.5), interpolation="nearest")
    for j in np.nonzero(cons)[0]:
        ax.scatter([j], [4.1], marker="^", s=FS["tri"], color=BLUE if cons[j] < 0 else RED, clip_on=False, zorder=5)
    ax.axvline((L - 1) * 0.5, color=INK, lw=0.9)
    ax.set_ylim(4.6, -0.5); ax.set_yticks(range(4)); ax.set_yticklabels(["pointer", "arith", "binding", "relational"], fontsize=FS["row"])
    ax.set_xticks([0, (L - 1) // 2, L - 1]); ax.set_xticklabels(["0", f"{(L - 1) // 2}", f"{L - 1}"], fontsize=FS["tick"])
    ax.grid(False); ax.set_title(f"{label(m)}  ({L} layers)", loc="left", fontsize=FS["title"], color=ORANGE if m in POST else INK)
    for sp in ax.spines.values(): sp.set_visible(False)
axes.flat[-1].axis("off")
axes.flat[-1].legend(handles=[Patch(color=BLUE, label=f"spreads ({QL} share falls with k, p < .05)"),
                              Patch(color=RED, label="concentrates (rises, p < .05)"), Patch(color="#ecebe6", label="no significant change"),
                              Line2D([], [], marker="^", lw=0, color=INK2, label="significant in the same direction in ≥ 3 tasks, none opposite"),
                              Line2D([], [], color=INK, lw=0.9, label="half depth")], loc="center", fontsize=FS["leg"])
finish(fig, "fig17_layer_maps_by_task", "17 · Layer by layer, every model and task",
       "Each panel is one model (orange title = post-trained); rows are the four serial tasks, columns the layers. Blue layers spread, red "
       "layers concentrate. Triangles mark layers with the same significant direction in at least three tasks and none opposite. Spreading "
       "is common in every task and gathers past half depth; which exact layers agree across tasks is only modestly above chance "
       "(64 consistently spreading layers against 48 expected if each task's pattern sat at a random depth).")
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(f"{OUT}/E2_consistency_figures.pdf") as pdf:
    for f in figs: pdf.savefig(f)
print("wrote", len(figs))

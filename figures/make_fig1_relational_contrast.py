"""Figure 1: relational lookup, top-10% head share -- Llama and OLMo-2 spread in the back half, Qwen2.5 (<= 7B) barely does.

(a) Per-layer heat strips, one row per model: z = OLS slope of the layer's top-10% head share on serial depth k,
    divided by the s.d. of its permutation null (consist.json, stat q10; items averaged over wordings w0/w1; k shuffled
    within prompt bodies).  Blue: the layer spreads activity as k grows; red: it concentrates.  Layer l fills
    [l/L, (l+1)/L] of relative depth, so for the even L of every model shown the 50% line is exactly the paper's
    back-half boundary l/(L-1) >= 0.5 (and l/(L-1) lies inside its own cell).  Triangles: per-layer permutation
    p < 0.05, coloured by the sign of z.
(b) One dot per model, on its strip row: T = mean z over back-half layers.  Grey bar: the range that holds 95% of
    the permutation null of T (+-q, q = 95th percentile of |T_null|), i.e. the acceptance region of the two-sided
    test; right column: two-sided permutation p of T.  The null draws B = 10000 fresh within-body shuffles of k
    (one shuffle for all layers, so the dependence between layers is kept) and standardises every layer by the
    same null s.d. that defines its z in consist.json, so T and its null are on one scale.

The per-layer null s.d. is recovered by replaying consist.py exactly (same items, wording average, OLMo-2 outlier
exclusion E2_CLEAN, rng = default_rng(0) consumed family by family with B = 2000); the script asserts that the
replayed z equal consist.json.  Qwen2.5-14B is computed for the caption clause but not drawn (author's choice).

    python3 make_fig1_relational_contrast.py [--recompute]
writes  fig1_relational_contrast.pdf (next to this script), a 200-dpi PNG preview and plotted_values.json (OUTDIR).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
CONSIST = f"{HERE}/fig1_data/consist_q10.json"          # q10 records of the pipeline's consist.json
ROOT = str(paths.E2_SWEEP)
OUTDIR = f"{HERE}/fig1_data"
PDF, PNG = f"{HERE}/fig1_relational_contrast.pdf", f"{OUTDIR}/fig1_relational_contrast.png"
CACHE, VALUES = f"{OUTDIR}/backhalf_null_cache.json", f"{OUTDIR}/plotted_values.json"
os.makedirs(OUTDIR, exist_ok=True)

TASKS = ["pointer", "arith", "binding", "relational"]; TASK, STAT = "relational", "q10"
B_CONSIST, B_BACK, SEED_BACK = 2000, 10000, 20260925
# models shown, top to bottom: (name in the data, row label, family, row position)
MODELS = [("Llama-3.2-1B", "Llama-3.2-1B", "Llama", 0.0), ("Llama-3.2-3B", "Llama-3.2-3B", "Llama", 1.0),
          ("Llama-3.1-8B", "Llama-3.1-8B", "Llama", 2.0), ("OLMo-2-1124-7B", "OLMo-2-7B", "OLMo", 3.4),
          ("Qwen2.5-0.5B", "Qwen2.5-0.5B", "Qwen", 4.8), ("Qwen2.5-1.5B", "Qwen2.5-1.5B", "Qwen", 5.8),
          ("Qwen2.5-3B", "Qwen2.5-3B", "Qwen", 6.8), ("Qwen2.5-7B", "Qwen2.5-7B", "Qwen", 7.8)]
NOT_SHOWN = ["Qwen2.5-14B"]

C = {r["model"]: r for r in json.load(open(CONSIST)) if r["stat"] == STAT}
ZL = {m: np.array(C[m]["families"][TASK]["z"]) for m in C}
PL = {m: np.array(C[m]["families"][TASK]["p"]) for m in C}
back_mask = lambda L: np.arange(L) / (L - 1) >= 0.5

# ── back-half test, from the raw captures ─────────────────────────────────────
def _load(model, cell, attn):
    fam, w = cell.split("/")
    hits = [f for f in glob.glob(f"{ROOT}/bf16_{attn}/*__{fam}_{w}_*.npz")
            if os.path.basename(f).split("__")[0].split("_", 1)[1] == model]
    assert len(hits) == 1, (model, cell, hits)
    z = np.load(hits[0], allow_pickle=False); items = json.loads(str(z["meta"]))["items"]
    S = z["contribution"].astype(float)
    keep = ~((S.max(-1) / np.maximum(np.median(S, -1), 1e-9)) > 50).any(1)   # E2_CLEAN (drops OLMo-2 items only)
    return S[keep], [t for t, kk in zip(items, keep) if kk]
def _q10(S):
    P = S / S.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.10 * S.shape[-1])))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def _family(model, fam, attn):
    (Sa, ia_), (Sb, ib_) = _load(model, fam + "/w0", attn), _load(model, fam + "/w1", attn)
    ka = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(ia_)}
    kb = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(ib_)}
    keys = sorted(set(ka) & set(kb), key=lambda x: (int(x[0]), int(x[1])))
    Y = 0.5 * (_q10(Sa[[ka[x] for x in keys]]) + _q10(Sb[[kb[x] for x in keys]]))
    return Y, np.array([int(x[0]) for x in keys], float), np.array([int(x[1]) for x in keys])
def _perms_consist(k, body, fam, rng):                  # exactly consist.perms (same rng consumption)
    P = np.empty((B_CONSIST, len(k)))
    groups = [np.nonzero(body == g)[0] for g in np.unique(body)]
    for i in range(B_CONSIST):
        if fam in ("binding", "relational"):
            kk = k.copy()
            for g in groups: kk[g] = k[g][rng.permutation(len(g))]
            P[i] = kk
        else: P[i] = k[rng.permutation(len(k))]
    return P
def _slopes(K, Y):
    Kc = K - K.mean(-1, keepdims=True); Yc = Y - Y.mean(0)
    return (Kc @ Yc) / (Kc ** 2).sum(-1, keepdims=True)
def backhalf_test(model):
    attn = C[model]["attn"]; rng = np.random.default_rng(0)
    for f in TASKS:                                      # replay consist.py family by family
        Y, k, body = _family(model, f, attn)
        null = _slopes(_perms_consist(k, body, f, rng), Y)
        if f == TASK: YT, kT, bodyT, obs, sd = Y, k, body, _slopes(k[None], Y)[0], null.std(0)
        _perms_consist(k, body, f, rng)                  # consist.py's window test draws its own shuffles
    z = obs / sd; L = len(z); back = back_mask(L)
    assert np.abs(z - ZL[model]).max() < 1e-9, (model, np.abs(z - ZL[model]).max())
    assert len(kT) == C[model]["families"][TASK]["n"]
    rng2 = np.random.default_rng(SEED_BACK); K = np.tile(kT, (B_BACK, 1))
    for g in np.unique(bodyT):                           # fresh within-body shuffles, one per draw for all layers
        i = np.nonzero(bodyT == g)[0]; K[:, i] = rng2.permuted(K[:, i], axis=1)
    Tn = (_slopes(K, YT)[:, back] / sd[back]).mean(1)    # null of T, same per-layer scale as z
    T = float(z[back].mean())
    return {"attn": attn, "L": L, "n_items": int(len(kT)), "n_back_layers": int(back.sum()),
            "back_mean_z": T, "null_q95_abs": float(np.quantile(np.abs(Tn), 0.95)), "null_sd": float(Tn.std()),
            "null_q025": float(np.quantile(Tn, 0.025)), "null_q975": float(np.quantile(Tn, 0.975)),
            "p_two_sided": float((np.sum(np.abs(Tn) >= abs(T) - 1e-12) + 1) / (B_BACK + 1)),
            "B": B_BACK, "seed": SEED_BACK, "z_replay_maxdiff": float(np.abs(z - ZL[model]).max())}

want = [m for m, *_ in MODELS] + NOT_SHOWN
BH = json.load(open(CACHE)) if os.path.exists(CACHE) and "--recompute" not in sys.argv else {}
if any(m not in BH for m in want):
    BH = {m: backhalf_test(m) for m in want}; json.dump(BH, open(CACHE, "w"), indent=1)
for m in want:
    b = BH[m]; assert abs(b["back_mean_z"] - ZL[m][back_mask(len(ZL[m]))].mean()) < 1e-9
    print(f"{m:16s} {b['attn']:5s} T {b['back_mean_z']:+.2f}  null 95% +-{b['null_q95_abs']:.2f}  p {b['p_two_sided']:.4f}")
fmt_p = lambda p: "<0.001" if p < 0.001 else f"{p:.2f}"

# ── style (model_figs.py) ─────────────────────────────────────────────────────
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FC = {"Llama": "#2a78d6", "Qwen": "#eb6834", "OLMo": "#1baf7a"}
BLUE, RED, NULLC = "#184f95", "#a92b2b", "#dcdbd4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7, "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 2, "ytick.major.size": 2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a92b2b"])
ZMAX = 6; norm = TwoSlopeNorm(0, -ZMAX, ZMAX)

# ── layout (inches) ───────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
W, H = 5.5, 2.45
BOT, TOP = 0.36, 0.45                                   # space below / above the strip rows
AX0, AX1 = 0.80, 4.00                                   # strips
BX0, BX1 = 4.30, 5.08                                   # back-half dot column
PX = 5.44                                               # right edge of the p column
TY, KY = H - 0.10, H - 0.285                            # centre lines of the title row and the key row
fx = lambda x: x / W; fy = lambda y: y / H
fig = plt.figure(figsize=(W, H))
axA = fig.add_axes([fx(AX0), fy(BOT), fx(AX1 - AX0), fy(H - BOT - TOP)])
axB = fig.add_axes([fx(BX0), fy(BOT), fx(BX1 - BX0), fy(H - BOT - TOP)])
YLIM = (8.35, -0.62)
for ax in (axA, axB): ax.set_ylim(*YLIM); ax.set_yticks([])
row_fy = lambda y: fy(BOT + (YLIM[0] - y) / (YLIM[0] - YLIM[1]) * (H - BOT - TOP))
SH = 0.62                                               # strip height (row pitch 1)
mark = lambda x, y, m, col, ms: fig.add_artist(Line2D([fx(x)], [fy(y)], marker=m, ms=ms, color=col, mew=0, ls="",
                                                      transform=fig.transFigure, clip_on=False))

# panel titles
fig.text(fx(0.04), fy(TY), "(a) Relational lookup: per-layer response z to serial depth k", fontsize=7, color=INK,
         ha="left", va="center")
fig.text(fx(BX0 - 0.12), fy(TY), "(b) Back-half mean z", fontsize=7, color=INK, ha="left", va="center")

# (a) heat strips
out = {"models": {}}
for m, lab, fam, y in MODELS:
    z, p = ZL[m], PL[m]; L = len(z)
    edges = 100 * np.arange(L + 1) / L; xc = 100 * (np.arange(L) + 0.5) / L
    axA.pcolormesh(edges, [y - SH / 2, y + SH / 2], z[None], cmap=cmap, norm=norm, shading="flat",
                   edgecolors="none", antialiased=False)
    sig = p < 0.05
    for sg, col in ((-1, BLUE), (1, RED)):
        i = sig & (np.sign(z) == sg)
        axA.scatter(xc[i], np.full(i.sum(), y - SH / 2 - 0.16), marker="v", s=6, color=col, lw=0, zorder=4, clip_on=False)
    axA.text(-1.6, y, lab, ha="right", va="center", fontsize=6.5, color=INK2)
    out["models"][m] = {"label": lab, "family": fam, "attn": C[m]["attn"], "L": L, "row_y": y,
                        "cell_edges_pct": edges.round(4).tolist(), "rel_depth_l_over_Lm1": (np.arange(L) / (L - 1)).round(4).tolist(),
                        "z": z.tolist(), "p": p.tolist(), "triangle": [int(np.sign(v)) if s else 0 for v, s in zip(z, sig)],
                        "n_sig_spread_back": int((sig & (z < 0) & back_mask(L)).sum()),
                        "n_sig_conc_back": int((sig & (z > 0) & back_mask(L)).sum()),
                        "colour_clipped_at": ZMAX}
axA.set_xlim(0, 100); axA.set_xticks([0, 25, 50, 75, 100]); axA.spines["left"].set_visible(False)
axA.set_xlabel("relative depth (%)", labelpad=1.5)
# back-half boundary and bracket (key row, over the back half)
x50, x100 = AX0 + 0.5 * (AX1 - AX0), AX1
fig.add_artist(Line2D([fx(x50)] * 2, [fy(BOT), fy(KY)], color=INK2, lw=0.5, ls=(0, (2.4, 1.6)), transform=fig.transFigure))
fig.add_artist(Line2D([fx(x100)] * 2, [fy(KY - 0.045), fy(KY)], color=INK2, lw=0.5, transform=fig.transFigure))
xm = 0.5 * (x50 + x100); gap = 0.27
for xa, xb in ((x50, xm - gap), (xm + gap, x100)):
    fig.add_artist(Line2D([fx(xa), fx(xb)], [fy(KY)] * 2, color=INK2, lw=0.5, transform=fig.transFigure))
fig.text(fx(xm), fy(KY), "back half", fontsize=6, color=INK2, ha="center", va="center")
# colour bar (key row, over the front half)
cb0, cb1 = 1.08, 1.62
cax = fig.add_axes([fx(cb0), fy(KY - 0.033), fx(cb1 - cb0), fy(0.066)])
cax.imshow(np.linspace(-ZMAX, ZMAX, 256)[None], cmap=cmap, norm=norm, aspect="auto", extent=(-ZMAX, ZMAX, 0, 1))
cax.set_xticks([]); cax.set_yticks([])
for sp in cax.spines.values(): sp.set_visible(False)
fig.text(fx(cb0 - 0.035), fy(KY), "spreads  ≤−6", color=BLUE, fontsize=6, ha="right", va="center")
fig.text(fx(cb1 + 0.035), fy(KY), "≥+6  concentrates", color=RED, fontsize=6, ha="left", va="center")
# triangle key (bottom left, under the row labels)
KB = 0.105
mark(0.10, KB, "v", BLUE, 2.45); mark(0.17, KB, "v", RED, 2.45)
fig.text(fx(0.225), fy(KB), "layer p < 0.05", fontsize=6, color=INK2, ha="left", va="center")

# (b) back-half dot column
XB = (-2.7, 1.1)
axB.set_xlim(*XB); axB.set_xticks([-2, -1, 0, 1]); axB.set_xticklabels(["−2", "−1", "0", "1"])
axB.spines["left"].set_visible(False)
for g in (-2, -1, 1): axB.plot([g, g], [YLIM[0], YLIM[1]], color=GRID, lw=0.4, zorder=0.5)
axB.plot([0, 0], [YLIM[0], YLIM[1]], color=AXIS, lw=0.6, zorder=3)
for m, lab, fam, y in MODELS:
    b = BH[m]; q = b["null_q95_abs"]
    axB.plot([-q, q], [y, y], color=NULLC, lw=4.2, solid_capstyle="butt", zorder=2)
    sig = b["p_two_sided"] < 0.05                     # filled: back-half response significant
    axB.scatter(b["back_mean_z"], y, s=15, facecolor=INK if sig else SURF, edgecolor=INK, lw=0.7, zorder=4)
    fig.text(fx(PX), row_fy(y), fmt_p(b["p_two_sided"]), fontsize=6, color=INK2, ha="right", va="center")
    out["models"][m]["back_half"] = {**b, "dot_x": b["back_mean_z"], "grey_bar": [-q, q],
                                     "significant_p_lt_0.05": b["p_two_sided"] < 0.05, "p_label": fmt_p(b["p_two_sided"])}
fig.text(fx(PX), fy(KY), "p", fontsize=6, color=INK2, ha="right", va="center")
fig.text(fx(BX0 - 0.12), fy(KY), "gray: 95% null; filled: p < 0.05", fontsize=6, color=INK2, ha="left", va="center")
# direction words under the axis, split at z = 0
zx = BX0 + (0 - XB[0]) / (XB[1] - XB[0]) * (BX1 - BX0); DY = KB
fig.text(fx(zx - 0.03), fy(DY), "spreads", fontsize=6, color=BLUE, ha="right", va="center")
fig.text(fx(zx + 0.03), fy(DY), "concentrates", fontsize=6, color=RED, ha="left", va="center")
fig.add_artist(Line2D([fx(zx)] * 2, [fy(DY - 0.04), fy(DY + 0.04)], color=AXIS, lw=0.6, transform=fig.transFigure))

# ── numbers behind the plot ───────────────────────────────────────────────────
def window_fraction(z, half=0.10):                      # not plotted: share of layers with z < 0 in a +-10% depth window
    x = np.arange(len(z)) / (len(z) - 1); g = np.linspace(0, 1, 101)
    return g.tolist(), [float(np.mean(z[np.abs(x - c) <= half + 1e-12] < 0)) for c in g]
for m, *_ in MODELS:
    g, fr = window_fraction(ZL[m]); out["models"][m]["not_plotted_window_fraction_spreading"] = {"grid": g, "fraction": fr}
out["not_shown"] = {m: BH[m] for m in NOT_SHOWN}
out["notes"] = {"stat": STAT, "task": TASK, "source_per_layer": CONSIST,
                "z": "OLS slope of top-10% head share on k / s.d. of its permutation null (k shuffled within prompt bodies, B=2000)",
                "triangle": "per-layer two-sided permutation p < 0.05; +1 concentrates (red), -1 spreads (blue)",
                "back_half": "layers with l/(L-1) >= 0.5; T = mean z; null from B=10000 within-body shuffles, same per-layer s.d.",
                "grey_bar": "+-95th percentile of |T_null| (acceptance region of the two-sided test)",
                "colour_scale": f"diverging, clipped at +-{ZMAX}", "figure_in": [W, H]}
json.dump(out, open(VALUES, "w"), indent=1)
fig.savefig(PDF); fig.savefig(PNG, dpi=200); plt.close(fig)
print("wrote", PDF, PNG, VALUES)

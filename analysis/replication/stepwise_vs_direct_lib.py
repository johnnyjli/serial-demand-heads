"""E2b stepwise vs direct: shared loading and per-cell statistics.

Reuses the paper's pipeline exactly: v2lib.zprof (OLS slope of each layer's statistic on k, standardised by
its exact permutation-null SD; within=False, k shuffled across items) and model_diff.dec_prof (average of the
per-layer z within relative-depth deciles l/(L-1)).  Statistics per item and layer: top-10% head share
(primary), top-25% share, negated effective number of heads exp(H(p))/H (negated so that, as in the paper,
negative z = spreading).  OLMo-2 exclusion rule: drop items where any head in any layer exceeds 50x its
layer's median head norm (applied to every model; only OLMo-2 items are affected)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, zlib
import numpy as np
sys.path.insert(0, str(paths.ANALYSIS / "vet3"))
sys.path.insert(0, str(paths.ANALYSIS))
sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof            # noqa: E402  (paper pipeline)
from model_diff import dec_prof    # noqa: E402  (paper pipeline)
import e2b                         # noqa: E402

ROOT = str(paths.E2B)
MODELS = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
          "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
LINEAGE = {"Llama-3.1-8B": "L8", "Llama-3.1-8B-Instruct": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7",
           "Qwen2.5-14B": "Q14", "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-Instruct": "O7", "Mistral-Small-24B-Base-2501": "M24"}
QWEN7 = ("Qwen2.5-7B", "Qwen2.5-7B-Instruct")
FAMS = ["pointer", "arith", "binding", "relational"]
MODES = ["direct", "stepwise"]
STATS = ["q10", "q25", "neff"]


def short(name): return name.split("/")[-1]


def topq(P, q):
    H = P.shape[-1]; m = max(1, int(np.ceil(q * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)


def item_layer_stats(C):
    """C [n, L, H] head norms -> dict of [n, L] statistics."""
    S = C.astype(np.float64); P = S / np.clip(S.sum(-1, keepdims=True), 1e-12, None)
    H = S.shape[-1]
    ent = -(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(-1)
    return {"q10": topq(P, .10), "q25": topq(P, .25), "neff": -np.exp(ent) / H}


def outlier_flag(C):
    S = np.nan_to_num(C.astype(np.float64))
    return ((S.max(-1) / np.maximum(np.median(S, -1), 1e-9)) > 50).any(1)


def load(exclude=True):
    """{(model, mode, family): cell dict} with measured items only (and OLMo-2 exclusion if exclude)."""
    items = json.load(open(f"{ROOT}/items.json"))
    aset = {x["id"]: x["answer_set"] for f in items for x in items[f]}
    out = {}
    for r in e2b.load_results(ROOT):
        if "_smoke" in r["model"]: continue
        m = short(r["model"]); its = r["items"]; C = r["C"]
        meas = np.array([x["measured"] for x in its]) & np.isfinite(C).all((1, 2))
        flag = outlier_flag(C)
        keep = meas & (~flag if exclude else True)
        idx = np.nonzero(keep)[0]
        sub = [its[i] for i in idx]
        DL = r["DL"][idx].astype(np.float64)
        ent = np.empty(len(idx)); pmax = np.empty(len(idx))
        for j, x in enumerate(sub):
            a = [int(d) for d in aset[x["id"]]]; lg = DL[j, a]; p = np.exp(lg - lg.max()); p /= p.sum()
            ent[j] = -(p * np.log(np.clip(p, 1e-300, None))).sum(); pmax[j] = p.max()
        d = {"model": m, "mode": r["mode"], "family": r["family"], "L": r["L"], "H": r["H"],
             "ids": [x["id"] for x in sub], "k": np.array([x["k"] for x in sub], float),
             "correct": np.array([bool(x["correct"]) for x in sub]),
             "trace_exact": np.array([bool(x.get("trace_exact", False)) for x in sub]),
             "ntok": np.array([x.get("n_tokens_measured", np.nan) for x in sub], float),
             "ent": ent, "pmax": pmax,
             "n_total": len(its), "n_measured": int(meas.sum()), "n_flagged_measured": int((meas & flag).sum()),
             "k_all": np.array([x["k"] for x in its]), "correct_all": np.array([bool(x["correct"]) for x in its]),
             "measured_all": meas}
        d.update(item_layer_stats(C[idx]))
        out[(m, r["mode"], r["family"])] = d
    return out


def seed(*parts): return zlib.crc32("|".join(map(str, parts)).encode()) + 12345


def cell_test(Y, k, L, rng, B=4000, body=None, within=False):
    """Per-layer z (paper's zprof: exact permutation SD), per-layer two-sided permutation p, and
    back-half / front-half mean z with a joint permutation null (the same shuffle of k for all layers,
    within strata if within=True).  Returns dict."""
    Y = np.asarray(Y, float); k = np.asarray(k, float)
    b, sd, z = zprof(Y, k, body, within)
    n = len(k); x = np.arange(L) / (L - 1); bk = x >= .5; fr = ~bk
    Yc = Y - Y.mean(0)
    if within:
        # centre k within strata for the observed statistic is not what zprof does (it centres globally);
        # the null permutes k within strata, so we use the same global-centred slope under permuted k.
        _, gid = np.unique(body, return_inverse=True); gid = gid.astype(float)
        base = np.argsort(gid, kind="stable")                                  # positions, grouped
        P = np.argsort(gid[base][None, :] * 2.0 + rng.random((B, n)), axis=1)  # random order within groups
        K = np.empty((B, n)); K[:, base] = k[base][P]
    else:
        K = np.stack([rng.permutation(k) for _ in range(B)])
    kc = k - k.mean(); den = kc @ kc
    Kc = K - K.mean(1, keepdims=True)
    null = Kc @ Yc / den
    zn = null / sd
    p = (np.sum(np.abs(null) >= np.abs(b) - 1e-15, 0) + 1) / (B + 1)
    Tb = z[bk].mean(); Tn = zn[:, bk].mean(1)
    return {"z": z, "slope": b, "sd": sd, "p": p, "back": float(Tb), "front": float(z[fr].mean()),
            "p_back_lo": float((np.sum(Tn <= Tb) + 1) / (B + 1)), "p_back_hi": float((np.sum(Tn >= Tb) + 1) / (B + 1)),
            "null_back_sd": float(Tn.std()), "n_sig_spread_back": int(((z < 0) & (p < .05) & bk).sum()),
            "n_sig_conc_back": int(((z > 0) & (p < .05) & bk).sum()), "n_back": int(bk.sum()),
            "back_slope_pp": float(100 * b[bk].mean()), "dec": dec_prof(z)}


def partial_k_given_len(Y, k, t, L, rng, B=4000):
    """Freedman-Lane: partial OLS slope of each layer's statistic on k given context length t (with intercept),
    standardised by the exact SD of its permutation null (residuals of Y ~ 1 + t permuted).  Returns None when
    k has (almost) no variance left after regressing it on t."""
    Y = np.asarray(Y, float); k = np.asarray(k, float); t = np.asarray(t, float); n = len(k)
    X = np.column_stack([np.ones(n), t])
    kr = k - X @ np.linalg.lstsq(X, k, rcond=None)[0]
    frac = float((kr @ kr) / ((k - k.mean()) @ (k - k.mean())))
    if frac < 1e-6: return {"frac_k_var_unexplained_by_len": frac}
    R = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    ss = kr @ kr; b = kr @ Y / ss
    sd = np.sqrt((R ** 2).sum(0) / (n - 1) / ss)
    z = b / sd
    null = np.stack([kr[rng.permutation(n)] for _ in range(B)]) @ R / ss   # permuting R == permuting kr
    zn = null / sd
    x = np.arange(L) / (L - 1); bk = x >= .5
    Tb = z[bk].mean(); Tn = zn[:, bk].mean(1)
    p = (np.sum(np.abs(null) >= np.abs(b) - 1e-15, 0) + 1) / (B + 1)
    return {"frac_k_var_unexplained_by_len": frac, "z": z, "back": float(Tb),
            "p_back_lo": float((np.sum(Tn <= Tb) + 1) / (B + 1)), "p_back_hi": float((np.sum(Tn >= Tb) + 1) / (B + 1)),
            "n_sig_back": int(((p < .05) & bk).sum()), "back_slope_pp": float(100 * b[bk].mean()), "dec": dec_prof(z)}


def within_centre(Y, k, g):
    """Centre Y and k within strata g (fixed effects), so that zprof(..., body=g, within=True) gives the
    pooled within-stratum slope and its exact within-stratum permutation SD.  Also returns the number of
    items in strata that contain at least two distinct k values (the items that carry information)."""
    Y = np.asarray(Y, float).copy(); k = np.asarray(k, float).copy(); n_inf = 0
    for v in np.unique(g):
        i = g == v
        if len(np.unique(k[i])) >= 2: n_inf += int(i.sum())
        Y[i] -= Y[i].mean(0); k[i] -= k[i].mean()
    return Y, k, n_inf

"""Loader for the correctness analyses (reads the raw E2 captures; writes a local cache only).

Per (model, family, wording): top-10% head share y10 [n, L] (primary statistic), k, prompt body, strict
correctness (correct_constrained AND first generated digit correct, = correct_strict in the captures),
answer-set entropy, gold, prompt text.  keep = the E2_CLEAN rule (no head > 50x its layer's median head);
for analyses we drop an instance if it is flagged in EITHER wording (as the paper does).
Qwen2.5-7B-Instruct is read from the SDPA captures, every other model from eager, as in the paper."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, glob, json, numpy as np
OLD = str(paths.ANALYSIS)
ROOT = str(paths.E2_SWEEP)
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
ORDER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
         "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
TASKS = ["pointer", "arith", "binding", "relational"]
FAMS = TASKS + ["fixed_digit"]
WITHIN = {"binding", "relational"}
ATTN = {"Qwen2.5-7B-Instruct": "sdpa"}
def attn_of(m): return ATTN.get(m, "eager")
def _path(model, fam, w, attn):
    fs = [f for f in glob.glob(f"{ROOT}/bf16_{attn}/*__{fam}_{w}_*.npz")
          if os.path.basename(f).split("__")[0].split("_", 1)[1] == model]
    assert len(fs) == 1, (model, fam, w, attn, fs); return fs[0]
def topq(P, q):
    H = P.shape[-1]; m = max(1, int(np.ceil(q * H))); return np.sort(P, -1)[..., ::-1][..., :m].sum(-1)
def raw(model, fam, w, attn=None):
    attn = attn or attn_of(model)
    fn = os.path.join(CACHE, f"{model}__{fam}__{w}__{attn}.npz")
    if os.path.exists(fn):
        z = np.load(fn); d = {k: z[k] for k in z.files}
        d["text"] = json.load(open(fn[:-4] + ".json")); return d
    z = np.load(_path(model, fam, w, attn), allow_pickle=False); meta = json.loads(str(z["meta"])); items = meta["items"]
    S = z["contribution"].astype(np.float64); S = np.where(np.isfinite(S), S, 0.0)
    keep = ~((S.max(-1) / np.maximum(np.median(S, -1), 1e-9)) > 50).any(1)
    P = S / np.clip(S.sum(-1, keepdims=True), 1e-12, None)
    y10 = topq(P, .10)
    ids = [t["id"].split("/") for t in items]; k = np.array([int(t[-2]) for t in ids]); body = np.array([int(t[-1]) for t in ids])
    assert (k == z["k"]).all()
    lg = z["digit_logits"].astype(float); am = z["answer_mask"]; lg = np.where(am, lg, -np.inf)
    lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    d = dict(y10=y10, keep=keep, k=k, body=body, correct=z["correct_strict"].astype(bool),
             correct_c=z["correct_constrained"].astype(bool), gold=z["gold"].astype(int), ent=ent,
             pred=z["pred_constrained"].astype(int), first=z["pred_first_digit"].astype(int),
             ntied=z["n_tied_at_max"].astype(int), nans=am.sum(1).astype(int), L=np.array(meta["n_layers"]), H=np.array(meta["n_heads"]))
    np.savez(fn, **d); json.dump([t["text"] for t in items], open(fn[:-4] + ".json", "w"))
    d["text"] = [t["text"] for t in items]; return d
def cell(model, fam, w, attn=None, clean_both=True):
    """One wording, instances flagged in either wording dropped."""
    a = raw(model, fam, w, attn); o = raw(model, fam, "w1" if w == "w0" else "w0", attn)
    assert (a["k"] == o["k"]).all() and (a["body"] == o["body"]).all()
    kp = a["keep"] & o["keep"] if clean_both else a["keep"]
    d = {key: (v[kp] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(kp) else v) for key, v in a.items() if key != "text"}
    d["text"] = [t for t, x in zip(a["text"], kp) if x]; d["L"] = int(a["L"]); d["H"] = int(a["H"]); d["n_drop"] = int((~kp).sum())
    d["x"] = np.arange(d["L"]) / (d["L"] - 1); d["within"] = fam in ("binding", "relational"); return d
def averaged(model, fam, attn=None):
    """Wording-averaged statistic (the paper's per-item unit): same instances in w0 and w1."""
    a, b = cell(model, fam, "w0", attn), cell(model, fam, "w1", attn)
    assert (a["k"] == b["k"]).all() and (a["body"] == b["body"]).all()
    return {"y10": 0.5 * (a["y10"] + b["y10"]), "k": a["k"].astype(float), "body": a["body"], "c0": a["correct"], "c1": b["correct"],
            "L": a["L"], "x": a["x"], "within": a["within"], "ent": 0.5 * (a["ent"] + b["ent"])}
# ---- per-layer standardised slope with its exact permutation-null SD ----
def zprof(Y, k, body=None, within=False, centre=False):
    """OLS slope of each column of Y on k and the exact SD of its permutation distribution (k shuffled across
    items, or within prompt body).  centre=False: the paper's estimator (plain slope over items; with a
    within-body null, the SD sums the within-body permutation variances; equals consist.py's B=2000 Monte Carlo
    SD in expectation).  centre=True (within-body designs only): slope on body-centred k and Y, i.e. the
    within-body slope, which equals the plain slope when every body appears at every k (the full design) but
    removes between-body differences when a subset leaves bodies unbalanced; singleton bodies carry no information."""
    Y = np.asarray(Y, float); k = np.asarray(k, float)
    if not within:
        kc = k - k.mean(); Skk = kc @ kc; Yc = Y - Y.mean(0); b = kc @ Yc / Skk
        var = (Yc ** 2).sum(0) / (len(k) - 1) / Skk
    else:
        num = np.zeros(Y.shape[1]); Sw = 0.0; var = np.zeros(Y.shape[1])
        for g in np.unique(body):
            i = body == g; ng = i.sum()
            if ng < 2: continue
            kg = k[i] - k[i].mean(); Yg = Y[i] - Y[i].mean(0); num += kg @ Yg; s = kg @ kg; Sw += s
            var += s * (Yg ** 2).sum(0) / (ng - 1)
        if Sw == 0: return (np.full(Y.shape[1], np.nan),) * 3
        if centre: b = num / Sw; var = var / Sw ** 2
        else:
            kc = k - k.mean(); Skk = kc @ kc; b = kc @ (Y - Y.mean(0)) / Skk; var = var / Skk ** 2
    sd = np.sqrt(var); return b, sd, b / np.where(sd > 0, sd, np.nan)
ND = 10
def dec_prof(z):
    z = np.asarray(z, float); L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * ND).astype(int), ND - 1)
    return np.array([np.nanmean(z[d == i]) for i in range(ND)])
def decompose(Z):
    mu = Z.mean((0, 1)); a = Z.mean(1) - mu; b = Z.mean(0) - mu; e = Z - mu - a[:, None] - b[None]; return mu, a, b, e
def bh(p, q=.05):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return np.zeros(0, bool)
    o = np.argsort(p); thr = q * np.arange(1, n + 1) / n; ok = p[o] <= thr
    kmax = np.nonzero(ok)[0].max() + 1 if ok.any() else 0; r = np.zeros(n, bool); r[o[:kmax]] = True; return r
def load_cell(model, fam, w, attn=None):
    d = cell(model, fam, w, attn); d["attn"] = attn or attn_of(model); d["text"] = np.array(d["text"], dtype=object)
    d["k"] = d["k"].astype(float); return d
def load_pair(model, fam, attn=None):
    P = averaged(model, fam, attn); P["Y"] = P["y10"]; return P

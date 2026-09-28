import os, numpy as np
PRIM = os.environ.get("VPRIM", "y10")          # primary statistic: top-10% share (y25 = top-25%)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ORDER = ["Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-3B-Instruct","Llama-3.1-8B","Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-7B-Instruct","Qwen2.5-14B",
         "OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1","Mistral-Small-24B-Base-2501"]
TASKS = ["pointer", "arith", "binding", "relational"]
FAMILY = {m: m.split("-")[0].split("2.")[0] if not m.startswith("OLMo") else "OLMo" for m in ORDER}
FAMILY = {m: ("Llama" if m.startswith("Llama") else "Qwen" if m.startswith("Qwen") else "OLMo" if m.startswith("OLMo") else "Mistral") for m in ORDER}
BASE = {"Llama-3.2-1B": "L1B", "Llama-3.2-3B": "L3B", "Llama-3.2-3B-Instruct": "L3B", "Llama-3.1-8B": "L8B", "Llama-3.1-8B-Instruct": "L8B",
        "Qwen2.5-0.5B": "Q0.5", "Qwen2.5-1.5B": "Q1.5", "Qwen2.5-3B": "Q3", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7", "Qwen2.5-14B": "Q14",
        "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-SFT": "O7", "OLMo-2-1124-7B-DPO": "O7", "OLMo-2-1124-7B-Instruct": "O7",
        "Mistral-7B-v0.1": "M7", "Mistral-Small-24B-Base-2501": "M24"}
def load(model, fam, w, attn="eager", clean=True):
    fn = os.path.join(D, f"{model}__{fam}__{w}__{attn}.npz")
    if not os.path.exists(fn): return None
    z = np.load(fn); d = {n: z[n] for n in z.files}
    if clean:
        kp = d["keep"]; n = len(kp)
        for key in list(d):
            if isinstance(d[key], np.ndarray) and d[key].ndim >= 1 and d[key].shape[0] == n and key != "sinkfrac": d[key] = d[key][kp]
    d["y25orig"] = d["y25"]
    if PRIM != "y25": d["y25"] = d[PRIM]; d["y25ns"] = d[PRIM + "ns"]
    d["k"] = d["k"].astype(float); d["L"] = d["y25"].shape[1]; d["x"] = np.arange(d["L"]) / (d["L"] - 1)
    d["within"] = fam in ("binding", "relational")
    return d
def zprof(Y, k, body=None, within=False):
    """OLS slope of each column of Y on k and its exact permutation-null SD (k shuffled across items,
    or within body).  Returns slope, sd, z = slope/sd.  NaN columns are handled by dropping NaN rows per column."""
    Y = np.asarray(Y, float); k = np.asarray(k, float)
    if np.isnan(Y).any():
        out = [zprof(Y[~np.isnan(Y[:, l]), l:l+1], k[~np.isnan(Y[:, l])], None if body is None else body[~np.isnan(Y[:, l])], within) if (~np.isnan(Y[:, l])).sum() > 10
               else (np.array([np.nan]),) * 3 for l in range(Y.shape[1])]
        return tuple(np.concatenate([o[i] for o in out]) for i in range(3))
    kc = k - k.mean(); Skk = kc @ kc; b = kc @ (Y - Y.mean(0)) / Skk
    if not within:
        n = len(k); var = ((Y - Y.mean(0)) ** 2).sum(0) / (n - 1) * Skk / Skk ** 2
    else:
        var = np.zeros(Y.shape[1])
        for g in np.unique(body):
            i = body == g; ng = i.sum()
            if ng < 2: continue
            kg = k[i] - k[i].mean(); Yg = Y[i] - Y[i].mean(0)
            var += (kg @ kg) * (Yg ** 2).sum(0) / (ng - 1)
        var = var / Skk ** 2
    sd = np.sqrt(var); return b, sd, b / sd
def cells(attn="eager", tasks=TASKS, models=ORDER, q7i_sdpa=False):
    for m in models:
        for f in tasks:
            for w in ("w0", "w1"):
                a = "sdpa" if (q7i_sdpa and m == "Qwen2.5-7B-Instruct") else attn
                yield m, f, w, a

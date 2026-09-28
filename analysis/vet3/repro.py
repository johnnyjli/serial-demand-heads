"""Independent reproduction of the E2 cross-task depth pattern (top-25% head share vs serial depth k).

Per cell (model x family x wording x attention implementation x variant): per item and layer, the
top-25% head share y = sum of the ceil(.25 H) largest head output norms / sum of all head norms.
Per layer OLS slope of y on k; permutation null (k shuffled across all items for pointer / arith,
within prompt body i for binding / relational); z = obs / sd(null).  Also an analytic (exact)
permutation-sd z for a Monte-Carlo-free check.
Writes cells.json (per-cell z profiles) and fam.json (wording-averaged per model x family, for fig 16 left).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(paths.ANALYSIS / "vet3"))
from myload import catalogue, read, MODELS, SERIAL, WITHIN_BODY

B = 2000
OUT = str(paths.ANALYSIS / "vet3")


def topq_share(S, q=float(os.environ.get("VQ", ".10")), drop=None, adapt=False):
    """S [n, L, H] head norms. drop: bool mask of heads to remove (same shape or [L, H]).
    adapt=False keeps m = ceil(q*H) of the full head count; adapt=True uses ceil(q*H_kept)."""
    H = S.shape[-1]
    if drop is not None:
        drop = np.broadcast_to(drop, S.shape)
        S = np.where(drop, 0.0, S)
    tot = S.sum(-1)
    srt = -np.sort(-S, axis=-1)
    cs = np.cumsum(srt, axis=-1)
    if drop is None or not adapt:
        m = max(1, int(np.ceil(q * H)))
        top = cs[..., m - 1]
    else:
        hk = H - drop.sum(-1)
        m = np.maximum(1, np.ceil(q * hk).astype(int))
        top = np.take_along_axis(cs, (m - 1)[..., None], axis=-1)[..., 0]
    return top / np.maximum(tot, 1e-12)


def groups_of(fam, ki):
    if fam in WITHIN_BODY:
        return [np.nonzero(ki[:, 1] == g)[0] for g in np.unique(ki[:, 1])]
    return [np.arange(len(ki))]


def zprofile(Y, k, groups, rng):
    n, L = Y.shape
    kc = k - k.mean(); Yc = Y - Y.mean(0)
    den = kc @ kc
    obs = kc @ Yc / den
    gid = np.empty(n)
    for j, g in enumerate(groups): gid[g] = j
    base = np.argsort(gid, kind="stable")                 # positions, grouped
    P = np.argsort(gid[None, :] * 2.0 + rng.random((B, n)), axis=1)   # same groups, random order within
    K = np.empty((B, n)); K[:, base] = k[P]
    assert np.allclose(np.sort(K[0]), np.sort(k))
    for g in groups[:3]:
        assert np.allclose(np.sort(K[0, g]), np.sort(k[g]))
    Kc = K - K.mean(1, keepdims=True)
    null = Kc @ Yc / den                       # denominator is permutation-invariant
    sd = null.std(0)
    z = obs / sd
    zc = (obs - null.mean(0)) / sd
    p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / (B + 1)
    # exact permutation variance of the numerator: sum_g SSk_g * SSy_g / (n_g - 1)
    var = np.zeros(L); mu = np.zeros(L)
    for g in groups:
        if len(g) < 2: continue
        kg = k[g] - k[g].mean(); yg = Y[g] - Y[g].mean(0)
        var += (kg @ kg) * (yg * yg).sum(0) / (len(g) - 1)
        mu += len(g) * (k[g].mean() - k.mean()) * (Y[g].mean(0) - Y.mean(0))
    zx = (obs * den - mu) / np.sqrt(var)
    return {"obs": obs, "z": z, "zc": zc, "zx": zx, "p": p}


def summarise(res, L):
    x = np.arange(L) / (L - 1)
    fr, bk = x < 0.5, x >= 0.5
    o = {}
    for key in ("z", "zc", "zx"):
        o[f"front_{key}"] = float(res[key][fr].mean()); o[f"back_{key}"] = float(res[key][bk].mean())
    for key in ("obs", "z", "zc", "zx", "p"):
        o[key] = res[key].tolist()
    return o


def seed_for(*parts):
    return zlib.crc32("|".join(parts).encode()) + 987654


def cell_job(args):
    model, fam, w, attn, variant = args
    cat = catalogue(attn)
    if (model, fam, w) not in cat:
        return None
    d = read(cat[(model, fam, w)], drop_massive=(variant != "noclean"))
    S, k, ki, L = d["S"], d["k"], d["k_i"], d["L"]
    if variant in ("all", "noclean"):
        Y = topq_share(S)
    elif variant in ("sink_item", "sink_item_adapt"):
        if d["pos0"] is None: return None
        Y = topq_share(S, drop=d["pos0"] > 0.9, adapt=(variant == "sink_item_adapt"))
    elif variant == "sink_head":
        if d["pos0"] is None: return None
        Y = topq_share(S, drop=(d["pos0"].mean(0) > 0.9), adapt=True)
    else:
        raise ValueError(variant)
    rng = np.random.default_rng(seed_for(model, fam, w, attn, variant))
    res = zprofile(Y, k, groups_of(fam, ki), rng)
    o = summarise(res, L)
    o.update({"model": model, "fam": fam, "w": w, "attn": attn, "variant": variant, "L": L, "n": int(len(k)),
              "n_drop": d["n_drop"],
              "sink_frac": None if d["pos0"] is None else float((d["pos0"] > 0.9).mean())})
    return o


def fam_job(args):
    """Wording-averaged per model x family (items matched on (k, i)); attention per model."""
    model, fam, attn, drop_massive = args
    cat = catalogue(attn)
    if (model, fam, "w0") not in cat or (model, fam, "w1") not in cat:
        return None
    a = read(cat[(model, fam, "w0")], drop_massive); b = read(cat[(model, fam, "w1")], drop_massive)
    ia = {tuple(t): j for j, t in enumerate(a["k_i"])}; ib = {tuple(t): j for j, t in enumerate(b["k_i"])}
    keys = sorted(set(ia) & set(ib))
    ja = np.array([ia[t] for t in keys]); jb = np.array([ib[t] for t in keys])
    Y = 0.5 * (topq_share(a["S"][ja]) + topq_share(b["S"][jb]))
    ki = np.array(keys); k = ki[:, 0].astype(float)
    rng = np.random.default_rng(seed_for(model, fam, attn, "fam", str(drop_massive)))
    res = zprofile(Y, k, groups_of(fam, ki), rng)
    o = summarise(res, a["L"])
    o.update({"model": model, "fam": fam, "attn": attn, "clean": drop_massive, "L": a["L"], "n": len(keys)})
    return o


if __name__ == "__main__":
    fams = SERIAL + ["fixed_digit"]
    jobs = [(m, f, w, "eager", v) for v in ("all", "sink_item", "sink_item_adapt", "sink_head") for m in MODELS for f in fams for w in ("w0", "w1")]
    jobs += [(m, f, w, "sdpa", "all") for m in MODELS for f in fams for w in ("w0", "w1")]
    jobs += [(m, f, w, "eager", "noclean") for m in MODELS if m.startswith("OLMo") for f in fams for w in ("w0", "w1")]
    fjobs = [(m, f, a, True) for m in MODELS for f in fams for a in ("eager", "sdpa")]
    fjobs += [(m, f, "eager", False) for m in MODELS if m.startswith("OLMo") for f in fams]
    with ProcessPoolExecutor(10) as ex:
        R = [r for r in ex.map(cell_job, jobs, chunksize=4) if r]
        F = [r for r in ex.map(fam_job, fjobs, chunksize=2) if r]
    json.dump(R, open(f"{OUT}/cells.json", "w")); json.dump(F, open(f"{OUT}/fam.json", "w"))
    print(len(R), len(F))

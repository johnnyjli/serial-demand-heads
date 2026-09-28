"""Independent loader for the E2 sweep (written for the vet; does not import e2lib)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, re
import numpy as np

DATA = str(paths.E2_SWEEP)
MODELS = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
          "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
          "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
          "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
SERIAL = ["pointer", "arith", "binding", "relational"]
WITHIN_BODY = {"binding", "relational"}

_PAT = re.compile(r"^(?P<org>[^_]+)_(?P<model>.+?)__(?P<fam>pointer|arith|binding|relational|fixed_digit)_(?P<w>w[01])_")


def catalogue(attn):
    cat = {}
    for f in sorted(glob.glob(os.path.join(DATA, f"bf16_{attn}", "*.npz"))):
        m = _PAT.match(os.path.basename(f))
        if not m:
            raise ValueError(f)
        key = (m["model"], m["fam"], m["w"])
        assert key not in cat, key
        cat[key] = f
    return cat


def massive_mask(S, thr=50.0):
    """True for items where, in some layer, the largest head output norm exceeds thr x the layer's median head."""
    med = np.median(S, axis=-1)
    ratio = S.max(-1) / np.maximum(med, 1e-9)
    return (ratio > thr).any(axis=1)


def read(path, drop_massive=True):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    items = meta["items"]
    S = z["contribution"].astype(np.float64)                     # [n, L, H]
    k = z["k"].astype(np.float64)
    ki = np.array([[int(p) for p in it["id"].split("/")[-2:]] for it in items])   # (k, i)
    assert (ki[:, 0] == k).all(), path
    pos0 = z["pos0_attn"].astype(np.float64) if "pos0_attn" in z.files else None
    n = len(k)
    bad = massive_mask(S) if drop_massive else np.zeros(n, bool)
    keep = ~bad
    out = {"S": S[keep], "k": k[keep], "k_i": ki[keep], "pos0": None if pos0 is None else pos0[keep],
           "L": int(meta["n_layers"]), "H": int(meta["n_heads"]), "n_raw": n, "n_drop": int(bad.sum()),
           "drop_k": k[bad].astype(int).tolist(), "attn_impl": meta.get("attn_impl"), "nan": int(np.isnan(S).sum())}
    assert S.shape[1:] == (out["L"], out["H"])
    return out

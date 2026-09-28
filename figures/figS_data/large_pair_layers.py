"""Per-layer response z (and two-sided permutation p) of the top-10% head share to k for Llama-3.1-70B and
Qwen2.5-72B, from the same clean captures and with the same statistic and seeds as checks/large_pair.py
(Appendix D): wording w0, all test items, bf16, SDPA; z = OLS slope on k / SD of its permutation null
(B = 2000; k shuffled within prompt bodies for binding and relational lookup, across items otherwise).

    python3 large_pair_layers.py   -> large_pair_layers.json
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, numpy as np
ROOT = str(paths.E4_ABLATION / "bf16_sdpa")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "large_pair_layers.json")
MODELS = {"Llama-3.1-70B": "meta-llama_Llama-3.1-70B", "Qwen2.5-72B": "Qwen_Qwen2.5-72B"}
TASKS = ["pointer", "arith", "binding", "relational"]
def q10(S):
    P = S / S.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.10 * S.shape[-1])))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def slopes(K, Y):
    Kc = K - K.mean(-1, keepdims=True); Yc = Y - Y.mean(0)
    return (Kc @ Yc) / (Kc ** 2).sum(-1, keepdims=True)
def perms(k, body, within, B, rng):
    K = np.tile(k, (B, 1))
    if within:
        for g in np.unique(body):
            i = np.nonzero(body == g)[0]; K[:, i] = rng.permuted(K[:, i], axis=1)
    else:
        K = rng.permuted(K, axis=1)
    return K
out = {}
for name, pre in MODELS.items():
    out[name] = {}
    for t in TASKS:
        f = glob.glob(f"{ROOT}/{pre}__{t}_w0_*__ablation.npz"); assert len(f) == 1, (name, t, f)
        d = np.load(f[0], allow_pickle=True); J = json.loads(str(d["meta"]))
        ids = [x["id"] for x in J["measurement_items"]] + list(J["selection_ids"])
        C = np.concatenate([d["meas_contribution"], d["sel_contribution"]]).astype(float)
        k = np.array([int(i.split("/")[-2]) for i in ids], float); body = np.array([int(i.split("/")[-1]) for i in ids])
        Y = q10(C); within = t in ("binding", "relational"); rng = np.random.default_rng(0)
        obs = slopes(k[None], Y)[0]; null = slopes(perms(k, body, within, 2000, rng), Y); sd = null.std(0)
        z = obs / sd; p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / 2001
        out[name][t] = {"L": int(Y.shape[1]), "n_items": int(len(k)), "z": z.round(4).tolist(), "p": p.round(5).tolist()}
        back = np.arange(Y.shape[1]) / (Y.shape[1] - 1) >= 0.5
        print(f"{name:14s} {t:10s} back mean z {z[back].mean():+.2f}")
json.dump(out, open(OUT, "w"), indent=1)

"""Head-activity response to serial depth k for Llama-3.1-70B and Qwen2.5-72B (80 layers, 64 heads each).

Data: the clean (unablated) captures stored in the E4 ablation runs (wording w0, test split, bf16, SDPA):
meas_contribution and sel_contribution are disjoint item halves; together they hold every test item
(42 per depth; 35 in pointer chasing's five depths).  The per-head activity is the same quantity as in the
main run (checked against the main run's SDPA captures of Llama-3.2-3B on identical items).
Statistic as in the paper: top-10% head share per layer; z = OLS slope on k / SD of its permutation null
(B = 2000; k shuffled within prompt bodies for binding and relational lookup, across items otherwise).
Back half: layers with l/(L-1) >= 0.5.  Back-half test: T = mean z over back-half layers, null from
B = 10000 fresh shuffles with every layer scaled by its own null SD (as in Figure 1).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, numpy as np
ROOT = str(paths.E4_ABLATION / "bf16_sdpa")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "large_pair.json")
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
def dec_prof(z):
    L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * 10).astype(int), 9)
    return np.array([z[d == j].mean() for j in range(10)])
out = {}
for name, pre in MODELS.items():
    out[name] = {}
    for t in TASKS:
        f = glob.glob(f"{ROOT}/{pre}__{t}_w0_*__ablation.npz"); assert len(f) == 1, (name, t, f)
        d = np.load(f[0], allow_pickle=True); J = json.loads(str(d["meta"]))
        ids = [x["id"] for x in J["measurement_items"]] + list(J["selection_ids"])
        C = np.concatenate([d["meas_contribution"], d["sel_contribution"]]).astype(float)
        k = np.array([int(i.split("/")[-2]) for i in ids], float); body = np.array([int(i.split("/")[-1]) for i in ids])
        assert np.array_equal(k[:len(J["measurement_items"])], d["k"].astype(float))
        Y = q10(C); L = Y.shape[1]; back = np.arange(L) / (L - 1) >= 0.5; within = t in ("binding", "relational")
        rng = np.random.default_rng(0)
        obs = slopes(k[None], Y)[0]; null = slopes(perms(k, body, within, 2000, rng), Y); sd = null.std(0)
        z = obs / sd; p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / 2001
        Tn = (slopes(perms(k, body, within, 10000, np.random.default_rng(20260926)), Y)[:, back] / sd[back]).mean(1)
        T = float(z[back].mean()); pT = float((np.sum(np.abs(Tn) >= abs(T) - 1e-12) + 1) / 10001)
        dp = dec_prof(z)
        out[name][t] = {"n_items": int(len(k)), "L": L, "H": int(C.shape[-1]), "back_mean_z": T, "back_p": pT,
                        "z_50_80": float(dp[5:8].mean()), "z_40_50": float(dp[4]), "decile_z": dp.round(3).tolist(),
                        "n_back_sig_spread": int(((p < .05) & (z < 0) & back).sum()), "n_back_sig_conc": int(((p < .05) & (z > 0) & back).sum()),
                        "n_back": int(back.sum()), "mean_top10_share_back": float(Y[:, back].mean())}
        print(f"{name:14s} {t:10s} n={len(k):3d}  back z {T:+.2f} (p={pT:.4f})  50-80 {dp[5:8].mean():+.2f}  40-50 {dp[4]:+.2f}  sig spr/conc {out[name][t]['n_back_sig_spread']}/{out[name][t]['n_back_sig_conc']} of {back.sum()}  share {Y[:, back].mean():.3f}")
json.dump(out, open(OUT, "w"), indent=1)

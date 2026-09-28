"""Extra checks: (a) per-cell permutation test of the back-half mean z and back-minus-front (joint null across
layers); (b) consistent-layer counts vs circular-shift null; (c) decile-boundary check; (d) centred z;
(e) sink removal with the all-heads-masked NaN fixed by dropping those items; (f) wording-averaged front/back."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(paths.ANALYSIS / "vet3"))
from myload import catalogue, read, MODELS, SERIAL
from repro import topq_share, groups_of, seed_for
SP = str(paths.ANALYSIS)
QI = "Qwen2.5-7B-Instruct"; B = 4000


def perms(k, groups, rng):
    n = len(k); gid = np.empty(n)
    for j, g in enumerate(groups): gid[g] = j
    base = np.argsort(gid, kind="stable"); P = np.argsort(gid[None] * 2.0 + rng.random((B, n)), axis=1)
    K = np.empty((B, n)); K[:, base] = k[P]; return K


def cell_test(args):
    model, fam, w, variant = args
    attn = "sdpa" if (model == QI and variant == "all") else "eager"
    d = read(catalogue(attn)[(model, fam, w)])
    S, k, ki, L = d["S"], d["k"], d["k_i"], d["L"]
    if variant == "sink_fixnan":
        mask = d["pos0"] > 0.9
        ok = ~mask.all(-1).any(-1)                 # drop items where some layer has every head masked
        S, k, ki, mask = S[ok], k[ok], ki[ok], mask[ok]
        Y = topq_share(S, drop=mask, adapt=False)
    else:
        Y = topq_share(S)
    rng = np.random.default_rng(seed_for(model, fam, w, variant, "extra"))
    K = perms(k, groups_of(fam, ki), rng)
    kc = k - k.mean(); Yc = Y - Y.mean(0); den = kc @ kc
    obs = kc @ Yc / den; Kc = K - K.mean(1, keepdims=True); null = Kc @ Yc / den
    sd = null.std(0); z = obs / sd; zn = null / sd
    x = np.arange(L) / (L - 1); bk, fr = x >= .5, x < .5
    Tb, Tn = z[bk].mean(), zn[:, bk].mean(1)
    Db, Dn = z[bk].mean() - z[fr].mean(), zn[:, bk].mean(1) - zn[:, fr].mean(1)
    return {"model": model, "fam": fam, "w": w, "variant": variant, "back": float(Tb), "front": float(z[fr].mean()),
            "p_back": float((np.sum(Tn <= Tb) + 1) / (B + 1)), "p_diff": float((np.sum(Dn <= Db) + 1) / (B + 1)),
            "null_back_sd": float(Tn.std()), "n": int(len(k))}


if __name__ == "__main__":
    jobs = [(m, f, w, v) for v in ("all", "sink_fixnan") for m in MODELS for f in SERIAL for w in ("w0", "w1")]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(cell_test, jobs, chunksize=4))
    json.dump(R, open(f"{SP}/vet3/extra_cells.json", "w"))
    for v in ("all", "sink_fixnan"):
        rs = [r for r in R if r["variant"] == v]
        rsq = [r for r in rs if r["model"] != QI]
        print(f"[{v}] n={len(rs)}  back<0 {100*np.mean([r['back']<0 for r in rs]):.1f}%  back<front {100*np.mean([r['back']<r['front'] for r in rs]):.1f}%"
              f"  | excl QwenInstr n={len(rsq)} back<0 {100*np.mean([r['back']<0 for r in rsq]):.1f}% back<front {100*np.mean([r['back']<r['front'] for r in rsq]):.1f}%")
        print(f"   cells with back-half spreading significant (one-sided perm p<.05): {sum(r['p_back']<.05 for r in rs)}/{len(rs)};"
              f" back-front significant: {sum(r['p_diff']<.05 for r in rs)}/{len(rs)};"
              f" back-half significantly CONCENTRATING (p>.95): {sum(r['p_back']>.95 for r in rs)}")
        for f in SERIAL:
            q = [r for r in rs if r["fam"] == f]
            print(f"     {f:10s} sig back spread {sum(r['p_back']<.05 for r in q)}/{len(q)}  sig back<front {sum(r['p_diff']<.05 for r in q)}/{len(q)}")
        print("   median null sd of back-half mean z:", round(float(np.median([r["null_back_sd"] for r in rs])), 3))

    # (b) consistent layers, wording-averaged, circular-shift null
    F = json.load(open(f"{SP}/vet3/fam.json"))
    def famrec(m, f):
        attn = "sdpa" if m == QI else "eager"
        return [r for r in F if r["model"] == m and r["fam"] == f and r["attn"] == attn and r["clean"]][0]
    def consistent(M):
        pos, neg = (M > 0).sum(0), (M < 0).sum(0)
        return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
    rng = np.random.default_rng(7); NB = 5000; tot = np.zeros(2); null = np.zeros((NB, 2)); per = []
    for m in MODELS:
        M = np.stack([np.sign(famrec(m, f)["obs"]) * (np.array(famrec(m, f)["p"]) < .05) for f in SERIAL]); L = M.shape[1]
        c = consistent(M); tot += [(c < 0).sum(), (c > 0).sum()]
        for b in range(NB):
            cc = consistent(np.stack([np.roll(M[i], rng.integers(L)) for i in range(4)])); null[b] += [(cc < 0).sum(), (cc > 0).sum()]
        per.append((m, int((c < 0).sum()), int((c > 0).sum())))
    print(f"\n(b) consistent layers (>=3 of 4 tasks same sig direction, none opposite), sum over 17 models: spread {int(tot[0])} "
          f"(shift-null mean {null[:,0].mean():.1f}, p={(np.sum(null[:,0]>=tot[0])+1)/(NB+1):.4f}), conc {int(tot[1])} "
          f"(null {null[:,1].mean():.1f}, p={(np.sum(null[:,1]>=tot[1])+1)/(NB+1):.4f})")
    print("   per model (spread, conc):", per)

    # (c) decile boundaries: float truncation vs epsilon floor
    for L in sorted({famrec(m, "pointer")["L"] for m in MODELS}):
        x = np.arange(L) / (L - 1); a = np.minimum((x * 10).astype(int), 9); b = np.minimum(np.floor(x * 10 + 1e-9).astype(int), 9)
        print(f"(c) L={L}: layers whose decile differs between (x*10).astype(int) and floor(x*10+eps): {np.nonzero(a != b)[0].tolist()}")

    # (d) centred vs uncentred z; (f) wording-averaged front/back
    C = json.load(open(f"{SP}/vet3/cells.json"))
    P = [r for r in C if r["fam"] in SERIAL and r["variant"] == "all" and ((r["attn"] == "eager" and r["model"] != QI) or (r["attn"] == "sdpa" and r["model"] == QI))]
    print(f"(d) centred z: back<0 {100*np.mean([r['back_zc']<0 for r in P]):.1f}%  back<front {100*np.mean([r['back_zc']<r['front_zc'] for r in P]):.1f}%;"
          f" max |back_z-back_zc| {max(abs(r['back_z']-r['back_zc']) for r in P):.3f}")
    FA = [famrec(m, f) for m in MODELS for f in SERIAL]
    print(f"(f) wording-averaged model x family (68): back<0 {100*np.mean([r['back_z']<0 for r in FA]):.1f}%  back<front {100*np.mean([r['back_z']<r['front_z'] for r in FA]):.1f}%  "
          f"mean front z {np.mean([r['front_z'] for r in FA]):+.2f}, mean back z {np.mean([r['back_z'] for r in FA]):+.2f}")
    for f in SERIAL:
        q = [famrec(m, f) for m in MODELS]
        print(f"     {f:10s} back<0 {sum(r['back_z']<0 for r in q)}/17  back<front {sum(r['back_z']<r['front_z'] for r in q)}/17  "
              f"mean front {np.mean([r['front_z'] for r in q]):+.2f} back {np.mean([r['back_z'] for r in q]):+.2f}")

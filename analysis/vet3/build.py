"""Cache per-cell item x layer statistics for the vetting analyses (read-only on the raw data).
Per (model, family, wording, attn): top-25% share (all heads; sink heads with pos0_attn>0.9 zeroed),
effective number of heads exp(H(p)), top-1 share, plus k, body, correctness, answer-set entropy.
All items stored; keep = not (one head > 50x its layer's median head), the E2_CLEAN rule, computed for every model."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
sys.path.insert(0, str(paths.ANALYSIS))
os.environ.pop("E2_CLEAN", None)
from e2lib import ORDER, index
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FAMS = ["pointer", "arith", "binding", "relational", "fixed_digit"]
def topq(P, q=.25):
    H = P.shape[-1]; m = max(1, int(np.ceil(q * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def one(args):
    model, fam, w, attn = args
    try: f = index(attn)[(model, f"{fam}/{w}")]
    except KeyError: return None
    z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"])); items = m["items"]
    S = z["contribution"].astype(float); S = np.where(np.isfinite(S), S, 0.0)
    keep = ~((S.max(-1) / np.maximum(np.median(S, -1), 1e-9)) > 50).any(1)
    tot = S.sum(-1, keepdims=True); P = S / np.clip(tot, 1e-12, None)
    y25 = topq(P); y10 = topq(P, .10); y50 = topq(P, .50); top1 = P.max(-1)
    neff = np.exp(-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(-1))
    sink = (z["pos0_attn"] > 0.9) if "pos0_attn" in z.files else np.zeros(S.shape, bool)
    has_sink = "pos0_attn" in z.files
    S2 = np.where(sink, 0.0, S); t2 = S2.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P2 = np.where(t2 > 0, S2 / t2, np.nan)
    y25ns = topq(np.nan_to_num(P2, nan=0.0)); y25ns[(t2[..., 0] <= 0)] = np.nan
    y10ns = topq(np.nan_to_num(P2, nan=0.0), .10); y10ns[(t2[..., 0] <= 0)] = np.nan
    if not has_sink: y25ns[:] = np.nan; y10ns[:] = np.nan
    lg = z["digit_logits"].astype(float); am = z["answer_mask"]
    lg = np.where(am, lg, -np.inf); lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    gold = z["gold"].astype(int); pg = p[np.arange(len(gold)), gold]
    ids = [t["id"].split("/") for t in items]
    k = np.array([int(t[-2]) for t in ids]); body = np.array([int(t[-1]) for t in ids])
    corr = z["correct_strict"].astype(bool)
    fn = os.path.join(OUT, f"{model}__{fam}__{w}__{attn}.npz")
    np.savez(fn, y25=y25, y10=y10, y50=y50, top1=top1, neff=neff, y25ns=y25ns, y10ns=y10ns, keep=keep,
             k=k, body=body, correct=corr, ent=ent, pgold=pg, nans=np.array(int(am[0].sum())),
             n_drop=np.array(int((~keep).sum())), H=np.array(S.shape[-1]), sinkfrac=sink[keep].mean(0))
    return (model, fam, w, attn, int((~keep).sum()), S.shape)
if __name__ == "__main__":
    jobs = [(m, f, w, a) for m in ORDER for f in FAMS for w in ("w0", "w1") for a in ("eager", "sdpa")]
    with ProcessPoolExecutor(10) as ex:
        for r in ex.map(one, jobs):
            if r and r[4]: print("dropped", r)
    print("done")

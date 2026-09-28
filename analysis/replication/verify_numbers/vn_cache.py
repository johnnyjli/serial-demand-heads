"""Verifier: load E2b once, compute per-item per-layer statistics, cache to a pickle.
Statistics: q10 (top-10% share), q25, neff (effective number of heads / H), from head norms C [items, L, H].
Also stores per-item metadata and the outlier flag (any head > 50x its layer's median head norm)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, pickle, numpy as np
sys.path.insert(0, str(paths.REPLICATION))
import e2b
OUT = str(paths.ANALYSIS / "replication" / "verify_numbers")
runs = e2b.load_results(str(paths.E2B))

def topq(C, q):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]; m = int(np.ceil(q * H))
    return np.sort(P, -1)[..., ::-1][..., :m].sum(-1)

def neff(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), -1)
    out = np.exp(ent) / H
    out[np.isnan(C).any(-1)] = np.nan
    return out

D = {}
for r in runs:
    m = r["model"].split("/")[-1]; C = r["C"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = C.max(-1) / np.maximum(np.median(C, -1), 1e-9)
    flag = np.nan_to_num(ratio, nan=0.0) > 50
    flag = flag.any(1)
    meas = ~np.isnan(C).any((1, 2))
    D[(m, r["mode"], r["family"])] = {
        "q10": topq(C, 0.1), "q25": topq(C, 0.25), "neff": neff(C),
        "flag": flag, "measured": meas, "L": r["L"], "H": r["H"], "items": r["items"], "DL": r["DL"],
        "maxratio": np.nanmax(ratio, 1) if meas.any() else None, "checks": r["checks"]}
    print(m, r["mode"], r["family"], C.shape, "flagged", int(flag.sum()), "measured", int(meas.sum()))
pickle.dump(D, open(f"{OUT}/vn_cache.pkl", "wb"))

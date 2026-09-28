import sys, json, glob, os, collections, numpy as np
sys.path.insert(0, "..")
import e2lib
rows = []
for attn in ("eager","sdpa"):
    for (model, cell), f in sorted(e2lib.index(attn).items()):
        z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
        c = m["checks"]; li = m["load_info"]
        ks = collections.Counter(int(x) for x in z["k"])
        splits = collections.Counter(it["split"] for it in m["items"])
        rows.append(dict(attn=attn, model=model, cell=cell, L=m["n_layers"], H=m["n_heads"], shape=z["contribution"].shape,
            gpu=li.get("gpu"), impl=m["attn_impl"], prec=m["precision"], split_err=c.get("split_err"), split_warn=c.get("split_warn"),
            verified=c.get("verified"), tol=c.get("split_tol"), ks=dict(ks), splits=dict(splits), ntok_by_k=m["n_tokens_by_k"],
            appended=m["appended_forms"], strategy=li.get("strategy"), fp=li.get("forced_precision"),
            first=m["items"][0]["text"][:20]))
json.dump(rows, open("survey.json","w"), default=str)
# per-model summary
by = collections.defaultdict(list)
for r in rows: by[(r["attn"], r["model"])].append(r)
for (attn, model), rs in sorted(by.items()):
    Ls = {r["L"] for r in rs}; Hs = {r["H"] for r in rs}; g = {r["gpu"] for r in rs}; pr = {r["prec"] for r in rs}; im={r["impl"] for r in rs}
    errs = [r["split_err"] for r in rs]
    ntok_const = all(len(v)==1 for r in rs for v in r["ntok_by_k"].values()) and all(len({tuple(v) for v in r["ntok_by_k"].values()})==1 for r in rs)
    print(attn, model, Ls, Hs, g, pr, im, "err %.4g-%.4g"%(min(errs), max(errs)), "warn", sum(r["split_warn"] for r in rs), "verified", all(r["verified"] for r in rs), "ntokconst", ntok_const, {r["strategy"] for r in rs})

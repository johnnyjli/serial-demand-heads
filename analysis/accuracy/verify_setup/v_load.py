"""Independent loader for the E2 accuracy verification.

Reads every E2 npz (eager and SDPA) straight from the raw capture folder; takes predictions and
correctness from the npz arrays (not the meta), and the item text / generated text / appended token
from the meta.  Cross-checks the arrays against the meta copy.  Writes v_items.pkl and v_runs.pkl.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, pickle
import numpy as np, pandas as pd

ROOT = str(paths.E2_SWEEP)
HERE = os.path.dirname(os.path.abspath(__file__))

recs, runs, meta_mismatch = [], [], 0
for kernel in ("eager", "sdpa"):
    for path in sorted(glob.glob(os.path.join(ROOT, f"bf16_{kernel}", "*.npz"))):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        model = meta["model"].split("/")[-1]
        fam, w = meta["family"], meta["wording"]
        assert meta["attn_impl"] == kernel, (path, meta["attn_impl"])
        n = int(meta["n_items"]); assert n == len(z["k"]) == len(meta["items"])
        runs.append(dict(model=model, kernel=kernel, fam=fam, w=w, n=n, checks=meta["checks"],
                         appended_forms=meta["appended_forms"], measured_tokens=meta["measured_tokens"],
                         n_tokens_by_k=meta["n_tokens_by_k"], precision=meta["precision"],
                         split=meta["split"], setting=meta["setting"], suite=meta["suite"],
                         suite_sha=meta["suite_sha256"], constrained_ties=meta["constrained_ties"]))
        dl = z["digit_logits"].astype(np.float64)
        for i, it in enumerate(meta["items"]):
            # arrays vs meta copy
            if (int(it["k"]) != int(z["k"][i]) or int(it["gold"]) != int(z["gold"][i])
                    or int(it["pred_constrained"]) != int(z["pred_constrained"][i])
                    or bool(it["correct_strict"]) != bool(z["correct_strict"][i])
                    or bool(it["correct_constrained"]) != bool(z["correct_constrained"][i])):
                meta_mismatch += 1
            recs.append(dict(model=model, kernel=kernel, fam=fam, w=w, iid=it["id"], split=it["split"],
                             k=int(z["k"][i]), gold=int(z["gold"][i]),
                             aset=tuple(int(a) for a in it["answer_set"]),
                             amask=tuple(bool(x) for x in z["answer_mask"][i]),
                             echo=None if it["echo"] is None else int(it["echo"]),
                             min_hops=it["min_hops"], text=it["text"], gen=it["gen_text"],
                             appended=tuple(it["appended"]), mtok=it["measured_token"],
                             ntok=int(z["n_tokens"][i]),
                             pc=int(z["pred_constrained"][i]), cc=bool(z["correct_constrained"][i]),
                             cs=bool(z["correct_strict"][i]), pfd=int(z["pred_first_digit"][i]),
                             ntie=int(z["n_tied_at_max"][i]), gtie_meta=bool(it["gold_tied_at_max"]),
                             dmass=float(z["digit_mass"][i]), gprob=float(z["gold_prob"][i]),
                             dl=dl[i]))
D = pd.DataFrame(recs)
print("items", len(D), "runs", len(runs), "array/meta mismatches", meta_mismatch)
print(D.groupby("kernel").size().to_dict(), D.groupby("kernel").model.nunique().to_dict())
pickle.dump(D, open(os.path.join(HERE, "v_items.pkl"), "wb"))
pickle.dump(runs, open(os.path.join(HERE, "v_runs.pkl"), "wb"))

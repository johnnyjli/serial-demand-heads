"""Load every E2 run (eager and SDPA) into one item table (no head arrays) + per-run meta checks."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, pickle, numpy as np
ROOT = str(paths.E2_SWEEP)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "items.pkl")
rows, runs = [], []
for attn in ("eager", "sdpa"):
    for f in sorted(glob.glob(f"{ROOT}/bf16_{attn}/*.npz")):
        b = os.path.basename(f); model = b.split("__")[0].split("_", 1)[1]
        suite = b.split("__")[1]; fam = suite.split("_w")[0]; w = "w" + suite.split("_w")[1][0]
        z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
        arr = {n: z[n] for n in ("k", "gold", "pred_constrained", "correct_constrained", "correct_strict",
                                 "pred_first_digit", "n_tied_at_max", "digit_mass", "gold_prob", "digit_logits",
                                 "answer_mask")}
        runs.append(dict(model=model, attn=attn, fam=fam, w=w, file=b, n_items=m["n_items"],
                         checks=m["checks"], measured_tokens=m["measured_tokens"],
                         appended_forms=m["appended_forms"], has_bos=m["has_bos"],
                         pos0_tokens=m.get("pos0_tokens"), n_tokens_by_k=m["n_tokens_by_k"],
                         constrained_ties=m["constrained_ties"], attn_impl=m["attn_impl"],
                         precision=m["precision"], setting=m["setting"], suite_sha=m["suite_sha256"],
                         proj_name=m.get("proj_name"), sandwich_layers=m.get("sandwich_layers"),
                         load_info=m.get("load_info")))
        for i, t in enumerate(m["items"]):
            assert int(t["k"]) == int(arr["k"][i]) and int(t["gold"]) == int(arr["gold"][i])
            rows.append(dict(model=model, attn=attn, fam=fam, w=w, cell=f"{fam}/{w}", id=t["id"], k=int(t["k"]),
                             gold=int(t["gold"]), answer_set=[int(a) for a in t["answer_set"]],
                             echo=(int(t["echo"]) if t["echo"] is not None else None),
                             min_hops=t.get("min_hops"), text=t["text"], gen_text=t.get("gen_text"),
                             measured_token=t["measured_token"], appended=t["appended"],
                             n_tokens=t["n_tokens"],
                             pred_c=int(arr["pred_constrained"][i]), corr_c=bool(arr["correct_constrained"][i]),
                             corr_s=bool(arr["correct_strict"][i]), pred_fd=int(arr["pred_first_digit"][i]),
                             pred_fd_meta=t.get("pred_first_digit"), pred_nonecho=t.get("pred_nonecho"),
                             pred_digit=int(t["pred_digit"]),
                             n_tied=int(arr["n_tied_at_max"][i]), gold_tied=bool(t["gold_tied_at_max"]),
                             digit_mass=float(arr["digit_mass"][i]), gold_prob=float(arr["gold_prob"][i]),
                             digit_logits=arr["digit_logits"][i].astype(float).tolist(),
                             top5=t.get("top5")))
pickle.dump(dict(rows=rows, runs=runs), open(OUT, "wb"))
print(len(rows), "item rows,", len(runs), "runs ->", OUT)

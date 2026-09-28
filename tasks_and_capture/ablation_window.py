#!/usr/bin/env python3
"""
Top-10% head ablation at the answer position, in fixed depth bands, for many
models: does a model whose head activity is more concentrated depend more on
its most active heads?  One model, several suites (serial_tasks.py JSON); the
model loads once.

Design, fixed before running:

  items       the suite's test split, halved by k as ablation_probe does
              (same seed, so the halves are E4's): a selection and a
              measurement half
  pass 1      selection half, one full forward per item: per-head activity
              ||z_h W_O^h|| at the measured position on every layer (the
              paper's activity), and each head's o_proj input z there.  From
              these: mean activity (heads ranked per layer), mean z (the
              mean-ablation value), and the top-q share of every layer
  heads       q = ceil(0.1 H) per layer, the paper's top-10% group, ranked by
              selection-half mean activity; random baseline: q heads per layer
              from the mid-ranked pool (activity between the layer's 25th and
              75th percentile, the top max(q, 3) excluded), ablation_probe's rule
  sets        win = the 40-80% depth window (ablation_probe's, the E4 window),
              and five depth bands b0..b4 by relative depth l/(L-1) in
              [0, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1]; every layer of a
              set is ablated jointly, whatever its response to k
  conditions  clean; per set topq/mean, topq/zero, randq/mean (--draws-win
              draws for win, --draws-band per band); for win also top1/mean
              and top3/mean (E4's sizes)
  outcome     per measurement item and condition: digit logits and the logit
              difference gold minus mean competitor (ablation_probe's
              logit_diff), at the measured position only (prefix KV cache,
              ablation_probe.Ablator: every check of E4 runs, the full
              per-item verification on each suite's first item and the
              full-forward cache check every 25th)

Analysis (--analyse), fixed before running:

  share       per model x task: top-q share of the win layers, averaged over
              selection items (all k) and layers; per band the same
  drop        per clean-correct measurement item, 1 - LD / LD_clean; averaged
              within each depth with >= MIN_CORRECT such items, then over
              those depths (depth-balanced).  Primary: win/topq/mean minus
              the mean of the win/randq/mean draws ("excess drop")
  within task across models: Spearman(share, excess drop) over the models,
              per task; a permutation null shuffles models within each task
  across tasks within a model: Spearman over its four tasks, averaged over
              models, reported beside it

    python3 ablation_window.py --model Qwen/Qwen2.5-0.5B --suites s1.json s2.json --out DIR
    python3 ablation_window.py --analyse DIR/*__abwin.npz --out REPORT_DIR
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

import ablation_probe as ap
import signature_analysis as sa
import sweep_probe as sp
from deep_probe import align_answer

DIGITS = "0123456789"
Q = 0.10
WINDOW = (0.4, 0.8)
N_BANDS = 5
SUFFIX = "__abwin.npz"
MIN_CORRECT = 5
MEASUREMENT_VERSION = 1


def top_q(H, q=Q):
    return max(1, int(math.ceil(q * H)))


def band_of(L):
    x = np.arange(L) / max(L - 1, 1)
    return np.minimum((x * N_BANDS).astype(int), N_BANDS - 1)


def window_layers(L, lo=WINDOW[0], hi=WINDOW[1]):
    return list(range(L))[sa.window(L, lo, hi)]


def top_share(S, m):
    """[..., H] -> [...]: summed share of the m largest entries."""
    S = np.asarray(S, float)
    return -np.sort(-S, -1)[..., :m].sum(-1) / np.clip(S.sum(-1), 1e-30, None)


def build_conditions(mean_activity, *, draws_win=5, draws_band=2, seed=0):
    """(conditions, sets, skipped).  Condition dicts as ablation_probe's
    (name, set, kind, size, mode, draw, layer, heads=[[layer, [heads]], ...]),
    clean first; names use 'topq' / 'randq' so they are the same for every
    model (the head count is in 'size')."""
    A = np.asarray(mean_activity, float)
    L, H = A.shape
    m = top_q(H)
    rank = ap.rank_heads(A)
    b = band_of(L)
    sets = {"win": window_layers(L)}
    sets.update({f"b{i}": [int(l) for l in np.nonzero(b == i)[0]] for i in range(N_BANDS)})
    conds = [dict(name="clean", set=None, kind="clean", size=0, mode=None, draw=None, layer=None,
                  heads=[])]
    skipped = []

    def top(s, n, label, mode):
        conds.append(dict(name=f"{s}/{label}/{mode}", set=s, kind="top", size=n, mode=mode,
                          draw=None, layer=None,
                          heads=[[l, sorted(int(h) for h in rank[l, :n])] for l in sets[s]]))

    for s, layers in sets.items():
        if not layers:
            skipped.append({"set": s, "reason": "no layer"})
            continue
        top(s, m, "topq", "mean")
        top(s, m, "topq", "zero")
        if s == "win":
            top(s, 1, "top1", "mean")
            if H > 3:
                top(s, 3, "top3", "mean")
        pool = {l: ap.mid_rank_heads(A[l], exclude_top=max(m, ap.EXCLUDE_TOP)) for l in layers}
        short = {l: len(p) for l, p in pool.items() if len(p) < m}
        if short:
            skipped.append({"set": s, "reason": f"mid-ranked pool smaller than {m}: {short}"})
            continue
        rng = ap._rng(seed, f"abwin/random/{s}/{m}")
        for r in range(draws_win if s == "win" else draws_band):
            conds.append(dict(name=f"{s}/randq/mean/r{r}", set=s, kind="rand", size=m, mode="mean",
                              draw=r, layer=None,
                              heads=[[l, sorted(int(h) for h in rng.choice(pool[l], m, replace=False))]
                                     for l in layers]))
    return conds, sets, skipped


def run_window(model, suite, *, attn_impl, split="test", seed=0, draws_win=5, draws_band=2,
               batch=None, max_items_per_k=None, log=print):
    """One model x suite; returns (arrays, meta)."""
    import torch
    t_start = time.time()
    tok = model._tok
    L, H, dh = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    name = getattr(model.cfg, "model_name", None)
    sp._check_items(suite)
    actual = getattr(model._model.config, "_attn_implementation", None)
    if actual is not None and actual != attn_impl:
        ap._abort(f"model runs {actual!r} attention but attn_impl={attn_impl!r}")
    items = suite["items"]
    sel, meas = ap.split_halves(items, split, seed, max_items_per_k)
    ap.check_competitors(items, sel + meas)
    allL = list(range(L))
    nS, nM = len(sel), len(meas)
    with ap.Ablator(model, allL, log=log) as ab:
        # ── pass 1 ──
        P1 = {"sel_index": np.array(sel, np.int32),
              "sel_k": np.array([items[i]["k"] for i in sel], np.int16),
              "sel_contribution": np.zeros((nS, L, H), np.float32),
              "sel_digit_logits": np.zeros((nS, 10), np.float32),
              "sel_correct": np.zeros(nS, bool)}
        zsum = np.zeros((L, H, dh), np.float64)
        for j, i in enumerate(sel):
            t = items[i]
            if j == 0 or (j + 1) % 25 == 0 or j + 1 == nS:
                log(f"  [select {j + 1}/{nS}] {t['id']}")
            feed, ANS, _ = align_answer(tok, t["text"])
            C, Z, lg = ab.capture(feed)
            P1["sel_contribution"][j] = C
            zsum += Z
            dlj = np.array([float(lg[ANS[d]]) for d in DIGITS], np.float32)
            P1["sel_digit_logits"][j] = dlj
            P1["sel_correct"][j] = sp._constrained(dlj, t["answer_set"], t["gold"])[1]
        P1["mean_z"] = (zsum / nS).astype(np.float32)
        P1["mean_activity"] = P1["sel_contribution"].mean(0)
        P1["head_rank"] = ap.rank_heads(P1["mean_activity"]).astype(np.int16)
        conds, sets, skipped = build_conditions(P1["mean_activity"], draws_win=draws_win,
                                                draws_band=draws_band, seed=seed)
        tables = ap.ablation_tables(conds, P1["mean_z"], allL)
        nC = len(conds)
        B = int(batch) if batch else min(nC, 32)
        log(f"  {name} | {suite['name']}: L={L} H={H} q={top_q(H)}  win layers "
            f"{sets['win'][0]}-{sets['win'][-1]}  {nC} conditions, {B} per forward"
            + (f"  skipped {skipped}" if skipped else ""))
        # ── pass 2 ──
        A = {"index": np.array(meas, np.int32),
             "k": np.array([items[i]["k"] for i in meas], np.int16),
             "gold": np.array([int(items[i]["gold"]) for i in meas], np.int8),
             "answer_mask": np.zeros((nM, 10), bool),
             "digit_logits": np.full((nM, nC, 10), np.nan, np.float32),
             "logit_diff": np.full((nM, nC), np.nan, np.float32),
             "frac": np.full((nM, nC), np.nan, np.float32),
             "clean_correct": np.zeros(nM, bool),
             "meas_contribution": np.zeros((nM, L, H), np.float32),
             "n_tokens": np.zeros(nM, np.int32)}
        for j, i in enumerate(meas):
            t = items[i]
            A["answer_mask"][j, [int(d) for d in t["answer_set"]]] = True
            if j == 0 or (j + 1) % 25 == 0 or j + 1 == nM:
                log(f"  [measure {j + 1}/{nM}] {t['id']}")
            feed, ANS, _ = align_answer(tok, t["text"])
            dl, _, act, _ = ab.measure(feed, tables, [ANS[d] for d in DIGITS], batch=B,
                                       verify=(j == 0), verify_cache=(j % ap.VERIFY_EVERY == 0))
            ld = ap.logit_diff(dl, t["gold"], list(t["answer_set"]))
            A["digit_logits"][j], A["logit_diff"][j] = dl, ld
            A["frac"][j] = ld / ld[0] if ld[0] != 0 else np.nan
            A["clean_correct"][j] = sp._constrained(dl[0], list(t["answer_set"]), t["gold"])[1]
            A["meas_contribution"][j] = act
            A["n_tokens"][j] = len(feed)
            if ab.dev.type == "cuda" and (j + 1) % 25 == 0:
                torch.cuda.empty_cache()
        checks = dict(ab.checks)
        layout = {str(l): md for l, (md, _) in ab.layout.items()}
        pname, head_checks = ab.pname, ab.head_checks
    A.update(P1)
    dtype = str(next(model._model.parameters()).dtype).replace("torch.", "")
    meta = {"kind": "ablation_window", "model": name, "suite": suite.get("name"),
            "family": suite.get("family"), "wording": suite.get("wording"),
            "setting": suite.get("setting"), "chance": suite.get("chance"),
            "suite_sha256": sp.suite_sha256(suite), "split": split, "seed": int(seed),
            "max_items_per_k": max_items_per_k, "q": Q, "q_heads": top_q(H), "window": list(WINDOW),
            "n_bands": N_BANDS, "draws_win": int(draws_win), "draws_band": int(draws_band),
            "batch_conditions": B, "attn_impl": attn_impl, "dtype": dtype,
            "n_layers": L, "n_heads": H, "d_head": dh, "sets": sets, "conditions": conds,
            "skipped": skipped, "n_selection": nS, "n_measurement": nM,
            "proj_name": pname, "head_layout": layout, "head_boundaries_checked": head_checks,
            "checks": checks, "measurement_version": MEASUREMENT_VERSION,
            "module_sha256": sp._sha256(Path(__file__).resolve()),
            "ablation_probe_sha256": ap._mod_sha(),
            "timing": {"total_s": round(time.time() - t_start, 1)}}
    return A, meta


def output_path(out_dir, model_name, suite_name):
    return Path(out_dir) / f"{model_name.replace('/', '_')}__{suite_name}{SUFFIX}"


def load_model(name, device, attn_impl, precision="bf16"):
    """ablation_probe.run's load path: native hooks, the given precision and kernel."""
    os.environ["ASD_FORCE_STRATEGY"] = "B"
    os.environ["ASD_FORCE_PRECISION"] = "bf16" if precision == "bf16" else ""
    os.environ["ASD_ATTN_IMPL"] = attn_impl
    import torch
    import active_subgraph_dot as asd
    if device.startswith("cuda") and not torch.cuda.is_available():
        ap._abort("--device cuda but CUDA is not available")
    model = asd.load_model(name, device=device)
    if not hasattr(model, "_model"):
        ap._abort("needs the _HFHookedModel wrapper (ASD_FORCE_STRATEGY=B)")
    on = sorted({p.device.type for p in model._model.parameters()})
    if device.split(":")[0] not in on:
        ap._abort(f"weights on {on}, not {device}; refusing")
    li = dict(asd.LAST_LOAD_INFO)
    want = {"bf16": "bfloat16", "fp32": "float32"}[precision]
    if li.get("strategy") != "B" or li.get("precision") != want:
        ap._abort(f"loaded with {li.get('strategy')!r} / {li.get('precision')!r}, not B / {want}")
    return model, li


def run(args, log=print):
    suites = []
    for p in args.suites:
        with open(p) as f:
            s = json.load(f)
        ap.split_halves(s["items"], args.split, args.seed, args.max_items_per_k)   # validate early
        suites.append((p, s))
    todo = [(p, s) for p, s in suites if not output_path(args.out, args.model, s["name"]).exists()]
    for p, s in suites:
        if (p, s) not in todo:
            log(f"  skip {s['name']}: done")
    if not todo:
        return []
    sp._git_commit()
    t0 = time.time()
    model, li = load_model(args.model, args.device, args.attn_impl, args.precision)
    load_s = round(time.time() - t0, 1)
    log(f"  {args.model} loaded in {load_s} s: {li}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    outs = []
    for p, s in todo:
        A, meta = run_window(model, s, attn_impl=args.attn_impl, split=args.split, seed=args.seed,
                             draws_win=args.draws_win, draws_band=args.draws_band,
                             batch=args.batch_conditions, max_items_per_k=args.max_items_per_k,
                             log=log)
        meta.update(model=args.model, suite_path=str(p), load_s=load_s, load_info=li,
                    requested_precision=args.precision,
                    next_runs_commit_env=os.environ.get("NEXT_RUNS_COMMIT"))
        out = output_path(args.out, args.model, s["name"])
        sp.save_output(out, A, meta)
        ok = A["clean_correct"]
        ci = {c["name"]: i for i, c in enumerate(meta["conditions"])}
        drop = lambda n: float(np.nanmean(1 - A["frac"][ok, ci[n]])) if n in ci and ok.any() else float("nan")
        rnd = [n for n in ci if n.startswith("win/randq/")]
        log(f"  -> {out.name}: clean correct {int(ok.sum())}/{len(ok)}; win share "
            f"{top_share(A['sel_contribution'][:, meta['sets']['win']], meta['q_heads']).mean():.3f}; "
            f"drop topq/mean {drop('win/topq/mean'):+.3f} topq/zero {drop('win/topq/zero'):+.3f} "
            f"randq {np.nanmean([drop(n) for n in rnd]) if rnd else float('nan'):+.3f}   checks: "
            f"split {meta['checks']['split_err']:.1e} delta {meta['checks']['delta_err']:.1e} "
            f"cache {meta['checks']['cache_err']:.1e} repeat {meta['checks']['repeat_err']:.1e}   "
            f"{meta['timing']['total_s']} s")
        outs.append(out)
    return outs


# ─────────────────────────────── analysis ───────────────────────────────────
def summarise(path, min_correct=MIN_CORRECT):
    A, meta = sp.load_output(path)
    ci = {c["name"]: i for i, c in enumerate(meta["conditions"])}
    ok, k = A["clean_correct"], A["k"]
    ld = A["logit_diff"].astype(float)
    depths = [int(d) for d in np.unique(k) if (ok & (k == d)).sum() >= min_correct]
    m = int(meta["q_heads"])

    def drop(cols, absolute=False):
        if not depths:
            return float("nan")
        v = (ld[:, [0]] - ld[:, cols]) if absolute else (1 - ld[:, cols] / ld[:, [0]])
        v = v.mean(1)
        return float(np.mean([v[ok & (k == d)].mean() for d in depths]))

    row = {"model": meta["model"].split("/")[-1], "task": meta["family"], "H": meta["n_heads"],
           "L": meta["n_layers"], "q_heads": m, "depths": depths,
           "n_correct": int(ok[np.isin(k, depths)].sum()), "accuracy": float(ok.mean()),
           "clean_ld": float(np.mean([ld[ok & (k == d), 0].mean() for d in depths])) if depths else float("nan")}
    for s, layers in meta["sets"].items():
        if not layers:
            continue
        sh = top_share(A["sel_contribution"][:, layers], m)
        row[f"share_{s}"] = float(sh.mean())
        row[f"share_{s}_k1"] = float(sh[A["sel_k"] == A["sel_k"].min()].mean())
        for n in ("topq/mean", "topq/zero", "top1/mean", "top3/mean"):
            if f"{s}/{n}" in ci:
                row[f"drop_{s}_{n.replace('/', '_')}"] = drop([ci[f"{s}/{n}"]])
        rc = [i for nm, i in ci.items() if nm.startswith(f"{s}/randq/")]
        if rc:
            row[f"drop_{s}_randq"] = drop(rc)
            row[f"excess_{s}"] = row[f"drop_{s}_topq_mean"] - row[f"drop_{s}_randq"]
            row[f"excess_{s}_abs"] = drop([ci[f"{s}/topq/mean"]], True) - drop(rc, True)
    return row


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return (float(spearmanr(x[ok], y[ok])[0]), int(ok.sum())) if ok.sum() >= 4 else (float("nan"), int(ok.sum()))


def _md_table(df):
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def analyse(paths, out_dir, n_perm=10000, seed=0, x="share_win", y="excess_win", min_correct=MIN_CORRECT):
    import pandas as pd
    rows = [summarise(p, min_correct) for p in paths]
    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "abwin_rows.csv", index=False)
    rng = np.random.default_rng(seed)
    res = {"x": x, "y": y, "within_task": {}, "across_tasks_within_model": {}}
    tasks = sorted(df.task.unique())
    obs = []
    for t in tasks:
        d = df[df.task == t]
        r, n = _spearman(d[x].to_numpy(float), d[y].to_numpy(float))
        res["within_task"][t] = {"rho": r, "n_models": n}
        obs.append(r)
    mean_obs = float(np.nanmean(obs))
    null = []
    for _ in range(n_perm):
        rr = []
        for t in tasks:
            d = df[df.task == t]
            rr.append(_spearman(d[x].to_numpy(float), rng.permutation(d[y].to_numpy(float)))[0])
        null.append(np.nanmean(rr))
    null = np.array(null)
    res["within_task_mean_rho"] = mean_obs
    res["within_task_p_one_sided_positive"] = float((np.sum(null >= mean_obs) + 1) / (n_perm + 1))
    per_model = {}
    for mdl, d in df.groupby("model"):
        per_model[mdl] = _spearman(d[x].to_numpy(float), d[y].to_numpy(float))[0] if len(d) >= 4 else float("nan")
    res["across_tasks_within_model"] = per_model
    res["across_tasks_within_model_mean_rho"] = float(np.nanmean(list(per_model.values())))
    json.dump(res, open(out / "abwin_report.json", "w"), indent=1)
    lines = [f"# Top-10% ablation in fixed depth bands\n",
             f"x = {x} (top-q share of the 40-80% window, selection half), y = {y} (drop of the "
             f"top-q heads minus random mid-ranked heads, fraction of clean LD, clean-correct items, "
             f"depth-balanced)\n",
             "## Within task, across models\n", "| task | rho | models |", "|---|---|---|"]
    lines += [f"| {t} | {v['rho']:+.2f} | {v['n_models']} |" for t, v in res["within_task"].items()]
    lines += [f"\nmean rho {mean_obs:+.2f}, permutation p (models shuffled within task, one-sided "
              f"positive) {res['within_task_p_one_sided_positive']:.4f}\n",
              "## Across tasks, within a model\n", "| model | rho over its tasks |", "|---|---|"]
    lines += [f"| {m} | {v:+.2f} |" for m, v in per_model.items()]
    lines += [f"\nmean {res['across_tasks_within_model_mean_rho']:+.2f}\n", "## Rows\n",
              _md_table(df[["model", "task", "H", "q_heads", "accuracy", "n_correct", "share_win",
                            "drop_win_topq_mean", "drop_win_randq", "excess_win", "drop_win_topq_zero"]])]
    (out / "abwin_report.md").write_text("\n".join(lines))
    return df, res


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--model")
    ap_.add_argument("--suites", nargs="+")
    ap_.add_argument("--out")
    ap_.add_argument("--device", default="cuda")
    ap_.add_argument("--attn-impl", default="sdpa", choices=["eager", "sdpa"])
    ap_.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--draws-win", type=int, default=5)
    ap_.add_argument("--draws-band", type=int, default=2)
    ap_.add_argument("--batch-conditions", type=int, default=None)
    ap_.add_argument("--max-items-per-k", type=int, default=None)
    ap_.add_argument("--analyse", nargs="+", metavar="NPZ")
    a = ap_.parse_args(argv)
    if a.analyse:
        df, res = analyse(a.analyse, a.out or ".")
        print(Path(a.out or ".", "abwin_report.md").read_text())
        return
    if not (a.model and a.suites and a.out):
        ap_.error("--model, --suites and --out are required (or --analyse)")
    run(a)


if __name__ == "__main__":
    main()

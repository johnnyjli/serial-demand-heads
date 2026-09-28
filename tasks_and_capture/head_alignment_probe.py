#!/usr/bin/env python3
"""
Head alignment and direct logit attribution at the answer position.

Separates two readings of the dispersion contrast (Llama spreads, Qwen funnels):

  consensus       heads that gain share as D rises write in the same direction
                  as the layer's leading head and push toward the answer, so
                  their writes add constructively
  specialisation  one head carries the answer; the others write elsewhere and
                  the leader's share of answer support grows with D

Norms alone cannot tell these apart: evenly spread norms are equally
consistent with heads writing one signal and with heads writing unrelated
ones.  This records directions.  Per item, at the measured position (the one
whose next token is the answer digit; deep_probe.align_answer), per head:

  contribution  ||z_h W_O^h||, exactly as deep_probe records it
  write         the head's write to the residual stream: z_h W_O^h, passed
                through the post-attention norm at its frozen scale on
                architectures that normalise the attention output before
                adding it (OLMo-2, Gemma-2)
  cos_top       cosine of the write with the layer's largest write
  cos_sum       cosine of the write with the layer's summed write
  dla           the write through the final norm at its frozen scale,
                projected on each digit's unembedding row          [L, H, 10]

and per layer sum_norm = ||sum of writes||, so coherence = sum_norm / sum of
write norms: 1 when every head writes one direction, ~1/sqrt(n) when unrelated.

Every linearisation is checked against the model on each forward and the run
aborts rather than emit numbers when one fails: the per-head split must
reproduce o_proj's output, the frozen post-attention norm its norm's output,
and the frozen final norm plus unembedding the model's digit logits.

    python3 head_alignment_probe.py --model Qwen/Qwen2.5-0.5B --suite serial_depth_fs \
        --out probe/align
    python3 head_alignment_probe.py --analyse probe/align/<model>_align.npz \
        [--band 0.30 0.94] [--plot align.png]
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np

from deep_probe import align_answer, _above_chance

DIGITS = "0123456789"
# max |error| / max |reference|.  bf16 rounding reaches ~2e-2 on GPT-2's large
# logits; a wrong linearisation (missing centring, (1+w) vs w) is O(1).
WARN, FAIL = 2e-2, 1e-1


# ────────────────────────────── linear pieces ───────────────────────────────
def _rel_err(pred, ref):
    return float((pred - ref).abs().max() / ref.abs().max().clamp_min(1e-9))


def fit_norm(x, y, w, b, eps, family=None):
    """Identify a norm's form from one input/output pair and return it frozen.

    Tries RMSNorm (x/rms * w), Gemma's RMSNorm (x/rms * (1+w)) and LayerNorm
    ((x-mean)/std * w + b), keeping whichever reproduces y.  Returns
    (kind, scale[d], center, bias, err): the frozen map is
    v -> (v - mean(v) if center else v) * scale, with bias added only to the
    whole output (it belongs to no head).
    """
    import torch
    x, y = x.float(), y.float()
    w = torch.ones_like(x) if w is None else w.float()
    rms = torch.sqrt((x * x).mean() + eps)
    mu = x.mean(); sd = torch.sqrt(((x - mu) ** 2).mean() + eps)
    bb = torch.zeros_like(x) if b is None else b.float()
    cands = {"rms": (w / rms, False, None), "rms1": ((1 + w) / rms, False, None),
             "ln": (w / sd, True, bb)}
    # When the residual's mean is near zero LayerNorm and RMSNorm fit one
    # input equally well, and bf16 rounding can pick the wrong one; the total
    # logit still matches but per-head attributions do not.  Use the module's
    # class to decide the family where it says.
    if family == "rms":
        cands.pop("ln")
    elif family == "rms1":
        cands = {"rms1": cands["rms1"]}
    elif family == "ln":
        cands = {"ln": cands["ln"]}
    best = None
    for kind, (g, center, bias) in cands.items():
        pred = ((x - mu) if center else x) * g + (0 if bias is None else bias)
        e = _rel_err(pred, y)
        if best is None or e < best[4]:
            best = (kind, g, center, bias, e)
    return best


def norm_family(module):
    """The norm's form from its class, for fit_norm.  Gemma's RMSNorm scales by
    (1 + w): where |w| is large, w and 1 + w fit one input equally well in bf16,
    and a per-item fit can flip between them (seen on Gemma-2-27B, layer 42), so
    the form is taken from the architecture rather than fitted."""
    cls = type(module).__name__.lower()
    if "rms" in cls:
        return "rms1" if "gemma" in cls else "rms"
    return "ln" if "layernorm" in cls else None


def _eps(m):
    for n in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(m, n, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 1e-6


def _find_final_norm(hf):
    for base in ("model", "transformer", "gpt_neox"):
        b = getattr(hf, base, None)
        for n in ("norm", "ln_f", "final_layer_norm", "final_layernorm", "final_norm"):
            m = getattr(b, n, None) if b is not None else None
            if m is not None and hasattr(m, "weight"):
                return m, f"{base}.{n}"
    raise SystemExit("final norm not found; unsupported architecture")


def _find_proj(attn):
    for n in ("o_proj", "dense", "c_proj", "out_proj", "wo"):
        p = getattr(attn, n, None)
        if p is not None and hasattr(p, "weight"):
            return p, n
    return None, None


def _patch(module, fn_factory, patched):
    at = "_old_forward" if hasattr(module, "_old_forward") else "forward"
    orig = getattr(module, at)
    setattr(module, at, fn_factory(orig)); patched.append((module, at, orig))


# ────────────────────────────── capture ─────────────────────────────────────
def capture(hf, tok, layers, get_attn, H, dh, tasks, labels, golds, log=print):
    """Run every task once and return per-item arrays.  Model-agnostic so the
    tests can drive it with small random models."""
    import torch
    L = len(layers)
    st = {"vec": {}, "post": {}, "final": None, "unemb": None}
    layout = {}                                   # layer -> ("linear"|"conv1d", err)
    ANS_IDS = []                                  # digit token ids, set per item

    def per_head_last(x, W, mode):
        Wm = W if mode == "linear" else W.t()
        Wr = Wm.reshape(Wm.shape[0], H, dh).float()
        xh = x[0, -1].reshape(H, dh).float()
        return torch.einsum("hd,ohd->ho", xh, Wr)             # [H, d_model]

    def proj_hook(l, module):
        def factory(orig):
            def f(*a, **kw):
                x = a[0] if a else next(iter(kw.values()))
                out = orig(*a, **kw)
                W = module.weight
                if x.dim() == 3 and W.device.type != "meta":
                    ref = out[0, -1].float()
                    bias = getattr(module, "bias", None)
                    bias = None if bias is None else bias.float()
                    if l not in layout:
                        best = None
                        for mode in ("linear", "conv1d"):
                            try:
                                rec = per_head_last(x, W, mode).sum(0)
                                e = _rel_err(rec + (0 if bias is None else bias), ref)
                            except Exception:
                                e = float("inf")
                            if best is None or e < best[1]:
                                best = (mode, e)
                        if best[1] > FAIL:
                            raise RuntimeError(
                                f"layer {l}: per-head split does not reproduce "
                                f"{type(module).__name__} (error {best[1]:.2e}); refusing")
                        layout[l] = best
                    v = per_head_last(x, W, layout[l][0])
                    st["vec"][l] = (v.cpu(), None if bias is None else bias.cpu(),
                                    ref.cpu())
                return out
            return f
        return factory

    def norm_hook(key, module):
        """Freeze a norm at the measured position.  key is a layer index for a
        post-attention norm, or "final"."""
        def factory(orig):
            def f(*a, **kw):
                x = a[0] if a else next(iter(kw.values()))
                out = orig(*a, **kw)
                if x.dim() == 3 and module.weight.device.type != "meta":
                    fam = norm_family(module)
                    fit = fit_norm(x[0, -1], out[0, -1], module.weight,
                                   getattr(module, "bias", None), _eps(module), fam)
                    rec = (fit[0], fit[1].cpu(), fit[2],
                           None if fit[3] is None else fit[3].cpu(), fit[4],
                           x[0, -1].float().cpu(), out[0, -1].float().cpu())
                    if key == "final":
                        st["final"] = rec
                    else:
                        st["post"][key] = rec
                return out
            return f
        return factory

    def unemb_hook(module):
        def factory(orig):
            def f(*a, **kw):
                out = orig(*a, **kw)
                if module.weight.device.type != "meta" and ANS_IDS:
                    idx = torch.tensor(ANS_IDS, device=module.weight.device)
                    b = getattr(module, "bias", None)
                    st["unemb"] = (module.weight[idx].float().cpu(),
                                   None if b is None else b[idx].float().cpu())
                return out
            return f
        return factory

    patched, pname, sandwich = [], None, []
    try:
        for l in range(L):
            o, n = _find_proj(get_attn(layers[l]))
            if o is None:
                raise SystemExit(f"layer {l}: no output projection; unsupported architecture")
            pname = pname or n
            _patch(o, proj_hook(l, o), patched)
            # OLMo-2 and Gemma-2 normalise the attention output before the
            # residual add; they are the architectures that also carry a
            # post_feedforward_layernorm.  (In Llama/Qwen, post_attention_layernorm
            # is the pre-MLP norm and must not be applied to the attention write.)
            if hasattr(layers[l], "post_feedforward_layernorm"):
                _patch(layers[l].post_attention_layernorm,
                       norm_hook(l, layers[l].post_attention_layernorm), patched)
                sandwich.append(l)
        fnorm, fname = _find_final_norm(hf)
        _patch(fnorm, norm_hook("final", fnorm), patched)
        lm = hf.get_output_embeddings()
        _patch(lm, unemb_hook(lm), patched)
        softcap = getattr(hf.config, "final_logit_softcapping", None)
        log(f"  output projection: {pname}   final norm: {fname}   "
            f"post-attention norm on {len(sandwich)}/{L} layers"
            + (f"   logit softcap {softcap}" if softcap else ""))

        n = len(tasks)
        A = {k: np.zeros((n, L, H), np.float32)
             for k in ("contribution", "write_norm", "cos_top", "cos_sum")}
        A["top_idx"] = np.zeros((n, L), np.int16)
        A["sum_norm"] = np.zeros((n, L), np.float32)
        A["dla"] = np.zeros((n, L, H, 10), np.float32)
        A["digit_logit"] = np.zeros((n, 10), np.float32)
        checks = {"post_err": 0.0, "final_err": 0.0, "logit_err": 0.0}
        items = []
        dev = next(hf.parameters()).device if not hasattr(hf, "hf_device_map") else hf.device

        for i, (text, label) in enumerate(zip(tasks, labels)):
            log(f"  [{i+1}/{n}] {label}")
            st["vec"].clear(); st["post"].clear(); st["final"] = st["unemb"] = None
            feed, ANS, appended = align_answer(tok, text)
            ANS_IDS[:] = [ANS[d] for d in DIGITS]
            ids = torch.tensor([feed], device=dev)
            with torch.no_grad():
                res = hf(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = res.logits[0, -1].float().cpu()
            del res
            if len(st["vec"]) < L or st["final"] is None or st["unemb"] is None:
                raise SystemExit(f"hooks fired on {len(st['vec'])}/{L} layers, final norm "
                                 f"{st['final'] is not None}, unembedding {st['unemb'] is not None}")

            # final norm, frozen at this position, then the digit rows
            kind, gF, centerF, biasF, eF, xF, _ = st["final"]
            WU, bU = st["unemb"]                                   # [10, d], [10]|None
            checks["final_err"] = max(checks["final_err"], eF)
            WUg = WU * gF                                          # [10, d]
            yF = ((xF - xF.mean()) if centerF else xF) * gF + (0 if biasF is None else biasF)
            lg = yF @ WU.T + (0 if bU is None else bU)
            if softcap:
                lg = softcap * torch.tanh(lg / softcap)
            ref = logits[ANS_IDS]
            checks["logit_err"] = max(checks["logit_err"], _rel_err(lg, ref))
            A["digit_logit"][i] = ref.numpy()

            for l in range(L):
                v, bias, ref_o = st["vec"][l]
                A["contribution"][i, l] = v.norm(dim=-1).numpy()
                if l in st["post"]:
                    kP, gP, cP, bP, eP, xP, yP = st["post"][l]
                    # the norm's input must be this layer's attention output
                    tot = v.sum(0) + (0 if bias is None else bias)
                    e_in = _rel_err(tot, xP)
                    checks["post_err"] = max(checks["post_err"], eP, e_in)
                    w = ((v - v.mean(-1, keepdim=True)) if cP else v) * gP
                else:
                    w = v
                wn = w.norm(dim=-1)
                s = w.sum(0)
                top = int(wn.argmax())
                A["write_norm"][i, l] = wn.numpy()
                A["top_idx"][i, l] = top
                A["sum_norm"][i, l] = float(s.norm())
                A["cos_top"][i, l] = (w @ w[top] / (wn * wn[top]).clamp_min(1e-12)).numpy()
                A["cos_sum"][i, l] = (w @ s / (wn * s.norm()).clamp_min(1e-12)).numpy()
                wc = (w - w.mean(-1, keepdim=True)) if centerF else w
                A["dla"][i, l] = (wc @ WUg.T).numpy()              # [H, 10]

            if max(checks.values()) > FAIL:
                raise SystemExit(f"linearisation check failed on item {i}: {checks}")
            probs = torch.softmax(logits, -1)
            gold = None if golds[i] is None else str(golds[i]).strip()
            pred = max(DIGITS, key=lambda d: float(probs[ANS[d]]))
            items.append({"label": label, "text": text, "gold": gold,
                          "pred_constrained": pred,
                          "correct_constrained": gold is not None and pred == gold,
                          "gold_prob": None if gold is None else float(probs[ANS[gold]]),
                          "digit_mass": float(sum(probs[a] for a in ANS.values())),
                          "appended": appended})
    finally:
        for m, at, orig in patched:
            setattr(m, at, orig)

    worst = max([e for _, e in layout.values()] + list(checks.values()))
    if worst > WARN:
        log(f"  WARNING: a linearisation check reached {worst:.1e} (warn {WARN}, abort {FAIL}); "
            f"expected from bf16 rounding only if well below {FAIL}")
    log(f"  checks (max relative error): per-head split "
        f"{max(e for _, e in layout.values()):.1e}   post-attn norm {checks['post_err']:.1e}"
        f"   final norm {checks['final_err']:.1e}   digit logits {checks['logit_err']:.1e}")
    meta = {"n_layers": L, "n_heads": H, "proj_name": pname, "sandwich_layers": sandwich,
            "final_norm": fname, "final_norm_kind": st["final"][0] if st["final"] else None,
            "softcap": softcap, "checks": checks,
            "head_layout": {str(l): m for l, (m, _) in layout.items()}, "items": items}
    return A, meta


def run(args):
    os.environ.setdefault("ASD_FORCE_STRATEGY", "B")
    os.environ.setdefault("ASD_FORCE_PRECISION", "bf16")
    os.environ.setdefault("ASD_ATTN_IMPL", args.attn_impl)
    import active_subgraph_dot as asd
    from active_subgraph_dot import load_model, TASK_SUITES

    model = load_model(args.model, device=args.device, hf_token=args.hf_token)
    if not hasattr(model, "_model"):
        raise SystemExit("needs the _HFHookedModel wrapper; set ASD_FORCE_STRATEGY=B")
    suite = TASK_SUITES[args.suite]
    tasks, labels = suite["tasks"], suite["labels"]
    golds = suite.get("_answers", [None] * len(tasks))
    if args.limit:
        tasks, labels, golds = tasks[:args.limit], labels[:args.limit], golds[:args.limit]
    print(f"  n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads}  "
          f"load_info={asd.LAST_LOAD_INFO}")
    A, meta = capture(model._model, model._tok, model._layers(), model._get_attn,
                      model.cfg.n_heads, model.cfg.d_head, tasks, labels, golds)
    meta.update(model=args.model, suite=args.suite, load_info=dict(asd.LAST_LOAD_INFO))
    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    f = p / f"{args.model.replace('/', '_')}_align.npz"
    np.savez_compressed(f, meta=np.array(json.dumps(meta)), **A)
    acc = np.mean([t["correct_constrained"] for t in meta["items"]])
    print(f"  digit-constrained accuracy {acc:.2f}   -> {f}  ({f.stat().st_size/1e6:.1f} MB)")


# ───────────────────────────── analysis ─────────────────────────────────────
# Per item and layer, from the arrays above (s = write norms, m = answer margin):
#   PR          participation ratio of s / H           spread of activity
#   top_share   max s / sum s                          dominance of the leader
#   coherence   ||sum w|| / sum s                      do the writes add up?  Falls
#               as ~1/sqrt(PR) when activity spreads even at fixed directions,
#               so it is descriptive only; pair_cos is the test.
#   pair_cos    s-weighted mean cosine between distinct heads' writes, exact:
#               (||sum w||^2 - sum s^2) / ((sum s)^2 - sum s^2)
#   align       s-weighted cos_top over non-top heads  do others follow the leader?
#   margin      sum_h m_h, m_h = dla[gold] - mean dla[competing states]
#   support_PR  PR of the positive m_h / H             how many heads carry the answer
#   support_top max m+ / sum m+                        does one head carry it?
#   agreement   sum m+ / sum |m|                       do heads agree on the answer?
PREDICT = {   # sign of d(metric)/dD inside the band under each reading
    "PR":          ("+", "0/-"), "top_share":  ("-", "+"),
    "coherence":   ("+", "0"),   "pair_cos":   ("+", "0"),
    "align":       ("+", "0"),
    "support_PR":  ("+", "-"),   "support_top": ("-", "+"),
    "agreement":   ("+", "0"),   "margin":     ("", ""),
}


def _pr(x):
    s, s2 = x.sum(-1), (x * x).sum(-1)
    return np.where(s2 > 0, s * s / np.where(s2 > 0, s2, 1), 0) / x.shape[-1]


def load(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    A = {k: z[k] for k in z.files if k != "meta"}
    return A, meta


def layer_metrics(A, meta):
    items = meta["items"]
    s = A["write_norm"].astype(float)                        # [n, L, H]
    n, L, H = s.shape
    gold = np.array([int(t["gold"]) for t in items])
    # competitors: the states named in the item's final rule table
    comp = np.zeros((n, 10), bool)
    for i, t in enumerate(items):
        states = {int(d) for d in re.findall(r"(\d) goes to", t["text"].split("Rules:")[-1])}
        comp[i, sorted(states - {gold[i]})] = True
    dla = A["dla"].astype(float)                             # [n, L, H, 10]
    g = np.take_along_axis(dla, gold[:, None, None, None], -1)[..., 0]
    o = (dla * comp[:, None, None, :]).sum(-1) / comp.sum(1)[:, None, None]
    m = g - o                                                # [n, L, H]
    mp = np.clip(m, 0, None)
    top = A["top_idx"].astype(int)
    notop = np.ones_like(s, bool)
    np.put_along_axis(notop, top[..., None], False, -1)
    sw = s * notop
    M = dict(
        PR=_pr(s),
        top_share=s.max(-1) / np.clip(s.sum(-1), 1e-12, None),
        coherence=A["sum_norm"] / np.clip(s.sum(-1), 1e-12, None),
        pair_cos=(A["sum_norm"] ** 2 - (s * s).sum(-1))
                 / np.clip(s.sum(-1) ** 2 - (s * s).sum(-1), 1e-12, None),
        align=(sw * A["cos_top"]).sum(-1) / np.clip(sw.sum(-1), 1e-12, None),
        margin=m.sum(-1),
        support_PR=_pr(mp),
        support_top=mp.max(-1) / np.clip(mp.sum(-1), 1e-12, None),
        agreement=mp.sum(-1) / np.clip(np.abs(m).sum(-1), 1e-12, None),
    )
    return M, m, s


def _slopes(Y, D, rng, nperm):
    """OLS slope of each column of Y on D, and a joint permutation null
    (one shuffle of D applied to every column)."""
    dc = D - D.mean(); ss = dc @ dc; Yc = Y - Y.mean(0)
    obs = dc @ Yc / ss
    P = np.stack([rng.permutation(dc) for _ in range(nperm)])
    return obs, P @ Yc / ss


def _p(obs, null):
    return float(((np.abs(null) >= abs(obs) - 1e-15).sum() + 1) / (len(null) + 1))


def analyse_one(path, band=None, correct_only=False, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    A, meta = load(path)
    items = meta["items"]
    L, H = meta["n_layers"], meta["n_heads"]
    D = np.array([int(re.match(r"D(\d+)", t["label"]).group(1)) for t in items], float)
    ok = np.array([t["correct_constrained"] for t in items])
    nstates = len(set(re.findall(r"(\d) goes to", items[0]["text"].split("Rules:")[-1])))
    M, m, s = layer_metrics(A, meta)

    print(f"\n{'#'*78}\n# {meta['model']}   suite={meta['suite']}   L={L} H={H}\n{'#'*78}")
    print(f"  checks: {meta['checks']}   post-attn norm layers: {len(meta['sandwich_layers'])}")
    print("  accuracy by D: " + "  ".join(
        f"D{int(k)} {ok[D == k].mean():.2f}" for k in np.unique(D)) + f"   chance {1/nstates:.2f}")
    sel = np.ones(len(D), bool)
    if correct_only:
        above = _above_chance(D.astype(int), ok, 1 / nstates)
        sel = ok & np.isin(D, above)
        print(f"  correct items at above-chance depths {above}: n={sel.sum()}")
        if len(above) < 3 or sel.sum() < 12:
            print("  not enough correct items for a depth slope"); return None

    if band:
        lo, hi = band
        inb = np.zeros(L, bool); inb[int(lo * (L - 1)):int(hi * (L - 1)) + 1] = True
        regions = [(f"band {lo:.2f}-{hi:.2f}", inb), ("outside", ~inb)]
    else:
        regions = [(f"{20*q}-{20*q+20}%", (np.arange(L) >= q * L / 5) & (np.arange(L) < (q + 1) * L / 5))
                   for q in range(5)]

    res = {"model": meta["model"], "L": L, "slopes": {}}
    print(f"\n  change per extra step, mean over layers in each region (p: joint D-permutation)")
    hdr = "".join(f"{r:>22}" for r, _ in regions)
    print(f"  {'metric':<12}{'consensus':>10}{'special.':>9}{hdr}")
    for k, Y in M.items():
        obs, null = _slopes(Y[sel], D[sel], rng, nperm)
        res["slopes"][k] = obs
        cells = []
        for _, mask in regions:
            o = obs[mask].mean()
            cells.append(f"{o:>+11.2e} (p={_p(o, null[:, mask].mean(1)):.3f})")
        c, sp = PREDICT[k]
        print(f"  {k:<12}{c:>10}{sp:>9}" + "".join(f"{x:>22}" for x in cells))

    # head level: do heads that gain share gain answer support and follow the leader?
    # In the band if given, else 40-80% depth, where dispersion rises in every
    # model measured so far (the first quintile is not where the question is).
    if band:
        hl_name, mask = regions[0]
    else:
        mask = np.zeros(L, bool); mask[int(.4 * (L - 1)):int(.8 * (L - 1)) + 1] = True
        hl_name = "40-80%"
    share = s / np.clip(s.sum(-1, keepdims=True), 1e-12, None)
    Ys = share[sel][:, mask].reshape(sel.sum(), -1)
    Ym = m[sel][:, mask].reshape(sel.sum(), -1)
    ct = A["cos_top"][sel][:, mask].reshape(sel.sum(), -1).astype(float)
    isnt = (A["top_idx"][sel][:, mask][..., None] != np.arange(H)).reshape(sel.sum(), -1)
    cnt = isnt.sum(0)
    ct_mean = np.where(cnt > 0, (ct * isnt).sum(0) / np.maximum(cnt, 1), np.nan)
    Dsel = D[sel]
    dc = Dsel - Dsel.mean(); ss = dc @ dc
    ds, dm = dc @ (Ys - Ys.mean(0)) / ss, dc @ (Ym - Ym.mean(0)) / ss
    base_m = Ym.mean(0)
    ok_h = np.isfinite(ct_mean)
    r_sm = float(np.corrcoef(ds, dm)[0, 1])
    r_sc = float(np.corrcoef(ds[ok_h], ct_mean[ok_h])[0, 1])
    r_sb = float(np.corrcoef(ds, base_m)[0, 1])
    null_sm, null_sc, null_sb = [], [], []
    for _ in range(min(nperm, 1000)):
        pc = rng.permutation(dc)
        a, b = pc @ (Ys - Ys.mean(0)) / ss, pc @ (Ym - Ym.mean(0)) / ss
        null_sm.append(np.corrcoef(a, b)[0, 1])
        null_sc.append(np.corrcoef(a[ok_h], ct_mean[ok_h])[0, 1])
        null_sb.append(np.corrcoef(a, base_m)[0, 1])
    print(f"\n  head level, {hl_name} ({int(mask.sum())} layers x {H} heads):")
    print(f"    r(d share/dD, d margin/dD)   = {r_sm:+.3f}  (p={_p(r_sm, np.array(null_sm)):.3f})"
          f"   consensus +, specialisation 0")
    print(f"    r(d share/dD, mean cos_top)  = {r_sc:+.3f}  (p={_p(r_sc, np.array(null_sc)):.3f})"
          f"   consensus + (gainers follow the leader)")
    print(f"    r(d share/dD, mean margin)   = {r_sb:+.3f}  (p={_p(r_sb, np.array(null_sb)):.3f})"
          f"   consensus + (gainers support the answer)")
    res.update(head=dict(r_share_margin=r_sm, r_share_cos=r_sc, r_share_base=r_sb))
    return res


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["PR", "pair_cos", "align", "support_PR", "support_top", "margin"]
    fig, ax = plt.subplots(len(results), len(keys), figsize=(2.9 * len(keys), 2.4 * len(results)),
                           squeeze=False, constrained_layout=True)
    for i, r in enumerate(results):
        xs = np.linspace(0, 1, r["L"])
        for j, k in enumerate(keys):
            a = ax[i, j]
            a.plot(xs, r["slopes"][k], lw=1.2); a.axhline(0, c="k", lw=.5)
            a.set_title(f"{r['model'].split('/')[-1]}: {k}" if j == 0 else k, fontsize=8)
            a.tick_params(labelsize=6)
    for a in ax[-1]:
        a.set_xlabel("relative depth", fontsize=7)
    fig.suptitle("change per extra step, by layer", fontsize=10)
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--suite", default="serial_depth_fs")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--hf_token", default=None)
    ap.add_argument("--out", default="probe/align")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    ap.add_argument("--analyse", nargs="+", metavar="NPZ")
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    help="relative depths, e.g. --band 0.30 0.94; default: quintiles")
    ap.add_argument("--correct-only", action="store_true",
                    help="correct items at above-chance depths only")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--plot")
    a = ap.parse_args()
    if a.analyse:
        rs = [r for r in (analyse_one(p, a.band, a.correct_only, a.nperm) for p in a.analyse) if r]
        if a.plot and rs:
            plot(rs, a.plot)
    elif a.model:
        run(a)
    else:
        ap.error("--model or --analyse required")

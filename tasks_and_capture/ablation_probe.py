#!/usr/bin/env python3
"""
Head ablation at the measured position: E4 of paper_manuscript/research_outline.tex
(RQ2, what the signature means), on one suite (serial_tasks.py JSON) per run.

Design, fixed before running (outline E4):

  items         the chosen split of the suite, divided 50/50 into a selection
                and a measurement half, stratified by k and seeded (at each
                depth the selection half gets floor(n/2) items)
  pass 1        selection half, one full forward per item.  At the measured
                position (the one whose next token is the answer digit;
                deep_probe.align_answer): per-head activity ||z_h W_O^h|| on
                every layer (the paper's `contribution`) and the heads' o_proj
                inputs z on the window layers (40-80% depth by default).  From
                these: the per-head mean z (the mean-ablation value), the mean
                activity (heads ranked per layer), and the layer classes:
                signature_analysis.classify_layers on per-layer top share, -1
                spreading / +1 funnelling, BH over all layers of the model and
                task as in the outline's rule; the target sets are the
                classified layers inside the window.
  pass 2        measurement half.  The prefix (every token but the last) runs
                once with a KV cache; each condition then runs only the last
                token against that cache, which is cropped back to the prefix
                afterwards (DynamicCache.crop).  Ablating a head at the measured
                position cannot change earlier positions, so this is exact.
  conditions    clean; for each set (spread, funnel) that has layers: the top 1
                and the top 3 heads of every layer of the set by selection-half
                activity, ablated jointly across the set, by mean- and by
                zero-ablation; a random baseline of 1 or 3 heads per layer drawn
                from the mid-ranked heads (activity between the layer's 25th
                and 75th percentile, top 3 excluded), R draws, mean-ablation
                only; with --per-layer also the top-1 mean-ablation of one layer
                at a time.  An empty set is skipped and reported, as is a
                random baseline a layer has too few mid-ranked heads for.
  intervention  the head's slice of the o_proj input at the last position is
                replaced by its selection-half mean (mean) or by zero (zero),
                inside the patched o_proj forward (_old_forward under
                accelerate offload, else forward, as deep_probe patches it)
  outcome       per item and condition: logit difference = gold logit minus
                the mean logit of the competitors (answer_set minus gold),
                absolute and as a fraction of clean; clean-correct items
                (digit-constrained, gold the unique maximum over answer_set, as
                sweep_probe scores it) are flagged

Verification (the run aborts rather than write numbers):
  - n_heads / head_dim against the model config and attention modules, and the
    per-head split against the output projection on every captured forward
    (sweep_probe's checks: exact fp32 map, module output within kernel
    rounding, Linear or GPT-2 Conv1D layout)
  - every output projection runs exactly once per forward; model in eval mode;
    no NaN / inf logits; no layer whose heads all write zero at the measured
    position of a selection item (its top share, and so its class, is
    undefined)
  - the cache holds exactly the prefix after every crop, and the first
    condition group re-run at the end of every item reproduces its first run
  - on the first measured item of a session and every VERIFY_EVERY-th after:
    the clean last-token logits through the cache reproduce a full forward
    (max relative error <= 1e-2)
  - on the first measured item of a session: replacing every head of every
    window layer with the item's own z reproduces clean; zero-ablating each
    window layer's largest head changes the logits; for every ablation on that
    item the o_proj input equals the intended one, rebuilt from the condition
    table (its heads and values, per-head column slices) bit for bit, so a
    replacement that hits another head, row or position aborts; and each
    o_proj output changes by exactly the listed heads' writes predicted from
    that intended input (twice the kernel tolerance, as two rounded outputs are
    compared)

Two run options leave the design unchanged.  --batch-conditions B runs B
conditions per forward (the prefix cache is repeated B times and cropped as
above; chunks are padded with clean rows, which must reproduce the clean row):
under CPU offload (70B / 72B on one A100) every forward streams the offloaded
weights once, so B conditions cost about one forward.  Per measured item a run
makes 1 prefix forward, ceil(C/B) condition forwards and 1 repeat of the first
chunk (plus the checks); at the 70B pair with --per-layer C is about 46, so the
default is B = 16 on cuda (5 forwards per item instead of 48) and 1 on the CPU.
Rows at different batch positions may round differently in bf16, so B is
recorded in the output.  A checkpoint next to the output (<output>.ckpt, not
.npz, so no '*.npz' glob picks it up) holds pass 1, the finished pass-2 items
and the weight layout; a disconnected Colab session resumes from it when every
setting and this file's version match, and aborts otherwise.

    python3 ablation_probe.py --model meta-llama/Llama-3.1-70B --suite suites/pointer_w0.json \\
        --out runs/e4 --device cuda --attn-impl eager --window 0.4 0.8 --split test \\
        --random-draws 5 --seed 0 --per-layer
    python3 ablation_probe.py --analyse runs/e4/meta-llama_Llama-3.1-70B__S__ablation.npz \\
        runs/e4/Qwen_Qwen2.5-72B__S__ablation.npz --out runs/e4/report

Output: OUT/<model with / -> _>__<suite name>__ablation.npz (arrays plus `meta`,
a JSON string with kind = "ablation", the design, the conditions and their
heads, provenance and checks); skipped when it exists with the same settings,
refused when it exists with others.  The per-item records are
meta["measurement_items"] and the measurement half's activity is
`meas_contribution`, never `items` / `contribution`: signature_analysis reads
those, and would otherwise take the half-suite ablation file for a capture of
the model (signature_analysis.load_run rejects these files, so a glob over
all outputs is safe, but keeping them apart keeps its report clean).

--analyse writes ablation_report.md and ablation_report.json: per model,
condition and k the mean fractional drop 1 - LD/LD_clean with bootstrap CIs on
clean-correct measurement items, the sub-additivity ratio top3 / (3 top1), the
excess over the random baseline, the slope of the drop on k, and the same slope
on the absolute scale (LD_clean - LD) with the clean LD by k: the clean LD
falls with k, so a fractional slope without an absolute one is flagged.  Slope
p-values come from a studentised permutation test (slope / HC2 SE, k
shuffled; slope_test): fewer items are correct at high k and the fraction is
noisier there, and a plain k permutation then rejects a zero slope far more
often than alpha.  For two different models on the same suite (the only
combination compared; any other is listed as not compared) the ablation x k x
model interaction with a bootstrap 90% CI and the outline's equivalence rule (a
null only if the CI lies within +/-0.10 of clean LD; a CI outside that
excluding 0 is a difference; anything else inconclusive), plus the
absolute-scale interaction.  Each model's layer classes come from its own
selection half, so a condition the two files hold only for different classes
(Llama spreading, Qwen funnelling) is compared across them, each model's own
target layers ('spread-vs-funnel/top1/mean'); a pair with no comparable
condition gets a note instead of an empty table, and a pair run with another
design (window, dtype, kernel, ...) or on other items gets a warning.  Glob
patterns in --analyse are expanded here (quote them for Drive paths).  Depths: the
eligible ones (outline rule on both halves' clean correctness) when there are
>= 3, else the depths solved above chance, reported as exploratory (the
outline's fallback); either way a depth needs --min-items clean-correct
measurement items, and depths below 20 analysed items are flagged.  Random
draws are pooled per item.  Every group's permutations and every pair's
bootstrap come from generators of their own, so a group's numbers do not
depend on the other groups in the file.
"""
import argparse, glob, hashlib, json, math, os, sys, time, zlib
from pathlib import Path
import numpy as np

from deep_probe import align_answer, _above_chance
from head_alignment_probe import _find_proj, _rel_err
import signature_analysis as sa
import sweep_probe as sp

DIGITS = "0123456789"
# Tolerances, max |error| / max |reference| (the repository's convention).
CACHE_TOL = 1e-2      # cached last-token logits vs a full forward (spec).  fp32
                      # reaches ~1e-6; a model that ignores the cache is O(1)
CACHE_TOL_BF16 = 0.25 # bf16: the cached and full forwards sum in a different
                      # order, and on 70B models with 400-token prompts they differ
                      # by a few percent (Llama-3.1-70B 3.8e-2, Qwen2.5-72B 6.6e-2,
                      # E4 Sep 24); a broken cache (wrong positions, stale rows)
                      # is still O(1).  Every condition, clean included, runs
                      # through the cached path, so comparisons stay consistent.
                      # Errors above CACHE_TOL are recorded and reported.
IDENT_TOL = 1e-3      # the same computation twice at the same batch position
                      # (repeat, own-z replacement): 0 on deterministic kernels;
                      # a stale cache or a misplaced slice is far larger
EXACT_TOL = sp.EXACT_TOL
VERIFY_EVERY = 25     # full-forward cache check: first item of a session, then every N
SETS = {"spread": -1, "funnel": 1}          # signature_analysis.classify_layers codes
TOP_SIZES = (1, 3)
MODES = ("mean", "zero")
MID_PCT = (25.0, 75.0)
EXCLUDE_TOP = 3
EQUIV = 0.10          # outline: a null needs the 90% CI within +/-10% of clean LD
SLOPE_TEST = "studentised permutation (slope / HC2 SE, k shuffled)"
# settings a pair's two files should share (the design is fixed before running
# and applies to both models); a mismatch is flagged in the interaction
PAIR_DESIGN_KEYS = ("window", "split", "max_items_per_k", "seed", "random_draws", "n_perm",
                    "q", "attn_impl", "dtype", "requested_precision")
# condition kinds paired across layer classes when the pair's classes differ
CROSS_KINDS = ("top", "rand")
# --batch-conditions default on CUDA.  Per measured item a run makes 1 prefix
# forward + ceil(C/B) condition forwards + 1 repeat of the first chunk (+ 1
# full forward on cache-check items, + 2 on the first item).  At the 70B pair
# (window 31-63, 5 draws, --per-layer) C is about 46: B = 1 is 48 forwards per
# item, each streaming the CPU-offloaded weights (~5 s), B = 16 is 5.  The KV
# cache grows B-fold (Llama-3.1-70B: 328 KB/token x B, ~2 GB at 400 tokens).
BATCH_CUDA = 16
SUFFIX = "__ablation.npz"
# settings that define an output; one written with any other value is not "done"
RESUME_KEYS = ("model", "suite", "suite_sha256", "split", "window", "max_items_per_k",
               "random_draws", "seed", "per_layer", "n_perm", "q", "batch_conditions",
               "attn_impl", "requested_precision", "measurement_version")
# An output or checkpoint is "done" only if measured the same way.
# MEASUREMENT_VERSION is bumped by any change that can alter the numbers; edits
# that cannot (a tolerance of a check, messages) keep it.  Files written before
# the field existed are mapped from their module_sha256.
MEASUREMENT_VERSION = 1
_COMPATIBLE_MODULES = {
    # Sep 24 (first E4 runs); differs from version 1 only in the bf16 cache-check tolerance
    "5ee967dee55448ca24ccf134407a04ba8118d0f49db1730462156c75b8c17585": 1,
}


def _meas_version(m):
    v = m.get("measurement_version")
    return v if v is not None else _COMPATIBLE_MODULES.get(m.get("module_sha256"))


def _abort(msg):
    raise SystemExit(msg)


def _rng(seed, tag):
    """A generator that depends only on the seed and what it is for, so the
    halves, the classification and each random baseline have their own
    streams (adding draws or conditions never moves the others)."""
    return np.random.default_rng([int(seed) % (1 << 63), zlib.crc32(str(tag).encode())])


def _mod_sha():
    return sp._sha256(Path(__file__).resolve())


# ───────────────────────────── design pieces ────────────────────────────────
def split_halves(items, split="test", seed=0, max_items_per_k=None):
    """(selection, measurement) suite indices: the items of `split`, per depth
    shuffled by a generator seeded with `seed`, capped at max_items_per_k,
    then halved (selection floor(n/2), measurement the rest).  Both lists are
    sorted.  Aborts when a depth has fewer than 2 items or there is only one
    depth (the design needs k to vary in both halves)."""
    idx = sp._select(items, split)
    byk = {}
    for i in idx:
        byk.setdefault(int(items[i]["k"]), []).append(i)
    if len(byk) < 2:
        _abort(f"split {split!r} has one depth ({sorted(byk)}); the ablation x k design "
               f"needs at least two")
    if max_items_per_k is not None and int(max_items_per_k) < 2:
        _abort("max_items_per_k must be >= 2 (each depth is split in two)")
    rng = _rng(seed, "halves")
    sel, meas = [], []
    for k in sorted(byk):
        ix = np.asarray(byk[k])[rng.permutation(len(byk[k]))]
        if max_items_per_k is not None:
            ix = ix[:int(max_items_per_k)]
        if len(ix) < 2:
            _abort(f"k={k}: {len(ix)} item in split {split!r}; each depth needs at least 2 "
                   f"to be divided into selection and measurement halves")
        h = len(ix) // 2
        sel += ix[:h].tolist()
        meas += ix[h:].tolist()
    return sorted(int(i) for i in sel), sorted(int(i) for i in meas)


def check_competitors(items, idx):
    """Abort unless every item in idx has a competitor (answer_set minus gold):
    the outcome, gold minus the mean competitor logit, is undefined without
    one.  sweep_probe._check_items accepts answer_set == [gold], so this runs
    before the model is loaded and before pass 1."""
    bad = [items[i]["id"] for i in idx
           if not [d for d in items[i]["answer_set"] if str(d) != str(items[i]["gold"])]]
    if bad:
        _abort(f"{len(bad)} item(s) have no competitor in their answer_set (e.g. {bad[:3]}); "
               f"the logit difference needs at least one admissible answer besides gold")


def rank_heads(mean_activity):
    """[L, H] head indices per layer by mean activity, largest first (ties to
    the lower index)."""
    return np.argsort(-np.asarray(mean_activity, float), axis=-1, kind="stable")


def mid_rank_heads(activity, lo=MID_PCT[0], hi=MID_PCT[1], exclude_top=EXCLUDE_TOP):
    """Heads of one layer whose mean activity lies between its lo-th and hi-th
    percentile (numpy's linear interpolation, inclusive), minus the
    exclude_top most active heads: the random baseline's pool.  Excluding the
    top heads matters when H is small (at H=4 the percentile band holds the
    2nd and 3rd heads)."""
    a = np.asarray(activity, float)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    top = set(np.argsort(-a, kind="stable")[:exclude_top].tolist())
    return np.array([h for h in range(len(a)) if p_lo <= a[h] <= p_hi and h not in top], int)


def build_conditions(layer_class, window_layers, mean_activity, *, random_draws=5, seed=0,
                     per_layer=False):
    """(conditions, skipped, sets).

    conditions: list of dicts {name, set, kind (clean|top|rand|top_layer),
    size, mode (mean|zero), draw, layer, heads: [[layer, [head, ...]], ...]},
    clean first.  sets: {"spread": [layers], "funnel": [layers]} (window
    layers classed -1 / +1).  Random draws for (set, size) come from their own
    generator seeded by (seed, set, size), so adding draws keeps earlier ones."""
    lc = np.asarray(layer_class, int)
    A = np.asarray(mean_activity, float)
    L, H = A.shape
    if lc.shape != (L,):
        raise ValueError(f"layer_class {lc.shape} does not match activity {A.shape}")
    rank = rank_heads(A)
    wl = [int(l) for l in window_layers]
    conds = [dict(name="clean", set=None, kind="clean", size=0, mode=None, draw=None,
                  layer=None, heads=[])]
    skipped, sets = [], {}
    for s, code in SETS.items():
        layers = [l for l in wl if lc[l] == code]
        sets[s] = layers
        if not layers:
            skipped.append({"set": s, "condition": f"{s}/*",
                            "reason": f"no {s} layer in the window (layers {wl[0]}-{wl[-1]})"})
            continue
        for n in TOP_SIZES:
            if n > H:
                skipped.append({"set": s, "condition": f"{s}/top{n}/*",
                                "reason": f"only {H} heads per layer"})
                continue
            for mode in MODES:
                conds.append(dict(name=f"{s}/top{n}/{mode}", set=s, kind="top", size=n,
                                  mode=mode, draw=None, layer=None,
                                  heads=[[l, sorted(int(h) for h in rank[l, :n])] for l in layers]))
        for n in TOP_SIZES:
            if random_draws < 1:
                break
            pool = {l: mid_rank_heads(A[l]) for l in layers}
            short = {l: len(p) for l, p in pool.items() if len(p) < n}
            if short:
                skipped.append({"set": s, "condition": f"{s}/rand{n}/mean",
                                "reason": f"layers with fewer than {n} mid-ranked heads "
                                          f"(25th-75th percentile, top {EXCLUDE_TOP} "
                                          f"excluded): {short}"})
                continue
            rng = _rng(seed, f"random/{s}/{n}")
            for r in range(int(random_draws)):
                conds.append(dict(name=f"{s}/rand{n}/mean/r{r}", set=s, kind="rand", size=n,
                                  mode="mean", draw=r, layer=None,
                                  heads=[[l, sorted(int(h) for h in
                                                    rng.choice(pool[l], n, replace=False))]
                                         for l in layers]))
        if per_layer:
            for l in layers:
                conds.append(dict(name=f"{s}/top1/mean/L{l}", set=s, kind="top_layer", size=1,
                                  mode="mean", draw=None, layer=l, heads=[[l, [int(rank[l, 0])]]]))
    return conds, skipped, sets


def ablation_tables(conditions, mean_z, window_layers):
    """Per condition {layer: ([heads], values [n, d_head] float32 tensor, or
    None for zero-ablation)}; the clean condition is {}."""
    import torch
    wi = {int(l): i for i, l in enumerate(window_layers)}
    mz = np.asarray(mean_z, np.float32)
    out = []
    for c in conditions:
        t = {}
        for l, heads in c["heads"]:
            if int(l) not in wi:
                raise ValueError(f"{c['name']}: layer {l} is not a window layer")
            heads = [int(h) for h in heads]
            vals = (None if c["mode"] == "zero" else
                    torch.tensor(mz[wi[int(l)]][heads], dtype=torch.float32))
            t[int(l)] = (heads, vals)
        out.append(t)
    return out


def logit_diff(digit_logits, gold, answer_set):
    """Gold logit minus the mean logit of the competitors (answer_set - gold);
    digit_logits [..., 10] indexed by digit."""
    dl = np.asarray(digit_logits, float)
    comp = [int(d) for d in answer_set if str(d) != str(gold)]
    if not comp:
        raise ValueError(f"answer_set {answer_set} has no competitor for gold {gold}")
    return dl[..., int(gold)] - dl[..., comp].mean(-1)


# ─────────────────────────────── cache ──────────────────────────────────────
def _as_cache(pkv):
    if pkv is None:
        _abort("the model returned no KV cache (use_cache=True); cannot run the last token "
               "against the prefix")
    if hasattr(pkv, "crop") and hasattr(pkv, "get_seq_length"):
        return pkv
    try:                                     # legacy tuple caches (older transformers)
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(pkv)
    except Exception as e:
        _abort(f"KV cache of type {type(pkv).__name__} cannot be cropped ({e})")


def _crop(cache, P):
    try:
        cache.crop(P)
    except Exception as e:
        _abort(f"cannot crop the KV cache back to the prefix ({type(e).__name__}: {e}); "
               f"sliding-window or static caches are not supported")
    n = int(cache.get_seq_length())
    if n != P:
        _abort(f"KV cache holds {n} positions after cropping, not the prefix's {P}; "
               f"conditions would see each other's last token")


# ─────────────────────────────── hooks ──────────────────────────────────────
class Ablator:
    """The model's output projections, patched: capture per-head activity and
    head inputs z at the last position, and replace head inputs there.

    Use as a context manager (patches removed on exit).  `model` is an
    _HFHookedModel-like wrapper (.cfg.n_layers/.n_heads/.d_head, ._layers(),
    ._get_attn(layer), ._model, ._tok).  Model-agnostic so the tests can drive
    it with small random models."""

    def __init__(self, model, window_layers, log=print):
        import torch
        self.torch, self.model, self.log = torch, model, log
        self.hf, self.tok = model._model, model._tok
        self.L, self.H, self.dh = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
        self.layers = model._layers()
        if len(self.layers) != self.L:
            _abort(f"config says {self.L} layers, model has {len(self.layers)}")
        self.window = [int(l) for l in window_layers]
        self.wset = set(self.window)
        self.dev = sp._input_device(self.hf)
        _dt = next(self.hf.parameters()).dtype
        self.cache_tol = CACHE_TOL if _dt == torch.float32 else CACHE_TOL_BF16
        self.layout, self.patched, self.pname, self.head_checks = {}, [], None, []
        self.checks = {"split_err": 0.0, "split_exact_err": 0.0, "split_tol": None,
                       "delta_err": 0.0, "delta_checks": 0, "input_checks": 0, "cache_err": 0.0,
                       "cache_checks": 0, "ownz_err": 0.0, "pad_err": 0.0,
                       "repeat_err": 0.0, "items": 0, "cache_tol": self.cache_tol}
        self._clear()

    def _clear(self):
        self.capture_on, self.check_delta = False, False
        self.ablate = {}       # layer -> [(row, [heads], values or None)]
        self.calls, self.contrib, self.z = {}, {}, {}

    # ── patch management ──
    def __enter__(self):
        torch = self.torch
        training = [n or type(self.hf).__name__ for n, m in self.hf.named_modules() if m.training]
        if training:
            _abort(f"model is in train mode ({len(training)} modules, e.g. {training[0]!r}): "
                   f"dropout would randomise every number; call .eval() first")
        self.head_checks = sp._check_heads(self.model, self.layers, self.H, self.dh)
        try:
            for l in range(self.L):
                o, nm = _find_proj(self.model._get_attn(self.layers[l]))
                if o is None:
                    _abort(f"layer {l}: no output projection; unsupported architecture")
                self.pname = self.pname or nm
                sp._patch(o, self._factory(l, o), self.patched)
        except BaseException:
            sp._unpatch(self.patched)
            raise
        # the exact split check compares fp32 products at EXACT_TOL; TF32 would
        # round both to 10 mantissa bits (restored on exit)
        self._mm = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        return self

    def __exit__(self, *exc):
        sp._unpatch(self.patched)
        self.torch.set_float32_matmul_precision(self._mm)
        self._clear()
        return False

    def _factory(self, l, module):
        torch, H, dh = self.torch, self.H, self.dh

        def factory(orig):
            def f(*a, **kw):
                self.calls[l] = self.calls.get(l, 0) + 1
                if a:
                    x = a[0]
                    call = lambda t: orig(t, *a[1:], **kw)
                else:
                    key = next(iter(kw))
                    x = kw[key]
                    call = lambda t: orig(**{**kw, key: t})
                rows = self.ablate.get(l)
                if rows:
                    if x.dim() != 3 or x.shape[-1] != H * dh:
                        _abort(f"layer {l}: o_proj input of shape {tuple(x.shape)}, expected "
                               f"[batch, seq, {H}*{dh}]; cannot locate head slices")
                    xm = x.clone(memory_format=torch.contiguous_format)
                    for row, heads, vals in rows:
                        v = xm[row, -1].view(H, dh)
                        if vals is None:
                            v[heads] = 0
                        else:
                            v[heads] = vals.to(device=x.device, dtype=x.dtype)
                    out = call(xm)
                else:
                    out = call(x)
                # row 0 is never ablated while capturing, so its output is clean
                if self.capture_on:
                    self._capture(l, module, x, out)
                if rows and self.check_delta:
                    self._check_delta(l, module, x, xm, out, call(x), rows)
                return out
            return f
        return factory

    def _capture(self, l, module, x, out):
        torch, H, dh = self.torch, self.H, self.dh
        W = module.weight
        if x.dim() != 3 or W.device.type == "meta":
            return                       # not recorded -> the capture check aborts
        ref = out[0, -1].float()
        if not torch.isfinite(ref).all():
            _abort(f"layer {l}: non-finite attention output at the measured position")
        b = getattr(module, "bias", None)
        b = 0 if b is None else b.float()
        ktol = sp._kernel_tol(x.dtype, W.dtype, out.dtype)
        self.checks["split_tol"] = max(self.checks["split_tol"] or 0.0, ktol)
        if l not in self.layout:
            # shape cannot tell Linear [out, in] from Conv1D [in, out] when the
            # matrix is square: keep the layout that reproduces the output
            best = ("none", float("inf"))
            for mode in ("linear", "conv1d"):
                try:
                    e = _rel_err(sp._split(x[0, -1], W, mode, H, dh)[0].sum(0) + b, ref)
                except Exception:
                    e = float("inf")
                e = e if math.isfinite(e) else float("inf")
                if e < best[1]:
                    best = (mode, e)
            if not sp._ok(best[1], ktol):
                _abort(f"layer {l}: per-head split does not reproduce {type(module).__name__} "
                       f"under either weight layout (best relative error {best[1]:.2e} > "
                       f"{ktol}, n_heads={H}, head_dim={dh}); refusing")
            self.layout[l] = best
        v, whole = sp._split(x[0, -1], W, self.layout[l][0], H, dh)
        tot = v.sum(0) + b
        e_exact = _rel_err(tot, whole + b)
        if not sp._ok(e_exact, EXACT_TOL):
            _abort(f"layer {l}: per-head split does not sum to the projection recomputed in "
                   f"fp32 (error {e_exact:.2e} > {EXACT_TOL}); refusing")
        e = _rel_err(tot, ref)
        if not sp._ok(e, ktol):
            _abort(f"layer {l}: per-head split error {e:.2e} > {ktol} "
                   f"(layout {self.layout[l][0]}); refusing")
        self.checks["split_err"] = max(self.checks["split_err"], e)
        self.checks["split_exact_err"] = max(self.checks["split_exact_err"], e_exact)
        self.contrib[l] = v.norm(dim=-1).cpu()
        if l in self.wset:
            self.z[l] = x[0, -1].float().reshape(H, dh).cpu()

    def _check_delta(self, l, module, x, xm, out, out_clean, rows):
        """The ablation the projection saw must be the one the table asks for,
        and it must change the output by exactly those heads' writes.

        (1) The intended input is rebuilt from `rows` (the condition's heads
        and values) with explicit column slices h*d_head:(h+1)*d_head, not
        from the replaced input: it must equal the input the projection
        received bit for bit, so a wrong head, a wrong value, a wrong row or a
        touched earlier position aborts (the own-z check replaces every head,
        so it cannot see which head a replacement hit).  (2) The ablated
        output minus the clean one must be the listed heads' (r_h - z_h)
        W_O^h, predicted from the intended input: both outputs went through
        the model's kernel, so the bound is twice the kernel tolerance,
        relative to the clean output's scale."""
        torch, H, dh = self.torch, self.H, self.dh
        W = module.weight
        if l not in self.layout:
            _abort(f"layer {l}: weight layout unknown at the ablation check")
        xe = x.clone(memory_format=torch.contiguous_format)
        for row, heads, vals in rows:
            for i, h in enumerate(heads):
                if not 0 <= int(h) < H:
                    _abort(f"layer {l}: head {h} outside 0..{H - 1} in the ablation table")
                cols = slice(int(h) * dh, (int(h) + 1) * dh)
                xe[row, -1, cols] = (0 if vals is None else
                                     vals[i].to(device=x.device, dtype=x.dtype))
        if xm.shape != xe.shape or not torch.equal(xm, xe):
            where = []
            if xm.shape == xe.shape:
                diff = (xm[:, -1] != xe[:, -1]).reshape(xm.shape[0], H, dh).any(-1)
                where = [(int(r), int(h)) for r, h in diff.nonzero().tolist()][:8]
                if xm.shape[1] > 1 and not torch.equal(xm[:, :-1], xe[:, :-1]):
                    where.append("an earlier position")
            _abort(f"layer {l}: the o_proj input differs from the intended ablation at "
                   f"(row, head) {where or 'shape ' + str(tuple(xm.shape))}; the replacement "
                   f"does not hit the heads and values of the condition table, refusing")
        self.checks["input_checks"] += 1
        ktol = 2 * sp._kernel_tol(x.dtype, W.dtype, out.dtype)
        rs = sorted({r for r, _, _ in rows})
        # one fp32 copy of W_O per call (268 MB at 8192 x 8192), not per row;
        # sum_h (r_h - z_h) W_O^h is W_O applied to the intended input change
        Wm = (W if self.layout[l][0] == "linear" else W.t()).float()
        d_pred_all = (xe[rs, -1].float() - x[rs, -1].float()) @ Wm.t()
        del Wm, xe
        for i, row in enumerate(rs):
            d_obs = out[row, -1].float() - out_clean[row, -1].float()
            d_pred = d_pred_all[i]
            scale = float(out_clean[row, -1].float().abs().max().clamp_min(1e-9))
            e = float((d_obs - d_pred).abs().max()) / scale
            if not sp._ok(e, ktol):
                _abort(f"layer {l}: ablating heads changed the o_proj output by something "
                       f"other than their write (error {e:.2e} > {ktol}); refusing")
            self.checks["delta_err"] = max(self.checks["delta_err"], e)
            self.checks["delta_checks"] += 1

    # ── forwards ──
    def forward(self, ids, cache=None, use_cache=False):
        """One model forward; (output, last-position logits [B, V] float32)."""
        torch = self.torch
        self.calls.clear()
        with torch.no_grad():
            o = self.hf(input_ids=ids, past_key_values=cache, use_cache=use_cache)
        bad = {l: self.calls.get(l, 0) for l in range(self.L) if self.calls.get(l, 0) != 1}
        if bad:
            _abort(f"output projection calls per layer {bad}: expected exactly one per "
                   f"forward on every layer")
        lg = o.logits[:, -1].float()
        if not torch.isfinite(lg).all():
            _abort("non-finite logits at the measured position")
        return o, lg

    def _collect(self, what):
        miss = [l for l in range(self.L) if l not in self.contrib]
        missz = [l for l in self.window if l not in self.z]
        if miss or missz:
            _abort(f"{what}: activity recorded on {self.L - len(miss)}/{self.L} layers, z on "
                   f"{len(self.window) - len(missz)}/{len(self.window)} window layers")
        C = self.torch.stack([self.contrib[l] for l in range(self.L)]).numpy()
        Z = (self.torch.stack([self.z[l] for l in self.window]).numpy() if self.window
             else np.zeros((0, self.H, self.dh), np.float32))
        if not (np.isfinite(C).all() and np.isfinite(Z).all()):
            _abort(f"{what}: activity or z has NaN/inf")
        return C.astype(np.float32), Z.astype(np.float32)

    def capture(self, feed):
        """Full forward of `feed` (token ids): (activity [L, H], z [len(window),
        H, d_head], last-position logits [V] float32 on the CPU)."""
        ids = self.torch.tensor([list(feed)], device=self.dev)
        self.contrib.clear(); self.z.clear()
        self.capture_on = True
        try:
            _, lg = self.forward(ids)
        finally:
            self.capture_on = False
        C, Z = self._collect("capture")
        return C, Z, lg[0].cpu()

    def measure(self, feed, tables, ans_ids, *, batch=1, verify=False, verify_cache=None):
        """Run every condition on one item.

        feed: token ids (align_answer's), measured at the last one.  tables:
        ablation_tables(...) output, tables[0] the clean condition ({}).
        ans_ids: the ten digit answer tokens.  Returns (digit logits [C, 10],
        clean last-position logits [V] (CPU), clean activity [L, H], clean z
        [len(window), H, d_head]).  verify runs the first-item checks (module
        docstring); verify_cache the full-forward cache check (default:
        verify)."""
        torch = self.torch
        if not tables or tables[0]:
            raise ValueError("tables[0] must be the clean condition ({})")
        batch = int(batch)
        if batch < 1:
            raise ValueError("batch must be >= 1")
        verify_cache = verify if verify_cache is None else verify_cache
        C = len(tables)
        ids = torch.tensor([list(feed)], device=self.dev)
        P = ids.shape[1] - 1
        if P < 1:
            _abort("the prompt is a single token; there is no prefix to cache")
        o, _ = self.forward(ids[:, :P], use_cache=True)
        cache = _as_cache(o.past_key_values)
        del o
        if int(cache.get_seq_length()) != P:
            _abort(f"prefix cache holds {int(cache.get_seq_length())} positions, not {P}")
        if batch > 1:
            cache.batch_repeat_interleave(batch)
        last = ids[:, P:].expand(batch, 1).contiguous()
        ans = list(ans_ids)
        dl = np.full((C, 10), np.nan, np.float32)

        def run(tabs, capture=False):
            tabs = list(tabs) + [{}] * (batch - len(tabs))       # pad with clean rows
            abl = {}
            for r, t in enumerate(tabs):
                for l, (heads, vals) in t.items():
                    abl.setdefault(int(l), []).append((r, heads, vals))
            if capture:
                self.contrib.clear(); self.z.clear()
            self.ablate, self.capture_on, self.check_delta = abl, capture, verify
            try:
                _, lg = self.forward(last, cache=cache, use_cache=True)
            finally:
                self.ablate, self.capture_on, self.check_delta = {}, False, False
                _crop(cache, P)
            return lg

        def pad_check(lg, n_used, clean):
            # padded rows are clean at other batch positions: a batched forward
            # must reproduce the clean row (bf16 may round rows differently)
            for r in range(n_used, batch):
                e = _rel_err(lg[r], clean)
                if not sp._ok(e, self.cache_tol):
                    _abort(f"batched clean row {r} differs from the clean row (error {e:.2e} "
                           f"> {self.cache_tol}); rows of one forward are not independent")
                self.checks["pad_err"] = max(self.checks["pad_err"], e)

        chunks = [list(range(i, min(i + batch, C))) for i in range(0, C, batch)]
        lg0 = run([tables[m] for m in chunks[0]], capture=True)
        clean = lg0[0].clone()
        act, z = self._collect("clean condition")
        pad_check(lg0, len(chunks[0]), clean)
        for r, m in enumerate(chunks[0]):
            dl[m] = lg0[r, ans].cpu().numpy()

        if verify_cache:
            _, lf = self.forward(ids)
            e = _rel_err(clean, lf[0])
            if not sp._ok(e, self.cache_tol):
                _abort(f"clean last-token logits through the KV cache do not reproduce a full "
                       f"forward (relative error {e:.2e} > {self.cache_tol}); the cached path is "
                       f"not exact here, refusing")
            self.checks["cache_err"] = max(self.checks["cache_err"], e)
            self.checks["cache_checks"] += 1
        if verify:
            zt = {l: self.torch.from_numpy(z[i]) for i, l in enumerate(self.window)}
            own = {l: (list(range(self.H)), zt[l]) for l in self.window}
            zero = {l: ([int(np.argmax(act[l]))], None) for l in self.window}
            # own z at row 0, the clean row's batch position: same computation
            lo = run([own])
            e = _rel_err(lo[0], clean)
            if not sp._ok(e, IDENT_TOL):
                _abort(f"replacing every window head's input with its own value changed the "
                       f"logits (error {e:.2e} > {IDENT_TOL}): the replacement does not hit "
                       f"the measured position's head slices")
            self.checks["ownz_err"] = max(self.checks["ownz_err"], e)
            lz = run([zero])
            if self.window and torch.equal(lz[0], clean):
                _abort("zero-ablating the largest head of every window layer left the logits "
                       "unchanged: the ablation does not reach the model")

        for ch in chunks[1:]:
            lg = run([tables[m] for m in ch])
            pad_check(lg, len(ch), clean)
            for r, m in enumerate(ch):
                dl[m] = lg[r, ans].cpu().numpy()

        # the first chunk again: a cache the crops did not restore changes it
        lr = run([tables[m] for m in chunks[0]])
        e = _rel_err(lr[:len(chunks[0])], lg0[:len(chunks[0])])
        if not sp._ok(e, IDENT_TOL):
            _abort(f"re-running the first conditions after the others changed their logits "
                   f"(error {e:.2e} > {IDENT_TOL}): the KV cache was not restored")
        self.checks["repeat_err"] = max(self.checks["repeat_err"], e)
        self.checks["items"] += 1
        del cache
        if not np.isfinite(dl).all():
            _abort("digit logits contain NaN/inf")
        return dl, clean.cpu(), act, z


# ────────────────────────────── the run ─────────────────────────────────────
def _provenance():
    import torch, transformers
    here = Path(__file__).resolve()
    commit, src = sp._git_commit()
    deps = {}
    for n in ("deep_probe.py", "head_alignment_probe.py", "active_subgraph_dot.py",
              "sweep_probe.py", "signature_analysis.py", "serial_tasks.py"):
        p = here.parent / n
        deps[n] = sp._sha256(p) if p.exists() else None
    return {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "numpy_version": np.__version__, "python_version": sys.version.split()[0],
            "git_commit": commit, "git_commit_source": src,
            "next_runs_commit_env": os.environ.get("NEXT_RUNS_COMMIT"),
            "module_sha256": _mod_sha(), "dependency_sha256": deps,
            "measurement_version": MEASUREMENT_VERSION}


def _ids_sha(items, idx):
    return hashlib.sha256(json.dumps([items[i]["id"] for i in idx]).encode()).hexdigest()


def _load_checkpoint(path, fingerprint, log):
    if path is None or not Path(path).exists():
        return None
    try:
        A, meta = sp.load_output(path)
    except Exception as e:
        _abort(f"checkpoint {path} exists but cannot be read ({e}); delete it to start over")
    have = dict(meta.get("fingerprint", {}))
    have["measurement_version"] = _meas_version(have)
    diff = [f"{k}: have {have.get(k)!r}, want {v!r}" for k, v in fingerprint.items()
            if have.get(k) != v]
    if diff:
        _abort(f"checkpoint {path} was written with other settings ({'; '.join(diff)}); "
               f"delete it or use another --out")
    log(f"  resuming from {Path(path).name}: pass 1 done, {meta['done']} measurement items done")
    return A, meta


def run_ablation(model, suite, *, attn_impl, window=(0.4, 0.8), split="test",
                 max_items_per_k=None, random_draws=5, seed=0, per_layer=False, n_perm=5000,
                 q=0.05, batch_conditions=1, layer_classes=None, checkpoint=None,
                 checkpoint_every=5, log=print):
    """E4 on one model and suite; returns (arrays, meta) (module docstring).

    model: an _HFHookedModel-like wrapper.  layer_classes [L] of -1/0/+1
    replaces the selection-half classification (tests; recorded in meta).
    checkpoint: path of a resumable checkpoint (written after pass 1 and every
    checkpoint_every measurement items, deleted by the caller when done)."""
    import torch
    t_start = time.time()
    hf, tok = model._model, model._tok
    L, H, dh = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    name = getattr(model.cfg, "model_name", None)
    sp._check_items(suite)
    actual = getattr(hf.config, "_attn_implementation", None)
    if actual is not None and actual != attn_impl:
        _abort(f"model runs {actual!r} attention but attn_impl={attn_impl!r}; the recorded "
               f"kernel must be the one that ran")
    lo, hi = (float(w) for w in window)
    if not 0 <= lo < hi <= 1:
        _abort(f"window must satisfy 0 <= lo < hi <= 1, got {window}")
    if int(batch_conditions) < 1 or int(random_draws) < 0:
        _abort("batch_conditions must be >= 1 and random_draws >= 0")
    items_all = suite["items"]
    sel, meas = split_halves(items_all, split, seed, max_items_per_k)
    check_competitors(items_all, sel + meas)
    wl = list(range(L))[sa.window(L, lo, hi)]
    if not wl:
        _abort(f"window {window} holds no layer of {L}")
    if 1 / (n_perm + 1) > q / L:
        log(f"  note: {n_perm} permutations, smallest p 1/{n_perm + 1} > q/L = {q / L:.2g}; "
            f"BH can then classify a layer only together with many others "
            f"(use --n-perm >= {math.ceil(L / q)})")
    asd = sys.modules.get("active_subgraph_dot")
    load_info = dict(getattr(asd, "LAST_LOAD_INFO", None) or {})
    if load_info.get("model") != name:
        load_info = {"note": "model not loaded through active_subgraph_dot.load_model "
                             "in this process"}
    dtype = str(next(hf.parameters()).dtype).replace("torch.", "")
    fingerprint = {"model": name, "suite_sha256": sp.suite_sha256(suite), "split": split,
                   "window": [lo, hi], "max_items_per_k": max_items_per_k,
                   "random_draws": int(random_draws), "seed": int(seed),
                   "per_layer": bool(per_layer), "n_perm": int(n_perm), "q": float(q),
                   "batch_conditions": int(batch_conditions), "attn_impl": attn_impl,
                   "dtype": dtype, "measurement_version": MEASUREMENT_VERSION,
                   "selection_sha256": _ids_sha(items_all, sel),
                   "measurement_sha256": _ids_sha(items_all, meas),
                   "layer_classes": None if layer_classes is None
                   else [int(c) for c in layer_classes]}
    ck = _load_checkpoint(checkpoint, fingerprint, log)
    nS, nM = len(sel), len(meas)
    log(f"  {name}: L={L} H={H} d_head={dh}   window layers {wl[0]}-{wl[-1]} ({len(wl)})   "
        f"selection {nS} / measurement {nM} items   batch {batch_conditions}")

    with Ablator(model, wl, log=log) as ab:
        # ── pass 1: selection half ──────────────────────────────────────────
        if ck is None:
            P1 = {"sel_index": np.array(sel, np.int32),
                  "sel_k": np.array([items_all[i]["k"] for i in sel], np.int16),
                  "sel_contribution": np.zeros((nS, L, H), np.float32),
                  "sel_digit_logits": np.zeros((nS, 10), np.float32),
                  "sel_correct": np.zeros(nS, bool)}
            zsum = np.zeros((len(wl), H, dh), np.float64)
            for j, i in enumerate(sel):
                t = items_all[i]
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
            if layer_classes is None:
                Y = sa.top_share(P1["sel_contribution"])
                bad = ~np.isfinite(Y)
                if bad.any():
                    # top share is 0/0 where every head of a layer writes zero;
                    # the joint null needs every item on every layer, and no
                    # trained layer is silent at the answer position
                    ls = np.nonzero(bad.any(0))[0]
                    _abort(f"layer(s) {ls.tolist()}: every head's activity is zero at the "
                           f"measured position on {int(bad.any(1).sum())}/{nS} selection "
                           f"items (e.g. {items_all[sel[int(np.nonzero(bad.any(1))[0][0])]]['id']})"
                           f", so the top share is undefined and the layers cannot be "
                           f"classified; a zeroed or bypassed output projection? refusing")
                cls = sa.classify_layers(Y, P1["sel_k"].astype(float), q, n_perm,
                                         _rng(seed, "classify"))
                src = "signature_analysis.classify_layers (selection half, all layers)"
            else:
                cls, src = np.asarray(layer_classes, int), "given"
                if cls.shape != (L,) or not set(np.unique(cls)) <= {-1, 0, 1}:
                    _abort(f"layer_classes must be [L={L}] of -1/0/+1")
            P1["layer_class"] = np.asarray(cls, np.int8)
            P1["head_rank"] = rank_heads(P1["mean_activity"]).astype(np.int16)
            P1["window_layers"] = np.array(wl, np.int16)
            p1meta = {"classes_source": src}
            done, recs, P2 = 0, [], None
            prev_checks = {}
        else:
            A0, m0 = ck
            P1 = {k: A0[k] for k in ("sel_index", "sel_k", "sel_contribution",
                                      "sel_digit_logits", "sel_correct", "mean_z",
                                      "mean_activity", "layer_class", "head_rank",
                                      "window_layers")}
            p1meta, done, recs = m0["pass1"], int(m0["done"]), list(m0["measurement_items"])
            P2 = {k: A0[k] for k in A0 if k not in P1}
            prev_checks = m0.get("checks", {})
        conds, skipped, sets = build_conditions(P1["layer_class"], wl, P1["mean_activity"],
                                                random_draws=random_draws, seed=seed,
                                                per_layer=per_layer)
        if ck is not None and [c["name"] for c in conds] != ck[1]["condition_names"]:
            _abort("checkpoint conditions differ from the ones rebuilt from its pass 1")
        tables = ablation_tables(conds, P1["mean_z"], wl)
        nC = len(conds)
        for s in SETS:
            log(f"  {s} layers in the window: {sets[s] or 'none'}")
        for sk in skipped:
            log(f"  skipped {sk['condition']}: {sk['reason']}")
        log(f"  {nC} conditions ({p1meta['classes_source']})")

        # ── pass 2: measurement half ────────────────────────────────────────
        A = {"index": np.array(meas, np.int32),
             "k": np.array([items_all[i]["k"] for i in meas], np.int16),
             "gold": np.array([int(items_all[i]["gold"]) for i in meas], np.int8),
             "answer_mask": np.zeros((nM, 10), bool),
             "digit_logits": np.full((nM, nC, 10), np.nan, np.float32),
             "logit_diff": np.full((nM, nC), np.nan, np.float32),
             "frac": np.full((nM, nC), np.nan, np.float32),
             "clean_correct": np.zeros(nM, bool),
             "meas_contribution": np.zeros((nM, L, H), np.float32),
             "n_tokens": np.zeros(nM, np.int32)}
        for j, i in enumerate(meas):
            A["answer_mask"][j, [int(d) for d in items_all[i]["answer_set"]]] = True
        if P2 is not None:
            for key in ("digit_logits", "logit_diff", "frac", "clean_correct",
                        "meas_contribution", "n_tokens"):
                A[key][:done] = P2[key][:done]

        # the weight layout found by the split check in any session; a session
        # resumed with nothing left to measure captures nothing and would
        # otherwise record none
        ck_layout = {} if ck is None else dict(ck[1].get("head_layout") or {})

        def layout_now():
            cur = {str(l): m for l, (m, _) in ab.layout.items()}
            clash = {l: (ck_layout[l], m) for l, m in cur.items()
                     if l in ck_layout and ck_layout[l] != m}
            if clash:
                _abort(f"weight layout differs from the checkpoint's on layers {clash}; the "
                       f"checkpoint was written for other weights")
            out = {**ck_layout, **cur}
            return {l: out[l] for l in sorted(out, key=int)}

        def save_ck(n_done):
            if checkpoint is None:
                return
            arr = dict(P1)
            arr.update({k: v for k, v in A.items()})
            sp.save_output(checkpoint, arr, {
                "kind": "ablation_checkpoint", "fingerprint": fingerprint, "done": n_done,
                "pass1": p1meta, "measurement_items": recs,
                "condition_names": [c["name"] for c in conds],
                "head_layout": layout_now(), "proj_name": ab.pname,
                "checks": _merge_checks(prev_checks, ab.checks)})

        if ck is None:
            save_ck(0)
        session = 0
        for j in range(done, nM):
            t = items_all[meas[j]]
            t0 = time.time()
            log(f"  [measure {j + 1}/{nM}] {t['id']}")
            feed, ANS, appended = align_answer(tok, t["text"])
            ans_ids = [ANS[d] for d in DIGITS]
            dl, _, act, _ = ab.measure(feed, tables, ans_ids, batch=batch_conditions,
                                       verify=(session == 0),
                                       verify_cache=(session % VERIFY_EVERY == 0))
            session += 1
            gold, aset = t["gold"], list(t["answer_set"])
            ld = logit_diff(dl, gold, aset)
            pred, correct, tied = sp._constrained(dl[0], aset, gold)
            A["digit_logits"][j], A["logit_diff"][j] = dl, ld
            A["frac"][j] = ld / ld[0] if ld[0] != 0 else np.nan
            A["clean_correct"][j] = correct
            A["meas_contribution"][j] = act
            A["n_tokens"][j] = len(feed)
            toks = tok.convert_ids_to_tokens(feed)
            recs.append({"id": t["id"], "suite_index": meas[j], "k": int(t["k"]),
                         "gold": gold, "answer_set": aset, "echo": t["echo"],
                         "n_tokens": len(feed), "measured_token": toks[-1],
                         "appended": appended, "pred_constrained": pred,
                         "correct_constrained": correct, "n_tied_at_max": len(tied),
                         "clean_logit_diff": float(ld[0]),
                         "time_s": round(time.time() - t0, 3)})
            if (j + 1) % max(1, int(checkpoint_every)) == 0:
                save_ck(j + 1)
            if ab.dev.type == "cuda":
                torch.cuda.empty_cache()
        layout = layout_now()
        if len(layout) != L:
            _abort(f"weight layout known for {len(layout)} of {L} layers; the provenance "
                   f"record would be incomplete")
        checks = _merge_checks(prev_checks, ab.checks)
        pname, head_checks = ab.pname, ab.head_checks

    A.update(P1)
    times = [r["time_s"] for r in recs] or [0.0]
    meta = {
        "kind": "ablation",
        "model": name, "suite": suite.get("name"), "family": suite.get("family"),
        "wording": suite.get("wording"), "setting": suite.get("setting"),
        "chance": suite.get("chance"), "suite_sha256": fingerprint["suite_sha256"],
        "split": split, "window": [lo, hi], "window_layers": wl,
        "max_items_per_k": max_items_per_k, "random_draws": int(random_draws),
        "seed": int(seed), "per_layer": bool(per_layer), "n_perm": int(n_perm), "q": float(q),
        "batch_conditions": int(batch_conditions), "attn_impl": attn_impl, "dtype": dtype,
        "n_layers": L, "n_heads": H, "d_head": dh, "load_info": load_info,
        **_provenance(),
        "proj_name": pname, "head_layout": layout, "head_boundaries_checked": head_checks,
        "classes_source": p1meta["classes_source"],
        "sets": sets, "conditions": conds, "skipped": skipped,
        "n_selection": nS, "n_measurement": nM,
        "selection_ids": [items_all[i]["id"] for i in sel],
        "tolerances": {"cache": CACHE_TOL, "cache_bf16": CACHE_TOL_BF16, "identical": IDENT_TOL, "split_exact": EXACT_TOL,
                       "split": sp.TOL, "split_fp32_kernel": sp.ROUNDING_TOL,
                       "delta": "2 x split tolerance", "input": "exact (intended ablation)",
                       "padded_rows": "as cache",
                       "verify_every": VERIFY_EVERY},
        "checks": checks, "measurement_items": recs,
        "timing": {"total_s": round(time.time() - t_start, 2),
                   "per_item_mean_s": round(float(np.mean(times)), 3)},
    }
    return A, meta


_COUNTS = ("delta_checks", "input_checks", "cache_checks", "items")


def _merge_checks(a, b):
    """Checks of an earlier session (a checkpoint) and this one: counts add,
    errors and tolerances take the maximum."""
    out = dict(a or {})
    for k, v in b.items():
        old = out.get(k)
        if k in _COUNTS:
            out[k] = (old or 0) + v
        elif isinstance(v, (int, float)) and isinstance(old, (int, float)):
            out[k] = max(old, v)
        elif v is not None or k not in out:
            out[k] = v
    return out


# ────────────────────────────── outputs ─────────────────────────────────────
def default_batch(device):
    """--batch-conditions when not given: BATCH_CUDA on a GPU (the offloaded
    70B runs, where every forward streams the weights), 1 on the CPU."""
    return BATCH_CUDA if str(device).split(":")[0] == "cuda" else 1


def output_path(out_dir, model_name, suite_name):
    return (Path(out_dir) /
            f"{model_name.replace('/', '_')}__{str(suite_name).replace('/', '_')}{SUFFIX}")


def checkpoint_path(out):
    out = Path(out)
    return out.with_name(out.name[:-len(".npz")] + ".ckpt")


def _summary(A, meta, log=print):
    ks = A["k"]
    log("  clean constrained by k: " + "  ".join(
        f"k{k} {int(A['clean_correct'][ks == k].sum())}/{int((ks == k).sum())}"
        for k in np.unique(ks)) + f"   chance {meta.get('chance')}")
    ok = A["clean_correct"]
    if ok.any():
        for i, c in enumerate(meta["conditions"]):
            if c["kind"] in ("top",):
                log(f"    {c['name']:<22} mean LD fraction of clean on correct items "
                    f"{np.nanmean(A['frac'][ok, i]):+.3f}")
    ch = meta["checks"]
    log(f"  checks: cache {ch.get('cache_err', 0):.1e} (abort {ch.get('cache_tol', CACHE_TOL)}"
        + (f"; above {CACHE_TOL}, the bf16 rounding band" if ch.get('cache_err', 0) > CACHE_TOL else "")
        + ", "
        f"{ch.get('cache_checks')} items)   own-z {ch.get('ownz_err', 0):.1e}   repeat "
        f"{ch.get('repeat_err', 0):.1e}   o_proj delta {ch.get('delta_err', 0):.1e} "
        f"({ch.get('input_checks', 0)} intended-input checks)   split "
        f"{ch.get('split_err', 0):.1e}")


def run(args, log=print):
    with open(args.suite) as f:
        suite = json.load(f)
    if not isinstance(suite, dict) or not isinstance(suite.get("name"), str) or not suite["name"]:
        _abort(f"{args.suite}: not a suite with a name (serial_tasks.save_suite)")
    # validate before the load (minutes to an hour for a 70B)
    try:
        sp._check_items(suite)
        sel, meas = split_halves(suite["items"], args.split, args.seed, args.max_items_per_k)
        check_competitors(suite["items"], sel + meas)
    except SystemExit as e:
        _abort(f"suite {args.suite}: {e}")
    B = getattr(args, "batch_conditions", None)
    B = default_batch(args.device) if B is None else int(B)
    out = output_path(args.out, args.model, suite["name"])
    want = {"model": args.model, "suite": suite["name"], "suite_sha256": sp.suite_sha256(suite),
            "split": args.split, "window": [float(w) for w in args.window],
            "max_items_per_k": args.max_items_per_k, "random_draws": args.random_draws,
            "seed": args.seed, "per_layer": bool(args.per_layer), "n_perm": args.n_perm,
            "q": float(args.q), "batch_conditions": B,
            "attn_impl": args.attn_impl, "requested_precision": args.precision,
            "measurement_version": MEASUREMENT_VERSION}
    if out.exists():
        try:
            _, m = sp.load_output(out)
        except Exception as e:
            _abort(f"{out} exists but cannot be read ({e}); delete it to re-run")
        m = dict(m, measurement_version=_meas_version(m))
        diff = [f"{k}: have {m.get(k)!r}, want {want[k]!r}" for k in RESUME_KEYS
                if m.get(k) != want[k]]
        if diff:
            _abort(f"{out} exists with different settings ({'; '.join(diff)}); use another "
                   f"--out or delete it")
        log(f"  skip {out.name}: already done")
        return None

    sp._git_commit()                 # before any tokenizer threads exist
    os.environ["ASD_FORCE_STRATEGY"] = "B"
    os.environ["ASD_FORCE_PRECISION"] = "bf16" if args.precision == "bf16" else ""
    os.environ["ASD_ATTN_IMPL"] = args.attn_impl
    import torch
    try:
        want_dev = torch.device(args.device).type
    except Exception:
        _abort(f"--device {args.device!r} is not a torch device")
    if want_dev == "cuda" and not torch.cuda.is_available():
        _abort("--device cuda but CUDA is not available: select a GPU runtime, or pass "
               "--device cpu explicitly")
    if args.precision == "fp32" and args.device != "cpu" and torch.cuda.is_available():
        _abort("load_model supports fp32 on CPU only; use --device cpu")
    import active_subgraph_dot as asd
    t0 = time.time()
    model = asd.load_model(args.model, device=args.device)
    load_s = time.time() - t0
    if not hasattr(model, "_model"):
        _abort("needs the _HFHookedModel wrapper (ASD_FORCE_STRATEGY=B)")
    on = sorted({p.device.type for p in model._model.parameters()})
    if want_dev != "cpu" and want_dev not in on:
        _abort(f"--device {args.device} requested but the model's weights are on {on} "
               f"(load_model fell back); refusing to run there")
    li = asd.LAST_LOAD_INFO
    want_prec = {"bf16": "bfloat16", "fp32": "float32"}[args.precision]
    if li.get("strategy") != "B" or li.get("precision") != want_prec:
        _abort(f"loaded with strategy {li.get('strategy')!r} precision "
               f"{li.get('precision')!r}; requested B / {want_prec}")
    log(f"  {args.model}: load_info={dict(li)}")
    ck = checkpoint_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    A, meta = run_ablation(model, suite, attn_impl=args.attn_impl,
                           window=tuple(args.window), split=args.split,
                           max_items_per_k=args.max_items_per_k,
                           random_draws=args.random_draws, seed=args.seed,
                           per_layer=args.per_layer, n_perm=args.n_perm, q=args.q,
                           batch_conditions=B, checkpoint=ck,
                           checkpoint_every=args.checkpoint_every, log=log)
    meta.update(want)
    meta.update(model=args.model, suite_path=str(args.suite), load_s=round(load_s, 1),
                cli={k: v for k, v in vars(args).items()})
    sp.save_output(out, A, meta)
    if ck.exists():
        ck.unlink()
    _summary(A, meta, log)
    log(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ───────────────────────────── analysis ─────────────────────────────────────
class RunError(ValueError):
    """An ablation output that cannot be analysed without emitting wrong numbers."""


def load_ablation(path):
    """One ablation output as a dict, after checking that its stored logit
    differences follow from its digit logits."""
    try:
        A, meta = sp.load_output(path)
    except Exception as e:
        raise RunError(f"{path}: cannot read ({e})")
    need = ("k", "gold", "answer_mask", "digit_logits", "logit_diff", "frac", "clean_correct",
            "sel_k", "sel_correct")
    miss = [k for k in need if k not in A]
    if miss or "conditions" not in meta:
        raise RunError(f"{path}: not an ablation_probe output (missing {miss or ['conditions']})")
    conds = meta["conditions"]
    n, C = A["logit_diff"].shape
    if C != len(conds) or conds[0]["kind"] != "clean" or A["digit_logits"].shape != (n, C, 10):
        raise RunError(f"{path}: {C} logit-difference columns for {len(conds)} conditions")
    ld = np.full((n, C), np.nan)
    for i in range(n):
        aset = [str(d) for d in np.nonzero(A["answer_mask"][i])[0]]
        gold = str(int(A["gold"][i]))
        ld[i] = logit_diff(A["digit_logits"][i], gold, aset)
        if sp._constrained(A["digit_logits"][i, 0], aset, gold)[1] != bool(A["clean_correct"][i]):
            raise RunError(f"{path}: item {i}: clean_correct contradicts its clean digit logits")
    if not np.allclose(ld, A["logit_diff"], rtol=1e-4, atol=1e-4, equal_nan=True):
        raise RunError(f"{path}: stored logit differences do not follow from the digit logits")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ld[:, :1] != 0, ld / ld[:, :1], np.nan)
    if not np.allclose(frac, A["frac"], rtol=1e-3, atol=1e-4, equal_nan=True):
        raise RunError(f"{path}: stored fractions do not follow from the logit differences")
    return {"path": str(path), "model": meta.get("model"), "suite": meta.get("suite"),
            "family": meta.get("family"), "meta": meta,
            # 1/|answer set| as the suite defines it; from the items if absent
            "chance": meta.get("chance") or 1 / int(A["answer_mask"].sum(1).max()),
            "conditions": conds, "k": A["k"].astype(int),
            "correct": A["clean_correct"].astype(bool), "ld": ld, "frac": frac,
            "sel_k": A["sel_k"].astype(int), "sel_correct": A["sel_correct"].astype(bool)}


def _groups(conds):
    """Condition group -> column indices; the random draws of one set and size
    form one group (an item's drop is its mean over the draws)."""
    g = {}
    for i, c in enumerate(conds):
        if c["kind"] == "clean":
            continue
        key = c["name"].rsplit("/r", 1)[0] if c["kind"] == "rand" else c["name"]
        g.setdefault(key, []).append(i)
    return g


def _strat_idx(k, n_boot, rng):
    """[n_boot, n] indices resampled with replacement within each depth; the
    columns of depth d are cols[d]."""
    k = np.asarray(k)
    idx = np.empty((n_boot, len(k)), int)
    cols, pos = {}, 0
    for d in np.unique(k):
        ix = np.nonzero(k == d)[0]
        idx[:, pos:pos + len(ix)] = ix[rng.integers(0, len(ix), (n_boot, len(ix)))]
        cols[int(d)] = slice(pos, pos + len(ix))
        pos += len(ix)
    return idx, cols


def _row_slopes(Y, K):
    Kc = K - K.mean(-1, keepdims=True)
    ss = (Kc * Kc).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss > 0, (Kc * (Y - Y.mean(-1, keepdims=True))).sum(-1) / ss, np.nan)


def _hc2_t(yc, KC, ss):
    """[B] studentised OLS slopes b / SE_HC2(b) of centred y [n] on each row of
    KC [B, n] (centred depths, sum of squares ss).  HC2 weighs each squared
    residual by 1 / (1 - leverage), which is unbiased when the variance
    differs by depth.  An item alone at one of only two depths is fitted
    exactly (leverage 1) and carries no residual.  A zero SE gives +-inf (a
    perfect fit), or 0 when the slope is 0 too."""
    n = KC.shape[-1]
    b = KC @ yc / ss
    e = yc[None, :] - b[:, None] * KC            # the intercept is mean(y): KC is centred
    lev = 1.0 / n + KC * KC / ss
    w = np.where(lev < 1 - 1e-9, 1.0 / np.maximum(1 - lev, 1e-9), 0.0)
    se = np.sqrt((KC * KC * e * e * w).sum(-1)) / ss
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(se > 0, b / np.where(se > 0, se, 1.0),
                        np.where(b != 0, np.copysign(np.inf, b), 0.0))


def slope_test(y, k, n_perm=5000, rng=None):
    """(OLS slope of y on k, two-sided p) that stays valid when the spread of
    y and the number of items change with k: a studentised permutation test,
    |b / SE_HC2(b)| against the same statistic with k shuffled across items;
    p = (count + 1) / (n_perm + 1).  Non-finite items dropped.

    Why not signature_analysis.slope_perm (the raw slope with k shuffled):
    that tests whether the drop's whole distribution depends on k, and E4
    changes its spread with k by construction (fewer clean-correct items at
    high k, and a noisier fraction where the clean margin is smaller), so it
    rejects a zero mean slope far more often than alpha (about 25% at 5% with
    40/28/17/10 items and SD 0.1-0.8).  Studentising makes the permutation
    test asymptotically valid for the slope alone (DiCiccio & Romano 2017)
    and keeps it exact when the drop does not depend on k at all."""
    y, k = np.asarray(y, float), np.asarray(k, float)
    ok = np.isfinite(y) & np.isfinite(k)
    y, k = y[ok], k[ok]
    if len(y) < 3 or np.ptp(k) == 0:
        return float("nan"), float("nan")
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    kc = k - k.mean()
    ss, yc = float(kc @ kc), y - y.mean()
    t0 = abs(float(_hc2_t(yc, kc[None, :], ss)[0]))
    cnt = 0
    for P in sa._perm_rows(kc, n_perm, rng):
        T = np.abs(_hc2_t(yc, P, ss))
        # sa._ge counts ties within rounding; an infinite t (perfect fit) only
        # ties another infinite one
        cnt += int(((T >= t0) | sa._ge(T, t0)).sum())
    return float(kc @ yc / ss), (cnt + 1) / (n_perm + 1)


def _ci(x, level):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None]
    a = (100 - level) / 2
    return [float(np.percentile(x, a)), float(np.percentile(x, 100 - a))]


def _depths(run, min_correct, min_items):
    """(depths analysed, mode, eligible depths, candidate depths).

    Correctness of both halves (clean) decides which depths qualify: the
    eligible ones (outline: >= min_correct correct, accuracy >= 2c/(1+c)) when
    there are at least 3; otherwise, as the outline's fallback for a pair
    below the rules, the depths solved above chance (one-sided binomial
    p < .05, deep_probe's rule), labelled exploratory.  Either way a
    candidate enters only with at least min_items clean-correct MEASUREMENT
    items: the bootstrap resamples items within a depth, so a depth with one
    or two items would enter the slope as if its mean were known exactly.
    The analysed n per depth is reported, and depths with fewer than
    min_correct analysed items are flagged."""
    kk = np.concatenate([run["sel_k"], run["k"]])
    ok = np.concatenate([run["sel_correct"], run["correct"]])
    c = run["chance"]
    n_meas = {int(d): int((run["correct"] & (run["k"] == d)).sum()) for d in np.unique(run["k"])}
    elig = [int(d) for d in sa.eligible_depths(kk, ok, c, min_correct)]
    if len(elig) >= 3:
        cand, mode = elig, "eligible"
    else:
        cand, mode = [int(d) for d in _above_chance(kk, ok, c)], "exploratory"
    need = max(1, int(min_items))
    return [d for d in cand if n_meas.get(d, 0) >= need], mode, elig, cand


def _significant_same_sign(s, p, s_ref, alpha):
    return bool(np.isfinite(p) and p < alpha and np.sign(s) == np.sign(s_ref) and s != 0)


def analyse_run(run, *, n_boot=2000, n_perm=5000, seed=0, min_correct=20, min_items=5,
                alpha=0.05):
    """Per condition group, on clean-correct measurement items at the analysed
    depths: the mean fractional drop (1 - LD/LD_clean) per k and overall with
    95% bootstrap CIs (items resampled within k) and its slope on k (95% CI;
    p from slope_test, a studentised permutation test, because the drop's
    spread and the number of clean-correct items change with k); the same on
    the absolute scale (LD_clean - LD, logits) and the ratio of means (sum of
    drops / sum of clean LD) per k; the sub-additivity ratio top3/(3 top1);
    the excess over the random baseline.  The slope CI is the percentile of
    the stratified bootstrap; with few items at a depth it runs a little
    narrow (about 9% outside at nominal 5% with 10 items and a spread 8x the
    k=1 spread), so decisions key on p.

    Scale check: the fractional drop divides by the clean LD, which falls with
    k (outline: the answer margin falls with k in both 70B models), so a head
    whose absolute effect is constant shows a fractional drop that grows with
    k, which is what specialisation predicts.  A group whose fractional slope
    is significant (p < alpha) while the absolute slope is not significant
    with the same sign is flagged `fractional_only`; the clean LD's own slope
    on k is reported.

    A group's statistics depend only on its own data and the seed: the
    bootstrap indices depend only on the items' depths (and are shared by all
    groups, as the ratios combine groups), and each group draws its
    permutations from a generator of its own, so adding a condition group
    (another set, --per-layer, random draws) never moves another group's p."""
    run = load_ablation(run) if isinstance(run, (str, Path)) else run
    tag = f"run/{run['model']}/{run['suite']}"
    depths, mode, elig, cand = _depths(run, min_correct, min_items)
    k, ok = run["k"], run["correct"]
    use = ok & np.isin(k, depths)
    K = k[use].astype(float)
    drop_all = 1 - run["frac"][use]
    ld = np.asarray(run["ld"], float)[use]
    groups = _groups(run["conditions"])
    n_by_k = {int(d): int((K == d).sum()) for d in depths}
    res = {"model": run["model"], "suite": run["suite"], "path": run["path"],
           "depth_mode": mode, "depths": [int(d) for d in depths],
           "eligible_depths": [int(d) for d in elig],
           "dropped_depths": [int(d) for d in cand if d not in depths],
           "analysed_n": n_by_k,
           "below_min_correct": [d for d, n in n_by_k.items() if n < min_correct],
           "min_items": int(min_items), "min_correct": int(min_correct), "alpha": float(alpha),
           "n_items": int(use.sum()),
           "accuracy": {int(d): [int(ok[k == d].sum()), int((k == d).sum())] for d in np.unique(k)},
           "sets": run["meta"].get("sets"), "skipped": run["meta"].get("skipped"),
           "clean_ld": {}, "groups": {}, "subadditivity": {}, "excess_over_random": {}}
    if use.sum() < 2 or len(depths) < 1:
        res["note"] = "no clean-correct items at the analysed depths"
        return res
    idx, cols = _strat_idx(K, n_boot, _rng(seed, f"{tag}/boot"))
    Kb = K[idx]
    clean = ld[:, 0]
    res["clean_ld"] = {"mean": float(clean.mean()),
                       "by_k": {d: float(clean[K == d].mean()) for d in cols}}
    if len(cols) >= 2:
        s, p = slope_test(clean, K, n_perm, _rng(seed, f"{tag}/perm/clean_ld"))
        res["clean_ld"]["slope"] = {"slope": s, "p": p, "test": SLOPE_TEST,
                                    "ci95": _ci(_row_slopes(clean[idx], Kb), 95)}
    G = {g: drop_all[:, ix].mean(1) for g, ix in groups.items()}
    labs = {g: (clean[:, None] - ld[:, ix]).mean(1) for g, ix in groups.items()}
    for g, y in G.items():
        a = labs[g]
        yb, ab_ = y[idx], a[idx]
        # the median guards against items whose clean LD is near zero, where
        # the fraction is large and noisy; the mean is the preregistered outcome
        ent = {"n_draws": len(groups[g]), "mean": float(np.mean(y)),
               "ci95": _ci(yb.mean(1), 95), "median": float(np.median(y)),
               "abs_drop": float(np.mean(a)), "abs_ci95": _ci(ab_.mean(1), 95),
               "ratio_of_means": float(a.sum() / clean.sum()), "by_k": {}}
        for d, sl in cols.items():
            m = K == d
            ent["by_k"][d] = {"n": int(m.sum()), "mean": float(y[m].mean()),
                              "ci95": _ci(yb[:, sl].mean(1), 95),
                              "median": float(np.median(y[m])),
                              "abs_drop": float(a[m].mean()),
                              "ratio_of_means": float(a[m].sum() / clean[m].sum())}
        if len(cols) >= 2:
            s, p = slope_test(y, K, n_perm, _rng(seed, f"{tag}/perm/{g}"))
            ent["slope"] = {"slope": s, "p": p, "test": SLOPE_TEST,
                            "ci95": _ci(_row_slopes(yb, Kb), 95)}
            s_a, p_a = slope_test(a, K, n_perm, _rng(seed, f"{tag}/perm_abs/{g}"))
            ent["abs_slope"] = {"slope": s_a, "p": p_a, "test": SLOPE_TEST,
                                "ci95": _ci(_row_slopes(ab_, Kb), 95)}
            ent["fractional_only"] = bool(np.isfinite(p) and p < alpha and
                                          not _significant_same_sign(s_a, p_a, s, alpha))
        if len(groups[g]) > 1:
            per_draw = drop_all[:, groups[g]].mean(0)
            ent["draw_means"] = [float(v) for v in per_draw]
        res["groups"][g] = ent
    for s in SETS:
        for m in MODES:
            g1, g3 = f"{s}/top1/{m}", f"{s}/top3/{m}"
            if g1 in G and g3 in G:
                r = _ratio(G[g3], G[g1], idx)
                r["by_k"] = {d: _ratio(G[g3][K == d], G[g1][K == d], None)["ratio"]
                             for d in cols}
                res["subadditivity"][f"{s}/{m}"] = r
        for n in TOP_SIZES:
            gt, gr = f"{s}/top{n}/mean", f"{s}/rand{n}/mean"
            if gt in G and gr in G:
                dlt = G[gt] - G[gr]
                res["excess_over_random"][f"{s}/{n}"] = {
                    "mean": float(dlt.mean()), "ci95": _ci(dlt[idx].mean(1), 95)}
    return res


def _ratio(y3, y1, idx):
    """top3 / (3 top1) on mean drops; undefined unless the top-1 drop is > 0."""
    def f(a, b):
        return a / (3 * b) if b > 0 else np.nan
    out = {"ratio": _nan_none(f(y3.mean(), y1.mean())), "top3": float(y3.mean()),
           "top1": float(y1.mean())}
    if idx is not None:
        m3, m1 = y3[idx].mean(1), y1[idx].mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ci95"] = _ci(np.where(m1 > 0, m3 / (3 * m1), np.nan), 95)
    if not (y1.mean() > 0):
        out["note"] = "top-1 drop not positive; ratio undefined"
    return out


def _nan_none(x):
    return None if x is None or not np.isfinite(x) else float(x)


def pair_reason(run_a, run_b):
    """None when two ablation outputs form the outline's pair (two models on
    one suite); otherwise why they are not compared.  One model on two
    suites, or two models on different suites, is not the ablation x k x
    model interaction and gets no verdict."""
    if run_a["model"] == run_b["model"]:
        return f"same model ({run_a['model']}); the interaction compares two models"
    if run_a["suite"] != run_b["suite"]:
        return (f"different suites ({run_a['suite']} / {run_b['suite']}); the interaction "
                f"compares two models on one suite")
    ha, hb = run_a["meta"].get("suite_sha256"), run_b["meta"].get("suite_sha256")
    if ha and hb and ha != hb:
        return (f"suite {run_a['suite']!r} has different contents in the two files "
                f"(suite_sha256 differs)")
    return None


def _same_items(run_a, run_b):
    """Whether both files measured the same items (None when a file does not
    record them)."""
    ids = []
    for r in (run_a, run_b):
        recs = r["meta"].get("measurement_items")
        if not recs:
            return None
        ids.append([t.get("id") for t in recs])
    return ids[0] == ids[1]


def design_mismatch(run_a, run_b):
    """{setting: [a, b]} for the PAIR_DESIGN_KEYS both files record with
    different values (another --window selects other target layers; another
    dtype or kernel rounds differently)."""
    ma, mb = run_a["meta"], run_b["meta"]
    return {k: [ma[k], mb[k]] for k in PAIR_DESIGN_KEYS
            if k in ma and k in mb and ma[k] != mb[k]}


def _group_info(conds, groups):
    """Group name -> (set, kind, rest of the name after the set), e.g.
    'spread/top1/mean' -> ('spread', 'top', 'top1/mean')."""
    out = {}
    for g, ix in groups.items():
        c = conds[ix[0]]
        s, _, rest = g.partition("/")
        out[g] = (c.get("set") or s, c["kind"], rest)
    return out


def pair_groups(run_a, run_b):
    """(primary, cross, not compared) pairings of the two files' condition
    groups, each pairing (key, group in a, group in b).

    primary: every group both files hold (same name, so the same layer
    class); and, for a condition (top / random baseline, size and mode) the
    files hold only for different layer classes, each model's own target
    layers: key 'SETa-vs-SETb/top1/mean' (e.g. Llama's spreading against Qwen's
    funnelling layers, which the pair's classes may well be).  cross: the
    other pairings of different classes of one condition (both files hold
    that condition for a common class too), reported apart.  Per-layer
    conditions are paired by name only.  not compared: groups in no
    pairing."""
    ga, gb = _groups(run_a["conditions"]), _groups(run_b["conditions"])
    ia, ib = _group_info(run_a["conditions"], ga), _group_info(run_b["conditions"], gb)
    primary = [(g, g, g) for g in sorted(set(ga) & set(gb))]
    cross = []
    rests = sorted({(kd, r) for _, kd, r in list(ia.values()) + list(ib.values())
                    if kd in CROSS_KINDS})
    for kd, rest in rests:
        sa_ = sorted(s for g, (s, k2, r) in ia.items() if (k2, r) == (kd, rest))
        sb_ = sorted(s for g, (s, k2, r) in ib.items() if (k2, r) == (kd, rest))
        pairs = [(s1, s2) for s1 in sa_ for s2 in sb_ if s1 != s2]
        dest = cross if set(sa_) & set(sb_) else primary
        dest += [(f"{s1}-vs-{s2}/{rest}", f"{s1}/{rest}", f"{s2}/{rest}") for s1, s2 in pairs]
    used_a = {x for _, x, _ in primary + cross}
    used_b = {y for _, _, y in primary + cross}
    rest_ = sorted((set(ga) - used_a) | (set(gb) - used_b))
    return primary, cross, rest_


def interaction(run_a, run_b, *, n_boot=2000, seed=0, equiv=EQUIV, min_correct=20,
                min_items=5):
    """Ablation x k x model, per pairing of the two runs' condition groups
    (pair_groups): the difference of the two models' slopes of fractional
    drop on k over their common analysed depths, scaled to that span (the
    change of the drop from the lowest to the highest common depth, in units
    of clean LD), with a bootstrap 90% CI (items resampled within k, each
    model separately).  Verdict (outline): 'null' only when the 90% CI lies
    within +/-equiv; 'difference' when it excludes 0; otherwise
    'inconclusive'.

    Pairings: `groups` holds the groups both files have (same layer class)
    and, where the pair's layer classes differ (each model's classes come
    from its own selection half; Llama spreading and Qwen funnelling is what
    E4 tests), each model's own target layers under 'SETa-vs-SETb/...';
    `cross_groups` the other cross-class pairings; every entry records the
    set each side used.  With no pairing at all, a note says why there is no
    verdict.

    Only two different models on one suite are compared (pair_reason); any
    other combination is returned with compared=False and a note.  Files
    that measured other items, or record another design (PAIR_DESIGN_KEYS:
    window, dtype, kernel, ...), get a warning.  The same interaction on the
    absolute scale (LD_clean - LD, logits; no margin, as the outline's is in
    units of clean LD) is reported beside it, and a 'difference' whose
    absolute CI does not exclude 0 on the same side is flagged
    fractional_only (analyse_run explains why).  The bootstrap indices
    depend only on the two runs' analysed depths and the seed, so a group's
    CI does not depend on which other groups the files hold."""
    ra = load_ablation(run_a) if isinstance(run_a, (str, Path)) else run_a
    rb = load_ablation(run_b) if isinstance(run_b, (str, Path)) else run_b
    out = {"a": ra["model"], "b": rb["model"], "suite_a": ra["suite"], "suite_b": rb["suite"],
           "compared": True, "depths": [], "depth_mode": None, "equivalence_margin": equiv,
           "min_correct": int(min_correct), "sets_a": ra["meta"].get("sets"),
           "sets_b": rb["meta"].get("sets"), "groups": {}, "cross_groups": {},
           "not_compared": []}
    why = pair_reason(ra, rb)
    if why:
        out.update(compared=False, note=f"not compared: {why}")
        return out
    same = _same_items(ra, rb)
    out["same_items"] = same
    warn = []
    if same is False:
        warn.append("the two files measured different items (another --seed, --split or "
                    "--max-items-per-k): the interaction compares independent samples of the "
                    "suite, not the same items")
    mism = design_mismatch(ra, rb)
    out["design_mismatch"] = mism
    if mism:
        warn.append("the two files were run with different designs (" + "; ".join(
            f"{k}: {v[0]!r} vs {v[1]!r}" for k, v in mism.items()) + "): E4's design is "
            "fixed before running and applies to both models")
    if warn:
        out["warnings"], out["warning"] = warn, "; ".join(warn)
    da, ma, _, _ = _depths(ra, min_correct, min_items)
    db, mb, _, _ = _depths(rb, min_correct, min_items)
    common = sorted(set(da) & set(db))
    out.update(depths=common, depth_mode="eligible" if ma == mb == "eligible" else "exploratory")
    if len(common) < 2:
        out["note"] = f"fewer than 2 common analysed depths ({common})"
        return out
    span = float(max(common) - min(common))
    ga, gb = _groups(ra["conditions"]), _groups(rb["conditions"])
    primary, cross, out["not_compared"] = pair_groups(ra, rb)
    if not primary:
        out["note"] = (f"no condition group common to both files, nor a condition both hold "
                       f"for different layer classes (sets {ra['model']}: "
                       f"{ra['meta'].get('sets')}, {rb['model']}: {rb['meta'].get('sets')}); "
                       f"no interaction verdict")
        return out
    ua = ra["correct"] & np.isin(ra["k"], common)
    ub = rb["correct"] & np.isin(rb["k"], common)
    ka, kb = ra["k"][ua].astype(float), rb["k"][ub].astype(float)
    out["n_by_k"] = {"a": {d: int((ka == d).sum()) for d in common},
                     "b": {d: int((kb == d).sum()) for d in common}}
    out["below_min_correct"] = [d for d in common
                                if min(out["n_by_k"]["a"][d], out["n_by_k"]["b"][d]) < min_correct]
    if len(np.unique(ka)) < 2 or len(np.unique(kb)) < 2:
        out["note"] = "a model has correct items at fewer than two common depths"
        return out
    tag = f"pair/{ra['model']}/{ra['suite']}/{rb['model']}/{rb['suite']}"
    ia, _ = _strat_idx(ka, n_boot, _rng(seed, f"{tag}/boot/a"))
    ib, _ = _strat_idx(kb, n_boot, _rng(seed, f"{tag}/boot/b"))
    lda, ldb = np.asarray(ra["ld"], float)[ua], np.asarray(rb["ld"], float)[ub]
    fa, fb = ra["frac"][ua], rb["frac"][ub]
    for dest, pairs in ((out["groups"], primary), (out["cross_groups"], cross)):
        for key, g_a, g_b in pairs:
            ca, cb = ga[g_a], gb[g_b]
            e = _pair_entry(fa[:, ca], fb[:, cb], lda[:, :1] - lda[:, ca],
                            ldb[:, :1] - ldb[:, cb], ka, kb, ia, ib, span, equiv)
            e.update(group_a=g_a, group_b=g_b, set_a=ra["conditions"][ca[0]].get("set"),
                     set_b=rb["conditions"][cb[0]].get("set"), cross_set=g_a != g_b)
            dest[key] = e
    return out


def _excludes_zero(ci):
    return ci[0] is not None and (ci[0] > 0 or ci[1] < 0)


def _pair_entry(fa, fb, da, db, ka, kb, ia, ib, span, equiv):
    """One pairing's interaction (interaction() documents the numbers).  fa /
    fb: [n, c] fractions LD/LD_clean of each model's columns of its group
    (random draws pooled per item); da / db: the matching absolute drops
    LD_clean - LD; ka / kb depths; ia / ib the stratified bootstrap indices."""
    Ka, Kb = ka[ia], kb[ib]
    ya, yb = (1 - fa).mean(1), (1 - fb).mean(1)
    sa_, sb_ = _row_slopes(ya, ka), _row_slopes(yb, kb)
    bsa, bsb = _row_slopes(ya[ia], Ka), _row_slopes(yb[ib], Kb)
    est = (sa_ - sb_) * span
    ci = _ci((bsa - bsb) * span, 90)
    main = float(ya.mean() - yb.mean())
    main_ci = _ci(ya[ia].mean(1) - yb[ib].mean(1), 90)
    within = ci[0] is not None and ci[0] >= -equiv and ci[1] <= equiv
    excl = _excludes_zero(ci)
    verdict = "null" if within else "difference" if excl else "inconclusive"
    # absolute scale (logits): LD_clean - LD, averaged over the group's columns
    xa, xb = da.mean(1), db.mean(1)
    abs_est = (_row_slopes(xa, ka) - _row_slopes(xb, kb)) * span
    abs_ci = _ci((_row_slopes(xa[ia], Ka) - _row_slopes(xb[ib], Kb)) * span, 90)
    abs_excl = _excludes_zero(abs_ci)
    abs_agrees = abs_excl and excl and (abs_ci[0] > 0) == (ci[0] > 0)
    return {
        "slope_a": float(sa_), "slope_b": float(sb_), "n_a": int(len(ya)), "n_b": int(len(yb)),
        "interaction_per_step": float(sa_ - sb_), "interaction": float(est), "ci90": ci,
        "within_margin": bool(within), "excludes_zero": bool(excl), "verdict": verdict,
        "main_effect": main, "main_ci90": main_ci,
        "abs_slope_a": float(_row_slopes(xa, ka)), "abs_slope_b": float(_row_slopes(xb, kb)),
        "abs_interaction": float(abs_est), "abs_ci90": abs_ci,
        "abs_excludes_zero": bool(abs_excl),
        "fractional_only": bool(verdict == "difference" and not abs_agrees)}


def _fmt(x, f="+.3f"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, f)


def _fci(ci, f="+.3f"):
    return "–" if not ci or ci[0] is None else f"[{format(ci[0], f)}, {format(ci[1], f)}]"


def report_md(res):
    L = ["# E4 ablation report", "",
         "Drop = 1 − LD/LD_clean (LD: gold logit minus the mean competitor logit), on "
         "clean-correct measurement items; 0 = no effect, 1 = the clean logit difference "
         "removed.  Absolute drop = LD_clean − LD, in logits.  CIs: bootstrap over items "
         "within k (95%; interactions 90%).  Slope p: studentised permutation test (slope / "
         "HC2 standard error, k shuffled across items), valid when the drop's spread and the "
         "number of correct items change with k.  † = fractional_only: the fractional slope "
         "or interaction is significant but the absolute one is not, on the same side; the "
         "clean LD changes with k, so the fraction's denominator may drive it.", ""]
    for r in res["runs"]:
        an = r.get("analysed_n") or {}
        L += [f"## {r['model']} — {r['suite']}", "",
              f"- depths analysed: {r['depths']} ({r['depth_mode']}; eligible "
              f"{r['eligible_depths']}), {r['n_items']} items",
              "- clean accuracy (measurement half) by k: " + ", ".join(
                  f"k{d} {a[0]}/{a[1]}" for d, a in
                  sorted(r["accuracy"].items(), key=lambda kv: int(kv[0]))),
              f"- sets: {r['sets']}"]
        if an:
            L.append("- analysed clean-correct items by k: " + ", ".join(
                f"k{d} {n}" for d, n in sorted(an.items(), key=lambda kv: int(kv[0]))))
        if r.get("below_min_correct"):
            L.append(f"- depths with fewer than {r.get('min_correct')} analysed items (the "
                     f"outline's rule counts both halves): {r['below_min_correct']}")
        if r.get("dropped_depths"):
            L.append(f"- qualifying depths left out, fewer than {r.get('min_items')} analysed "
                     f"items: {r['dropped_depths']}")
        cl = r.get("clean_ld") or {}
        if cl.get("by_k"):
            sl = cl.get("slope") or {}
            L.append("- clean LD by k: " + ", ".join(
                f"k{d} {_fmt(v, '.3f')}" for d, v in
                sorted(cl["by_k"].items(), key=lambda kv: int(kv[0])))
                + (f"; slope {_fmt(sl.get('slope'))} {_fci(sl.get('ci95'))}, "
                   f"p {_fmt(sl.get('p'), '.4f')}" if sl else ""))
        for sk in r.get("skipped") or []:
            L.append(f"- skipped {sk['condition']}: {sk['reason']}")
        if r.get("note"):
            L += ["", f"_{r['note']}_", ""]
            continue
        ds = r["depths"]
        L += ["", "| condition | mean drop [95% CI] | median | slope on k [95% CI] | p | "
              "abs slope (logits) [95% CI] | p (abs) | "
              + " | ".join(f"k={d}" for d in ds) + " |",
              "|---|---|---|---|---|---|---|" + "---|" * len(ds)]
        for g, e in r["groups"].items():
            sl, sa_ = e.get("slope") or {}, e.get("abs_slope") or {}
            L.append(f"| {g}{' (' + str(e['n_draws']) + ' draws)' if e['n_draws'] > 1 else ''} "
                     f"| {_fmt(e['mean'])} {_fci(e['ci95'])} | {_fmt(e['median'])} "
                     f"| {_fmt(sl.get('slope'))} "
                     f"{_fci(sl.get('ci95'))}{' †' if e.get('fractional_only') else ''} "
                     f"| {_fmt(sl.get('p'), '.4f')} "
                     f"| {_fmt(sa_.get('slope'))} {_fci(sa_.get('ci95'))} "
                     f"| {_fmt(sa_.get('p'), '.4f')} | "
                     + " | ".join(_fmt((e["by_k"].get(str(d)) or {}).get("mean")) for d in ds)
                     + " |")
        if r["subadditivity"]:
            L += ["", "Sub-additivity top3 / (3 × top1) (outline: consensus predicts a top-3 drop well "
                  "below three times the top-1 drop, not growing with k):", "",
                  "| set/mode | ratio [95% CI] | top1 | top3 |", "|---|---|---|---|"]
            for key, e in r["subadditivity"].items():
                L.append(f"| {key} | {_fmt(e['ratio'], '.3f')} {_fci(e.get('ci95'), '.3f')} | "
                         f"{_fmt(e['top1'])} | {_fmt(e['top3'])} |")
        if r["excess_over_random"]:
            L += ["", "Top heads minus the random mid-rank baseline (mean-ablation):", "",
                  "| set/size | excess drop [95% CI] |", "|---|---|"]
            for key, e in r["excess_over_random"].items():
                L.append(f"| {key} | {_fmt(e['mean'])} {_fci(e['ci95'])} |")
        L.append("")
    skipped_pairs = [p for p in res["pairs"] if p.get("compared") is False]
    for p in res["pairs"]:
        if p.get("compared") is False:
            continue
        L += [f"## Interaction: {p['a']} ({p['suite_a']}) minus {p['b']} ({p['suite_b']})", "",
              f"Common depths {p['depths']} ({p['depth_mode']}).  Interaction = difference of "
              f"the slopes of the drop on k, times the depth span: the change of the drop over "
              f"the common depths in units of clean LD.  Null claimed only if the 90% CI lies "
              f"within ±{p['equivalence_margin']}.  The absolute interaction (logits) has no "
              f"margin.", ""]
        for w in p.get("warnings") or ([p["warning"]] if p.get("warning") else []):
            L += [f"**Warning:** {w}.", ""]
        L += [f"Layer classes (window): {p['a']} {p.get('sets_a')}; "
              f"{p['b']} {p.get('sets_b')}", ""]
        if p.get("n_by_k"):
            L += ["Analysed items by k: " + "; ".join(
                f"{who} " + ", ".join(f"k{d} {n}" for d, n in
                                      sorted(p["n_by_k"][ab].items(), key=lambda kv: int(kv[0])))
                for who, ab in ((p["a"], "a"), (p["b"], "b"))), ""]
        if p.get("below_min_correct"):
            L += [f"Common depths with fewer than {p.get('min_correct')} analysed items in a "
                  f"model: {p['below_min_correct']}", ""]
        if p.get("note"):
            L += [f"_{p['note']}_", ""]
            continue
        if any(e.get("cross_set") for e in p["groups"].values()):
            L += [f"Rows named SETa-vs-SETb pair a condition the two files hold only for "
                  f"different layer classes: {p['a']}'s SETa layers against {p['b']}'s SETb "
                  f"layers (each model's own target layers).", ""]
        L += _pair_table(p["groups"])
        if p.get("cross_groups"):
            L += ["", "Cross-class pairings of conditions both files also hold for a common "
                      "class (secondary; SETa is " + str(p["a"]) + "'s layers, SETb "
                      + str(p["b"]) + "'s):", ""]
            L += _pair_table(p["cross_groups"])
        if p.get("not_compared"):
            L += ["", f"Condition groups in no pairing (not compared): "
                      f"{', '.join(p['not_compared'])}"]
        L.append("")
    if skipped_pairs:
        L += ["## Files not compared", "",
              "The interaction is defined for two models on one suite; no verdict for:", ""]
        for p in skipped_pairs:
            L.append(f"- {p['a']} ({p['suite_a']}) and {p['b']} ({p['suite_b']}): "
                     f"{p['note'].removeprefix('not compared: ')}")
        L.append("")
    return "\n".join(L) + "\n"


def _pair_table(groups):
    L = ["| condition | slope a | slope b | interaction [90% CI] | verdict | "
         "abs interaction (logits) [90% CI] | a − b mean drop [90% CI] |",
         "|---|---|---|---|---|---|---|"]
    for g, e in groups.items():
        # the outline's rule: within the margin is a null even when the CI
        # excludes 0 (equivalent, not identical); say so
        v = e["verdict"] + (" (CI excludes 0)" if e["verdict"] == "null"
                            and e["excludes_zero"] else "")
        v += " †" if e.get("fractional_only") else ""
        L.append(f"| {g} | {_fmt(e['slope_a'])} | {_fmt(e['slope_b'])} | "
                 f"{_fmt(e['interaction'])} {_fci(e['ci90'])} | {v} | "
                 f"{_fmt(e.get('abs_interaction'))} {_fci(e.get('abs_ci90'))} | "
                 f"{_fmt(e['main_effect'])} {_fci(e['main_ci90'])} |")
    return L


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def expand_paths(paths, log=print):
    """--analyse arguments as files.  A glob pattern is expanded here (Drive
    folders hold spaces, so the notebook quotes the pattern and the shell
    leaves it), keeping ablation outputs (*__ablation.npz) only, so 'DIR/*.npz'
    over a folder that also holds sweep captures works; '**' matches any number
    of folders, as in signature_analysis; a pattern matching none aborts.
    Plain paths are kept as given (load_ablation checks them).
    Duplicates are dropped, first occurrence kept."""
    out = []
    for p in (str(x) for x in paths):
        if any(c in p for c in "*?[") and not Path(p).exists():
            # recursive, as in signature_analysis: one Drive pattern such as
            # 'runs/**/*.npz' must reach the same files in both analyses
            hits = sorted(glob.glob(p, recursive=True))
            keep = [h for h in hits if h.endswith(SUFFIX)]
            if not keep:
                _abort(f"--analyse {p!r}: no ablation output (*{SUFFIX}) matches "
                       f"({len(hits)} other file(s))")
            if len(keep) < len(hits):
                log(f"  {p}: {len(hits) - len(keep)} match(es) not ending in {SUFFIX} left out")
            out += keep
        else:
            out.append(p)
    return list(dict.fromkeys(out))


def analyse(paths, out_dir=None, *, n_boot=2000, n_perm=5000, seed=0, equiv=EQUIV,
            min_correct=20, min_items=5, log=print):
    """Analyse ablation outputs (paths or glob patterns, expand_paths).  Every
    file gets analyse_run; every two files get an entry in `pairs`, but only
    two different models on the same suite (pair_reason) get the interaction
    and a verdict: one model on two suites or a cross-suite combination is
    listed as not compared.  Writes ablation_report.md / .json to out_dir
    (default: the first file's folder) and returns the report dict."""
    paths = expand_paths(paths, log)
    runs = [load_ablation(p) for p in paths]
    kw = dict(n_boot=n_boot, seed=seed, min_correct=min_correct, min_items=min_items)
    res = {"runs": [analyse_run(r, n_perm=n_perm, **kw) for r in runs], "pairs": []}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            x = interaction(runs[i], runs[j], equiv=equiv, **kw)
            if not x["compared"]:
                log(f"  {x['a']} ({x['suite_a']}) and {x['b']} ({x['suite_b']}): {x['note']}")
            res["pairs"].append(x)
    res = _plain(res)
    out = Path(out_dir) if out_dir else Path(paths[0]).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_report.json").write_text(json.dumps(res, indent=1))
    (out / "ablation_report.md").write_text(report_md(res))
    log(f"  -> {out / 'ablation_report.md'}")
    return res


# ─────────────────────────────── CLI ────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--suite", help="suite JSON (serial_tasks.save_suite)")
    ap.add_argument("--out", help="output directory (with --analyse: the report's)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"],
                    help="fp32 is supported on CPU only (load_model)")
    ap.add_argument("--window", nargs=2, type=float, default=[0.4, 0.8], metavar=("LO", "HI"),
                    help="target layers floor(LO(L-1))..floor(HI(L-1))")
    ap.add_argument("--split", default="test", choices=list(sp.SPLITS))
    ap.add_argument("--max-items-per-k", type=int, default=None)
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-layer", action="store_true",
                    help="also top-1 mean-ablation one layer at a time")
    ap.add_argument("--n-perm", type=int, default=5000,
                    help="permutations for the layer classification (and --analyse slopes)")
    ap.add_argument("--q", type=float, default=0.05, help="BH level of the layer classes")
    ap.add_argument("--batch-conditions", type=int, default=None,
                    help=f"conditions per forward (default {BATCH_CUDA} on cuda, 1 otherwise). "
                         f"Forwards per measured item: 1 prefix + ceil(C/B) conditions + 1 "
                         f"repeat of the first chunk (+1 on cache-check items, +2 on the "
                         f"first); C is about 46 for the 70B pair with --per-layer, so B=1 is "
                         f"about 10x slower than B={BATCH_CUDA} under CPU offload")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--analyse", nargs="+", metavar="NPZ",
                    help="ablation outputs; quoted glob patterns are expanded here")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--equiv", type=float, default=EQUIV)
    ap.add_argument("--min-correct", type=int, default=20)
    ap.add_argument("--min-items", type=int, default=5)
    a = ap.parse_args(argv)
    if a.analyse:
        if a.model or a.suite:
            ap.error("--analyse takes ablation outputs only")
        try:
            return analyse(a.analyse, a.out, n_boot=a.n_boot, n_perm=a.n_perm, seed=a.seed,
                           equiv=a.equiv, min_correct=a.min_correct, min_items=a.min_items)
        except RunError as e:
            _abort(f"--analyse: {e}")
    if not (a.model and a.suite and a.out):
        ap.error("--model, --suite and --out are required (or --analyse)")
    if not 0 <= a.window[0] < a.window[1] <= 1:
        ap.error("--window needs 0 <= LO < HI <= 1")
    if a.random_draws < 0 or (a.batch_conditions is not None and a.batch_conditions < 1) \
            or a.n_perm < 1:
        ap.error("--random-draws >= 0, --batch-conditions >= 1, --n-perm >= 1")
    if a.batch_conditions is None:
        a.batch_conditions = default_batch(a.device)
    if a.max_items_per_k is not None and a.max_items_per_k < 2:
        ap.error("--max-items-per-k must be >= 2")
    return run(a)


if __name__ == "__main__":
    main()

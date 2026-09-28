#!/usr/bin/env python3
"""
Item-level analysis of the next runs (E1-E6 of paper_manuscript/research_outline.tex).

Turns per-item capture files into the paper's statistics.  Activity s(l,h) is
the per-head contribution ||z_h W_O^h|| at the position whose next token is the
answer digit (`contribution`; `write_norm`, the post-norm write, is reported
alongside for OLMo-2 / Gemma-2).

  primary     top-head share max_h s / sum_h s, averaged over layers
              floor(0.4(L-1))..floor(0.8(L-1)) ("40-80% depth"); its item-level
              OLS slope on the serial depth k with a permutation null, and the
              k=1 -> kmax change.  PR/H over the same window is secondary.
  signature   spread (share falls) / funnel (rises) / flat / mixed (share and
              PR/H point to opposite responses) / inconclusive (correct items
              and all items disagree in sign) / missing (no correct-item
              analysis: fewer than 3 depths with >= 20 correct items and
              accuracy >= 2c/(1+c))
  RQ1         invariance of a model's pre-registered sign across the four task
              families (pointer, arith, binding, relational) and wordings
              (stratified bootstrap CI, Holm across families), on each model's
              pinned configuration only; the numeral control (fixed_digit),
              the single-prompt ladder and kernel / precision re-runs are
              reported beside it, never counted as families or models
  RQ3         per-layer classes (BH within model and task, joint k-permutation
              null), layer map, final-20% region, contiguous PR band
  RQ4         pooled scale-normalised slopes under ONE relabelling of k shared
              by all models on the same items; paired differences
  checks      kernel comparison, power targets, BOS / attention-sink control,
              mass centroid, and the E1 choice of one setting per family and
              comparison group

Input files (load_run), all item-level:
  <model>__<suite>.npz   sweep_probe.py           arrays + meta (JSON string)
  <model>_align.npz      head_alignment_probe.py  contribution, items in meta
  <model>_deep.json      deep_probe.py            tasks[] with contribution
Older files carry no item metadata; depth, answer set and echo are then parsed
from the pointer-chase template (label D{k}.., "a goes to b.", "Take k steps").

Measured position: files written with deep_probe.align_answer record it per
item ('appended' / 'measured_token').  A file without that record whose own
predictions show a position that does not predict the answer digit (top-1
next token whitespace on most items, or median digit mass < 0.5; the 17
September deep_probe runs, which the outline says are "not evidence yet") is
rejected by the CLI; one with no evidence either way is analysed with a note.

Every analysis checks its input and refuses rather than report numbers from a
file that is inconsistent (NaN activity, arrays and item records that disagree,
correctness flags that contradict the stored predictions, the same item id
with different depths, prompts or answers in two runs).  The CLI lists such
files as rejected and reports nothing for them; an analysis that fails on one
file is listed as refused and the rest of the report is still written.

When several captures of one model, suite and configuration are globbed
(an E1 calibration run, a --limit smoke run and the E2 capture all share the
file name sweep_probe gives them), files with identical contents (a copy in
another folder) are rejected as duplicates and one capture is analysed: the
one with activity, then without --limit, then on the test split, then with the
most items; the others are listed as superseded.  Captures that tie on all of
that but differ in content (a re-run saved in another folder) are conflicts:
the files cannot tell which is current, so none of them is analysed.

Each run's permutations and bootstraps are drawn from a generator seeded by
(--seed, the run's model / suite / kernel / precision, the analysis), so a
run's p-values do not change when other files join the glob.

    python3 signature_analysis.py --runs 'runs/e2/*.npz' --out report \\
        --preregistered qwen2.5=funnel llama=spread OLMo-2-1124-7B-Instruct=spread
    python3 signature_analysis.py --calib 'runs/e1/*.npz' --groups groups.json --out e1

--preregistered MODEL=SIGN (SIGN funnel/spread, or none to leave a model out
of RQ1): MODEL occurs in the model name, case-insensitively; a post-trained
model (-Instruct, -SFT, -DPO, -it, -chat, -RLVR) is covered only by a pattern
that names its marker, so qwen2.5=funnel is "Qwen2.5 base: funnel" and does not
reach Qwen2.5-7B-Instruct (post-training moves the signature: OLMo).  A
pattern ending in $ is anchored at the end of the name; re:REGEX is a regular
expression with no marker rule.  An exact model name wins, then the longest
pattern.  --pinned ATTN PRECISION names E3's configuration; without it each
model's RQ1 uses the configuration holding most of its task-family cells.
"""
import argparse, glob, hashlib, json, math, os, re, subprocess, sys, time, warnings, zlib
from collections import Counter
from pathlib import Path
from statistics import NormalDist
import numpy as np

DIGITS = "0123456789"
CLASSES = ("spread", "funnel", "flat", "mixed", "inconclusive", "missing")
# Arrays not read from capture files: large and unused by these statistics.
_SKIP_ARRAYS = ("dla", "cos_top", "cos_sum", "top_idx", "sum_norm")
POST_TRAINING_SUFFIXES = ("-instruct", "-sft", "-dpo", "-it", "-chat", "-rlvr")
# post-training marker anywhere in a name (Mistral-7B-Instruct-v0.3, gemma-2-27b-it)
_POST_MARK = re.compile(r"(?:^|[-_. /])(instruct|sft|dpo|it|chat|rlvr)(?=$|[-_. /])", re.I)
# RQ1 tests invariance over these; anything else is reported beside it
TASK_FAMILIES = ("pointer", "arith", "binding", "relational")
CONTROL_FAMILIES = {"fixed_digit": "numeral control (fixed_digit)",
                    "ladder": "single-prompt ladder (one item per depth)"}


class RunError(ValueError):
    """A capture file that cannot be analysed without emitting wrong numbers."""


# ────────────────────────────── small helpers ───────────────────────────────
def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(0 if rng is None else rng)


def _sub_rng(seed, *parts):
    """A generator that depends only on the seed and `parts` (a run's identity
    and the analysis): a run's p-values must not move when another file joins
    the glob, as one stream shared in file order would make them."""
    ent = [int(seed) % (1 << 63)] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(ent)


def _ge(null_abs, obs_abs):
    """|null| >= |obs| with a relative tolerance: a relabelling that reproduces
    the observed arrangement must count, whatever the floating-point order."""
    return null_abs >= obs_abs - (1e-9 * obs_abs + 1e-15)


def _perm_rows(x, n_perm, rng, batch=2000):
    """Yield [b, n] blocks whose rows are independent random permutations of x."""
    x = np.asarray(x, float)
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        yield rng.permuted(np.tile(x, (b, 1)), axis=1)
        done += b


def _ols_slope(y, k):
    y, k = np.asarray(y, float), np.asarray(k, float)
    kc = k - k.mean()
    ss = kc @ kc
    return float(kc @ (y - y.mean()) / ss) if ss > 0 else float("nan")


def _row_slopes(Y, K):
    """OLS slope of each row of Y on the matching row of K."""
    Kc = K - K.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    ss = (Kc * Kc).sum(1)
    return np.where(ss > 0, (Kc * Yc).sum(1) / np.where(ss > 0, ss, 1), np.nan)


def _p_two_sided(obs, null):
    null = np.asarray(null, float)
    if not np.isfinite(obs) or len(null) == 0:
        return float("nan")
    return float((_ge(np.abs(null), abs(obs)).sum() + 1) / (len(null) + 1))


def _sign(x):
    return 0 if x is None or not np.isfinite(x) else int(np.sign(x))


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _short(model):
    return str(model).split("/")[-1]


def _as_run(r):
    return load_run(r) if isinstance(r, (str, Path)) else r


# ────────────────────────────── statistics ──────────────────────────────────
def window(L, lo=0.4, hi=0.8):
    """Layers floor(lo(L-1))..floor(hi(L-1)) inclusive: 10-21 of 28, 12-24 of
    32, 31-63 of 80.  The epsilon keeps an exact product such as 0.4*10 from
    rounding down to the layer below."""
    if L < 2:
        raise ValueError(f"window needs L >= 2, got {L}")
    return slice(math.floor(lo * (L - 1) + 1e-9), math.floor(hi * (L - 1) + 1e-9) + 1)


def _kept(S, exclude):
    """Activity with excluded heads zeroed (activity is a norm, so zero adds
    nothing to a max or a sum) and the number of heads kept per layer."""
    S = np.asarray(S, float)
    if exclude is None:
        return S, np.full(S.shape[:-1], S.shape[-1], float)
    ex = np.asarray(exclude, bool)
    if ex.shape != S.shape:
        raise ValueError(f"exclude mask {ex.shape} does not match activity {S.shape}")
    return np.where(ex, 0.0, S), (~ex).sum(-1).astype(float)


def top_share(S, exclude=None):
    """[n, L] top-head share max_h s / sum_h s (NaN where no head is kept)."""
    S, _ = _kept(S, exclude)
    tot = S.sum(-1)
    return np.where(tot > 0, S.max(-1) / np.where(tot > 0, tot, 1), np.nan)


def pr_h(S, exclude=None):
    """[n, L] participation ratio (sum s)^2 / sum s^2 over the kept heads,
    divided by their number (H when nothing is excluded)."""
    S, nk = _kept(S, exclude)
    s, s2 = S.sum(-1), (S * S).sum(-1)
    ok = (s2 > 0) & (nk > 0)
    return np.where(ok, s * s / np.where(ok, s2, 1) / np.where(ok, nk, 1), np.nan)


def top_share_window(S, L, lo=0.4, hi=0.8, exclude=None):
    """[n] the primary statistic: top share averaged over the 40-80% window."""
    S = np.asarray(S, float)
    if S.ndim != 3 or S.shape[1] != L:
        raise ValueError(f"activity of shape {S.shape} is not [n, {L}, H]")
    return _nanmean(top_share(S, exclude)[:, window(L, lo, hi)], 1)


def slope_perm(y, k, n_perm=10000, rng=None):
    """OLS slope of y on k and its two-sided permutation p-value (k shuffled
    across items), p = (count + 1) / (n_perm + 1).  Non-finite items dropped."""
    y, k = np.asarray(y, float), np.asarray(k, float)
    ok = np.isfinite(y) & np.isfinite(k)
    y, k = y[ok], k[ok]
    if len(y) < 3 or np.ptp(k) == 0:
        return float("nan"), float("nan")
    rng = _rng(rng)
    kc = k - k.mean()
    ss, yc = kc @ kc, y - y.mean()
    obs = float(kc @ yc / ss)
    cnt = 0
    for P in _perm_rows(kc, n_perm, rng):
        cnt += int(_ge(np.abs(P @ yc / ss), abs(obs)).sum())
    return obs, (cnt + 1) / (n_perm + 1)


def eligible_depths(k, correct, chance, min_correct=20):
    """Depths with >= min_correct correct items and accuracy >= 2c/(1+c).  If a
    fraction g of items is solved and the rest guessed at chance c, accuracy is
    a = g + (1-g)c, and g/a >= 1/2 exactly when a >= 2c/(1+c): at least half
    of the correct items are genuine solves."""
    k, ok = np.asarray(k), np.asarray(correct, bool)
    thr = 2 * chance / (1 + chance)
    out = []
    for d in np.unique(k):
        m = k == d
        nc = int(ok[m].sum())
        if nc >= min_correct and nc / m.sum() >= thr - 1e-12:
            out.append(int(d))
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (NaN entries stay NaN, not counted)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    m = len(x)
    if m:
        o = np.argsort(x, kind="stable")
        adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * x[o]), 1.0)
        r = np.empty(m)
        r[o] = adj
        out[ok] = r
    return out


def bh(pvals, q=0.05):
    """Benjamini-Hochberg rejections at FDR q (bool mask; NaN never rejected)."""
    p = np.asarray(pvals, float)
    rej = np.zeros(p.shape, bool)
    ok = np.isfinite(p)
    x = p[ok]
    m = len(x)
    if m:
        o = np.argsort(x, kind="stable")
        below = np.nonzero(x[o] <= q * np.arange(1, m + 1) / m)[0]
        if len(below):
            r = np.zeros(m, bool)
            r[o[:below[-1] + 1]] = True
            rej[ok] = r
    return rej


def _bh_rows(P, q):
    """bh() applied to every row of P at once."""
    R, m = P.shape
    o = np.argsort(P, 1, kind="stable")
    below = np.take_along_axis(P, o, 1) <= q * np.arange(1, m + 1) / m
    last = np.where(below.any(1), m - 1 - np.argmax(below[:, ::-1], 1), -1)
    rank = np.argsort(o, 1)
    return rank <= last[:, None]


def _joint_slopes(Y, k, n_perm, rng):
    """Per-column OLS slopes of Y [n, L] on k and their null [n_perm, L] under
    ONE shuffle of k per draw applied to every column, which keeps the
    correlation of sampling noise between columns (adjacent layers)."""
    Y, k = np.asarray(Y, float), np.asarray(k, float)
    if Y.ndim != 2 or len(k) != len(Y):
        raise ValueError(f"Y {Y.shape} and k {k.shape} do not match")
    if not np.isfinite(Y).all() or not np.isfinite(k).all():
        raise ValueError("non-finite values in Y or k; a joint null needs every item")
    kc = k - k.mean()
    ss = kc @ kc
    if ss == 0:
        raise ValueError("k does not vary")
    Yc = Y - Y.mean(0)
    obs = kc @ Yc / ss
    null = np.concatenate([P @ Yc / ss for P in _perm_rows(kc, n_perm, rng)])
    return obs, null


def _perm_p_cols(obs, null):
    return (_ge(np.abs(null), np.abs(obs)).sum(0) + 1) / (len(null) + 1)


def _null_p_rows(null):
    """Per-layer p-value of every null draw against the whole null set, so each
    draw can be classified exactly as the observed slopes are."""
    R, L = null.shape
    A = np.abs(null)
    P = np.empty_like(A)
    for l in range(L):
        s = np.sort(A[:, l])
        P[:, l] = (R - np.searchsorted(s, A[:, l] - (1e-9 * A[:, l] + 1e-15), "left")) / R
    return P


def classify_layers(Y, k, q=0.05, n_perm=5000, rng=None):
    """Per layer -1 (spreading: top-share slope < 0), +1 (funnelling) or 0.

    Y [n, L] per-layer top share.  Item-level OLS slopes per layer, two-sided
    p from a joint permutation null (one shuffle of k applied to all layers),
    Benjamini-Hochberg at q within the model and task."""
    obs, null = _joint_slopes(Y, k, n_perm, _rng(rng))
    rej = bh(_perm_p_cols(obs, null), q)
    return np.where(rej, np.sign(obs), 0).astype(int)


def _seq_stats(c):
    """(sign changes, longest same-sign run) among the classified layers, in
    layer order with unclassified layers skipped."""
    s = np.asarray(c)[np.asarray(c) != 0]
    if len(s) == 0:
        return 0, 0
    ch = np.nonzero(s[1:] != s[:-1])[0]
    bounds = np.concatenate([[-1], ch, [len(s) - 1]])
    return int(len(ch)), int(np.diff(bounds).max())


def _matched_classes(null, npos, nneg):
    """Classes of each null draw with exactly the observed numbers of
    funnelling and spreading layers: the npos layers with the largest and the
    nneg with the smallest standardised slope (slope / its null SD, so layers
    with noisier statistics are not favoured)."""
    R, L = null.shape
    npos, nneg = int(npos), int(nneg)
    if npos + nneg > L:
        raise ValueError("more classified layers than layers")
    sd = null.std(0)
    o = np.argsort(null / np.where(sd > 0, sd, 1), 1)
    C = np.zeros((R, L), int)
    rows = np.arange(R)[:, None]
    if nneg:
        C[rows, o[:, :nneg]] = -1
    if npos:
        C[rows, o[:, L - npos:]] = 1
    return C


def layer_map(classes, Y, k, n_perm=2000, rng=None, q=0.05, mass=None, final=0.8):
    """RQ3 layer map: are spreading and funnelling layers contiguous,
    interleaved or randomly ordered?

    Statistics among classified layers: number of sign changes, longest
    same-sign run.  Nulls:
      joint (primary)   ONE shuffle of k per draw for all layers, which keeps
                        the correlation of noise between adjacent layers.  It
                        shuffles k against each layer's residuals (Y minus its
                        fitted slope): shuffling raw Y would give every layer
                        with a true effect b_l the common null term
                        b_l*slope(k, shuffled k), so the null would copy the
                        observed block pattern.  Each draw is reclassified
                        with the observed numbers of spreading and funnelling
                        layers held fixed (the most negative / most positive
                        standardised slopes), so only the arrangement is tested
      joint_bh          raw-Y draws reclassified by BH at q, as the spec's text
                        literally says ("recompute classes under each
                        shuffle").  Degenerate: under the null BH classifies
                        almost nothing, so its null has ~0 sign changes and it
                        only tests whether ANY layer responds (p_more is then
                        1/(n_perm+1) for every responsive model).  Reported for
                        completeness; never read as an arrangement test
      layer_shuffle     the observed classes in random layer order (secondary)
    p_fewer = P(null changes <= observed) (contiguous), p_more = P(>=)
    (interleaved), p_longer = P(null run >= observed) (contiguous).

    `arrangement` holds the p-values to read.  Contiguity (p_contiguous,
    p_contiguous_run) is judged against the joint null: correlated noise
    between adjacent layers makes contiguous blocks by itself, and the joint
    null keeps that correlation (the layer shuffle would over-call contiguity).
    Interleaving (p_interleaved) is judged against the layer shuffle: the
    joint null's draws inherit the noise correlation, so a random arrangement
    of true signs looks more interleaved than they are (AR(1) layer noise with
    rho 0.8: joint p_more <= 0.05 in ~60% of random arrangements, layer
    shuffle ~1%; with independent noise both hold their level).  This splits
    the spec's single "primary null" by direction, a documented deviation.
    Also the fraction and relative position of each class, and the final-20%
    region (layers after the 40-80% window): per-layer slopes, region mean with
    its joint p, and the mass response if `mass` [n, L] is given."""
    rng = _rng(rng)
    classes = np.asarray(classes, int)
    Y = np.asarray(Y, float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"layer_map needs [n, L] with L >= 2 layers, got {Y.shape}")
    L = Y.shape[1]
    if classes.shape != (L,):
        raise ValueError(f"classes {classes.shape} do not match {L} layers")
    obs, null = _joint_slopes(Y, k, n_perm, rng)
    ch, run = _seq_stats(classes)
    Pn = _null_p_rows(null)
    kf = np.asarray(k, float)
    _, rnull = _joint_slopes(Y - np.outer(kf - kf.mean(), obs), kf, n_perm, rng)
    nulls = {
        "joint": _matched_classes(rnull, (classes > 0).sum(), (classes < 0).sum()),
        "joint_bh": np.where(_bh_rows(Pn, q), np.sign(null), 0).astype(int),
        "layer_shuffle": rng.permuted(np.tile(classes, (n_perm, 1)), axis=1),
    }
    tests = {}
    for name, C in nulls.items():
        st = np.array([_seq_stats(c) for c in C])
        tests[name] = {
            "p_fewer": float(((st[:, 0] <= ch).sum() + 1) / (len(st) + 1)),
            "p_more": float(((st[:, 0] >= ch).sum() + 1) / (len(st) + 1)),
            "p_longer": float(((st[:, 1] >= run).sum() + 1) / (len(st) + 1)),
            "null_changes_mean": float(st[:, 0].mean()),
            "null_run_mean": float(st[:, 1].mean()),
            "null_classified_mean": float((C != 0).sum(1).mean()),
        }
    x = np.arange(L) / (L - 1)
    sp, fu = np.nonzero(classes < 0)[0], np.nonzero(classes > 0)[0]
    fin = np.arange(L) > math.floor(final * (L - 1) + 1e-9)
    out = {
        "L": L, "signmap": "".join("+" if c > 0 else "-" if c < 0 else "." for c in classes),
        "n_spread": int(len(sp)), "n_funnel": int(len(fu)),
        "frac_spread": len(sp) / L, "frac_funnel": len(fu) / L,
        "pos_spread": [round(float(x[l]), 3) for l in sp],
        "pos_funnel": [round(float(x[l]), 3) for l in fu],
        "mean_pos_spread": float(x[sp].mean()) if len(sp) else None,
        "mean_pos_funnel": float(x[fu].mean()) if len(fu) else None,
        "sign_changes": ch, "longest_run": run, "tests": tests,
        "arrangement": {"p_contiguous": tests["joint"]["p_fewer"],
                        "p_contiguous_run": tests["joint"]["p_longer"],
                        "p_interleaved": tests["layer_shuffle"]["p_more"],
                        "nulls": "contiguity vs joint (matched counts, residual "
                                 "shuffle); interleaving vs layer shuffle"},
        "slopes": obs.tolist(), "p": _perm_p_cols(obs, null).tolist(),
        "final_region": {
            "layers": np.nonzero(fin)[0].tolist(),
            "slopes": obs[fin].tolist(),
            "mean_slope": float(obs[fin].mean()) if fin.any() else None,
            "p": _p_two_sided(obs[fin].mean(), null[:, fin].mean(1)) if fin.any() else None,
            "n_spread": int((classes[fin] < 0).sum()), "n_funnel": int((classes[fin] > 0).sum()),
        },
    }
    if mass is not None and fin.any():
        mo, mn = _joint_slopes(np.asarray(mass, float), k, n_perm, rng)
        base = np.asarray(mass, float).mean(0)
        out["final_region"]["mass_slope_rel"] = float((mo[fin] / base[fin]).mean())
        out["final_region"]["mass_p"] = _p_two_sided((mo[fin] / base[fin]).mean(),
                                                     (mn[:, fin] / base[fin]).mean(1))
        out["mass_slope_rel"] = (mo / base).tolist()
    return out


def band_rule(slope, p, window_=0.10, call=0.5, sustain=0.15):
    """The band from per-layer PR/H slopes and permutation p-values, under the
    contiguous rule and the superseded one, in relative depth l/(L-1).

    A layer is callable if p < call.  frac[l] = share of the callable layers
    within +/- window_ of l that disperse (slope > 0).
      contiguous   onset = first layer of the first run of layers with
                   frac > 1/2 whose span (last - first)/(L-1) >= sustain; the
                   band closes where that run ends
      superseded   onset = first layer followed by int(sustain*L) layers with
                   frac > 1/2 (a count, not a span); close = the last layer
                   with frac > 1/2 anywhere, which bridges gaps.  Kept only so
                   old results can be compared (sweep_analysis.py,
                   replicate_analysis.py)."""
    slope, p = np.asarray(slope, float), np.asarray(p, float)
    L = len(slope)
    if L < 2 or len(p) != L:
        raise ValueError("band_rule needs matching slope and p over >= 2 layers")
    x = np.arange(L) / (L - 1)
    callable_ = p < call
    frac = np.full(L, np.nan)
    for l in range(L):
        w = (np.abs(x - x[l]) <= window_ + 1e-12) & callable_
        if w.any():
            frac[l] = float((slope[w] > 0).mean())
    above = frac > 0.5                    # NaN (nothing callable nearby) is not above

    runs, start = [], None
    for l in range(L + 1):
        if l < L and above[l]:
            start = l if start is None else start
        elif start is not None:
            runs.append((start, l - 1))
            start = None
    qual = [(a, b) for a, b in runs if (b - a) / (L - 1) >= sustain - 1e-9]
    if qual:
        a, b = qual[0]
        cont = {"onset": a / (L - 1), "close": b / (L - 1), "width": (b - a) / (L - 1),
                "first_layer": a, "last_layer": b, "n_layers": b - a + 1}
    else:
        cont = {"onset": None, "close": None, "width": 0.0,
                "first_layer": None, "last_layer": None, "n_layers": 0}
    cont["runs"] = [[a / (L - 1), b / (L - 1)] for a, b in runs]

    need, on = int(sustain * L), None
    for l in range(L):
        if above[l] and all(above[j] for j in range(l, min(L, l + need))):
            on = l
            break
    cl = max(np.nonzero(above)[0]) if on is not None else None
    sup = ({"onset": on / (L - 1), "close": cl / (L - 1), "width": (cl - on) / (L - 1),
            "width_legacy": (cl - on + 1) / L, "first_layer": int(on), "last_layer": int(cl)}
           if on is not None else
           {"onset": None, "close": None, "width": 0.0, "width_legacy": 0.0,
            "first_layer": None, "last_layer": None})
    differs = (cont["onset"], cont["close"]) != (sup["onset"], sup["close"])
    return {"frac": frac.tolist(), "callable": int(callable_.sum()),
            "contiguous": cont, "superseded": sup, "differs": bool(differs)}


def band(S, k, rng=None, n_perm=2000, window_=0.10, call=0.5, sustain=0.15, exclude=None):
    """Contiguous band rule on per-layer PR/H slopes (item-level OLS on k, p from
    a joint k-permutation null), with the superseded rule for comparison."""
    if np.ndim(S) != 3 or np.shape(S)[1] < 2:
        raise ValueError(f"band needs activity [n, L, H] with L >= 2, got {np.shape(S)}")
    P = pr_h(S, exclude)
    obs, null = _joint_slopes(P, k, n_perm, _rng(rng))
    p = _perm_p_cols(obs, null)
    out = band_rule(obs, p, window_, call, sustain)
    out.update(slopes=obs.tolist(), p=p.tolist(), L=int(P.shape[1]))
    return out


# ────────────────────────────── loading ─────────────────────────────────────
_RULE = re.compile(r"(\d) goes to (\d)")


def _pointer_parse(text):
    """(rule table, start, written k) of the query in a pointer-chase prompt.
    The last "Rules:" block is the query; earlier blocks are worked examples."""
    q = (text or "").split("Rules:")[-1]
    table = {int(a): int(b) for a, b in _RULE.findall(q)}
    s = re.search(r"Start at (\d+)", q)
    kk = re.search(r"Take (\d+) steps", q)
    return table, (int(s.group(1)) if s else None), (int(kk.group(1)) if kk else None)


def _ring_min_hops(table, s, g):
    """Fewest lookups from s to g walking a permutation table forward or
    backward (serial_tasks._min_hops): on a cycle of length c with g f steps
    ahead, min(f, c - f).  None when s or g is not on a cycle of the table."""
    if s not in table or g is None:
        return None
    ring, x = [s], table.get(s)
    while x is not None and x != s and len(ring) <= len(table):
        ring.append(x)
        x = table.get(x)
    if x != s or g not in ring:
        return None
    f = ring.index(g)
    return min(f, len(ring) - f)


def _item_min_hops(t, family):
    """An item's shortest lookup path: recorded by serial_tasks (sweep_probe
    copies it), else recomputed from the prompt for the pointer-chase families
    (Sep 22 and 70B files predate the field).  NaN where undefined."""
    v = t.get("min_hops")
    if v is None:
        v = (t.get("setting") or {}).get("min_hops")
    if v is None and family in ("pointer", "ladder", "fixed_digit"):
        table, s, _ = _pointer_parse(t.get("text"))
        g = t.get("gold")
        try:
            g = int(g) if g is not None else None
        except (TypeError, ValueError):
            g = None
        v = _ring_min_hops(table, s, g) if table and s is not None else None
    return np.nan if v is None else float(v)


def _legacy_k(label, text):
    """Depth of an item without a "k" field.  D{k}...: k.  K{k}R{rule}
    (the original fixed_digit control): the number of distinct states the
    k-step walk visits AFTER the start (ident 0, cyc2 1, cyc3 2, cyc5 up to
    4), serial_tasks' fixed_digit convention len(set(s_0..s_K)) - 1, so the
    depths of old and new numeral controls line up."""
    m = re.match(r"D(\d+)", label or "")
    if m:
        return int(m.group(1))
    m = re.match(r"K(\d+)R", label or "")
    if m:
        table, s, _ = _pointer_parse(text)
        if not table or s is None:
            raise RunError(f"label {label!r}: cannot parse the rule table")
        path = [s]
        for _ in range(int(m.group(1))):
            if path[-1] not in table:          # the walk leaves the table: not a permutation
                raise RunError(f"label {label!r}: state {path[-1]} has no rule in the table")
            path.append(table[path[-1]])
        return len(set(path)) - 1
    raise RunError(f"item label {label!r} carries no depth and the item has no 'k'")


def _legacy_family(suite):
    s = str(suite or "").lower()
    if s.startswith("fixed_digit"):
        return "fixed_digit"
    if s == "serial_depth":
        return "ladder"
    if s.startswith("serial_depth") or s.startswith("serial_replicates"):
        return "pointer"
    return s or "unknown"


def _norm_precision(p):
    s = str(p or "").lower().replace("torch.", "")
    return {"bfloat16": "bf16", "bf16": "bf16", "float32": "fp32", "fp32": "fp32",
            "float16": "fp16", "fp16": "fp16", "nf4": "nf4"}.get(s, s or None)


def load_run(path):
    """One capture file as a dict: arrays (activity etc.), meta, and per-item
    ids, k, gold, answer-set sizes, correctness flags, gold_prob, digit_mass.

    `correct` holds the correctness used by the analysis: correct_strict when
    the run generated, else correct_constrained (else the legacy `correct`)."""
    path = str(path)
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        if "tasks" not in d:
            raise RunError(f"{path}: not a deep_probe file (no 'tasks')")
        # content fingerprint: only files with identical contents are copies
        content = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()
        tasks = d["tasks"]
        meta = {k: v for k, v in d.items() if k != "tasks"}
        drop = {"contribution", "anchor_pos", "anchor_mass", "attn_entropy", "bos_mass",
                "hidden_sim", "tokens"}
        items = [{k: v for k, v in t.items() if k not in drop} for t in tasks]
        arrays = {}
        if tasks and all("contribution" in t for t in tasks):
            try:
                arrays["contribution"] = np.array([t["contribution"] for t in tasks], float)
            except ValueError as e:
                raise RunError(f"{path}: ragged contribution arrays ({e})")
        fmt = "deep"
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            if "meta" not in z.files:
                raise RunError(f"{path}: no meta")
            meta = json.loads(str(z["meta"]))
            # E5 globs every output on Drive, E4's included.  Refuse ablation
            # files by what they are, not only because their per-item keys
            # happen to differ from a capture's: the half-suite measurement
            # would otherwise read as a capture of the model.
            kind = str(meta.get("kind") or "")
            if kind.startswith("ablation"):
                raise RunError(f"{path}: an ablation_probe output (kind {kind!r}, E4), not a "
                               f"capture; analyse it with ablation_probe.py --analyse")
            arrays = {k: z[k] for k in z.files if k != "meta" and k not in _SKIP_ARRAYS}
            content = _npz_fingerprint(z, meta, arrays)
        items = meta.get("items")
        fmt = "align" if path.endswith("_align.npz") else "sweep"
    else:
        raise RunError(f"{path}: unknown file type")
    if not items:
        raise RunError(f"{path}: no per-item records")
    n = len(items)

    sm = meta.get("suite") if isinstance(meta.get("suite"), dict) else {}
    suite = sm.get("name") if sm else meta.get("suite")
    family = meta.get("family") or sm.get("family") or _legacy_family(suite)
    wording = meta.get("wording") or sm.get("wording") or "w0"
    setting = meta.get("setting") or sm.get("setting") or {}
    li = meta.get("load_info") or {}

    def col(name, conv):
        """Per-item field from the item records, else from a 1-D array."""
        if all(name in t for t in items):
            return [conv(t[name]) for t in items]
        a = arrays.get(name)
        if a is not None and a.ndim == 1 and len(a) == n:
            return [conv(v) for v in a.tolist()]
        return None

    ids, ks, golds, asz, echo, fams, words = [], [], [], [], [], [], []
    for i, t in enumerate(items):
        label, text = t.get("label") or "", t.get("text") or ""
        k = t.get("k")
        k = _legacy_k(label, text) if k is None else k
        if isinstance(k, bool) or int(k) != k or k < 0:
            raise RunError(f"{path}: item {i} has depth {k!r}")
        lm = re.match(r"D(\d+)", label)
        if t.get("k") is not None and lm and int(lm.group(1)) != int(k):
            raise RunError(f"{path}: item {i} label {label!r} but k = {k}")
        ks.append(int(k))
        g = t.get("gold")
        golds.append(None if g is None else str(g).strip())
        aset = t.get("answer_set")
        if aset is None:
            table, _, _ = _pointer_parse(text)
            aset = [str(s) for s in sorted(table)] if table else None
        if aset is not None and golds[-1] is not None and golds[-1] not in [str(a) for a in aset]:
            raise RunError(f"{path}: item {i} gold {golds[-1]!r} not in answer set {aset}")
        asz.append(len(aset) if aset else 0)
        if "echo" in t:
            echo.append(t["echo"])
        else:
            _, _, kw = _pointer_parse(text)
            echo.append(None if kw is None else str(kw))
        ids.append(str(t.get("id") or f"{suite}/{label or i}"))
        fams.append(t.get("family") or family)
        words.append(t.get("wording") or wording)
    if len(set(ids)) != n:
        raise RunError(f"{path}: duplicate item ids")
    ks = np.array(ks, int)
    if "k" in arrays and not np.array_equal(np.asarray(arrays["k"], int), ks):
        raise RunError(f"{path}: k array and item records disagree")
    if 0 in asz:                          # no table and no answer_set: the golds seen
        seen = sorted({g for g in golds if g is not None})
        asz = [a or len(seen) for a in asz]
    asz = np.array(asz, int)

    flags = {}
    for name in ("correct_strict", "correct_constrained", "correct"):
        v = col(name, lambda x: bool(x))
        if v is not None:
            flags[name] = np.array(v, bool)
            a = arrays.get(name)
            if (a is not None and a.ndim == 1 and len(a) == n
                    and not np.array_equal(a.astype(bool), flags[name])):
                raise RunError(f"{path}: {name} array and item records disagree")
    key = next((k for k in ("correct_strict", "correct_constrained", "correct") if k in flags), None)
    if key is None:
        raise RunError(f"{path}: no correctness field")
    _check_predictions(path, items, golds, arrays, flags, n)
    fl = lambda v: np.nan if v is None else float(v)
    gold_prob = np.array(col("gold_prob", fl) or [np.nan] * n, float)
    digit_mass = np.array(col("digit_mass", fl) or [np.nan] * n, float)
    for name, a in (("gold_prob", gold_prob), ("digit_mass", digit_mass)):
        f = a[np.isfinite(a)]
        if len(f) and (f.min() < -1e-4 or f.max() > 1 + 1e-3):   # probabilities
            raise RunError(f"{path}: {name} outside [0, 1] ({f.min():.4g}..{f.max():.4g})")
    hb = [t.get("has_bos") for t in items if t.get("has_bos") is not None]
    position = _measured_position(items, digit_mass)

    # activity: shapes agree with the records and nothing is NaN
    L, H = meta.get("n_layers"), meta.get("n_heads")
    C = arrays.get("contribution")
    if C is not None:
        if C.ndim != 3 or C.shape[0] != n:
            raise RunError(f"{path}: contribution of shape {C.shape} for {n} items")
        if (L is not None and C.shape[1] != L) or (H is not None and C.shape[2] != H):
            raise RunError(f"{path}: contribution {C.shape} but meta says L={L} H={H}")
        L, H = C.shape[1], C.shape[2]
        if not np.isfinite(C).all():
            raise RunError(f"{path}: contribution contains NaN/inf (head_dim bug in the "
                           f"capture); re-run it")
        if (C < 0).any():
            raise RunError(f"{path}: negative contribution; not a norm")
    sandwich = [int(l) for l in (meta.get("sandwich_layers") or [])]
    Wn = arrays.get("write_norm")
    if Wn is not None:
        if C is None or Wn.shape != C.shape:
            raise RunError(f"{path}: write_norm without matching contribution")
        Wn = Wn.astype(float)
        plain = np.ones(L, bool)
        plain[sandwich] = False
        # layers without a post-attention norm write the pre-norm vector itself
        Wn[:, plain] = np.where(np.isfinite(Wn[:, plain]), Wn[:, plain], C[:, plain])
        if not np.isfinite(Wn).all():
            raise RunError(f"{path}: write_norm NaN/inf on post-norm layers")
        arrays["write_norm"] = Wn
    A0 = arrays.get("pos0_attn")
    if A0 is not None:
        if C is None or A0.shape != C.shape:
            raise RunError(f"{path}: pos0_attn of shape {A0.shape} does not match activity")
        if not np.isfinite(A0).all() or A0.min() < -1e-3 or A0.max() > 1 + 1e-3:
            raise RunError(f"{path}: pos0_attn outside [0, 1] or NaN")

    chance = meta.get("chance") if meta.get("chance") is not None else sm.get("chance")
    if chance is None:
        chance = float(np.mean(1.0 / np.maximum(asz, 1)))
    # what makes two files with the same name different captures (sweep_probe
    # names every output <model>__<suite>.npz whatever the split or --limit)
    splits = sorted({str(t["split"]) for t in items if t.get("split")})
    split = meta.get("split") or (splits[0] if len(splits) == 1 else "all" if splits else None)
    return {
        "path": path, "format": fmt, "model": meta.get("model") or Path(path).stem,
        "suite": suite, "family": family, "wording": wording, "setting": setting,
        "split": split, "limit": meta.get("limit"),
        "accuracy_only": bool(meta.get("accuracy_only")) or "contribution" not in arrays,
        "suite_sha256": meta.get("suite_sha256"),
        "chance": float(chance),
        "attn_impl": meta.get("attn_impl") or li.get("attn_impl"),
        "precision": _norm_precision(meta.get("precision") or li.get("precision")),
        "L": L, "H": H, "n": n, "arrays": arrays, "meta": meta, "items": items,
        "ids": np.array(ids), "k": ks, "gold": golds, "answer_set_size": asz,
        "item_family": np.array(fams), "item_wording": np.array(words),
        "echo": echo, "correctness": key, "correct": flags[key],
        "correct_strict": flags.get("correct_strict"),
        "correct_constrained": flags.get("correct_constrained"),
        "gold_prob": gold_prob, "digit_mass": digit_mass,
        "has_bos": (sum(bool(b) for b in hb) / len(hb)) if hb else None,
        "sandwich_layers": sandwich, "position": position, "content_sha1": content,
        # on a cycle, k steps forward = n - k steps back, so depth is clean only
        # where min_hops == k (NaN where the family has no such shortcut data)
        "min_hops": np.array([_item_min_hops(t, t.get("family") or family) for t in items], float),
    }


def _npz_fingerprint(z, meta, arrays):
    """SHA-1 over a capture's meta and every array, so that only files with
    identical contents count as copies (two captures of one suite share all
    their metadata but not their numbers).  Arrays not loaded (_SKIP_ARRAYS)
    enter by the CRC-32 and size the zip directory holds for them."""
    h = hashlib.sha1(json.dumps(meta, sort_keys=True).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"|{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    zf = getattr(z, "zip", None)
    for name in sorted(n for n in z.files if n in _SKIP_ARRAYS):
        try:
            info = zf.getinfo(name + ".npy")
            h.update(f"|{name}|crc{info.CRC}|{info.file_size}|".encode())
        except (AttributeError, KeyError):
            a = np.ascontiguousarray(z[name])
            h.update(f"|{name}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _measured_position(items, digit_mass):
    """Was activity measured at the position whose next token is the answer
    digit?  Returns {"status", "evidence"}:
      aligned     the items record deep_probe.align_answer ('appended' or
                  'measured_token'): the position is right by construction
      verified    no record, but the position predicts a digit (median digit
                  mass >= 0.5, or top-1 next token a digit on most items);
                  GPT-2 and Pythia keep " 6" whole and are right at the last
                  prompt token
      misaligned  no record and the file's own predictions show a position
                  that does not predict the answer: top-1 next token is
                  whitespace on at least half the items (the space that
                  split-digit tokenizers put before the digit: Llama-3, Qwen2.5,
                  OLMo-2 'Ġ'; Mistral, Yi, Phi-3, Gemma-2 '▁', which decodes
                  to ''), or median digit mass < 0.5.  The 17 September
                  deep_probe runs look like this
      unverified  no record and no evidence either way"""
    n = len(items)
    rec = sum(("appended" in t) or ("measured_token" in t) for t in items)
    if rec == n:
        return {"status": "aligned", "evidence": "align_answer record on every item"}
    top = []
    for t in items:
        t5 = t.get("top5")
        if isinstance(t5, (list, tuple)) and t5 and isinstance(t5[0], (list, tuple)) and t5[0]:
            top.append(str(t5[0][0]))
    ws = sum(s.strip() == "" for s in top)
    dg = sum(s.strip().isdigit() for s in top)
    dm = np.asarray(digit_mass, float)
    dm = dm[np.isfinite(dm)]
    what = "no align_answer record" + (f" (only {rec}/{n} items)" if rec else "")
    if top and 2 * ws >= len(top):
        return {"status": "misaligned",
                "evidence": f"{what}; top-1 next token is whitespace on {ws}/{len(top)} items"}
    if len(dm) and np.median(dm) < 0.5:
        return {"status": "misaligned", "evidence": f"{what}; median digit mass "
                                                    f"{np.median(dm):.2f} at the measured position"}
    if len(dm):
        return {"status": "verified",
                "evidence": f"{what}; median digit mass {np.median(dm):.2f}"}
    if top and 2 * dg > len(top):
        return {"status": "verified",
                "evidence": f"{what}; top-1 next token is a digit on {dg}/{len(top)} items"}
    return {"status": "unverified",
            "evidence": what + ("; no top-5 predictions or digit mass" if not top else
                                f"; top-1 next token a digit on {dg}/{len(top)} items, "
                                f"whitespace on {ws}")}


def _dig(x):
    """A stored digit (str or int) as a string; None if absent."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    return str(x).strip()


def _check_predictions(path, items, golds, arrays, flags, n):
    """The correctness flags drive the correct-item analysis, so they must
    agree with the predictions stored beside them (both probes define
    correct_constrained = pred_constrained == gold and correct_strict =
    correct_constrained and first generated digit == gold).  A contradiction
    means a corrupted or hand-edited file."""
    def one_d(name):
        a = arrays.get(name)
        return a if a is not None and a.ndim == 1 and len(a) == n else None

    gold = list(golds)
    for name in ("gold", "pred_constrained"):             # arrays and records agree
        a = one_d(name)
        if a is None:
            continue
        for i, t in enumerate(items):
            v = gold[i] if name == "gold" else _dig(t.get(name))
            if v is not None and v.isdigit() and int(v) != int(a[i]):
                raise RunError(f"{path}: item {i} {name} {v!r} but the {name} array holds "
                               f"{int(a[i])}")
        if name == "gold":
            gold = [g if g is not None else str(int(a[i])) for i, g in enumerate(gold)]
    pred = [_dig(t.get("pred_constrained")) for t in items]
    if all(p is None for p in pred) and one_d("pred_constrained") is not None:
        pred = [str(int(v)) for v in one_d("pred_constrained")]
    cc, cs = flags.get("correct_constrained"), flags.get("correct_strict")
    for i in range(n):
        g = gold[i]
        if cc is not None and g is not None and pred[i] is not None and bool(cc[i]) != (pred[i] == g):
            raise RunError(f"{path}: item {i} correct_constrained={bool(cc[i])} but "
                           f"pred_constrained {pred[i]!r} and gold {g!r}")
        if cs is not None and cc is not None:
            if cs[i] and not cc[i]:
                raise RunError(f"{path}: item {i} correct_strict without correct_constrained")
            if g is not None and "pred_first_digit" in items[i]:
                want = bool(cc[i]) and _dig(items[i]["pred_first_digit"]) == g
                if bool(cs[i]) != want:
                    raise RunError(f"{path}: item {i} correct_strict={bool(cs[i])} but first "
                                   f"generated digit {items[i]['pred_first_digit']!r}, gold {g!r}")


def _subset(run, idx):
    """The run restricted to the items idx (every per-item field and array)."""
    idx = np.asarray(idx, int)
    n = run["n"]
    out = dict(run)
    out["arrays"] = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
                     for k, v in run["arrays"].items()}
    for f in ("ids", "k", "answer_set_size", "item_family", "item_wording", "correct",
              "gold_prob", "digit_mass", "correct_strict", "correct_constrained"):
        if run.get(f) is not None:
            out[f] = np.asarray(run[f])[idx]
    for f in ("gold", "echo", "items"):
        out[f] = [run[f][i] for i in idx]
    out["n"] = int(len(idx))
    return out


def _activity(run, metric):
    return run["arrays"].get(metric)


def _run_label(run):
    return f"{_short(run['model'])} / {run['suite']}"


# ────────────────────────────── per-run analyses ────────────────────────────
def _by_k(k, ok, y, yp):
    rows = []
    for d in np.unique(k):
        m = k == d
        rows.append({"k": int(d), "n": int(m.sum()), "correct": int(ok[m].sum()),
                     "acc": float(ok[m].mean()), "top_share": float(_nanmean(y[m])),
                     "pr_h": float(_nanmean(yp[m]))})
    return rows


def _pct(y, k, lo_k, hi_k):
    a, b = _nanmean(y[k == lo_k]), _nanmean(y[k == hi_k])
    return float(100 * (b / a - 1)) if np.isfinite(a) and a != 0 else float("nan")


def _correct_vs_wrong(y, k, ok, n_perm, rng):
    """Correct minus wrong at fixed k, averaged over depths with >= 3 of each;
    null: correctness shuffled within each depth."""
    strata = [np.nonzero((k == d) & np.isfinite(y))[0] for d in np.unique(k)]
    strata = [ix for ix in strata if ok[ix].sum() >= 3 and (~ok[ix]).sum() >= 3]
    if not strata:
        return None
    def diff(o):
        return float(np.mean([y[ix][o[ix]].mean() - y[ix][~o[ix]].mean() for ix in strata]))
    obs, null, o = diff(ok), np.empty(n_perm), ok.copy()
    for j in range(n_perm):
        for ix in strata:
            o[ix] = rng.permutation(ok[ix])
        null[j] = diff(o)
    return {"diff": obs, "p": _p_two_sided(obs, null),
            "depths": [int(k[ix[0]]) for ix in strata]}


def _adjusted_slope(y, k, ok, gp):
    """Descriptive slope on k with correctness and gold probability as
    covariates (outline: all-items results reported with both as covariates)."""
    m = np.isfinite(y) & np.isfinite(gp)
    if m.sum() < 5:
        return None
    X = [np.ones(m.sum()), k[m].astype(float)]
    if 0 < ok[m].sum() < m.sum():
        X.append(ok[m].astype(float))
    X.append(gp[m])
    beta, *_ = np.linalg.lstsq(np.stack(X, 1), y[m], rcond=None)
    return float(beta[1])


def signature(run, metric="contribution", *, alpha=0.05, n_perm=10000, rng=None,
              lo=0.4, hi=0.8, min_correct=20, head_mask=None):
    """A run's signature on the primary statistic (see module docstring).

    head_mask [n, L, H] (True = drop the head from the statistic) is how the
    sink control removes sink-dominated heads.  Classification:
      classification_all_items  spread / funnel / flat, or mixed when the
                                top-share and PR/H slopes are both significant
                                with the same sign
      classification            missing without a correct-item analysis
                                (fewer than 3 eligible depths) or without the
                                metric; inconclusive when the correct-item and
                                all-items slopes disagree in sign; else the
                                all-items class"""
    run = _as_run(run)
    rng = _rng(rng)
    base = {"model": run["model"], "suite": run["suite"], "family": run["family"],
            "wording": run["wording"], "metric": metric, "n": run["n"], "L": run["L"],
            "H": run["H"], "attn_impl": run["attn_impl"], "precision": run["precision"],
            "correctness": run["correctness"], "chance": run["chance"],
            "acc_threshold": 2 * run["chance"] / (1 + run["chance"])}
    S = _activity(run, metric)
    if S is None:
        return {**base, "classification": "missing", "classification_all_items": "missing",
                "reason": f"no {metric} in this run (accuracy-only?)"}
    L, H = run["L"], run["H"]
    w = window(L, lo, hi)
    TS, PR = top_share(S, head_mask), pr_h(S, head_mask)
    fin = TS[np.isfinite(TS)]
    if len(fin) and (fin.min() < 1 / H - 1e-6 or fin.max() > 1 + 1e-6):
        raise RunError(f"{_run_label(run)}: top share outside [1/H, 1]")
    y, yp = _nanmean(TS[:, w], 1), _nanmean(PR[:, w], 1)
    k, ok = run["k"], run["correct"]
    kmin, kmax = int(k.min()), int(k.max())
    base.update(window=[w.start, w.stop - 1], by_k=_by_k(k, ok, y, yp))
    if kmin == kmax:
        return {**base, "classification": "missing", "classification_all_items": "missing",
                "reason": "k does not vary"}

    s, p = slope_perm(y, k, n_perm, rng)
    spr, ppr = slope_perm(yp, k, n_perm, rng)
    if not np.isfinite(s):                  # e.g. every head masked out
        return {**base, "classification": "missing", "classification_all_items": "missing",
                "reason": "primary statistic undefined on too many items"}
    allr = {"slope": s, "p": p, "pct_change": _pct(y, k, kmin, kmax), "k_from": kmin,
            "k_to": kmax, "mean": float(_nanmean(y)),
            "slope_adjusted": _adjusted_slope(y, k, ok, run["gold_prob"])}
    clean = _clean_depth(y, k, run.get("min_hops"), run["family"], n_perm, rng)
    elig = eligible_depths(k, ok, run["chance"], min_correct)
    if len(elig) >= 3:
        sel = ok & np.isin(k, elig)
        sc, pc = slope_perm(y[sel], k[sel], n_perm, rng)
        corr = {"status": "ok", "slope": sc, "p": pc, "n": int(sel.sum()), "depths": elig,
                "pct_change": _pct(y[sel], k[sel], min(elig), max(elig))}
    else:
        corr = {"status": "missing", "depths": elig,
                "reason": f"{len(elig)} depth(s) with >= {min_correct} correct and accuracy "
                          f">= {base['acc_threshold']:.3f}; need 3"}

    sig, sig_pr = p < alpha, ppr < alpha
    if sig and sig_pr and _sign(s) == _sign(spr):
        cls_all = "mixed"
    elif sig:
        cls_all = "spread" if s < 0 else "funnel"
    else:
        cls_all = "flat"
    if corr["status"] != "ok":
        cls, reason = "missing", corr["reason"]
    elif _sign(corr["slope"]) != _sign(s):
        cls, reason = "inconclusive", "correct-item and all-items slopes disagree in sign"
    else:
        cls, reason = cls_all, None
    return {**base, "all_items": allr, "pr_h": {"slope": spr, "p": ppr},
            "eligible_depths": elig, "correct_items": corr, "clean_depth": clean,
            "correct_vs_wrong": _correct_vs_wrong(y, k, ok, min(n_perm, 2000), rng),
            "classification_all_items": cls_all, "classification": cls, "reason": reason}


def _clean_depth(y, k, mh, family, n_perm, rng):
    """The primary statistic's slope on items whose shortest lookup path equals
    k.  On an n-cycle, k steps forward land where n - k steps backward do, so
    for k > n/2 the written depth overstates the work a model that can read
    the rules backwards must do; the slope over the clean items is the depth
    effect free of that shortcut.  Also the slope on min_hops over all items."""
    if mh is None or not len(mh) or not np.isfinite(mh).all():
        why = ("no cycle shortcut in this family" if family in ("arith", "binding", "relational")
               else "min_hops not recorded and not recoverable from the prompts")
        return {"status": "not_applicable", "reason": why}
    sel = mh == k
    out = {"confounded_depths": sorted({int(d) for d in k[~sel]}),
           "min_hops_by_k": {int(d): sorted({int(x) for x in mh[k == d]}) for d in np.unique(k)}}
    if sel.all():
        return {**out, "status": "all_clean"}
    ds = sorted({int(d) for d in k[sel]})
    if len(ds) >= 2 and sel.sum() >= 6:
        s, p = slope_perm(y[sel], k[sel], n_perm, rng)
        out.update(status="ok", slope=s, p=p, n=int(sel.sum()), depths=ds,
                   pct_change=_pct(y[sel], k[sel], min(ds), max(ds)))
    else:
        out.update(status="missing", depths=ds, reason="fewer than 2 clean depths")
    if len(np.unique(mh)) >= 2:
        sm, pm = slope_perm(y, mh, n_perm, rng)
        out["slope_on_min_hops"] = {"slope": sm, "p": pm}
    return out


def power(run, delta=0.02, alpha=0.05, power=0.8, *, metric="contribution", lo=0.4, hi=0.8):
    """Items per depth needed to detect a relative change `delta` in the primary
    statistic between the lowest and highest depth, with the design's slope test
    (two-sided, level alpha) at the given power.  Normal approximation with the
    pilot's OLS residual variance (y on k) and a linear change across depths;
    on the Sep 22 pilots this gives 7 items for Qwen2.5-7B and 82 for
    OLMo-2-7B-Instruct."""
    run = _as_run(run)
    S = _activity(run, metric)
    if S is None:
        raise ValueError(f"{_run_label(run)}: no {metric}")
    y, k = top_share_window(S, run["L"], lo, hi), run["k"].astype(float)
    m = np.isfinite(y)
    y, k = y[m], k[m]
    depths = np.unique(k)
    if len(depths) < 2 or len(y) < 4:
        raise ValueError("power needs at least two depths and four items")
    res = y - y.mean() - _ols_slope(y, k) * (k - k.mean())
    var = float(res @ res / (len(y) - 2))
    beta = delta * y[k == depths[0]].mean() / (depths[-1] - depths[0])
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    sxx = float(((depths - depths.mean()) ** 2).sum())
    return int(max(2, math.ceil(z * z * var / (beta * beta * sxx))))


def sink_control(run, threshold=0.9, *, metric="contribution", **kw):
    """Signature with sink-dominated heads (attention from the measured position
    to position 0 above `threshold`) removed per item and layer.  Qwen2.5 has
    no BOS, so its position 0 is a content token; Llama-3's is a BOS."""
    run = _as_run(run)
    A0 = run["arrays"].get("pos0_attn")
    if A0 is None:
        return {"status": "missing", "reason": "no pos0_attn (run without --pos0)",
                "model": run["model"], "suite": run["suite"]}
    ex = A0 > threshold
    w = window(run["L"], kw.get("lo", 0.4), kw.get("hi", 0.8))
    # one seed for both: with nothing removed the two signatures must be
    # identical, not differ by Monte Carlo noise that can flip a class near alpha
    seed = int(_rng(kw.pop("rng", None)).integers(1 << 62))
    orig = signature(run, metric, rng=seed, **kw)
    masked = signature(run, metric, head_mask=ex, rng=seed, **kw)
    s0 = (orig.get("all_items") or {}).get("slope")
    s1 = (masked.get("all_items") or {}).get("slope")
    return {"status": "ok", "model": run["model"], "suite": run["suite"],
            "threshold": threshold, "has_bos": run["has_bos"],
            "frac_removed": float(ex.mean()), "frac_removed_window": float(ex[:, w].mean()),
            "frac_removed_by_layer": ex.mean((0, 2)).tolist(),
            "signature": orig, "signature_masked": masked,
            "sign_agree": _sign(s0) == _sign(s1),
            "class_agree": orig["classification_all_items"] == masked["classification_all_items"]}


def mass_centroid(run, metric="contribution", n_perm=5000, rng=None):
    """Centre of the attention-output mass sum_h s over layers, per item, and
    its slope on k: does the work move earlier or later as k rises?"""
    run = _as_run(run)
    S = _activity(run, metric)
    if S is None:
        return {"status": "missing", "reason": f"no {metric}"}
    if run["L"] is None or run["L"] < 2:
        raise ValueError(f"mass_centroid needs L >= 2 layers, got {run['L']}")
    M = np.asarray(S, float).sum(-1)
    c = (M * np.arange(run["L"])).sum(1) / M.sum(1)
    k = run["k"]
    s, p = slope_perm(c, k, n_perm, _rng(rng))
    lo, hi = k.min(), k.max()
    shift = float(c[k == hi].mean() - c[k == lo].mean())
    return {"status": "ok", "model": run["model"], "suite": run["suite"],
            "by_k": {int(d): float(c[k == d].mean()) for d in np.unique(k)},
            "slope": s, "p": p, "shift_layers": shift, "shift_rel": shift / (run["L"] - 1),
            "centroid": c}


# ────────────────────────────── cross-run analyses ──────────────────────────
def _item_fields(run, ix):
    """(prompt text, gold) of the run's items ix; None where not recorded."""
    items, gold = run.get("items"), run.get("gold")
    out = []
    for j in ix:
        t = items[j].get("text") if items is not None and isinstance(items[j], dict) else None
        g = gold[j] if gold is not None else None
        out.append((None if t is None else str(t).strip(), None if g is None else str(g)))
    return out


def _check_same_items(runs, idx, ids):
    """The same id must mean the same item in every run: same depth, prompt
    and gold wherever recorded.  Ids alone are not enough: legacy files get
    ids from the suite name and label ('serial_depth_fs/D1N00'), which two
    files of prompts drawn differently (another seed or generator version)
    share although no item is the same."""
    ref_k = runs[0]["k"][idx[0]]
    ref = [[None, None] for _ in ids]             # first recorded text / gold per id
    src = [[None, None] for _ in ids]
    for r, ix in zip(runs, idx):
        if not np.array_equal(r["k"][ix], ref_k):
            raise RunError(f"{_run_label(r)}: the same item ids carry different depths")
        bad = []
        for i, fields in enumerate(_item_fields(r, ix)):
            for f, v in enumerate(fields):
                if v is None:
                    continue
                if ref[i][f] is None:
                    ref[i][f], src[i][f] = v, r
                elif v != ref[i][f]:
                    bad.append((ids[i], ("prompt", "gold")[f], src[i][f]))
        if bad:
            i0, what, other = bad[0]
            raise RunError(f"{_run_label(r)}: {len({b[0] for b in bad})} of {len(ids)} shared item "
                           f"ids hold a different prompt or gold than in {_run_label(other)} "
                           f"(e.g. {i0!r}: {what}); the runs do not measure the same items")


def _align_items(runs, metric, lo=0.4, hi=0.8):
    """Primary statistic of each run on the item ids all runs share, in the
    first run's order.  The same id must mean the same item everywhere
    (depth, prompt and gold; _check_same_items)."""
    shared = set(runs[0]["ids"])
    for r in runs[1:]:
        shared &= set(r["ids"])
    ids = [i for i in runs[0]["ids"] if i in shared]
    if len(ids) < 3:
        raise ValueError(f"runs share {len(ids)} item ids; need >= 3")
    Y, idx = [], []
    for r in runs:
        S = _activity(r, metric)
        if S is None:
            raise ValueError(f"{_run_label(r)}: no {metric}")
        pos = {i: j for j, i in enumerate(r["ids"])}
        ix = np.array([pos[i] for i in ids])
        idx.append(ix)
        Y.append(top_share_window(S, r["L"], lo, hi)[ix])
    _check_same_items(runs, idx, ids)
    return ids, runs[0]["k"][idx[0]], np.stack(Y, 1)


def joint_relabel(runs, metric="contribution", n_perm=5000, *, rng=None, pairs=None,
                  scale="mean", lo=0.4, hi=0.8):
    """Cross-model test on shared items under ONE relabelling of k.

    Per run: OLS slope of the primary statistic on k, scale-normalised (divided
    by the run's mean, i.e. relative change per step; scale="sd" divides by
    its SD).  Each null draw shuffles k once and applies that shuffle to every
    run, the design-respecting null when models see the same items.  Reports
    each run's p, the pooled mean slope and its p, and paired differences
    (runs[j] - runs[i]; default for two runs: second minus first, e.g.
    instruct minus base)."""
    runs = [_as_run(r) for r in runs]
    rng = _rng(rng)
    ids, k, Y = _align_items(runs, metric, lo, hi)
    if not np.isfinite(Y).all():
        raise ValueError("non-finite primary statistic on shared items")
    norm = Y.mean(0) if scale == "mean" else Y.std(0, ddof=1)
    kc = k - k.mean()
    ss = kc @ kc
    if ss == 0:
        raise ValueError("k does not vary on the shared items")
    Yc = (Y - Y.mean(0)) / norm
    obs = kc @ Yc / ss
    null = np.concatenate([P @ Yc / ss for P in _perm_rows(kc, n_perm, rng)])  # [n_perm, R]
    names = [f"{_short(r['model'])}" for r in runs]
    if pairs is None:
        pairs = [(0, 1)] if len(runs) == 2 else []
    return {
        "n_items": len(ids), "models": [r["model"] for r in runs], "suite": runs[0]["suite"],
        "scale": scale, "slopes": obs.tolist(), "p_each": _perm_p_cols(obs, null).tolist(),
        "pooled": float(obs.mean()), "p_pooled": _p_two_sided(obs.mean(), null.mean(1)),
        "diffs": [{"a": names[i], "b": names[j], "diff": float(obs[j] - obs[i]),
                   "p": _p_two_sided(obs[j] - obs[i], null[:, j] - null[:, i])}
                  for i, j in pairs],
    }


def kernel_compare(run_a, run_b, *, metric="contribution", late=0.8, **kw):
    """Same model and items under two kernels (or precisions; pass the
    reference, e.g. fp32, as run_b): score correlation, median relative
    difference in the final 20% of layers, per-layer slope agreement for top
    share and PR/H, and both signatures.

    Everything, the two signatures included, is computed on the item ids the
    runs share (a kernel-check re-run may cover a subset, e.g. --limit), and
    both signatures use the same permutation seed, so identical activity gives
    identical signatures."""
    a, b = _as_run(run_a), _as_run(run_b)
    if not _model_match(a["model"], b["model"]):
        raise RunError(f"kernel_compare compares ONE model under two configurations; got "
                       f"{a['model']} and {b['model']}")
    if _activity(a, metric) is None or _activity(b, metric) is None:
        raise ValueError(f"kernel_compare needs {metric} in both runs")
    if (a["L"], a["H"]) != (b["L"], b["H"]):
        raise ValueError("runs differ in shape; not the same model")
    shared = set(a["ids"]) & set(b["ids"])
    ids = [i for i in a["ids"] if i in shared]
    if len(ids) < 3:
        raise ValueError("fewer than 3 shared items")
    pa = {i: j for j, i in enumerate(a["ids"])}
    pb = {i: j for j, i in enumerate(b["ids"])}
    ia, ib = np.array([pa[i] for i in ids]), np.array([pb[i] for i in ids])
    _check_same_items([a, b], [ia, ib], ids)       # same depth, prompt and gold per id
    Sa, Sb = _activity(a, metric)[ia].astype(float), _activity(b, metric)[ib].astype(float)
    k, L = a["k"][ia], a["L"]
    fin = np.arange(L) > math.floor(late * (L - 1) + 1e-9)
    den = np.abs(Sa) + np.abs(Sb)
    rel = np.where(den > 0, 2 * np.abs(Sa - Sb) / np.where(den > 0, den, 1), 0.0)
    per_layer = {}
    for name, f in (("top_share", top_share), ("pr_h", pr_h)):
        kc = k - k.mean()
        la = kc @ (f(Sa) - f(Sa).mean(0)) / (kc @ kc)
        lb = kc @ (f(Sb) - f(Sb).mean(0)) / (kc @ kc)
        per_layer[name] = {"r": float(np.corrcoef(la, lb)[0, 1]),
                           "same_sign": int((np.sign(la) == np.sign(lb)).sum()), "L": L}
    # one seed for both: the comparison is paired, not two independent nulls
    seed = int(_rng(kw.pop("rng", None)).integers(1 << 62))
    sa = signature(_subset(a, ia), metric, rng=seed, **kw)
    sb = signature(_subset(b, ib), metric, rng=seed, **kw)
    s1 = (sa.get("all_items") or {}).get("slope")
    s2 = (sb.get("all_items") or {}).get("slope")
    sign_agree = _sign(s1) == _sign(s2)
    class_agree = sa["classification_all_items"] == sb["classification_all_items"]
    outside = {"a": a["n"] - len(ids), "b": b["n"] - len(ids)}
    warn = ([f"run {s} has {v} item(s) outside the shared set; compared on the {len(ids)} "
             f"shared items" for s, v in outside.items() if v]) or None
    return {"model": a["model"], "suite": a["suite"], "n_items": len(ids),
            "n_items_a": a["n"], "n_items_b": b["n"], "warning": warn,
            "a": {"attn_impl": a["attn_impl"], "precision": a["precision"], "path": a["path"]},
            "b": {"attn_impl": b["attn_impl"], "precision": b["precision"], "path": b["path"]},
            "score_r": float(np.corrcoef(Sa.ravel(), Sb.ravel())[0, 1]),
            "late_rel_diff_median": float(np.median(rel[:, fin])) if fin.any() else None,
            "rel_diff_median": float(np.median(rel)),
            "per_layer": per_layer,
            "signature_a": sa, "signature_b": sb,
            "sign_agree": sign_agree, "class_agree": class_agree,
            "agree": bool(sign_agree and class_agree)}


def _sign_of(pre):
    if isinstance(pre, (int, float, np.integer, np.floating)) and not isinstance(pre, bool):
        if pre == 0:
            raise ValueError("preregistered sign must be non-zero")
        return int(np.sign(pre))
    s = str(pre).strip().lower()
    if s in ("funnel", "funnelling", "+", "+1", "1", "positive", "up"):
        return 1
    if s in ("spread", "spreading", "-", "-1", "negative", "down"):
        return -1
    raise ValueError(f"preregistered sign {pre!r}: use funnel/spread or +1/-1")


_NOT_PREREGISTERED = ("none", "skip", "exclude")


def _prereg_value(v):
    """A --preregistered value: +1 / -1, or None (none / skip / exclude) for a
    model that gets no RQ1 verdict."""
    if v is None or (isinstance(v, str) and v.strip().lower() in _NOT_PREREGISTERED):
        return None
    return _sign_of(v)


def _post_marks(name):
    return {m.lower() for m in _POST_MARK.findall(str(name))}


def _prereg_match(model, prereg):
    """(value, pattern) of the --preregistered entry that covers `model`, or
    (None, None).  value None: excluded on purpose (MODEL=none).

      text       occurs in the model name (case-insensitive).  A post-trained
                 model (-Instruct, -SFT, -DPO, -it, -chat, -RLVR) is covered
                 only by a pattern naming its marker: the outline pre-registers
                 base and post-trained models separately ("Qwen2.5 base:
                 funnel", "post-trained OLMo: spread"), so qwen2.5 must not
                 reach Qwen2.5-7B-Instruct
      text$      the same, anchored at the end of the name
      re:REGEX   regular expression searched in the name; no marker rule
    Precedence: a pattern equal to the full or short model name, then the
    longest pattern.  Equally specific patterns with different values raise."""
    full = str(model).lower()
    short = _short(full)
    marks = _post_marks(short)
    hits = []
    for pat, val in prereg.items():
        p = str(pat).strip()
        pl = p.lower()
        if pl.startswith("re:"):
            ok, exact = re.search(p[3:], str(model), re.I) is not None, False
        else:
            body = pl[:-1] if pl.endswith("$") else pl
            if not body:
                continue
            ok = ((full.endswith(body) if pl.endswith("$") else body in full)
                  and marks <= _post_marks(body))
            exact = body in (full, short)
        if ok:
            hits.append((exact, len(p), p, val))
    if not hits:
        return None, None
    best = max(h[:2] for h in hits)
    top = [h for h in hits if h[:2] == best]
    if len({h[3] for h in top}) > 1:
        raise ValueError(f"{model}: equally specific --preregistered patterns disagree: "
                         f"{', '.join(h[2] for h in top)}")
    return top[0][3], top[0][2]


def _pre_sign_for(model, prereg):
    """The pre-registered sign of `model` (None: not pre-registered)."""
    v, _ = _prereg_match(model, {p: _prereg_value(s) for p, s in prereg.items()})
    return v


def _stratified_boot_idx(ks, n_boot, rng):
    idx = np.empty((n_boot, len(ks)), int)
    pos = 0
    for d in np.unique(ks):
        ix = np.nonzero(ks == d)[0]
        idx[:, pos:pos + len(ix)] = ix[rng.integers(0, len(ix), (n_boot, len(ix)))]
        pos += len(ix)
    return idx


def _mixedlm(present):
    """Secondary: random intercepts and slopes by family, variance components
    for wording (cell) intercepts and slopes within family.  Reported only if
    statsmodels imports and the optimiser converges.  A variance on the
    boundary (zero), which is the MLE when families agree, keeps status "ok"
    but sets boundary=True: the Hessian is then not positive definite and the
    slope's SE ignores between-family variation, so the report flags it."""
    fams = {c[0] for c in present}
    if len(fams) < 2:
        return {"status": "skipped", "reason": "fewer than 2 families"}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as e:
        return {"status": "unavailable", "reason": str(e)}
    # present is keyed by (family, cell label); a cell is one wording of one suite
    rows = [(float(y), float(k), f, cell) for (f, cell), (ys, ks) in present.items()
            for y, k in zip(ys, ks)]
    df = pd.DataFrame(rows, columns=["y", "k", "family", "cell"])
    sd = float(df["y"].std()) or 1.0
    # unit scale and centred k: the optimiser fails far more often without
    df["y"] = (df["y"] - df["y"].mean()) / sd
    df["k"] = df["k"] - df["k"].mean()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = smf.mixedlm("y ~ k", df, groups=df["family"], re_formula="~k",
                             vc_formula={"cell": "0 + C(cell)", "cell_k": "0 + C(cell):k"})
            fit = md.fit(reml=True, method=["powell", "lbfgs"])
        msgs = sorted({str(w.message).split(".")[0] for w in caught})
        failed = [m for m in msgs if "failed" in m.lower()]
        if not fit.converged or failed or not np.isfinite(fit.bse["k"]):
            return {"status": "not converged", "warnings": msgs[:4]}
        boundary = any(w in m.lower() for m in msgs
                       for w in ("not positive definite", "boundary", "singular"))
        return {"status": "ok", "boundary": bool(boundary),
                "slope": float(fit.fe_params["k"] * sd),
                "se": float(fit.bse["k"] * sd), "p": float(fit.pvalues["k"]), "notes": msgs[:4]}
    except Exception as e:
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}


def _cell_labels(keys):
    """Report names of RQ1 cells keyed (family, wording, suite, suite_sha256):
    'family/wording', with the suite added when one family and wording has
    several suites (and a hash prefix if their names also coincide)."""
    fw = Counter((f, w) for f, w, _, _ in keys)
    fws = Counter((f, w, s) for f, w, s, _ in keys)
    out = {}
    for f, w, s, h in keys:
        lab = f"{f}/{w}"
        if fw[(f, w)] > 1:
            lab += f" [{s}]"
            if fws[(f, w, s)] > 1:
                lab += f" sha {str(h)[:8]}"
        out[(f, w, s, h)] = lab
    return out


def _invariance_core(cells, mode, sign, alpha, n_boot, n_perm, rng, min_correct, family_frac,
                     min_families):
    """cells {(family, wording, suite, suite_sha256): (y, k, correct, chance)};
    a (family, wording) key is one cell of an unnamed suite."""
    cells = {(tuple(c) + (None, None) if len(c) == 2 else tuple(c)): v for c, v in cells.items()}
    labels = _cell_labels(list(cells))
    cell_out, present, where = {}, {}, {}
    for key in sorted(cells, key=lambda c: (c[0], c[1], str(c[2]), str(c[3]))):
        fam, wd, suite, _ = key
        y, k, ok, ch = cells[key]
        lab = labels[key]
        info = {"family": fam, "wording": wd, "suite": suite}
        fin = np.isfinite(y)
        if mode == "correct":
            elig = eligible_depths(k[fin], ok[fin], float(np.mean(ch)), min_correct)
            if len(elig) < 3:
                cell_out[lab] = {"status": "missing", "eligible_depths": elig, **info}
                continue
            sel = fin & ok & np.isin(k, elig)
        else:
            elig, sel = sorted(int(d) for d in np.unique(k[fin])), fin
            if len(elig) < 2:
                cell_out[lab] = {"status": "missing", "eligible_depths": elig, **info}
                continue
        present[(fam, lab)] = (y[sel], k[sel].astype(float))
        where[lab] = (wd, suite)
        cell_out[lab] = {"status": "ok", "eligible_depths": elig, **info,
                         "n": int(sel.sum()), "slope": _ols_slope(y[sel], k[sel])}
    all_fams = sorted({c[0] for c in cells})
    if not present:
        return {"verdict": "missing", "cells": cell_out, "families": {},
                "families_missing": all_fams, "n_families_present": 0}

    keys = list(present)
    obs = np.array([_ols_slope(*present[c]) for c in keys])
    boot = np.empty((n_boot, len(keys)))
    perm = np.empty((n_perm, len(keys)))
    for j, c in enumerate(keys):
        ys, ks = present[c]
        bi = _stratified_boot_idx(ks, n_boot, rng)
        boot[:, j] = _row_slopes(ys[bi], ks[bi])
        pk = np.concatenate(list(_perm_rows(ks, n_perm, rng)))
        perm[:, j] = _row_slopes(np.broadcast_to(ys, pk.shape), pk)
    pooled = float(obs.mean())
    ci = [float(np.percentile(boot.mean(1), 2.5)), float(np.percentile(boot.mean(1), 97.5))]
    p_pooled = _p_two_sided(pooled, perm.mean(1))

    fams = sorted({c[0] for c in keys})
    fres = {}
    for f in fams:
        cols = [j for j, c in enumerate(keys) if c[0] == f]
        s = float(obs[cols].mean())                # mean over the family's (wording, suite) cells
        cl = [keys[j][1] for j in cols]
        fres[f] = {"slope": s, "p": _p_two_sided(s, perm[:, cols].mean(1)),
                   "wordings": sorted({where[c][0] for c in cl}), "cells": cl,
                   "suites": [where[c][1] for c in cl]}
    ph = holm([fres[f]["p"] for f in fams])
    for f, h in zip(fams, ph):
        fres[f]["p_holm"] = float(h)
    n_present = len(fams)
    n_with = sum(_sign(fres[f]["slope"]) == sign for f in fams)
    need = math.ceil(family_frac * n_present - 1e-9)
    opposite = [f for f in fams if fres[f]["p_holm"] < alpha and _sign(fres[f]["slope"]) == -sign]
    a = (ci[0] > 0) if sign > 0 else (ci[1] < 0)
    # one family cannot show invariance ACROSS families, whatever its sign
    b = n_with >= need and n_present >= min_families
    c = not opposite
    verdict = ("task-dependent" if opposite else "invariant" if (a and b and c)
               else "inconclusive")
    return {"verdict": verdict, "pooled": pooled, "ci95": ci, "p_pooled": p_pooled,
            "criteria": {"a_pooled_ci_excludes_0_with_sign": bool(a),
                         "b_families_with_sign": f"{n_with}/{n_present} (need {need}, "
                                                 f"and >= {min_families} families)",
                         "b": bool(b), "c_no_significant_opposite_family": bool(c)},
            "opposite_families": opposite, "cells": cell_out, "families": fres,
            "families_missing": [f for f in all_fams if f not in fams],
            "n_families_present": n_present, "_present": present}


def invariance(runs_of_one_model, preregistered_sign, alpha=0.05, *, metric="contribution",
               n_boot=2000, n_perm=5000, rng=None, min_correct=20, lo=0.4, hi=0.8,
               family_frac=0.75, min_families=2, mixedlm=True, task_families=TASK_FAMILIES):
    """RQ1 for one model: is its pre-registered direction invariant across task
    families and wordings?

    Pass the runs of ONE configuration (kernel, precision).  Only items of the
    task families (`task_families`, default pointer / arith / binding /
    relational; None keeps every family) form cells: the numeral control
    (fixed_digit) and the single-prompt ladder are not task families, and
    counting them would change both the verdict and the 3-of-4 rule; their
    cells are listed under excluded_cells.

    Items are pooled into (family, wording, suite) cells: two suites of one
    family and wording (E1 picks a setting per family AND comparison group, so
    a model on the matched pair and a ladder can run two settings) are two
    cells, never one regression, whose pooled OLS could show a slope that
    neither suite has (Simpson's paradox from different levels and k ranges).
    Per cell: OLS slope of the primary statistic on k over the items correct
    at eligible depths (cells with fewer than 3 eligible depths are missing
    and do not count).  Pooled slope = mean of cell slopes, 95% CI by a
    bootstrap that resamples items within each (cell, k) stratum; family slope
    = mean over its cells (wordings x suites), p from shuffling k within
    cells.  Verdict:
      invariant       (a) pooled CI excludes 0 with the pre-registered sign,
                      (b) at least ceil(3/4 of the families present) have
                      that sign, and at least min_families (2) are present,
                      (c) no family significant in the opposite sign after
                      Holm across families
      task-dependent  some family significant in the opposite sign (Holm)
      inconclusive    otherwise;  missing: no cell meets the rules
    The same on all items is reported, with a flag when the verdicts differ.
    p_pooled (permutation) is the per-model primary p for Holm across models."""
    rng = _rng(rng)
    sign = _sign_of(preregistered_sign)
    runs = [_as_run(r) for r in runs_of_one_model]
    seen = [i for r in runs for i in r["ids"]]
    if len(set(seen)) != len(seen):
        raise RunError("the same item ids appear in two runs of this model; one run per "
                       "suite and configuration")
    tf = None if task_families is None else {str(f) for f in task_families}
    cells, skipped, excluded = {}, [], {}
    for r in runs:
        S = _activity(r, metric)
        if S is None:
            skipped.append(_run_label(r))
            continue
        y = top_share_window(S, r["L"], lo, hi)
        ch = 1.0 / np.maximum(r["answer_set_size"], 1)
        suite, sha = str(r.get("suite")), r.get("suite_sha256")
        for fam, wd in sorted(set(zip(r["item_family"], r["item_wording"]))):
            if tf is not None and fam not in tf:
                excluded[f"{fam}/{wd}"] = CONTROL_FAMILIES.get(fam, "not a task family")
                continue
            m = (r["item_family"] == fam) & (r["item_wording"] == wd)
            parts = (y[m], r["k"][m], r["correct"][m], ch[m])
            key = (fam, wd, suite, sha)
            if key in cells:          # one suite over two captures (its calib and test halves)
                parts = tuple(np.concatenate([o, p]) for o, p in zip(cells[key], parts))
            cells[key] = parts
    args = (sign, alpha, n_boot, n_perm, rng, min_correct, family_frac, min_families)
    corr = _invariance_core(cells, "correct", *args)
    alli = _invariance_core(cells, "all", *args)
    present = corr.pop("_present", None)
    alli.pop("_present", None)
    out = {"models": sorted({r["model"] for r in runs}), "metric": metric,
           "preregistered_sign": sign, "alpha": alpha, "skipped_runs": skipped,
           "task_families": None if tf is None else sorted(tf), "excluded_cells": excluded,
           "correct_items": corr, "all_items": alli,
           "verdict": corr["verdict"], "verdict_all_items": alli["verdict"],
           "disagree": (corr["verdict"] != alli["verdict"]
                        and "missing" not in (corr["verdict"], alli["verdict"]))}
    if mixedlm and present:
        out["mixedlm"] = _mixedlm(present)
    return out


# ────────────────────────────── E1 calibration ──────────────────────────────
def _model_match(run_model, name):
    a, b = str(run_model).lower(), str(name).lower()
    return a == b or _short(a) == _short(b)


def group_by_setting(runs):
    """{setting key: [runs]} from each run's suite setting (canonical JSON), the
    input choose_setting expects."""
    out = {}
    for r in runs:
        r = _as_run(r)
        out.setdefault(json.dumps(r["setting"], sort_keys=True), []).append(r)
    return out


def _test_per_k(run):
    """Items per depth on the TEST split of the run's suite, for a capture of
    the calibration split whose setting records n_per and calib_frac
    (serial_tasks puts round(calib_frac * n_per) items of every depth, or of
    every shared body, in calib); None otherwise (the run's own counts are
    then judged, as for a test-split or legacy run)."""
    if run.get("split") != "calib":
        return None
    s = run.get("setting") or {}
    try:
        n_per, f = int(s["n_per"]), float(s["calib_frac"])
    except (KeyError, TypeError, ValueError):
        return None
    return n_per - int(round(f * n_per))


def _cell_eligibility(k, ok, chance, min_correct, n_test):
    """Eligible depths of one (family, wording) cell and its by-k rows.  With
    n_test (items per depth on the test split) the rule is judged for the
    test split the setting is chosen for: calibration accuracy >= 2c/(1+c)
    and projected correct items acc * n_test >= min_correct.  (Applied to the
    calibration split itself, the absolute count could never be met: 12
    calibration items per depth at n_per 40.)"""
    if n_test is None:
        elig = eligible_depths(k, ok, chance, min_correct)
    thr = 2 * chance / (1 + chance)
    rows, proj = [], []
    for d in np.unique(k):
        m = k == d
        nc, acc = int(ok[m].sum()), float(ok[m].mean())
        row = {"k": int(d), "n": int(m.sum()), "correct": nc, "acc": acc}
        if n_test is not None:
            row["projected_test_correct"] = round(acc * n_test, 1)
            if acc * n_test >= min_correct - 1e-9 and acc >= thr - 1e-12:
                proj.append(int(d))
        rows.append(row)
    return (elig if n_test is None else proj), rows


def choose_setting(calib_runs_by_setting, groups, *, min_depths=3, min_correct=20, wordings=None):
    """E1: one setting per task family and comparison group.

    calib_runs_by_setting {setting: [runs or paths]} (accuracy-only runs on the
    calibration split are enough); groups {name: [model, ...]}.  Eligibility
    is judged per (family, wording) cell, as invariance() does, and for the
    TEST split: a calibration-split run whose setting has n_per and calib_frac
    is projected to the test split's n_per - round(calib_frac * n_per) items
    per depth (_cell_eligibility); other runs are judged on their own counts.
    A model's number of eligible depths in a family is that of its weakest
    wording, over every wording the family needs: `wordings` ({family:
    [wording, ...]} or one list for all families; default: every wording seen
    for the family under any setting and model).  A wording with no run for
    the model under a setting counts as 0 eligible depths (listed under
    missing_wordings): E2 runs every wording, so a setting must not win
    because a harder wording's calibration run crashed or has not run yet.
    For each family and group, the setting under which the WEAKEST model of
    the group (fewest eligible depths, then lowest accuracy) has the most
    eligible depths, at least `min_depths`; ties go to more serial depth (the
    weakest model's deepest eligible depth, then the setting's largest k).  A
    setting where a group model has no run is not a candidate.  All
    settings' accuracy tables are returned, with the projected counts."""
    tables, pooled = {}, {}
    for setting, runs in calib_runs_by_setting.items():
        for r in (_as_run(x) for x in runs):
            nt = _test_per_k(r)
            for fam, wd in sorted(set(zip(r["item_family"], r["item_wording"]))):
                m = (r["item_family"] == fam) & (r["item_wording"] == wd)
                cells = pooled.setdefault((setting, fam, r["model"]), {})
                parts = (r["k"][m], r["correct"][m], 1.0 / np.maximum(r["answer_set_size"][m], 1))
                ntw = nt
                if wd in cells:
                    old = cells[wd]
                    parts = tuple(np.concatenate([o, p]) for o, p in zip(old[:3], parts))
                    ntw = min((x for x in (old[3], nt) if x is not None), default=None)
                cells[wd] = (*parts, ntw)
    seen = {}
    for (_, fam, _), cells in pooled.items():
        seen.setdefault(fam, set()).update(cells)

    def needed(fam, cells):
        if wordings is None:
            want = seen.get(fam, set())
        elif isinstance(wordings, dict):
            want = set(wordings.get(fam, seen.get(fam, set())))
        else:
            want = set(wordings)
        return sorted(want | set(cells))

    for (setting, fam, model), cells in pooled.items():
        k = np.concatenate([c[0] for c in cells.values()])
        ok = np.concatenate([c[1] for c in cells.values()])
        ch = np.concatenate([c[2] for c in cells.values()])
        wout, missing = {}, []
        for wd in needed(fam, cells):
            if wd not in cells:
                missing.append(wd)
                wout[wd] = {"status": "missing", "chance": None, "acc": None,
                            "eligible_depths": [], "n_test_per_k": None,
                            "basis": "no run under this setting (counts as 0 eligible depths)",
                            "by_k": []}
                continue
            kw, okw, chw, nt = cells[wd]
            e, rows = _cell_eligibility(kw, okw, float(chw.mean()), min_correct, nt)
            wout[wd] = {"status": "ok", "chance": float(chw.mean()), "acc": float(okw.mean()),
                        "eligible_depths": e, "n_test_per_k": nt,
                        "basis": "projected to the test split" if nt is not None else "own counts",
                        "by_k": rows}
        sets = [set(v["eligible_depths"]) for v in wout.values()]
        tables.setdefault(setting, {}).setdefault(fam, {})[model] = {
            "chance": float(ch.mean()), "acc": float(ok.mean()),
            "eligible_depths": sorted(set.intersection(*sets)),    # in every wording
            "n_eligible": min(len(s) for s in sets),               # weakest wording
            "max_eligible_depth": (None if any(not s for s in sets)
                                   else min(max(s) for s in sets)),
            "missing_wordings": missing,
            "k_max": int(k.max()), "wordings": wout,
            "by_k": [{"k": int(d), "n": int((k == d).sum()), "correct": int(ok[k == d].sum()),
                      "acc": float(ok[k == d].mean())} for d in np.unique(k)]}
    families = sorted({f for t in tables.values() for f in t})
    choices = {}
    for fam in families:
        for g, models in groups.items():
            cands = []
            for setting, t in tables.items():
                rows = t.get(fam, {})
                got = {}
                for mname in models:
                    hit = [v for rm, v in rows.items() if _model_match(rm, mname)]
                    if hit:
                        got[mname] = hit[0]
                if len(got) < len(models):
                    continue
                weak = min(got, key=lambda mm: (got[mm]["n_eligible"], got[mm]["acc"]))
                wv = got[weak]
                cands.append({"setting": setting, "weakest": weak,
                              "n_eligible": wv["n_eligible"],
                              "max_eligible_depth": wv["max_eligible_depth"],
                              "k_max": max(v["k_max"] for v in got.values()),
                              "acc_weakest": wv["acc"],
                              "per_model": {mm: {wd: c["eligible_depths"]
                                                 for wd, c in v["wordings"].items()}
                                            for mm, v in got.items()},
                              "missing_wordings": {mm: v["missing_wordings"]
                                                   for mm, v in got.items()
                                                   if v["missing_wordings"]}})
            cands.sort(key=lambda c: (c["n_eligible"], c["max_eligible_depth"] or -1,
                                      c["k_max"], c["acc_weakest"]), reverse=True)
            if not cands:
                res = {"setting": None, "status": "missing",
                       "reason": "no setting has runs for every model of the group"}
            elif cands[0]["n_eligible"] < min_depths:
                mw = cands[0]["missing_wordings"]
                res = {"setting": None, "status": "fails",
                       "reason": f"best setting gives the weakest model "
                                 f"{cands[0]['n_eligible']} eligible depth(s); need {min_depths}"
                                 + (f" (no run for wording(s) {mw})" if mw else "")}
            else:
                res = {"setting": cands[0]["setting"], "status": "ok",
                       "weakest": cands[0]["weakest"], "n_eligible": cands[0]["n_eligible"]}
            res["candidates"] = cands
            choices.setdefault(fam, {})[g] = res
    return {"choices": choices, "tables": tables}


# ────────────────────────────── report ──────────────────────────────────────
def _fp(p):
    if p is None or not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fs(x, fmt="{:+.2e}"):
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _provenance():
    here = Path(__file__).resolve()
    h = hashlib.sha256(here.read_bytes()).hexdigest()
    commit = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here.parent,
                           capture_output=True, text=True, timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        pass
    return {"module_sha256": h, "git_commit": commit or os.environ.get("NEXT_RUNS_COMMIT"),
            "numpy_version": np.__version__, "python_version": sys.version.split()[0]}


def model_family(name):
    s = re.sub(r"^meta-", "", _short(name).lower())     # meta-llama/Meta-Llama-3-70B
    for f in ("qwen", "llama", "olmo", "gemma", "mixtral", "mistral", "phi", "yi", "gpt2", "pythia"):
        if s.startswith(f):
            return f
    m = re.match(r"[a-z]+", s)           # unknown: the leading word ("m-1b-instruct" -> "m")
    return m.group(0) if m else s


def _post_pairs(models):
    out = []
    for a in models:
        for b in models:
            sa, sb = _short(a).lower(), _short(b).lower()
            if sb.startswith(sa) and sb[len(sa):] in POST_TRAINING_SUFFIXES:
                out.append((a, b))
    return out


def _size_b(name):
    """Parameter count in billions from a model name (Qwen2.5-0.5B -> 0.5)."""
    m = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", _short(name))
    return float(m[-1]) if m else float("nan")


def _config(r):
    return (r["attn_impl"], r["precision"])


def _has_activity(r):
    return "contribution" in r["arrays"]


def _run_key(r):
    """A run's identity for its random streams (not its path: moving a file
    must not move its p-values)."""
    return "|".join(str(x) for x in (r["model"], r["suite"], r["attn_impl"], r["precision"]))


_SPLIT_RANK = {"test": 2, "calib": 0}          # "all" / unknown: 1


def _capture_identity(r):
    """Two files are the same capture only if all of this matches, contents
    included (an output copied to two folders).  The name alone does not tell
    apart the E1 calibration run, a --limit smoke run and the E2 capture of
    one suite (sweep_probe names all of them <model>__<suite>.npz), and the
    metadata alone does not tell apart two captures of the same suite (a
    re-run after a probe fix, saved in another folder): content_sha1 does."""
    return (r["model"], r["suite"], r["attn_impl"], r["precision"], r.get("split"),
            r.get("limit"), bool(r.get("accuracy_only")), r.get("suite_sha256"),
            _has_activity(r), frozenset(r["ids"].tolist()), r.get("content_sha1"))


def _capture_rank(r):
    """Which capture of one model, suite and configuration to analyse: with
    activity, then without --limit, then the test split, then more items."""
    return (_has_activity(r), r.get("limit") is None, _SPLIT_RANK.get(r.get("split"), 1), r["n"])


def _why_superseded(r, best):
    a, b = _capture_rank(r), _capture_rank(best)
    if a[0] != b[0]:
        return "no activity (accuracy-only capture)"
    if a[1] != b[1]:
        return f"--limit {r.get('limit')} capture"
    if a[2] != b[2]:
        return f"split {r.get('split')} (the test split is analysed)"
    return f"{r['n']} items against {best['n']}"


def _capture_diff(a, b):
    """What differs between two captures of one model, suite and
    configuration, so a conflict says where to look."""
    out = []
    ia, ib = set(a["ids"].tolist()), set(b["ids"].tolist())
    if ia != ib:
        out.append(f"item ids ({len(ia ^ ib)} not shared)")
    if a.get("suite_sha256") != b.get("suite_sha256"):
        out.append("suite hash")
    for key in ("git_commit", "module_sha256"):
        if a["meta"].get(key) != b["meta"].get(key):
            out.append(key)
    shared = [i for i in a["ids"].tolist() if i in ib]
    if shared:
        pa = {i: j for j, i in enumerate(a["ids"].tolist())}
        pb = {i: j for j, i in enumerate(b["ids"].tolist())}
        xa, xb = np.array([pa[i] for i in shared]), np.array([pb[i] for i in shared])
        nd = int((a["correct"][xa] != b["correct"][xb]).sum())
        if nd:
            out.append(f"correctness on {nd} shared item(s)")
        Sa, Sb = _activity(a, "contribution"), _activity(b, "contribution")
        if Sa is not None and Sb is not None and Sa.shape[1:] == Sb.shape[1:]:
            d = float(np.abs(Sa[xa].astype(float) - Sb[xb].astype(float)).max())
            if d > 0:
                out.append(f"activity (max |difference| {d:.3g})")
    return ", ".join(out) or "contents"


def _select_captures(loaded, rejected, log):
    """One run per (model, suite, kernel, precision).  Files with identical
    contents are rejected as duplicates; among the rest the best-ranked
    capture (_capture_rank) is analysed and the others are listed as
    superseded.  Several best-ranked captures with different contents are a
    conflict: nothing recorded in the files (sweep_probe stores no wall-clock
    time) says which is current, and taking the first in glob order would
    analyse an obsolete capture silently, so none of them is analysed and each
    is listed as rejected with what differs."""
    groups = {}
    for r in loaded:
        groups.setdefault((r["model"], r["suite"]) + _config(r), []).append(r)
    keep, superseded = set(), []
    for rs in groups.values():
        uniq = []
        for r in rs:
            twin = next((u for u in uniq if _capture_identity(u) == _capture_identity(r)), None)
            if twin is not None:
                rejected.append({"path": r["path"], "reason": f"duplicate of {twin['path']}"})
                log(f"  SKIPPED {r['path']}: duplicate of {twin['path']}")
                continue
            uniq.append(r)
        top = max(_capture_rank(u) for u in uniq)
        best = [u for u in uniq if _capture_rank(u) == top]
        if len(best) > 1:
            for u in best:
                others = [b for b in best if b is not u]
                why = (f"conflicting capture: same model, suite and configuration as "
                       f"{', '.join(repr(b['path']) for b in others)} with different contents "
                       f"({_capture_diff(u, others[0])}); the files do not say which is current, "
                       f"so none is analysed: pass only the current one to --runs")
                rejected.append({"path": u["path"], "reason": why})
                log(f"  CONFLICT {u['path']}: {why}")
        else:
            keep.add(id(best[0]))
        for r in uniq:
            if any(r is b for b in best):          # identity: run dicts hold arrays
                continue
            why = _why_superseded(r, best[0])
            kept = best[0]["path"] if len(best) == 1 else None
            if kept is None:
                why += "; the better-ranked captures conflict, so none is analysed"
            superseded.append({"path": r["path"], "kept": kept, "reason": why})
            log(f"  SUPERSEDED {r['path']}: {why}" + (f"; analysing {kept}" if kept else ""))
    return [r for r in loaded if id(r) in keep], superseded


def _pin_configs(runs, pinned=None, task_families=TASK_FAMILIES):
    """{model: (attn_impl, precision)}: the one configuration per model that
    RQ1 tests and whose report rows carry no [kernel precision] tag.
    `pinned` (E3's choice) when the model has runs under it; else the
    configuration holding the most task-family cells, then the most cells and
    items, then the one most runs share (the sweep's kernel), then bf16 and
    eager (the outline's pinned environment).  Other configurations are kernel
    / precision checks: not extra models in any Holm correction."""
    act = [r for r in runs if _has_activity(r)]
    freq = Counter(_config(r) for r in act)
    tf = None if task_families is None else set(task_families)
    stats = {}
    for r in act:
        s = stats.setdefault(r["model"], {}).setdefault(_config(r), [set(), set(), 0])
        cells = set(zip(r["item_family"].tolist(), r["item_wording"].tolist()))
        s[0] |= {c for c in cells if tf is None or c[0] in tf}
        s[1] |= cells
        s[2] += r["n"]
    pins = {}
    for model, cfgs in stats.items():
        if pinned is not None and tuple(pinned) in cfgs:
            pins[model] = tuple(pinned)
            continue
        pins[model] = max(cfgs, key=lambda c: (len(cfgs[c][0]), len(cfgs[c][1]), cfgs[c][2],
                                               freq[c], c[1] == "bf16", c[0] == "eager", str(c)))
    return pins


def _load_runs(paths, rejected, notes, log):
    """load_run on every path.  A file that cannot be read or is inconsistent
    goes to `rejected` (a truncated .npz from a Drive sync raises BadZipFile,
    not RunError), as does one measured where the next token is not the
    answer (_measured_position 'misaligned'); a file whose measured position
    cannot be checked is kept with a note."""
    loaded = []
    for p in paths:
        try:
            r = load_run(p)
        except Exception as e:
            rejected.append({"path": str(p), "reason": f"{type(e).__name__}: {e}"})
            log(f"  REJECTED {p}: {e}")
            continue
        pos = r["position"]
        if pos["status"] == "misaligned":
            why = (f"measured position does not predict the answer digit ({pos['evidence']}): a "
                   f"capture made before deep_probe.align_answer, measured at the last prompt "
                   f"token, which predicts the space before the digit for split-digit tokenizers "
                   f"(the outline's 17 September runs, not evidence); re-run it with the current "
                   f"probe")
            rejected.append({"path": str(p), "reason": why})
            log(f"  REJECTED {p}: {why}")
            continue
        if pos["status"] == "unverified":
            notes.append(f"`{p}`: measured position unverified ({pos['evidence']}); a capture "
                         f"made before deep_probe.align_answer was measured at the last prompt "
                         f"token, which predicts a space, not the answer, for tokenizers that "
                         f"split ' 6' (Llama-3, Qwen2.5, OLMo-2, Mistral, Yi, Phi-3, Gemma-2)")
            log(f"  NOTE {p}: measured position unverified ({pos['evidence']})")
        loaded.append(r)
        log(f"  loaded {p}")
    return loaded


def analyse(paths, out_dir, *, preregistered=None, pairs=None, n_perm=5000, n_boot=2000,
            alpha=0.05, q=0.05, seed=0, mixedlm=True, log=print, task_families=TASK_FAMILIES,
            pinned=None, min_families=2, notes=None):
    """Everything the report needs, from a list of capture files.

    task_families   the families RQ1 pools; runs of other families (the
                    numeral control fixed_digit, the ladder) are reported in
                    their own table, never as families
    pinned          (attn_impl, precision) chosen by E3; RQ1 and its Holm
                    correction across models use one configuration per model
                    (_pin_configs), re-runs under another kernel or precision
                    go to the kernel checks
    min_families    passed to invariance() (criterion (b))"""
    prereg = {str(p): _prereg_value(v) for p, v in (preregistered or {}).items()}
    tf = None if task_families is None else tuple(task_families)
    if pinned is not None:
        pinned = (pinned[0], _norm_precision(pinned[1]))
    t0 = time.time()
    rejected, notes = [], list(notes or [])
    loaded = _load_runs(paths, rejected, notes, log)
    runs, superseded = _select_captures(loaded, rejected, log)
    pins = _pin_configs(runs, pinned, tf)
    # resolved before any number is computed: conflicting patterns abort here
    pre = {m: _prereg_match(m, prereg) for m in sorted({r["model"] for r in runs})}
    R = {"provenance": _provenance(), "settings": {
            "n_perm": n_perm, "n_boot": n_boot, "alpha": alpha, "q": q, "seed": seed,
            "preregistered": prereg, "task_families": list(tf) if tf else None,
            "pinned": list(pinned) if pinned else None, "min_families": min_families},
         "rejected": rejected, "superseded": superseded, "notes": notes,
         "runs": [], "signatures": [], "layer_maps": [], "bands": [], "power": [], "sink": [],
         "centroid": [], "preregistration": [], "invariance": [], "rq1_excluded": [],
         "joint": [], "pairs": [], "kernel": [], "errors": []}
    if pinned is not None:
        R["notes"] += [f"{m}: no run under the pinned {pinned[0]} {pinned[1]}; its RQ1 uses "
                       f"{c[0]} {c[1]}" for m, c in sorted(pins.items()) if c != pinned]

    # a model and suite under several configurations: tag all but the pinned one
    multi = Counter((r["model"], r["suite"]) for r in runs)

    def label(r):
        lab = _run_label(r)
        if multi[(r["model"], r["suite"])] > 1 and pins.get(r["model"]) != _config(r):
            lab += f" [{r['attn_impl']} {r['precision']}]"
        return lab

    def attempt(section, what, fn):
        try:
            return fn()
        except Exception as e:           # one odd file must not cost the whole report
            R["errors"].append({"section": section, "what": what,
                                "reason": f"{type(e).__name__}: {e}"})
            log(f"  refused {section} / {what}: {type(e).__name__}: {e}")
            return None

    for r in runs:
        lab, rid = label(r), _run_key(r)
        R["runs"].append({"label": lab, "path": r["path"], "format": r["format"],
                          "model": r["model"], "suite": r["suite"], "family": r["family"],
                          "wording": r["wording"], "n": r["n"], "L": r["L"], "H": r["H"],
                          "attn_impl": r["attn_impl"], "precision": r["precision"],
                          "split": r["split"], "limit": r["limit"],
                          "correctness": r["correctness"], "chance": r["chance"],
                          "has_bos": r["has_bos"], "sandwich_layers": len(r["sandwich_layers"]),
                          "activity": _has_activity(r), "position": r["position"]["status"],
                          "content_sha1": r["content_sha1"]})
        metrics = ["contribution"] + (["write_norm"] if r["sandwich_layers"]
                                      and "write_norm" in r["arrays"] else [])
        sigs = {}
        for m in metrics:
            s = attempt("signature", lab, lambda: signature(
                r, m, alpha=alpha, n_perm=n_perm, rng=_sub_rng(seed, rid, "signature", m)))
            if s:
                sigs[m] = s
                R["signatures"].append({"label": lab, **s})
        if len(sigs) == 2:
            a, b = sigs["contribution"], sigs["write_norm"]
            sa_, sb_ = (a.get("all_items") or {}).get("slope"), (b.get("all_items") or {}).get("slope")
            R["signatures"][-1]["definition_dependent"] = _sign(sa_) != _sign(sb_)
        S = _activity(r, "contribution")
        if S is None or len(np.unique(r["k"])) < 2:
            continue
        TS = top_share(S)

        def lm():
            g = _sub_rng(seed, rid, "layer_map")
            cls = classify_layers(TS, r["k"], q, n_perm, g)
            out = {"label": lab, **layer_map(cls, TS, r["k"], n_perm, g, q, mass=S.sum(-1))}
            if 1 / (n_perm + 1) > q / r["L"]:
                out["bh_note"] = (f"{n_perm} permutations: the smallest p, 1/{n_perm + 1}, is above "
                                  f"q/L = {q / r['L']:.2g}, so BH cannot classify a layer on its "
                                  f"own; use --n-perm >= {math.ceil(r['L'] / q)}")
            return out
        x = attempt("layer_map", lab, lm)
        if x:
            R["layer_maps"].append(x)
        x = attempt("band", lab, lambda: {"label": lab, **band(
            S, r["k"], _sub_rng(seed, rid, "band"), n_perm)})
        if x:
            R["bands"].append(x)
        x = attempt("power", lab, lambda: {"label": lab, "items_per_depth_2pct": power(r)})
        if x:
            R["power"].append(x)
        if "pos0_attn" in r["arrays"]:
            x = attempt("sink", lab, lambda: {"label": lab, **sink_control(
                r, alpha=alpha, n_perm=n_perm, rng=_sub_rng(seed, rid, "sink"))})
            if x:
                R["sink"].append(x)
        x = attempt("centroid", lab, lambda: {"label": lab, **mass_centroid(
            r, n_perm=n_perm, rng=_sub_rng(seed, rid, "centroid"))})
        if x:
            x.pop("centroid", None)
            R["centroid"].append(x)

    # Holm across models within each suite and configuration (a kernel-check
    # re-run is not another model), on the primary all-items p
    by_suite = {}
    for s in R["signatures"]:
        if s["metric"] == "contribution" and s.get("all_items"):
            by_suite.setdefault((s["suite"], s["attn_impl"], s["precision"]), []).append(s)
    for ss in by_suite.values():
        for s, h in zip(ss, holm([x["all_items"]["p"] for x in ss])):
            s["all_items"]["p_holm_models"] = float(h)

    # RQ1: one configuration per model, task families only
    for model in sorted(pins):
        cfg = pins[model]
        val, pat = pre.get(model, (None, None))
        R["preregistration"].append({"model": model, "sign": val, "pattern": pat,
                                     "attn_impl": cfg[0], "precision": cfg[1]})
        rs = []
        for r in runs:
            if r["model"] != model or not _has_activity(r):
                continue
            fams = set(r["item_family"].tolist())
            if _config(r) != cfg:
                why = (f"{r['attn_impl']} {r['precision']} re-run; the model's RQ1 configuration "
                       f"is {cfg[0]} {cfg[1]} (see kernel checks)")
            elif tf is not None and not fams & set(tf):
                why = "; ".join(CONTROL_FAMILIES.get(f, f"family {f!r} is not a task family")
                                for f in sorted(fams))
            else:
                rs.append(r)
                continue
            R["rq1_excluded"].append({"label": label(r), "model": model, "suite": r["suite"],
                                      "family": r["family"], "attn_impl": r["attn_impl"],
                                      "precision": r["precision"], "reason": why})
        if val is None or not rs:
            continue
        x = attempt("invariance", model, lambda: invariance(
            rs, val, alpha, n_boot=n_boot, n_perm=n_perm, mixedlm=mixedlm,
            rng=_sub_rng(seed, "invariance", model, *cfg), task_families=tf,
            min_families=min_families))
        if x:
            x.update(model=model, attn_impl=cfg[0], precision=cfg[1], preregistered_pattern=pat)
            R["invariance"].append(x)
    ph = holm([x["correct_items"].get("p_pooled", np.nan) for x in R["invariance"]])
    pa = holm([x["all_items"].get("p_pooled", np.nan) for x in R["invariance"]])
    for x, h, h2 in zip(R["invariance"], ph, pa):
        x["correct_items"]["p_pooled_holm_models"] = float(h)
        x["all_items"]["p_pooled_holm_models"] = float(h2)
        # the pooled slope is each model's one primary outcome: an invariant
        # verdict also needs it to survive Holm across models
        for mode, key, hp in (("correct_items", "verdict_holm", h),
                              ("all_items", "verdict_all_items_holm", h2)):
            v = x[mode]["verdict"]
            x[key] = "inconclusive" if v == "invariant" and not hp < alpha else v

    # E6 ladders: per suite and model family, models in order of size
    lad = {}
    for s in R["signatures"]:
        if s["metric"] == "contribution" and s.get("all_items"):
            lad.setdefault((s["suite"], model_family(s["model"]), s["attn_impl"], s["precision"]),
                           []).append(s)
    R["ladders"] = [{"suite": k[0], "family": k[1], "attn_impl": k[2], "precision": k[3],
                     "models": [{"model": s["model"], "size_b": _size_b(s["model"]),
                                 "pct_change": s["all_items"]["pct_change"],
                                 "slope": s["all_items"]["slope"], "p": s["all_items"]["p"],
                                 "classification_all_items": s["classification_all_items"],
                                 "classification": s["classification"]}
                                for s in sorted(v, key=lambda s: (_size_b(s["model"]), s["model"]))]}
                    for k, v in sorted(lad.items(), key=lambda kv: str(kv[0])) if len(v) >= 2]

    # RQ4: runs sharing items (same suite and configuration)
    groups = {}
    for r in runs:
        if _has_activity(r):
            groups.setdefault((r["suite"],) + _config(r), []).append(r)
    for (suite, ai, pr), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        fams = {}
        for r in rs:
            fams.setdefault(model_family(r["model"]), []).append(r)
        for fam, fr in sorted(fams.items()):
            if len(fr) >= 2:
                x = attempt("joint", f"{suite}/{fam}", lambda: joint_relabel(
                    fr, n_perm=n_perm, pairs=[], rng=_sub_rng(
                        seed, "joint", suite, ai, pr, *sorted(y["model"] for y in fr))))
                if x:
                    x.update(family=fam, attn_impl=ai, precision=pr)
                    R["joint"].append(x)
        byname = {r["model"]: r for r in rs}
        want = _post_pairs(list(byname)) + [
            (a, b) for pa_, pb_ in (pairs or []) for a in byname for b in byname
            if pa_.lower() in a.lower() and pb_.lower() in b.lower() and a != b]
        for a, b in dict.fromkeys(want):
            x = attempt("pairs", f"{suite}: {a} vs {b}", lambda: joint_relabel(
                [byname[a], byname[b]], n_perm=n_perm,
                rng=_sub_rng(seed, "pair", suite, ai, pr, a, b)))
            if x:
                x.update(attn_impl=ai, precision=pr)
                R["pairs"].append(x)

    # kernel / precision checks: same model and suite, different configuration
    km = {}
    for r in runs:
        if _has_activity(r):
            km.setdefault((r["model"], r["suite"]), []).append(r)
    for (model, suite), rs in sorted(km.items(), key=lambda kv: str(kv[0])):
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if _config(a) == _config(b):
                    continue
                if a["precision"] == "fp32":           # the reference goes second
                    a, b = b, a
                x = attempt("kernel", f"{model}/{suite}", lambda: kernel_compare(
                    a, b, alpha=alpha, n_perm=n_perm,
                    rng=_sub_rng(seed, "kernel", _run_key(a), _run_key(b))))
                if x:
                    R["kernel"].append(x)
    R["elapsed_s"] = round(time.time() - t0, 1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(_jsonable(R), indent=1))
    (out / "report.md").write_text(render_markdown(R))
    log(f"  -> {out / 'report.md'}   {out / 'report.json'}   ({R['elapsed_s']} s)")
    return R


def render_markdown(R):
    L = []
    P = R["provenance"]
    st = R["settings"]
    L += ["# Signature analysis report", "",
          f"signature_analysis.py sha256 `{P['module_sha256'][:16]}`, commit "
          f"`{P['git_commit']}`; permutations {st['n_perm']}, bootstrap {st['n_boot']}, "
          f"alpha {st['alpha']}, BH q {st['q']}, seed {st['seed']}.", "",
          "Primary statistic: top-head share at 40-80% depth (activity = `contribution`, "
          "pre-norm). Slopes are per unit k; p-values two-sided, permutation.", ""]
    L += ["## Runs", "",
          "A `[kernel precision]` tag marks a re-run outside the model's pinned configuration "
          "(RQ1 uses the pinned one).", "",
          "Position: `aligned` (deep_probe.align_answer recorded), `verified` (no record; the "
          "position predicts a digit), `unverified` (no evidence either way; see notes).", "",
          "| run | family/wording | split | n | L x H | kernel | precision | correctness | chance "
          "| BOS | position |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["runs"]:
        sp = (r.get("split") or "-") + (f" (limit {r['limit']})" if r.get("limit") else "")
        L.append(f"| {r['label']} | {r['family']}/{r['wording']} | {sp} | {r['n']} | "
                 f"{r['L']}x{r['H']} | {r['attn_impl']} | {r['precision']} | {r['correctness']} | "
                 f"{r['chance']:.3f} | {_fs(r['has_bos'], '{:.2f}')} | {r.get('position', '-')} |")
    if R["rejected"]:
        L += ["", "**Rejected files (no numbers reported):**", ""]
        L += [f"- `{x['path']}`: {x['reason']}" for x in R["rejected"]]
    if R.get("superseded"):
        L += ["", "**Superseded captures (same model, suite and configuration; not analysed):**", ""]
        L += [f"- `{x['path']}`: {x['reason']}"
              + (f"; analysed `{x['kept']}`" if x.get("kept") else "") for x in R["superseded"]]
    if R.get("notes"):
        L += ["", "**Notes:**", ""] + [f"- {x}" for x in R["notes"]]
    if R["errors"]:
        L += ["", "**Analyses refused:**", ""]
        L += [f"- {x['section']} / {x['what']}: {x['reason']}" for x in R["errors"]]

    L += ["", "## 1. Signatures", "",
          "| run | metric | all-items slope | p | p Holm (models) | % change | PR/H slope | p "
          "| eligible depths | correct slope | p | all items | signature |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in R["signatures"]:
        a, c = s.get("all_items") or {}, s.get("correct_items") or {}
        pct = (f"{a['pct_change']:+.1f}% (k={a['k_from']}->{a['k_to']})"
               if a.get("pct_change") is not None and np.isfinite(a["pct_change"]) else "-")
        L.append(f"| {s['label']} | {s['metric']} | {_fs(a.get('slope'))} | {_fp(a.get('p'))} | "
                 f"{_fp(a.get('p_holm_models'))} | {pct} | "
                 f"{_fs((s.get('pr_h') or {}).get('slope'))} | {_fp((s.get('pr_h') or {}).get('p'))} | "
                 f"{s.get('eligible_depths', '-')} | {_fs(c.get('slope'))} | {_fp(c.get('p'))} | "
                 f"{s['classification_all_items']} | **{s['classification']}** |")
    cd = [s for s in R["signatures"] if (s.get("clean_depth") or {}).get("status") == "ok"]
    if cd:
        L += ["", "**Depth confound.** On a cycle, k steps forward land where n - k steps back do, "
              "so depths above n/2 need fewer lookups than k for a model that reads the rules "
              "backwards. Slope of the primary statistic on the items whose shortest lookup "
              "path equals k:", "",
              "| run | metric | clean depths | n | slope | p | % change | confounded depths | "
              "slope on min_hops (p) |", "|---|---|---|---|---|---|---|---|---|"]
        for s in cd:
            c = s["clean_depth"]
            smh = c.get("slope_on_min_hops") or {}
            L.append(f"| {s['label']} | {s['metric']} | {c['depths']} | {c['n']} | {_fs(c['slope'])} | "
                     f"{_fp(c['p'])} | {c['pct_change']:+.1f}% | {c['confounded_depths']} | "
                     f"{_fs(smh.get('slope'))} ({_fp(smh.get('p'))}) |")
    dd = [s for s in R["signatures"] if s.get("definition_dependent")]
    if dd:
        L += ["", "Definition-dependent (pre-norm and post-norm slopes differ in sign): "
              + ", ".join(s["label"] for s in dd)]
    L += ["", "Accuracy by k (correct / n):", ""]
    for s in R["signatures"]:
        if s["metric"] == "contribution" and s.get("by_k"):
            L.append(f"- {s['label']}: " + "  ".join(
                f"k{b['k']} {b['correct']}/{b['n']}" for b in s["by_k"])
                + f"  (threshold {s['acc_threshold']:.2f})")

    L += ["", "## 2. Invariance (RQ1)", ""]
    tfs = st.get("task_families")
    L += [f"Task families: {', '.join(tfs) if tfs else 'all'}. One configuration per model "
          f"(pinned: {' '.join(map(str, st['pinned'])) if st.get('pinned') else 'the one holding most task-family cells'}"
          "); Holm across models counts each model once.", ""]
    if R.get("preregistration"):
        L += ["| model | RQ1 configuration | pre-registered | by pattern |", "|---|---|---|---|"]
        for p in R["preregistration"]:
            v = ("funnel (+)" if p["sign"] == 1 else "spread (-)" if p["sign"] == -1
                 else "excluded (none)" if p["pattern"] is not None else "not pre-registered")
            L.append(f"| {p['model']} | {p['attn_impl']} {p['precision']} | {v} | "
                     f"{'`' + p['pattern'] + '`' if p['pattern'] else '-'} |")
        L.append("")
    if not R["invariance"]:
        L.append("No model with a pre-registered sign.")
    for x in R["invariance"]:
        sgn = "funnel (+)" if x["preregistered_sign"] > 0 else "spread (-)"
        L += [f"### {x['model']}  ({x['attn_impl']}, {x['precision']}); pre-registered {sgn} "
              f"(pattern `{x.get('preregistered_pattern')}`)", ""]
        if x.get("excluded_cells"):
            L += ["Not task families (excluded): " + ", ".join(
                f"{c} ({why})" for c, why in x["excluded_cells"].items()), ""]
        for mode in ("correct_items", "all_items"):
            v = x[mode]
            L.append(f"**{mode.replace('_', ' ')}: {v['verdict']}**")
            if v.get("pooled") is not None:
                L.append(f"pooled slope {_fs(v['pooled'])}, 95% CI [{_fs(v['ci95'][0])}, "
                         f"{_fs(v['ci95'][1])}], p {_fp(v['p_pooled'])}, Holm across models "
                         f"{_fp(v.get('p_pooled_holm_models'))}; criteria {v['criteria']}")
            L += ["", "| family | slope | p | p Holm | cells (wording: suite) |",
                  "|---|---|---|---|---|"]
            for f, fr in v.get("families", {}).items():
                cells = [f"{v['cells'][c]['wording']}: `{v['cells'][c]['suite']}`"
                         for c in fr.get("cells", [])] or fr["wordings"]
                L.append(f"| {f} | {_fs(fr['slope'])} | {_fp(fr['p'])} | {_fp(fr['p_holm'])} "
                         f"| {', '.join(cells)} |")
            miss = [c for c, cv in v["cells"].items() if cv["status"] != "ok"]
            if miss:
                L.append(f"\nmissing cells: {', '.join(miss)}")
            L.append("")
        L.append(f"After Holm across models: correct items **{x.get('verdict_holm')}**, "
                 f"all items **{x.get('verdict_all_items_holm')}**.")
        if x["disagree"]:
            L.append("**Correct-item and all-items verdicts disagree.**")
        mx = x.get("mixedlm")
        if mx:
            tag = (" (boundary: a variance component is estimated at 0, so the SE ignores "
                   "between-family variation)" if mx.get("boundary") else "")
            L.append(f"MixedLM (secondary){tag}: {mx}")
        L.append("")
    if R.get("rq1_excluded"):
        sig = {s["label"]: s for s in R["signatures"] if s["metric"] == "contribution"}
        L += ["### Outside RQ1", "",
              "Numeral control, ladder and kernel / precision re-runs: reported here, never "
              "counted as a task family or as another model.", "",
              "| run | why not in RQ1 | all-items slope | p | correct slope | p | signature |",
              "|---|---|---|---|---|---|---|"]
        for e in R["rq1_excluded"]:
            s = sig.get(e["label"], {})
            a, c = s.get("all_items") or {}, s.get("correct_items") or {}
            L.append(f"| {e['label']} | {e['reason']} | {_fs(a.get('slope'))} | {_fp(a.get('p'))} "
                     f"| {_fs(c.get('slope'))} | {_fp(c.get('p'))} | {s.get('classification', '-')} |")
        L.append("")

    L += ["## 3. Layer maps (RQ3)", "",
          "`+` funnelling, `-` spreading, `.` neither (BH q within model and task). "
          "Contiguity (fewer sign changes, longer runs than chance) is tested against the joint "
          "k-permutation null, which keeps the noise correlation between adjacent layers; "
          "interleaving (more sign changes) against the layer-order shuffle, since the joint "
          "null over-calls interleaving when layer noise is correlated. The BH-reclassified "
          "joint null (`joint_bh` in report.json) only tests whether any layer responds.", "",
          "| run | map | spread / funnel | sign changes | longest run | contiguous p (joint) | "
          "longest-run p (joint) | interleaved p (layer shuffle) | contiguous p (layer shuffle) "
          "| final 20% slope | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in R["layer_maps"]:
        t, fr = m["tests"], m["final_region"]
        L.append(f"| {m['label']} | `{m['signmap']}` | {m['n_spread']} / {m['n_funnel']} | "
                 f"{m['sign_changes']} | {m['longest_run']} | {_fp(t['joint']['p_fewer'])} | "
                 f"{_fp(t['joint']['p_longer'])} | {_fp(t['layer_shuffle']['p_more'])} | "
                 f"{_fp(t['layer_shuffle']['p_fewer'])} | "
                 f"{_fs(fr['mean_slope'])} | {_fp(fr['p'])} |")
    notes = [f"- {m['label']}: {m['bh_note']}" for m in R["layer_maps"] if m.get("bh_note")]
    if notes:
        L += [""] + notes

    L += ["", "## 4. Bands (PR/H, contiguous rule; superseded rule for comparison)", "",
          "| run | onset | close | width | superseded onset-close | differs |",
          "|---|---|---|---|---|---|"]
    for b in R["bands"]:
        c, s = b["contiguous"], b["superseded"]
        sup = f"{s['onset']:.2f}-{s['close']:.2f}" if s["onset"] is not None else "none"
        L.append(f"| {b['label']} | {_fs(c['onset'], '{:.2f}')} | {_fs(c['close'], '{:.2f}')} | "
                 f"{c['width']:.2f} | {sup} | {b['differs']} |")

    L += ["", "## 5. Ladders and joint relabelling (RQ4)", ""]
    for lad in R.get("ladders", []):
        L += [f"**{lad['family']} on {lad['suite']}** ({lad['attn_impl']}, {lad['precision']}): "
              + ", ".join(f"{_short(m['model'])} {_fs(m['pct_change'], '{:+.1f}%')} "
                          f"(p {_fp(m['p'])}, {m['classification_all_items']})"
                          for m in lad["models"]), ""]
    L += ["",
          "Scale-normalised slopes (relative change per step), one shuffle of k shared by all "
          "runs.", "", "| suite | family | models | pooled | p | per-model slopes |",
          "|---|---|---|---|---|---|"]
    for j in R["joint"]:
        L.append(f"| {j['suite']} | {j['family']} | {', '.join(_short(m) for m in j['models'])} | "
                 f"{_fs(j['pooled'])} | {_fp(j['p_pooled'])} | "
                 + ", ".join(f"{_fs(s)} (p {_fp(p)})" for s, p in zip(j["slopes"], j["p_each"])) + " |")
    if R["pairs"]:
        L += ["", "| suite | pair (b - a) | difference | p |", "|---|---|---|---|"]
        for j in R["pairs"]:
            for d in j["diffs"]:
                L.append(f"| {j['suite']} | {d['b']} - {d['a']} | {_fs(d['diff'])} | {_fp(d['p'])} |")

    L += ["", "## 6. Kernel checks", ""]
    if not R["kernel"]:
        L.append("No model and suite measured under two configurations.")
    else:
        L += ["Everything, both signatures included, on the items the two runs share.", "",
              "| model / suite | a vs b (ref) | shared items | score r | late rel. diff | "
              "top-share slope r (same sign) | PR/H slope r | signatures | agree |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in R["kernel"]:
            pl = x["per_layer"]
            L.append(f"| {_short(x['model'])} / {x['suite']} | {x['a']['attn_impl']} "
                     f"{x['a']['precision']} vs {x['b']['attn_impl']} {x['b']['precision']} | "
                     f"{x['n_items']} | "
                     f"{x['score_r']:.3f} | {_fs(x['late_rel_diff_median'], '{:.3f}')} | "
                     f"{pl['top_share']['r']:.2f} ({pl['top_share']['same_sign']}/{pl['top_share']['L']}) | "
                     f"{pl['pr_h']['r']:.2f} | {x['signature_a']['classification_all_items']} / "
                     f"{x['signature_b']['classification_all_items']} | {x['agree']} |")
        warn = [f"- {_short(x['model'])} / {x['suite']}: {w}" for x in R["kernel"]
                for w in (x.get("warning") or [])]
        if warn:
            L += [""] + warn

    L += ["", "## 7. Power targets", "",
          "Items per depth to detect a 2% change in the primary statistic between the lowest "
          "and highest depth (alpha 0.05, power 0.8, pilot OLS residual variance).", ""]
    L += [f"- {p['label']}: {p['items_per_depth_2pct']}" for p in R["power"]]

    L += ["", "## 8. Sink control", ""]
    if not R["sink"]:
        L.append("No run recorded position-0 attention (--pos0).")
    for s in R["sink"]:
        a = s["signature"].get("all_items") or {}
        b = s["signature_masked"].get("all_items") or {}
        L.append(f"- {s['label']}: BOS {_fs(s['has_bos'], '{:.2f}')}, heads removed "
                 f"{100 * s['frac_removed_window']:.1f}% at 40-80%; slope {_fs(a.get('slope'))} "
                 f"(p {_fp(a.get('p'))}) -> {_fs(b.get('slope'))} (p {_fp(b.get('p'))}); "
                 f"{s['signature']['classification_all_items']} -> "
                 f"{s['signature_masked']['classification_all_items']}; sign agrees {s['sign_agree']}")

    L += ["", "## 9. Mass centroid", "",
          "| run | slope (layers per step) | p | shift k_min->k_max (layers) |", "|---|---|---|---|"]
    for c in R["centroid"]:
        L.append(f"| {c['label']} | {_fs(c['slope'])} | {_fp(c['p'])} | "
                 f"{_fs(c['shift_layers'], '{:+.2f}')} |")
    L += ["", f"_{R.get('elapsed_s')} s_", ""]
    return "\n".join(L)


def render_choice_markdown(res):
    L = ["# E1 setting choice", "",
         "Eligibility is judged for the test split: a calibration-split run is projected to "
         "n_per - round(calib_frac * n_per) test items per depth (calibration accuracy >= "
         "2c/(1+c) and accuracy x test items >= the correct-item minimum), per family and "
         "wording; a model counts the eligible depths of its weakest wording, and a wording "
         "the family needs but the model has no run for under a setting counts as 0.", ""]
    for fam, gs in res["choices"].items():
        L += [f"## {fam}", ""]
        for g, c in gs.items():
            L.append(f"- **{g}**: {c['status']}" + (f", setting `{c['setting']}` (weakest "
                     f"{c['weakest']}, {c['n_eligible']} eligible depths)" if c["setting"] else
                     f" ({c.get('reason')})"))
        L.append("")
    L += ["## Accuracy tables", "",
          "Per depth: correct / calibration items, and in brackets the projected correct items "
          "on the test split.", ""]
    for setting, t in res["tables"].items():
        L += [f"### setting `{setting}`", ""]
        for fam, rows in t.items():
            for model, v in rows.items():
                for wd, c in v.get("wordings", {}).items():
                    if c.get("status") == "missing":
                        L.append(f"- {fam}/{wd} / {_short(model)}: **no run under this setting** "
                                 f"(counts as 0 eligible depths)")
                        continue
                    L.append(f"- {fam}/{wd} / {_short(model)}: " + "  ".join(
                        f"k{b['k']} {b['correct']}/{b['n']}"
                        + (f" ({b['projected_test_correct']:g})" if "projected_test_correct" in b
                           else "") for b in c["by_k"])
                        + f"  eligible {c['eligible_depths']} ({c['basis']})")
        L.append("")
    return "\n".join(L)


def _expand(patterns, unmatched=None):
    """Files matching the globs, in order, without repeats; '**' matches any
    number of folders (runs/**/*.npz reaches runs/ itself and every subfolder).
    A pattern that matches nothing is reported (a typo must not silently drop a
    family of runs) and appended to `unmatched`."""
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        hits = hits if hits else ([p] if Path(p).exists() else [])
        if not hits:
            print(f"  WARNING: {p!r} matches no file", file=sys.stderr)
            if unmatched is not None:
                unmatched.append(p)
        out += hits
    return list(dict.fromkeys(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=[],
                    help="capture files or quoted globs (*.npz from sweep_probe or "
                         "head_alignment_probe, *_deep.json from deep_probe)")
    ap.add_argument("--out", required=True, help="report directory")
    ap.add_argument("--preregistered", nargs="*", default=[], metavar="MODEL=SIGN",
                    help="e.g. qwen2.5=funnel llama=spread OLMo-2-1124-7B-Instruct=spread; "
                         "SIGN funnel/spread or none (no RQ1 verdict).  MODEL occurs in the "
                         "model name (case-insensitive); a post-trained model (-Instruct, -SFT, "
                         "-DPO, -it, -chat, -RLVR) is covered only by a pattern naming its "
                         "marker; MODEL$ anchors at the end, re:REGEX is a regular expression; "
                         "an exact name wins, then the longest pattern")
    ap.add_argument("--pair", nargs="*", default=[], metavar="A:B",
                    help="extra paired tests B minus A (name substrings); base/post-trained "
                         "pairs such as X and X-Instruct are found automatically")
    ap.add_argument("--task-families", nargs="+", default=list(TASK_FAMILIES),
                    help="families RQ1 pools (default: %(default)s); fixed_digit and the ladder "
                         "are reported beside it")
    ap.add_argument("--pinned", nargs=2, metavar=("ATTN", "PRECISION"),
                    help="E3's configuration, e.g. eager bf16; RQ1 uses it for every model that "
                         "has runs under it (default: per model, the configuration holding "
                         "most task-family cells)")
    ap.add_argument("--min-families", type=int, default=2,
                    help="RQ1 criterion (b) needs at least this many families present "
                         "(default 2, the spec's 'of those present'; 3 requires three "
                         "families with the sign whatever is missing)")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mixedlm", action="store_true",
                    help="skip the secondary MixedLM fit in the invariance test (slow on CPU)")
    ap.add_argument("--calib", nargs="+", help="E1: calibration runs; writes the setting choice")
    ap.add_argument("--groups", help="E1: JSON file {group: [model, ...]}")
    ap.add_argument("--min-correct", type=int, default=20,
                    help="E1: correct items a depth needs on the test split (default 20)")
    ap.add_argument("--min-depths", type=int, default=3,
                    help="E1: eligible depths the weakest model needs (default 3)")
    a = ap.parse_args(argv)
    unmatched = []
    if a.calib:
        if not a.groups:
            ap.error("--calib needs --groups")
        paths = _expand(a.calib, unmatched)
        if not paths:
            ap.error("no calibration files match")
        with open(a.groups) as f:
            groups = json.load(f)
        rej, notes = [], []
        # an unreadable file (a Drive sync cut short) is listed, not fatal; a
        # calibration file copied to two folders must not count twice
        runs, sup = _select_captures(_load_runs(paths, rej, notes, print), rej, print)
        res = choose_setting(group_by_setting(runs), groups, min_depths=a.min_depths,
                             min_correct=a.min_correct)
        res["skipped"] = rej + sup
        res["notes"] = notes
        res["unmatched_patterns"] = unmatched
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "choose_setting.json").write_text(json.dumps(_jsonable(res), indent=1))
        (out / "choose_setting.md").write_text(render_choice_markdown(res))
        print(f"  -> {out / 'choose_setting.md'}")
        return res
    paths = _expand(a.runs, unmatched)
    if not paths:
        ap.error("no run files match --runs")
    prereg = {}
    for x in a.preregistered:
        if "=" not in x:
            ap.error(f"--preregistered {x!r}: use MODEL=funnel|spread|none")
        k, v = x.rsplit("=", 1)
        try:
            prereg[k] = _prereg_value(v)
        except ValueError as e:
            ap.error(f"--preregistered {x!r}: {e}")
        if k.strip().lower().startswith("re:"):
            try:                          # before any file is loaded, not deep in analyse()
                re.compile(k.strip()[3:])
            except re.error as e:
                ap.error(f"--preregistered {x!r}: invalid regular expression ({e})")
    bad = [x for x in a.pair if ":" not in x or not all(s.strip() for s in x.split(":", 1))]
    if bad:
        ap.error(f"--pair {' '.join(bad)}: use A:B (name substrings, B minus A)")
    pairs = [tuple(x.split(":", 1)) for x in a.pair]
    notes = [f"--runs pattern {p!r} matches no file" for p in unmatched]
    return analyse(paths, a.out, preregistered=prereg, pairs=pairs, n_perm=a.n_perm,
                   n_boot=a.n_boot, alpha=a.alpha, q=a.q, seed=a.seed, mixedlm=not a.no_mixedlm,
                   task_families=a.task_families, pinned=a.pinned, min_families=a.min_families,
                   notes=notes)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Per-item capture over a serial-task suite (serial_tasks.py JSON) for the E1-E3
runs of research_outline.tex.  One model, one or more suites, one .npz each.

Per item, at the measured position (the one whose next token is the answer
digit; deep_probe.align_answer):

  contribution  [L,H]  ||z_h W_O^h||: the head's slice of the attention output
                       projection before any norm (the paper's activity; same
                       quantity as deep_probe's `contribution`)
  write_norm    [L,H]  OLMo-2 / Gemma-2 only (layers that normalise the
                       attention output before the residual add): norm of the
                       head's post-norm write at the frozen norm scale, as
                       head_alignment_probe defines it
  pos0_attn     [L,H]  attention weight from the measured position to position
                       0 (--pos0; eager only), for the BOS / attention-sink
                       control.  Read from each attention module's returned
                       weights, the tensor transformers collects for
                       output_attentions, keeping only the measured row: all
                       L full [H,T,T] maps held at once are GBs at 80 x 64
                       heads.  The two sources are compared once per suite on
                       a short prefix.
  digit_logits  [10]   the model's logits of the ten answer-digit tokens,
                       index = digit
  predictions          pred_constrained (argmax over the item's answer_set),
                       gold_prob, digit_mass, correct_constrained; with
                       --generate a greedy KV-cached continuation (<= 16
                       tokens, stopping after a digit other than the item's
                       echo) giving pred_first_digit and correct_strict

An item is correct_constrained only when gold is the UNIQUE maximum over its
answer_set.  bf16 logits are coarse (0.0625 apart at |logit| 8-16), so digits
often tie exactly; a tie is no preference for gold, and without this rule the
verdict would depend on the order answer_set lists the digits.  A tie is
resolved against gold (pred_constrained = the lowest tied digit other than
gold, so pred_constrained == gold exactly when correct), n_tied_at_max records
it, and gold_rank counts tied tokens as ahead of gold.

has_bos: position 0 is the tokenizer's BOS or another special token (Qwen's
<|endoftext|> prepended for the sink control; Qwen has no bos_token), i.e. not
a content token.

Depth, answer set and echo come from the item metadata, never from parsing the
prompt, so every task family goes through the same code.

Every linearisation is checked on every item and the run aborts rather than
write numbers when one fails.  Errors are max |error| / max |reference| at the
measured position, and each tolerance follows from how many roundings a
correct computation goes through (bf16 unit roundoff 2^-8 = 3.9e-3):

  exact         the per-head split must sum to the output projection's map
                recomputed in fp32 from the same input and weights; only fp32
                summation order differs (EXACT_TOL 1e-4)
  kernel        the split must reproduce the projection's own output
                (split_err), which the model's GEMM rounded to bf16, and
                cuBLAS may also reduce split-K partial sums in bf16 (PyTorch's
                default), one more rounding each.  Correct OLMo-2-7B bf16 runs
                on an L4 reached 9.6e-3; 16-64 bf16-reduced slices reach
                1.3e-2 to 2.4e-2 at width 8192.  Above WARN_TOL 1e-2 the run
                logs and counts it; it aborts above TOL 5e-2 for bf16/fp16
                kernels (1e-2 for fp32 ones).  A wrong Linear / GPT-2 Conv1D
                layout, a transposed weight or permuted head inputs give
                5e-2 to O(1).  Only the module's output can reveal these, since
                the exact check holds under either layout, so this check
                also chooses the layout
  one rounding  the frozen post-attention norm must reproduce its norm's
                output (OLMo-2 and Gemma-2 norms compute in fp32 and round
                once), the norm's input must be that layer's projection output,
                and attention rows must sum to 1 (ROUNDING_TOL 1e-2)

Every hook must fire exactly once per layer, nothing may be NaN, and the model
must be in eval mode (dropout would make every number a random draw).  fp32
matmuls run at full precision during capture (no TF32), whatever the global
setting.  The split check verifies the weight layout and that the attention
output decomposes linearly over heads.  Any column partition of W_O sums to
the same output, though, so it cannot see head boundaries.  n_heads and
head_dim are therefore checked against the model's config and its attention
modules' head_dim (what their forward reshapes by) before anything runs.

    python3 sweep_probe.py --model Qwen/Qwen2.5-7B --suite suites/pointer_w0.json \\
        --suite suites/binding_w0.json --out runs/e2 --device cuda \\
        --attn-impl eager --precision bf16 --pos0 --generate --split test
    python3 sweep_probe.py --model meta-llama/Llama-3.2-1B --suite calib/*.json \\
        --out runs/e1 --accuracy-only --split calib

Output: OUT/<model with / -> _>__<suite name>.npz holding the arrays above
(stacked over items, plus per-item k, gold, answer_mask and prediction arrays)
and `meta`, a JSON string with provenance, checks and per-item records.  An
output that already exists is skipped, so a disconnected Colab session resumes
where it stopped; one written with different settings, or by a version of
this file whose measurement differs (MEASUREMENT_VERSION), aborts the run
instead, so a sweep never mixes measurements.  Every suite is validated, and output paths checked for
collisions, before the model is loaded; --device cuda is enforced (a CPU
fallback of load_model aborts instead of running for days).
"""
import argparse, functools, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

from deep_probe import align_answer
from head_alignment_probe import fit_norm, norm_family, _eps, _find_proj, _rel_err

DIGITS = "0123456789"
# Tolerances, max |error| / max |reference| at the measured position (module
# docstring).  bf16 unit roundoff is 2^-8 = 3.9e-3.
EXACT_TOL = 1e-4      # two fp32 computations of one map: summation order only
ROUNDING_TOL = 1e-2   # at most one bf16 rounding: frozen norm fit, attention
                      # rows; also the kernel limit when the GEMM is fp32
WARN_TOL = 1e-2       # kernel error above this is logged and counted
# The projection's bf16 output against the exact map.  A single bf16 rounding
# is only the floor: split-K GEMMs reduce partial sums in bf16 (cuBLAS under
# PyTorch's default allow_bf16_reduced_precision_reduction=True).  The
# repository's correct OLMo-2-7B runs (L4, 240 items) reached 9.6e-3, so 1e-2
# would abort correct 70B sweeps.  Structural faults are 5e-2 to O(1).
TOL = 5e-2
MAX_NEW = 16
# prefix length for the once-per-suite check that the hooked attention rows
# are output_attentions' rows; short, so the check itself never holds L full
# [H,T,T] maps at the prompt's length
POS0_CHECK_TOKENS = 16
SPLITS = ("calib", "test", "all")
REQUIRED = ("id", "k", "text", "gold", "answer_set", "echo", "split")


# ────────────────────────────── helpers ─────────────────────────────────────
def _ok(err, tol=TOL):
    """NaN-safe: a NaN error must fail the check, and `nan > tol` is False."""
    return err is not None and err <= tol


def _kernel_tol(*dtypes):
    """Abort limit for a module's own output against its exact fp32 map: TOL
    when any operand or the output is a 16-bit float (bf16/fp16 rounding,
    possibly per split-K partial sum), ROUNDING_TOL for fp32 kernels."""
    import torch
    return TOL if any(d in (torch.bfloat16, torch.float16) for d in dtypes) else ROUNDING_TOL


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def suite_sha256(suite):
    """Content hash of a suite, so a resumed run can tell a regenerated suite
    with the same name from the one it already measured."""
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _git_commit():
    # once per process: the subprocess forks, which the tokenizers library
    # warns about on every fork once its thread pool has been used
    here = Path(__file__).resolve().parent
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip(), "git"
    except Exception:
        pass
    env = os.environ.get("NEXT_RUNS_COMMIT")
    return (env, "env") if env else (None, None)


def _provenance():
    import torch, transformers
    here = Path(__file__).resolve()
    commit, src = _git_commit()
    deps = {}
    for name in ("deep_probe.py", "head_alignment_probe.py", "active_subgraph_dot.py"):
        p = here.parent / name
        deps[name] = _sha256(p) if p.exists() else None
    return {"torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "git_commit": commit, "git_commit_source": src,
            "next_runs_commit_env": os.environ.get("NEXT_RUNS_COMMIT"),
            "module_sha256": _sha256(here), "dependency_sha256": deps,
            "measurement_version": MEASUREMENT_VERSION}


def _check_items(suite):
    """Abort on items the capture cannot score correctly."""
    items = suite.get("items")
    if not items:
        raise SystemExit(f"suite {suite.get('name')!r} has no items")
    seen = set()
    for t in items:
        miss = [f for f in REQUIRED if f not in t]
        if miss:
            raise SystemExit(f"item {t.get('id')!r} lacks {miss}")
        if t["id"] in seen:
            raise SystemExit(f"duplicate item id {t['id']!r}")
        seen.add(t["id"])
        g, A, e = t["gold"], t["answer_set"], t["echo"]
        if not (isinstance(g, str) and len(g) == 1 and g in DIGITS):
            raise SystemExit(f"item {t['id']!r}: gold {g!r} is not a single digit")
        if (not A or len(set(A)) != len(A)
                or any(not (isinstance(a, str) and len(a) == 1 and a in DIGITS) for a in A)):
            raise SystemExit(f"item {t['id']!r}: answer_set {A!r} is not a set of digits")
        if g not in A:
            raise SystemExit(f"item {t['id']!r}: gold {g!r} not in answer_set {A!r}")
        if e is not None and not (isinstance(e, str) and len(e) == 1 and e in DIGITS):
            raise SystemExit(f"item {t['id']!r}: echo {e!r} is not a single digit or None")
        # echoing the written numeral must never score as correct
        if e is not None and e == g:
            raise SystemExit(f"item {t['id']!r}: gold equals echo {e!r}")
        if t["split"] not in ("calib", "test"):
            raise SystemExit(f"item {t['id']!r}: split {t['split']!r} not calib/test")
        if isinstance(t["k"], bool) or not isinstance(t["k"], int):
            raise SystemExit(f"item {t['id']!r}: k {t['k']!r} is not an int")


def _select(items, split="all", limit=None):
    """Indices of the items to run: the split, then at most `limit` taken
    round-robin across k, so a smoke test still covers every depth."""
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}, got {split!r}")
    idx = [i for i, t in enumerate(items) if split == "all" or t["split"] == split]
    if limit is not None and limit < len(idx):
        byk = {}
        for i in idx:
            byk.setdefault(items[i]["k"], []).append(i)
        queues = [list(v) for _, v in sorted(byk.items())]
        pick = []
        while len(pick) < limit:
            for q in queues:
                if q and len(pick) < limit:
                    pick.append(q.pop(0))
        idx = sorted(pick)
    if not idx:
        raise SystemExit(f"no items in split {split!r}")
    return idx


def _patch(module, factory, patched):
    # accelerate's offload hook moves weights onto the device and then calls
    # _old_forward, so that is where the weights are real (as in deep_probe)
    at = "_old_forward" if hasattr(module, "_old_forward") else "forward"
    had = at in module.__dict__
    orig = getattr(module, at)
    setattr(module, at, factory(orig))
    patched.append((module, at, orig, had))


def _unpatch(patched):
    for module, at, orig, had in reversed(patched):
        if had:
            setattr(module, at, orig)
        else:
            delattr(module, at)       # back to the class method, no stale wrapper
    patched.clear()


def _hook_device(module):
    """The execution device accelerate's hook on `module` moves weights and
    inputs to, or None (no hook, or none that says)."""
    import torch
    hook = getattr(module, "_hf_hook", None)
    for h in [hook] + list(getattr(hook, "hooks", None) or []):
        d = getattr(h, "execution_device", None)
        if d is not None:
            return torch.device(d) if not isinstance(d, int) else torch.device("cuda", d)
    return None


def _input_device(hf):
    """Where input ids must live: the embedding's device.  When the embedding
    is offloaded (accelerate cpu_offload, or a device_map putting it on
    cpu/disk), its weight is on 'meta' outside the forward, and so is
    hf.device; the execution device is then on its accelerate hook."""
    try:
        emb = hf.get_input_embeddings()
    except Exception:
        emb = None
    if emb is not None:
        d = _hook_device(emb)
        if d is not None and d.type != "meta":
            return d
        try:
            d = emb.weight.device
            if d.type != "meta":
                return d
        except Exception:
            pass
    d = _hook_device(hf)
    if d is not None and d.type != "meta":
        return d
    if hf.device.type == "meta":
        raise SystemExit("cannot tell which device the input ids belong on: the embedding "
                         "is offloaded with no accelerate execution device")
    return hf.device


def _precision(hf, load_info):
    if load_info.get("precision") == "nf4":
        return "nf4"
    for p in hf.parameters():
        if p.is_floating_point():
            return str(p.dtype).replace("torch.", "")
    return None


def _eos_ids(hf, tok):
    ids = set()
    for v in (getattr(tok, "eos_token_id", None),
              getattr(getattr(hf, "generation_config", None), "eos_token_id", None)):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    return ids


def _check_heads(model, layers, H, dh):
    """Abort unless the wrapper's n_heads / d_head are the model's own.  The
    split check cannot see head boundaries (every column partition of W_O sums
    to the same output), so 8 heads of 8 would pass where the model runs 4
    heads of 16 and give meaningless per-head scores.  The attention module's
    head_dim is what its forward reshapes by, so it is the ground truth."""
    cfg = model._model.config
    try:
        cfg = cfg.get_text_config()
    except Exception:
        pass
    seen = []
    nh = getattr(cfg, "num_attention_heads", None)
    if isinstance(nh, int) and not isinstance(nh, bool):
        if nh != H:
            raise SystemExit(f"per-head split does not reproduce the model's heads: the "
                             f"wrapper says n_heads={H}, the model config "
                             f"num_attention_heads={nh}; refusing to emit contribution scores")
        seen.append("config.num_attention_heads")
    for l, ly in enumerate(layers):
        at = model._get_attn(ly)
        for attr, want in (("head_dim", dh), ("num_heads", H)):
            v = getattr(at, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                if v != want:
                    raise SystemExit(f"layer {l}: per-head split does not reproduce the "
                                     f"model's heads: the wrapper says {attr}={want}, the "
                                     f"attention module {attr}={v}; refusing to emit "
                                     f"contribution scores")
                if l == 0:
                    seen.append(f"attention.{attr}")
    # one of H or dh confirmed is enough: the split check enforces H*dh = the
    # projection's input width, which pins the other
    if not seen:
        raise SystemExit("cannot verify head boundaries: the model has neither "
                         "config.num_attention_heads nor attention head_dim/num_heads")
    return seen


def _special_start_ids(tok):
    """Token ids that make position 0 a start token rather than content: the
    BOS and the other special tokens (Qwen2.5 has no bos_token; the outline's
    sink control prepends <|endoftext|>).  unk is content that failed to
    tokenize, not a start token."""
    ids = set()
    try:
        ids.update(int(i) for i in (getattr(tok, "all_special_ids", None) or []))
    except Exception:
        pass
    unk, bos = getattr(tok, "unk_token_id", None), getattr(tok, "bos_token_id", None)
    if unk is not None and unk != bos:
        ids.discard(int(unk))
    if bos is not None:
        ids.add(int(bos))
    return ids


def _constrained(dl, answer_set, gold):
    """(pred, correct, tied digits) over the item's answer_set.  Correct only
    when gold is the unique maximum: bf16 logits tie often, and a tie is no
    preference for gold.  The tie is resolved against gold (lowest other tied
    digit), so pred == gold exactly when correct and nothing depends on the
    order in which answer_set lists the digits."""
    vals = {d: float(dl[int(d)]) for d in answer_set}
    top = max(vals.values())
    tied = sorted(d for d, v in vals.items() if v == top)
    correct = tied == [gold]
    pred = gold if correct else next(d for d in tied if d != gold)
    return pred, correct, tied


def _split(x_last, W, mode, H, dh):
    """(per-head outputs [H, d_out], the whole map [d_out]) at one position,
    both in fp32 and without the bias (it belongs to no head).  The whole map
    is a separate matrix-vector product, not the sum of the heads, so the two
    agree only if the per-head reshape is right.  Linear weights are
    [out, in]; GPT-2's Conv1D is [in, out]."""
    import torch
    Wm = (W if mode == "linear" else W.t()).float()          # one fp32 copy
    if Wm.dim() != 2 or Wm.shape[1] != H * dh:
        raise ValueError(f"projection weight {tuple(Wm.shape)} does not take "
                         f"n_heads*head_dim {H}*{dh} inputs")
    x = x_last.float()
    v = torch.einsum("hd,ohd->ho", x.reshape(H, dh), Wm.reshape(Wm.shape[0], H, dh))
    return v, Wm @ x


def _per_head(x_last, W, mode, H, dh):
    """Per-head outputs [H, d_out] at one position, bias excluded (it belongs
    to no head).  Linear weights are [out, in]; GPT-2's Conv1D is [in, out]."""
    return _split(x_last, W, mode, H, dh)[0]


def _greedy(hf, tok, first_logits, pkv, echo, dev, eos):
    """Greedy continuation reusing the probe forward's KV cache (its logits
    give the first token).  Stops at EOS, after MAX_NEW tokens, or once a
    digit other than the echo has appeared: echoing the written numeral is not
    an answer, and nothing after the first real digit is read."""
    import torch
    gen, logits = [], first_logits
    for step in range(MAX_NEW):
        nx = int(logits.argmax())
        gen.append(nx)
        if nx in eos:
            break
        if any(c in DIGITS and c != echo for c in tok.decode(gen)):
            break
        if step == MAX_NEW - 1:
            break
        with torch.no_grad():
            o = hf(input_ids=torch.tensor([[nx]], device=dev), past_key_values=pkv,
                   use_cache=True)
        logits, pkv = o.logits[0, -1].float(), o.past_key_values
    return gen


# ────────────────────────────── capture ─────────────────────────────────────
def capture_suite(model, suite, *, attn_impl, record_pos0=False, generate=False,
                  split="all", limit=None, log=print, accuracy_only=False):
    """Run the selected items of `suite` through `model` (an _HFHookedModel-like
    wrapper: .cfg.n_layers/.n_heads/.d_head/.model_name, ._layers(),
    ._get_attn(layer), ._model, ._tok) and return (arrays, meta).

    attn_impl is the kernel the model runs; it is recorded, so it is checked
    against the model's config.  accuracy_only skips activation capture
    (logits and predictions only, for E1 calibration).
    """
    import torch
    t_start = time.time()
    hf, tok = model._model, model._tok
    L, H, dh = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    name = getattr(model.cfg, "model_name", None)
    layers = model._layers()
    if len(layers) != L:
        raise SystemExit(f"config says {L} layers, model has {len(layers)}")
    _check_items(suite)
    if record_pos0 and accuracy_only:
        raise SystemExit("record_pos0 captures activations; it cannot be combined "
                         "with accuracy_only")
    actual = getattr(hf.config, "_attn_implementation", None)
    if actual is not None and actual != attn_impl:
        raise SystemExit(f"model runs {actual!r} attention but attn_impl={attn_impl!r}; "
                         f"the recorded kernel must be the one that ran")
    if record_pos0 and attn_impl != "eager":
        raise SystemExit("position-0 attention needs eager attention (other kernels "
                         "return no attention weights)")
    # dropout in train mode makes every logit and activity a random draw, and
    # no check below would notice
    training = [nm or type(hf).__name__ for nm, m in hf.named_modules() if m.training]
    if training:
        raise SystemExit(f"model is in train mode ({len(training)} modules, e.g. "
                         f"{training[0]!r}): dropout would randomise every number; "
                         f"call .eval() first")

    items_all = suite["items"]
    sel = _select(items_all, split, limit)
    items = [items_all[i] for i in sel]
    n = len(items)
    dev = _input_device(hf)
    eos = _eos_ids(hf, tok)
    asd = sys.modules.get("active_subgraph_dot")
    load_info = dict(getattr(asd, "LAST_LOAD_INFO", None) or {})
    if load_info.get("model") != name:
        load_info = {"note": "model not loaded through active_subgraph_dot.load_model "
                             "in this process"}

    # OLMo-2 and Gemma-2 normalise the attention output before the residual
    # add; they are the architectures that also carry a
    # post_feedforward_layernorm.  (In Llama/Qwen, post_attention_layernorm is
    # the pre-MLP norm.)  Model metadata, so recorded in both modes.
    sandwich = [l for l in range(L) if hasattr(layers[l], "post_feedforward_layernorm")]

    # ── hooks: output projection per layer, post-attention norm where present ──
    st = {"on": False, "contrib": {}, "pending": {}, "write": {}, "calls": {}, "ncalls": {},
          "attn_on": False, "attn": {}, "acalls": {}}
    layout, norm_kind = {}, {}
    # split_err: per-head split vs the projection's own output (kernel
    # rounding included); split_exact_err: vs its map recomputed in fp32;
    # post_norm_input_err: the norm's input vs the projection's output.
    # split_warn counts (item, layer) checks above WARN_TOL, per layer too.
    checks = {"split_err": 0.0, "split_exact_err": 0.0, "post_norm_err": 0.0,
              "post_norm_input_err": 0.0, "attn_row_err": 0.0,
              "split_tol": None, "split_warn": 0, "split_warn_layers": {}, "split_checks": 0}
    pname, patched, handles, head_checks = None, [], [], []

    def attn_hook(l):
        # the attention module's (output, weights) pair: transformers collects
        # output[1] of this module for output_attentions.  Only the measured
        # row is kept, as a copy (a view would keep the whole map alive), so
        # each layer's [H,T,T] map is freed as soon as the layer is done.
        def hook(module, args, output):
            if not st["attn_on"]:
                return None
            st["acalls"][l] = st["acalls"].get(l, 0) + 1
            w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            st["attn"][l] = (None if w is None or w.dim() != 4 else
                             w[0, :, -1, :].detach().to(torch.float32, copy=True))
            return None
        return hook

    def attn_rows(what, T):
        """[L, H, T] measured-position rows recorded by the hooks, or abort."""
        rows = [st["attn"].get(l) for l in range(L)]
        if any(r is None for r in rows):
            raise SystemExit(f"{what}: the model returned no attention weights for every "
                             f"layer (is attention eager?)")
        if any(st["acalls"].get(l) != 1 for l in range(L)):
            raise SystemExit(f"{what}: attention hook calls per layer "
                             f"{dict(sorted(st['acalls'].items()))}; expected exactly one")
        P = torch.stack(rows)
        if tuple(P.shape) != (L, H, T) or not torch.isfinite(P).all():
            raise SystemExit(f"{what}: attention weights of shape {tuple(P.shape)} "
                             f"(want {(L, H, T)}) or non-finite")
        return P

    def pos0_source_check(feed):
        """The rows come from hooks, the spec's source is output_attentions:
        on a short prefix of the first item, run once with
        output_attentions and require the two to be identical."""
        ids = torch.tensor([feed[:POS0_CHECK_TOKENS]], device=dev)
        st["attn"].clear(); st["acalls"].clear()
        st["attn_on"] = True
        try:
            with torch.no_grad():
                r = hf(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       output_attentions=True)
        finally:
            st["attn_on"] = False
        att = getattr(r, "attentions", None)
        if att is None or len(att) != L or any(a is None for a in att):
            raise SystemExit("pos0 source check: the model returned no attention weights "
                             "for every layer (is attention eager?)")
        P = attn_rows("pos0 source check", ids.shape[1])
        ref = torch.stack([a[0, :, -1, :].float() for a in att])
        del r, att
        if ref.shape != P.shape or not torch.equal(ref, P):
            err = float((ref - P).abs().max()) if ref.shape == P.shape else float("inf")
            raise SystemExit(f"attention weights returned by the attention modules differ "
                             f"from output_attentions (max |diff| {err:.2e}); refusing to "
                             f"emit pos0_attn")
        st["attn"].clear(); st["acalls"].clear()

    def proj_hook(l, module):
        def factory(orig):
            def f(*a, **kw):
                out = orig(*a, **kw)
                if not st["on"]:
                    return out
                st["calls"][l] = st["calls"].get(l, 0) + 1
                x = a[0] if a else next(iter(kw.values()))
                W = module.weight
                if x.dim() != 3 or W.device.type == "meta":
                    return out            # not recorded -> the fired-on-all check aborts
                ref = out[0, -1].float()
                if not torch.isfinite(ref).all():
                    raise SystemExit(f"layer {l}: non-finite attention output at the "
                                     f"measured position (NaN/inf in the model)")
                b = getattr(module, "bias", None)
                b = 0 if b is None else b.float()
                # bf16/fp16 GEMMs round their output and possibly each split-K
                # partial sum; an fp32 one only reorders the sum
                ktol = _kernel_tol(x.dtype, W.dtype, out.dtype)
                checks["split_tol"] = max(checks["split_tol"] or 0.0, ktol)
                if l not in layout:
                    # shape cannot tell the layouts apart when the matrix is
                    # square, and the exact check below holds under either
                    # (both sides use the same x and W): keep the layout that
                    # reproduces the module's own output
                    best = ("none", float("inf"))
                    for mode in ("linear", "conv1d"):
                        try:
                            e = _rel_err(_split(x[0, -1], W, mode, H, dh)[0].sum(0) + b, ref)
                        except Exception:
                            e = float("inf")
                        e = e if math.isfinite(e) else float("inf")
                        if e < best[1]:
                            best = (mode, e)
                    if not _ok(best[1], ktol):
                        raise SystemExit(
                            f"layer {l}: per-head split does not reproduce "
                            f"{type(module).__name__} under either weight layout (best "
                            f"relative error {best[1]:.2e} > {ktol}, n_heads={H}, "
                            f"head_dim={dh}). Wrong head_dim, or attention that does not "
                            f"decompose into per-head slices; refusing to emit "
                            f"contribution scores.")
                    layout[l] = best
                v, whole = _split(x[0, -1], W, layout[l][0], H, dh)
                tot = v.sum(0) + b
                # the scores are the exact per-head split of the module's map
                # (independent of how the model's kernel rounded it)
                e_exact = _rel_err(tot, whole + b)
                if not _ok(e_exact, EXACT_TOL):
                    raise SystemExit(f"layer {l}: per-head split does not sum to the "
                                     f"projection recomputed in fp32 (error {e_exact:.2e} > "
                                     f"{EXACT_TOL}); refusing")
                # ... and that map is what the module computed, up to rounding
                e = _rel_err(tot, ref)
                if not _ok(e, ktol):
                    raise SystemExit(f"layer {l}: per-head split error {e:.2e} > {ktol} "
                                     f"(layout {layout[l][0]}); refusing")
                checks["split_checks"] += 1
                if e > WARN_TOL:
                    checks["split_warn"] += 1
                    wl = checks["split_warn_layers"]
                    wl[str(l)] = wl.get(str(l), 0) + 1
                checks["split_err"] = max(checks["split_err"], e)
                checks["split_exact_err"] = max(checks["split_exact_err"], e_exact)
                st["contrib"][l] = v.norm(dim=-1).cpu()
                if l in sandwich:
                    # consumed by the post-norm hook, which checks that its
                    # input is this output (a copy: a view would keep the
                    # whole [T, d] output alive)
                    st["pending"][l] = (v, out[0, -1].detach().to(torch.float32, copy=True))
                return out
            return f
        return factory

    def norm_hook(l, module):
        fam = norm_family(module)
        def factory(orig):
            def f(*a, **kw):
                out = orig(*a, **kw)
                if not st["on"]:
                    return out
                st["ncalls"][l] = st["ncalls"].get(l, 0) + 1
                x = a[0] if a else next(iter(kw.values()))
                if x.dim() != 3 or module.weight.device.type == "meta" or l not in st["pending"]:
                    return out
                kind, g, center, _bias, e_fit = fit_norm(
                    x[0, -1], out[0, -1], module.weight, getattr(module, "bias", None),
                    _eps(module), fam)
                v, proj_out = st["pending"].pop(l)
                # the norm must act on this layer's attention output, else the
                # frozen map is applied to the wrong vector.  Compared with the
                # projection's own output, not the per-head sum, so the GEMM's
                # rounding (checked in the projection hook) does not enter:
                # the two are the same tensor, up to at most a dtype cast
                e_in = _rel_err(proj_out.to(x.device), x[0, -1].float())
                e_fit = e_fit if math.isfinite(e_fit) else float("inf")
                if not (_ok(e_fit, ROUNDING_TOL) and _ok(e_in, ROUNDING_TOL)):
                    raise SystemExit(
                        f"layer {l}: frozen post-attention norm check failed (fit "
                        f"{kind} error {e_fit:.2e}, input-vs-attention-output error "
                        f"{e_in:.2e}, tolerance {ROUNDING_TOL}); refusing to emit write_norm")
                if norm_kind.setdefault(l, kind) != kind:
                    raise SystemExit(f"layer {l}: post-attention norm fitted as "
                                     f"{norm_kind[l]} and {kind} on different items")
                checks["post_norm_err"] = max(checks["post_norm_err"], e_fit)
                checks["post_norm_input_err"] = max(checks["post_norm_input_err"], e_in)
                w = ((v - v.mean(-1, keepdim=True)) if center else v) * g.to(v.device)
                st["write"][l] = w.norm(dim=-1).cpu()
                return out
            return f
        return factory

    # the exact check compares two fp32 products at EXACT_TOL, and the stored
    # scores are fp32 products: TF32 (a global opt-in on GPUs) would round
    # both to 10 mantissa bits.  Restored afterwards.
    mm_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        if not accuracy_only:
            for l in sandwich:
                if not hasattr(layers[l], "post_attention_layernorm"):
                    raise SystemExit(f"layer {l}: post_feedforward_layernorm without "
                                     f"post_attention_layernorm; unsupported")
            head_checks = _check_heads(model, layers, H, dh)
            for l in range(L):
                o, nm = _find_proj(model._get_attn(layers[l]))
                if o is None:
                    raise SystemExit(f"layer {l}: no output projection; unsupported "
                                     f"architecture")
                pname = pname or nm
                _patch(o, proj_hook(l, o), patched)
                if l in sandwich:
                    _patch(layers[l].post_attention_layernorm,
                           norm_hook(l, layers[l].post_attention_layernorm), patched)
            if record_pos0:
                for l in range(L):
                    handles.append(model._get_attn(layers[l]).register_forward_hook(attn_hook(l)))
                pos0_source_check(align_answer(tok, items[0]["text"])[0])
            log(f"  output projection: {pname}   post-attention norm on "
                f"{len(sandwich)}/{L} layers   attention {attn_impl}   heads checked "
                f"against {', '.join(head_checks)}"
                + ("   pos0 (hooked rows = output_attentions)" if record_pos0 else "")
                + ("   generate" if generate else ""))
        else:
            log(f"  accuracy only: no activation hooks   attention {attn_impl}"
                + ("   generate" if generate else ""))

        A = {"k": np.zeros(n, np.int16), "gold": np.zeros(n, np.int8),
             "answer_mask": np.zeros((n, 10), bool),
             "digit_logits": np.zeros((n, 10), np.float32),
             "pred_constrained": np.zeros(n, np.int8),
             "correct_constrained": np.zeros(n, bool),
             "n_tied_at_max": np.zeros(n, np.int8),
             "gold_prob": np.zeros(n, np.float32), "digit_mass": np.zeros(n, np.float32),
             "n_tokens": np.zeros(n, np.int32)}
        if not accuracy_only:
            A["contribution"] = np.zeros((n, L, H), np.float32)
            if sandwich:
                A["write_norm"] = np.zeros((n, L, H), np.float32)
        if record_pos0:
            A["pos0_attn"] = np.zeros((n, L, H), np.float32)
        if generate:
            A["pred_first_digit"] = np.full(n, -1, np.int8)
            A["correct_strict"] = np.zeros(n, bool)

        start_ids = _special_start_ids(tok)
        recs, times = [], []
        for i, t in enumerate(items):
            t0 = time.time()
            log(f"  [{i+1}/{n}] {t['id']}")
            for d in ("contrib", "pending", "write", "calls", "ncalls", "attn", "acalls"):
                st[d].clear()
            feed, ANS, appended = align_answer(tok, t["text"])
            ids = torch.tensor([feed], device=dev)
            # no output_attentions: the attention hooks keep the measured rows
            kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=bool(generate))
            st["on"] = not accuracy_only
            st["attn_on"] = bool(record_pos0)
            try:
                with torch.no_grad():
                    res = hf(**kw)
            finally:
                # generation forwards must not overwrite anything
                st["on"] = st["attn_on"] = False
            lg = res.logits[0, -1].float()
            if not torch.isfinite(lg).all():
                raise SystemExit(f"item {t['id']}: non-finite logits at the measured position")

            if not accuracy_only:
                fired = sorted(st["contrib"])
                if len(fired) < L or any(st["calls"].get(l) != 1 for l in range(L)):
                    raise SystemExit(
                        f"item {t['id']}: output-projection hook recorded {len(fired)}/{L} "
                        f"layers (calls per layer {dict(sorted(st['calls'].items()))}); "
                        f"expected exactly one call on every layer")
                if (len(st["write"]) < len(sandwich)
                        or any(st["ncalls"].get(l) != 1 for l in sandwich)):
                    raise SystemExit(f"item {t['id']}: post-attention norm hook recorded "
                                     f"{len(st['write'])}/{len(sandwich)} layers")
                C = torch.stack([st["contrib"][l] for l in range(L)]).numpy()
                if not np.isfinite(C).all():
                    raise SystemExit(f"item {t['id']}: contribution has NaN/inf")
                A["contribution"][i] = C
                if sandwich:
                    Wn = C.copy()   # non-sandwich layers write v itself
                    for l in sandwich:
                        Wn[l] = st["write"][l].numpy()
                    if not np.isfinite(Wn).all():
                        raise SystemExit(f"item {t['id']}: write_norm has NaN/inf")
                    A["write_norm"][i] = Wn
            if record_pos0:
                P = attn_rows(f"item {t['id']}", len(feed))                   # [L, H, seq]
                row = float((P.sum(-1) - 1).abs().max())
                if not _ok(row, ROUNDING_TOL):
                    raise SystemExit(f"item {t['id']}: attention rows sum to 1 +/- {row:.2e} "
                                     f"(tolerance {ROUNDING_TOL})")
                checks["attn_row_err"] = max(checks["attn_row_err"], row)
                A["pos0_attn"][i] = P[:, :, 0].cpu().numpy()
                del P
                st["attn"].clear()

            # predictions at the measured position, over the exact answer tokens
            probs = torch.softmax(lg, -1)
            dl = torch.stack([lg[ANS[d]] for d in DIGITS]).cpu().numpy()
            gold, aset, echo = t["gold"], list(t["answer_set"]), t["echo"]
            pred, correct, tied = _constrained(dl, aset, gold)
            n_tied = len(tied)
            gold_prob = float(probs[ANS[gold]])
            digit_mass = float(sum(probs[ANS[d]] for d in DIGITS))
            tv, ti = probs.topk(5)
            toks = tok.convert_ids_to_tokens(feed)
            rec = {"id": t["id"], "suite_index": sel[i], "label": t.get("label"),
                   "k": int(t["k"]), "family": t.get("family"), "wording": t.get("wording"),
                   # fewest lookups from start to answer walking the table either
                   # way (serial_tasks); below k on a cycle (k and n-k coincide)
                   "min_hops": (t.get("setting") or {}).get("min_hops"),
                   "k_written": t.get("k_written"), "echo": echo, "split": t["split"],
                   "gold": gold, "answer_set": aset, "text": t["text"],
                   "n_tokens": len(feed), "measured_token": toks[-1], "appended": appended,
                   # position 0 is a start token (BOS, or a special token such
                   # as a prepended <|endoftext|>) rather than content; without
                   # one, attention to it is not a sink in the usual sense
                   "has_bos": int(feed[0]) in start_ids,
                   "pos0_token": toks[0],
                   "pred_constrained": pred, "correct_constrained": correct,
                   "n_tied_at_max": n_tied,
                   # scored incorrect only because of the tie
                   "gold_tied_at_max": n_tied > 1 and gold in tied,
                   "pred_digit": DIGITS[int(np.argmax(dl))],
                   # tokens tied with gold count as ahead of it, as in correctness
                   "gold_prob": gold_prob, "gold_rank": int((probs >= probs[ANS[gold]]).sum()),
                   "digit_mass": digit_mass,
                   "top5": [[tok.decode([int(j)]), round(float(v), 4)] for v, j in zip(tv, ti)]}
            if generate:
                gen = _greedy(hf, tok, lg, res.past_key_values, echo, dev, eos)
                text = tok.decode(gen)
                digs = [c for c in text if c in DIGITS]
                nonecho = [c for c in digs if c != echo]
                rec.update(gen_text=text, pred_first_digit=digs[0] if digs else None,
                           pred_nonecho=nonecho[0] if nonecho else None,
                           correct_strict=correct and bool(digs) and digs[0] == gold)
                A["pred_first_digit"][i] = int(digs[0]) if digs else -1
                A["correct_strict"][i] = rec["correct_strict"]
            del res

            A["k"][i], A["gold"][i] = int(t["k"]), int(gold)
            A["answer_mask"][i, [int(d) for d in aset]] = True
            A["digit_logits"][i] = dl
            A["pred_constrained"][i] = int(pred)
            A["correct_constrained"][i] = correct
            A["n_tied_at_max"][i] = n_tied
            A["gold_prob"][i], A["digit_mass"][i] = gold_prob, digit_mass
            A["n_tokens"][i] = len(feed)
            rec["time_s"] = round(time.time() - t0, 3)
            times.append(rec["time_s"])
            recs.append(rec)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        st["on"] = st["attn_on"] = False
        st["attn"].clear()
        for h in handles:
            h.remove()
        _unpatch(patched)
        torch.set_float32_matmul_precision(mm_prec)

    if not np.isfinite(A["digit_logits"]).all():
        raise SystemExit("digit logits contain NaN/inf")
    if layout:
        log(f"  checks (max relative error): per-head split {checks['split_err']:.1e} vs the "
            f"module's output (abort {checks['split_tol']}), {checks['split_exact_err']:.1e} "
            f"vs its fp32 map (abort {EXACT_TOL}); layout "
            f"{sorted({m for m, _ in layout.values()})}"
            + (f"   post-attn norm {checks['post_norm_err']:.1e}, its input "
               f"{checks['post_norm_input_err']:.1e} (abort {ROUNDING_TOL})" if sandwich else "")
            + (f"   attention rows {checks['attn_row_err']:.1e} (abort {ROUNDING_TOL})"
               if record_pos0 else ""))
        if checks["split_warn"]:
            log(f"  WARNING: per-head split error above {WARN_TOL} on {checks['split_warn']} "
                f"of {checks['split_checks']} (item, layer) checks (per layer "
                f"{checks['split_warn_layers']}; max {checks['split_err']:.1e}, abort "
                f"{checks['split_tol']}).  Expected from bf16 split-K rounding only if "
                f"sporadic and well below the abort limit; the same layers on most items "
                f"point to a projection that is not the plain linear map of its weights")

    nt_by_k = {}
    for r in recs:
        nt_by_k.setdefault(str(r["k"]), set()).add(r["n_tokens"])
    measured = {}
    for r in recs:
        measured[r["measured_token"]] = measured.get(r["measured_token"], 0) + 1
    meta = {
        "model": name, "suite": suite.get("name"), "family": suite.get("family"),
        "wording": suite.get("wording"), "setting": suite.get("setting"),
        "chance": suite.get("chance"), "suite_sha256": suite_sha256(suite),
        "split": split, "limit": limit, "accuracy_only": bool(accuracy_only),
        "record_pos0": bool(record_pos0), "generate": bool(generate),
        "n_items": n, "n_layers": L, "n_heads": H, "d_head": dh,
        "load_info": load_info, "attn_impl": attn_impl,
        "precision": _precision(hf, load_info), "device": str(dev),
        **_provenance(),
        "proj_name": pname, "sandwich_layers": sandwich,
        "post_norm_kind": {str(l): k for l, k in sorted(norm_kind.items())},
        "head_layout": {str(l): m for l, (m, _) in sorted(layout.items())},
        "checks": {**checks,
                   # abort limits (module docstring); split_tol above is the
                   # kernel limit actually applied (TOL for bf16/fp16 GEMMs)
                   "tolerances": {"split": TOL, "split_fp32_kernel": ROUNDING_TOL,
                                  "split_exact": EXACT_TOL, "split_warn": WARN_TOL,
                                  "post_norm": ROUNDING_TOL, "post_norm_input": ROUNDING_TOL,
                                  "attn_row": ROUNDING_TOL},
                   "float32_matmul_precision": {"during": "highest", "outside": mm_prec},
                   "layers_hooked": 0 if accuracy_only else L,
                   "verified": not accuracy_only, "head_boundaries_checked": head_checks,
                   # hooked attention rows identical to output_attentions'
                   "pos0_source_checked": bool(record_pos0),
                   "pos0_source_check_tokens": POS0_CHECK_TOKENS if record_pos0 else None},
        "digit_order": DIGITS,
        # items whose answer_set maximum is shared by several digits (scored
        # incorrect, see the module docstring), and those among them where
        # gold is one of the tied digits
        "constrained_ties": {"items": sum(r["n_tied_at_max"] > 1 for r in recs),
                             "gold_among_tied": sum(r["gold_tied_at_max"] for r in recs)},
        "measured_tokens": measured,
        "appended_forms": [list(a) for a in sorted({tuple(r["appended"]) for r in recs})],
        "has_bos": sum(r["has_bos"] for r in recs),
        "pos0_tokens": sorted({r["pos0_token"] for r in recs}),
        "n_tokens_by_k": {k: sorted(v) for k, v in sorted(nt_by_k.items(), key=lambda kv: int(kv[0]))},
        "timing": {"total_s": round(time.time() - t_start, 2),
                   "per_item_mean_s": round(float(np.mean(times)), 3),
                   "per_item_max_s": round(float(np.max(times)), 3)},
        "items": recs,
    }
    return A, meta


# ────────────────────────────── outputs ─────────────────────────────────────
def output_path(out_dir, model_name, suite_name):
    return (Path(out_dir) /
            f"{model_name.replace('/', '_')}__{str(suite_name).replace('/', '_')}.npz")


def save_output(path, arrays, meta):
    """Write atomically: a disconnect mid-write must not leave a truncated file
    that the next session would skip as done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_output(path):
    """(arrays, meta) from one output file."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return {k: z[k] for k in z.files if k != "meta"}, meta


# An output is "done" only if it was measured the same way.  MEASUREMENT_VERSION
# is bumped by any change that can alter the recorded numbers; edits that cannot
# (messages, or a fix that only changes which models run, such as taking
# Gemma's (1 + w) norm form from its class) keep it, so a resumed sweep does not
# redo hours of finished work.  Outputs written before the field existed are
# mapped from their module_sha256 in _COMPATIBLE_MODULES.
MEASUREMENT_VERSION = 1
_COMPATIBLE_MODULES = {
    # Sep 24 (E3 and the first E2 runs); differs from version 1 only in how the
    # post-attention norm family is chosen, which a successful run never changes
    "4ac79510b759a3a28853167ad0f859cf96f1d334b1f89ec4785182b68dc1c1b7": 1,
}
RESUME_KEYS = ("model", "suite", "suite_sha256", "split", "limit", "accuracy_only",
               "record_pos0", "generate", "attn_impl", "requested_precision",
               "measurement_version")


def _measurement_version(meta):
    v = meta.get("measurement_version")
    return v if v is not None else _COMPATIBLE_MODULES.get(meta.get("module_sha256"))


def _resume_state(path, want):
    """None if the output is missing; else the list of settings that differ."""
    if not Path(path).exists():
        return None
    try:
        _, meta = load_output(path)
    except Exception as e:
        raise SystemExit(f"{path} exists but cannot be read ({e}); delete it to re-run")
    have = dict(meta, measurement_version=_measurement_version(meta))
    return [f"{k}: have {have.get(k)!r}, want {want[k]!r}"
            for k in RESUME_KEYS if have.get(k) != want[k]]


def _summary(meta, log=print):
    byk = {}
    for r in meta["items"]:
        byk.setdefault(r["k"], []).append(r)
    log("  constrained by k: " + "  ".join(
        f"k{k} {sum(r['correct_constrained'] for r in v)}/{len(v)}" for k, v in sorted(byk.items()))
        + f"   chance {meta.get('chance')}")
    if meta["generate"]:
        log("  strict by k:      " + "  ".join(
            f"k{k} {sum(r['correct_strict'] for r in v)}/{len(v)}" for k, v in sorted(byk.items())))
    ties = meta.get("constrained_ties") or {}
    log(f"  measured tokens {meta['measured_tokens']}   appended {meta['appended_forms']}   "
        f"BOS {meta['has_bos']}/{meta['n_items']}   ties at the constrained max "
        f"{ties.get('items')} (gold among the tied, scored incorrect: "
        f"{ties.get('gold_among_tied')})")


# ─────────────────────────────── CLI ────────────────────────────────────────
def run(args, log=print):
    suites = []
    for p in args.suite:
        with open(p) as f:
            s = json.load(f)
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not s["name"]:
            raise SystemExit(f"{p}: not a suite with a name (serial_tasks.save_suite)")
        suites.append((p, s))
    # compare the files, not the names: 'grp/a' and 'grp_a' both write
    # MODEL__grp_a.npz, and case-insensitive disks (macOS, some Drive mounts)
    # also merge names differing only in case
    plan, by_file = [], {}
    for p, s in suites:
        f = output_path(args.out, args.model, s["name"])
        key = str(f).casefold()
        if key in by_file:
            raise SystemExit(f"suites {by_file[key]!r} and {p!r} would both write {f.name}; "
                             f"rename one")
        by_file[key] = p
        plan.append((p, s, f))
    common = dict(model=args.model, split=args.split, limit=args.limit,
                  accuracy_only=args.accuracy_only, record_pos0=args.pos0,
                  generate=args.generate, attn_impl=args.attn_impl,
                  requested_precision=args.precision,
                  measurement_version=MEASUREMENT_VERSION)
    todo = []
    for p, s, f in plan:
        want = dict(common, suite=s["name"], suite_sha256=suite_sha256(s))
        diff = _resume_state(f, want)
        if diff is None:
            todo.append((p, s, f))
        elif diff:
            hint = (" (written by a version of sweep_probe.py that measures differently)"
                    if any(d.startswith("measurement_version") for d in diff)
                    else "")
            raise SystemExit(f"{f} exists with different settings ({'; '.join(diff)}){hint}; "
                             f"use another --out or delete it")
        else:
            log(f"  skip {f.name}: already done")
    if not todo:
        log("  nothing to do")
        return []
    # validate every pending suite before the load (minutes to an hour for a
    # 70B) and before earlier suites spend their Colab time
    for p, s, f in todo:
        try:
            _check_items(s)
            _select(s["items"], args.split, args.limit)
        except SystemExit as e:
            raise SystemExit(f"suite {p}: {e}")

    _git_commit()                 # before any tokenizer threads exist
    # pin the load path; load_model reads these
    os.environ["ASD_FORCE_STRATEGY"] = "B"
    os.environ["ASD_FORCE_PRECISION"] = "bf16" if args.precision == "bf16" else ""
    os.environ["ASD_ATTN_IMPL"] = args.attn_impl
    import torch
    try:
        want_dev = torch.device(args.device).type
    except Exception:
        raise SystemExit(f"--device {args.device!r} is not a torch device")
    # load_model quietly falls back to the CPU when CUDA is missing (or a GPU
    # load fails): a 70B would then run for days or exhaust host RAM
    if want_dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available: select a GPU runtime, "
                         "or pass --device cpu explicitly")
    if args.precision == "fp32" and args.device != "cpu" and torch.cuda.is_available():
        raise SystemExit("load_model supports fp32 on CPU only; use --device cpu")
    import active_subgraph_dot as asd
    t0 = time.time()
    model = asd.load_model(args.model, device=args.device)
    load_s = time.time() - t0
    if not hasattr(model, "_model"):
        raise SystemExit("needs the _HFHookedModel wrapper (ASD_FORCE_STRATEGY=B)")
    on = sorted({p.device.type for p in model._model.parameters()})
    if want_dev != "cpu" and want_dev not in on:
        raise SystemExit(f"--device {args.device} requested but the model's weights are on "
                         f"{on} (load_model fell back); refusing to run there")
    li = asd.LAST_LOAD_INFO
    want_prec = {"bf16": "bfloat16", "fp32": "float32"}[args.precision]
    if li.get("strategy") != "B" or li.get("precision") != want_prec:
        raise SystemExit(f"loaded with strategy {li.get('strategy')!r} precision "
                         f"{li.get('precision')!r}; requested B / {want_prec}")
    log(f"  {args.model}: n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} "
        f"d_head={model.cfg.d_head}  load_info={dict(li)}")

    written = []
    for p, s, f in todo:
        log(f"\n  suite {s.get('name')} ({p})")
        A, meta = capture_suite(model, s, attn_impl=args.attn_impl, record_pos0=args.pos0,
                                generate=args.generate, split=args.split, limit=args.limit,
                                log=log, accuracy_only=args.accuracy_only)
        meta.update(model=args.model, requested_precision=args.precision,
                    suite_path=str(p), load_s=round(load_s, 1),
                    cli={k: v for k, v in vars(args).items()})
        save_output(f, A, meta)
        _summary(meta, log)
        log(f"  -> {f}  ({f.stat().st_size / 1e6:.1f} MB)")
        written.append(f)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True, action="append", nargs="+",
                    help="suite JSON (serial_tasks.save_suite); repeatable")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"],
                    help="attention kernel; it changes the contribution metric, so it "
                         "is pinned and recorded")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"],
                    help="fp32 is supported on CPU only (load_model)")
    ap.add_argument("--pos0", action="store_true",
                    help="record attention to position 0 (needs --attn-impl eager)")
    ap.add_argument("--generate", action="store_true",
                    help="greedy continuation for pred_first_digit / correct_strict")
    ap.add_argument("--split", default="all", choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None,
                    help="at most N items, taken round-robin across k (smoke tests)")
    ap.add_argument("--accuracy-only", action="store_true",
                    help="logits and predictions only, no activation capture (E1)")
    a = ap.parse_args(argv)
    a.suite = [p for group in a.suite for p in group]
    if a.pos0 and a.attn_impl != "eager":
        ap.error("--pos0 needs --attn-impl eager")
    if a.pos0 and a.accuracy_only:
        ap.error("--pos0 records activations; drop it with --accuracy-only")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be positive")
    return run(a)


if __name__ == "__main__":
    main()

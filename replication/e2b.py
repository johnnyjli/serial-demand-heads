#!/usr/bin/env python3
"""
E2b: serial-depth tasks set up for accuracy, in two modes on the same items.

  direct    8 worked examples whose depths cover the whole k range, each on its
            own table and never showing the query's answer.  The query ends
            where the answer digit goes.  Head activity is captured at that
            position (E2's measurement), and the model also generates a short
            continuation, so an answer it gives later is recorded too.
  stepwise  the same queries and the same worked examples, but every worked
            example writes out its intermediate states ("Path: 4 -> 9 -> 1 -> 7.
            You end at 7.").  The model generates its own trace and final
            answer; head activity is captured at the position whose next token
            is that final answer digit, with the model's own trace in context.

Families (one wording each): pointer chase on a 10-cycle (k = 1..5, so k steps
forward are never fewer lookups backward), composed addition mod 7 (k = 1..6),
variable binding and relational lookup (3 interleaved chains of 6, k = 1..6;
the prompt has the same length at every k).  Answers are balanced within each
depth, never equal the written step count, and every item and worked example is
re-solved from its text by an independent parser before it is used.

Activity for head h in layer l is ||z_h W_O^h|| at the measured position, the
E2 definition, recomputed from o_proj's input and checked against o_proj's
output on every item.

    import e2b
    items = e2b.make_all(n_per=30, shots=8, seed=0)
    model, tok = e2b.load("Qwen/Qwen2.5-7B")
    e2b.run(model, tok, "Qwen/Qwen2.5-7B", items, "/content/drive/MyDrive/e2b")
"""
import json, os, random, re, time
from pathlib import Path
import numpy as np

FAMILIES = ("pointer", "arith", "binding", "relational")
K_RANGE = {"pointer": (1, 5), "arith": (1, 6), "binding": (1, 6), "relational": (1, 6)}
N_STATES, MOD, N_CHAINS, CHAIN_LEN = 10, 7, 3, 6
LOWER = list("bcdefghjklmnopqrstuvwxyz")        # no a / i: they read as English words
UPPER = list("BCDEFGHJKLMNOPQRSTUVWXYZ")
DIGITS = "0123456789"
# how the final answer is introduced; group 1 (if any) is the queried name
PHRASE_RE = {"pointer": r"You end at", "arith": r"You end at",
             "binding": r"Therefore (\w) =", "relational": r"The top boss above (\w) is"}


# ───────────────────────────── items ────────────────────────────────────────
def _pointer(rng, k, gold):
    order = list(range(N_STATES)); rng.shuffle(order)
    nxt = {order[i]: order[(i + 1) % N_STATES] for i in range(N_STATES)}
    prev = {v: u for u, v in nxt.items()}
    s = gold
    for _ in range(k): s = prev[s]
    path = [s]
    for _ in range(k): path.append(nxt[path[-1]])
    head = "Rules: " + " ".join(f"{a} goes to {nxt[a]}." for a in range(N_STATES)) + f" Start at {s}. Take {k} steps."
    return head, "Path:", " -> ".join(map(str, path)), "You end at", list(DIGITS)

def _arith(rng, k, gold):
    d = rng.randint(1, MOD - 1)                   # k*d is never 0 mod 7 for k < 7
    s = (gold - k * d) % MOD
    path = [s]
    for _ in range(k): path.append((path[-1] + d) % MOD)
    head = f"Rule: add {d} and keep the remainder after dividing by {MOD}. Start at {s}. Apply the rule {k} times."
    return head, "Steps:", " -> ".join(map(str, path)), "You end at", [str(x) for x in range(MOD)]

def _chains(rng, gold, names):
    roots = [gold] + rng.sample([d for d in range(10) if d != gold], N_CHAINS - 1)
    chains = [names[c * CHAIN_LEN:(c + 1) * CHAIN_LEN] for c in range(N_CHAINS)]
    return roots, chains

def _binding(rng, k, gold):
    roots, chains = _chains(rng, gold, rng.sample(LOWER, N_CHAINS * CHAIN_LEN))
    st = []
    for vs, r in zip(chains, roots):
        st.append(f"{vs[0]} = {r}.")
        st += [f"{vs[i]} = {vs[i - 1]}." for i in range(1, CHAIN_LEN)]
    rng.shuffle(st)
    q = chains[0][k - 1]
    body = " = ".join([chains[0][i] for i in range(k - 1, -1, -1)] + [str(gold)])
    head = "Assignments: " + " ".join(st) + f" Question: what is {q}?"
    return head, "Trace:", body, f"Therefore {q} =", sorted(str(r) for r in roots)

def _relational(rng, k, gold):
    roots, chains = _chains(rng, gold, rng.sample(UPPER, N_CHAINS * CHAIN_LEN))
    st = []
    for vs, r in zip(chains, roots):
        st.append(f"The boss of {vs[0]} is {r}.")
        st += [f"The boss of {vs[i]} is {vs[i - 1]}." for i in range(1, CHAIN_LEN)]
    rng.shuffle(st)
    q = chains[0][k - 1]
    body = " -> ".join([chains[0][i] for i in range(k - 1, -1, -1)] + [str(gold)])
    head = "Chart: " + " ".join(st) + f" Question: who is the top boss above {q}?"
    return head, "Chain:", body, f"The top boss above {q} is", sorted(str(r) for r in roots)

BUILD = {"pointer": _pointer, "arith": _arith, "binding": _binding, "relational": _relational}
WRITES_K = {"pointer": True, "arith": True, "binding": False, "relational": False}
GOLDS = {"pointer": range(N_STATES), "arith": range(MOD), "binding": range(10), "relational": range(10)}

def _admissible(fam, k):
    return [g for g in GOLDS[fam] if not (WRITES_K[fam] and g == k)]

def _render(inst, gold, mode, with_answer):
    head, label, body, phrase = inst
    if not with_answer:
        return f"{head} {phrase}" if mode == "direct" else f"{head} {label}"
    return f"{head} {phrase} {gold}." if mode == "direct" else f"{head} {label} {body}. {phrase} {gold}."

def make_items(family, n_per=30, shots=8, seed=0):
    """Items of one family.  Direct and stepwise texts share every instance."""
    lo, hi = K_RANGE[family]; build = BUILD[family]
    rng = random.Random(f"e2b/{family}/{seed}")
    items = []
    for k in range(lo, hi + 1):
        golds = _admissible(family, k); rng.shuffle(golds)
        for i in range(n_per):
            gold = golds[i % len(golds)]
            head, label, body, phrase, aset = build(rng, k, gold)
            q = (head, label, body, phrase)
            # worked examples: depths cycle over the whole range, answers never the query's
            ks = [lo + j % (hi - lo + 1) for j in range(shots)]; rng.shuffle(ks)
            ex = []
            for kk in ks:
                g = rng.choice([x for x in _admissible(family, kk) if x != gold])
                h, lb, bd, ph, _ = build(rng, kk, g)
                ex.append(((h, lb, bd, ph), g, kk))
            item = {"id": f"{family}/{k}/{i}", "family": family, "k": k, "gold": str(gold), "answer_set": aset,
                    "chance": 1 / len(aset), "echo": str(k) if WRITES_K[family] else None,
                    "gold_trace": body, "shot_ks": ks, "shot_golds": [g for _, g, _ in ex]}
            for mode in ("direct", "stepwise"):
                blocks = [_render(e, g, mode, True) for e, g, _ in ex] + [_render(q, gold, mode, False)]
                item[f"text_{mode}"] = "\n\n".join(blocks)
            items.append(item)
    verify(items)
    return items

def make_all(n_per=30, shots=8, seed=0, families=FAMILIES):
    return {f: make_items(f, n_per, shots, seed) for f in families}


# ───────────────── independent solver (checks every item) ──────────────────
def _solve_block(fam, b):
    """(answer, trace body or None, answer shown or None) from one block's text."""
    shown = re.search(r"(\d)\.$", b.strip())
    if fam == "pointer":
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) goes to (\d)\.", b))
        s, k = map(int, re.search(r"Start at (\d)\. Take (\d+) steps\.", b).groups())
        path = [s]
        for _ in range(k): path.append(tbl[path[-1]])
        return path[-1], " -> ".join(map(str, path))
    if fam == "arith":
        d, mod, s, k = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)\. "
                                          r"Apply the rule (\d+) times\.", b).groups())
        path = [s]
        for _ in range(k): path.append((path[-1] + d) % mod)
        return path[-1], " -> ".join(map(str, path))
    if fam == "binding":
        st = dict(re.findall(r"\b(\w) = (\w)\.", b.split("Question:")[0]))
        q = re.search(r"Question: what is (\w)\?", b).group(1); sep = " = "
    else:
        st = dict(re.findall(r"The boss of (\w) is (\w)\.", b.split("Question:")[0]))
        q = re.search(r"Question: who is the top boss above (\w)\?", b).group(1); sep = " -> "
    path = [q]
    while not path[-1].isdigit(): path.append(st[path[-1]])
    return int(path[-1]), sep.join(path)

def verify(items):
    for it in items:
        fam = it["family"]
        for mode in ("direct", "stepwise"):
            bl = it[f"text_{mode}"].split("\n\n")
            ans, trace = _solve_block(fam, bl[-1])
            assert str(ans) == it["gold"], f"{it['id']}: gold {it['gold']} but the text gives {ans}"
            assert trace == it["gold_trace"], f"{it['id']}: trace mismatch"
            for b, g in zip(bl[:-1], it["shot_golds"]):
                a, tr = _solve_block(fam, b)
                assert a == g and b.strip().endswith(f"{g}."), f"{it['id']}: wrong worked example"
                if mode == "stepwise": assert f" {tr}. " in b, f"{it['id']}: worked example trace wrong"
            assert it["gold"] not in [str(g) for g in it["shot_golds"]]
            if it["echo"] is not None: assert it["gold"] != it["echo"]


# ───────────────────────────── model ────────────────────────────────────────
def load(name, token=None, attn="eager"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, token=token)
    kw = dict(device_map="cuda", attn_implementation=attn, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:                                  # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    return model.eval(), tok

def answer_feed(tok, text):
    """Token ids to feed so that the next token is the answer digit, and the
    token id of every digit there.  Raises unless each digit is one token
    after the same prefix (Llama-3 / Qwen / OLMo feed the space, Mistral '▁')."""
    base, dig = None, {}
    for d in DIGITS:
        ids = tok(text + " " + d)["input_ids"]
        if tok.decode(ids[-1:]).strip() != d:
            raise ValueError(f"'{d}' is not a single token after the prompt")
        if base is None: base = ids[:-1]
        elif ids[:-1] != base: raise ValueError("prompt tokens differ across answer digits")
        dig[d] = ids[-1]
    return base, [dig[d] for d in DIGITS]

class Capture:
    """Hooks every layer's o_proj; per-head write norms at the last position."""
    def __init__(self, model):
        import torch
        self.torch = torch
        layers = model.model.layers
        self.o = [l.self_attn.o_proj for l in layers]
        self.H = model.config.num_attention_heads
        self.L = len(layers)
        self.buf = {}
        self.hs = [o.register_forward_hook(self._hook(i)) for i, o in enumerate(self.o)]
        self.on = False
    def _hook(self, i):
        def f(mod, inp, out):
            if self.on:
                self.buf[i] = (inp[0][0, -1].detach(), out[0, -1].detach())
        return f
    def norms(self):
        t = self.torch; S = np.zeros((self.L, self.H), np.float32); err = 0.0
        for i, o in enumerate(self.o):
            x, y = self.buf[i]
            W = o.weight.float(); dh = W.shape[1] // self.H
            v = t.einsum("hk,dhk->hd", x.float().view(self.H, dh), W.view(W.shape[0], self.H, dh))
            rec = v.sum(0) + (o.bias.float() if o.bias is not None else 0)
            yf = y.float(); e = float((rec - yf).abs().max() / yf.abs().max().clamp_min(1e-6))
            err = max(err, e)
            S[i] = v.norm(dim=-1).cpu().numpy()
        self.buf = {}
        return S, err
    def remove(self):
        for h in self.hs: h.remove()

def _forward(model, cap, ids):
    t = cap.torch
    with t.inference_mode():
        cap.on = True
        out = model(t.tensor([ids], device=model.device))
        cap.on = False
        S, err = cap.norms()
        logits = out.logits[0, -1].float().cpu().numpy()
    return S, logits, err

def _generate(model, tok, ids, max_new):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    n0 = len(ids)
    class Blank(StoppingCriteria):                    # stop when the model starts a new block
        def __call__(self, input_ids, scores, **kw):
            done = "\n\n" in tok.decode(input_ids[0, n0:])
            return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)
    with torch.inference_mode():
        g = model.generate(torch.tensor([ids], device=model.device), attention_mask=torch.ones(1, n0, dtype=torch.long, device=model.device),
                           max_new_tokens=max_new, do_sample=False, stopping_criteria=StoppingCriteriaList([Blank()]),
                           pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(g[0, n0:], skip_special_tokens=True).split("\n\n")[0]

def _trace_score(gen_body, gold_body, sep):
    """Steps of the gold trace reproduced before the first error, as a fraction."""
    g = [x.strip() for x in gold_body.split(sep.strip())]
    m = [x.strip() for x in gen_body.split(sep.strip())]
    n = 0
    for a, b in zip(g, m):
        if a != b: break
        n += 1
    return (n - 1) / max(len(g) - 1, 1), gen_body.strip() == gold_body.strip()


# ───────────────────────────── run ──────────────────────────────────────────
MAX_NEW = {"direct": 12, "stepwise": 72}
ERR_WARN, ERR_FAIL = 5e-2, 1.5e-1

def run_one(model, tok, cap, items, mode, log=print, generate_fn=None):
    """Arrays and per-item records for one family in one mode."""
    gen_fn = generate_fn or (lambda ids, n: _generate(model, tok, ids, n))
    n = len(items)
    C = np.full((n, cap.L, cap.H), np.nan, np.float16); DL = np.full((n, 10), np.nan, np.float32)
    recs, t0, maxerr, nwarn = [], time.time(), 0.0, 0
    for i, it in enumerate(items):
        fam, gold, aset = it["family"], it["gold"], it["answer_set"]
        r = {"id": it["id"], "family": fam, "k": it["k"], "gold": gold, "chance": it["chance"], "mode": mode}
        if mode == "direct":
            feed, dig = answer_feed(tok, it["text_direct"])
            S, logits, err = _forward(model, cap, feed)
            gen = gen_fn(feed, MAX_NEW["direct"])
            first = re.search(r"\d", gen)
            r.update(gen=gen, pred_gen=first.group(0) if first else None, parsed=True)
            measured = True
        else:
            feed0 = tok(it["text_stepwise"])["input_ids"]
            gen = gen_fn(feed0, MAX_NEW["stepwise"])
            m = re.search(PHRASE_RE[fam] + r" (\d)", gen)
            r.update(gen=gen, parsed=bool(m))
            label_sep = " = " if fam == "binding" else " -> "
            body = gen.split(".")[0] if "." in gen else gen
            frac, exact = _trace_score(body, it["gold_trace"], label_sep)
            r.update(trace_steps_correct=frac, trace_exact=exact,
                     gold_in_gen=bool(re.search(rf"(?<!\d){gold}(?!\d)", gen)))
            if m:
                r["pred_gen"] = m.group(m.lastindex)
                cap_text = it["text_stepwise"] + gen[:m.start(m.lastindex)].rstrip()
                feed, dig = answer_feed(tok, cap_text)
                S, logits, err = _forward(model, cap, feed)
                r["n_tokens_measured"] = len(feed)
                measured = True
            else:
                r["pred_gen"] = None; measured = False
        if measured:
            if err > ERR_FAIL:
                raise RuntimeError(f"{it['id']}: per-head split does not reproduce o_proj (error {err:.2e})")
            nwarn += err > ERR_WARN; maxerr = max(maxerr, err)
            C[i] = S.astype(np.float16); DL[i] = logits[dig]
            lg = {d: DL[i][int(d)] for d in aset}
            r["pred_constrained"] = max(lg, key=lg.get)
            p = np.exp(DL[i] - DL[i].max()); p /= p.sum(); r["gold_prob"] = float(p[int(gold)])
        else:
            r["pred_constrained"] = None
        r["correct"] = r.get("pred_gen") == gold                      # the answer the model wrote
        r["correct_constrained"] = r["pred_constrained"] == gold      # argmax over the answer set
        r["measured"] = measured
        recs.append(r)
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - t0
            log(f"    {mode:<8} {fam:<10} {i + 1:4d}/{n}  acc {np.mean([x['correct'] for x in recs]):.2f}  "
                f"{el / (i + 1):.2f}s/item  ~{el / (i + 1) * (n - i - 1) / 60:.1f} min left")
    return C, DL, recs, {"max_split_err": maxerr, "split_warn": int(nwarn)}

def out_path(root, model_name, mode, family):
    return Path(root) / model_name.replace("/", "_") / f"{mode}__{family}.npz"

def run(model, tok, model_name, items_by_family, root, modes=("direct", "stepwise"), log=print, deadline=None):
    """All families and modes for one loaded model; finished files are skipped.
    Returns False if it stopped early at `deadline` (time.time() value)."""
    cap = Capture(model)
    try:
        for mode in modes:
            for fam, items in items_by_family.items():
                f = out_path(root, model_name, mode, fam)
                if f.exists():
                    log(f"    done already: {f.name}"); continue
                if deadline and time.time() > deadline:
                    log("    time budget reached; stopping this model"); return False
                t0 = time.time()
                C, DL, recs, checks = run_one(model, tok, cap, items, mode, log)
                meta = {"model": model_name, "mode": mode, "family": fam, "n_layers": cap.L, "n_heads": cap.H,
                        "items": recs, "checks": checks, "seconds": time.time() - t0,
                        "texts": {it["id"]: it[f"text_{mode}"] for it in items}}
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, contribution=C, digit_logits=DL, meta=np.array(json.dumps(meta)))
                os.replace(tmp, f)
                log(f"    saved {f.name}  ({time.time() - t0:.0f}s, max split error {checks['max_split_err']:.1e})")
    finally:
        cap.remove()
    return True


# ───────────────────────────── analysis ─────────────────────────────────────
def load_results(root):
    runs = []
    for f in sorted(Path(root).glob("*/*.npz")):
        if f.name.endswith(".tmp.npz"): continue
        z = np.load(f, allow_pickle=False); meta = json.loads(str(z["meta"]))
        runs.append({"model": meta["model"], "mode": meta["mode"], "family": meta["family"],
                     "C": z["contribution"].astype(np.float32), "DL": z["digit_logits"], "items": meta["items"],
                     "L": meta["n_layers"], "H": meta["n_heads"], "checks": meta["checks"]})
    return runs

def accuracy_table(runs):
    """{(model, mode, family): [accuracy at each k]} using the written answer."""
    out = {}
    for r in runs:
        ks = sorted({x["k"] for x in r["items"]})
        out[(r["model"], r["mode"], r["family"])] = [np.mean([x["correct"] for x in r["items"] if x["k"] == k]) for k in ks]
    return out

def window(L, lo=.4, hi=.8): return slice(int(lo * (L - 1)), int(hi * (L - 1)) + 1)

def item_stats(C, L, H):
    """Per item, 40-80% window: top-head share and effective number of heads / H.
    Items where one head outputs > 50x its layer's median head (rare, OLMo-2)
    are flagged so they can be excluded."""
    w = window(L); S = C[:, w].astype(np.float64)
    P = S / S.sum(-1, keepdims=True)
    top = P.max(-1).mean(1)
    ent = np.exp(-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(-1)).mean(1) / H
    flag = ((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
    return top, ent, flag

def slope_p(y, k, n_perm=2000, seed=0):
    k = np.asarray(k, float); kc = k - k.mean(); ss = kc @ kc; yc = y - y.mean(); o = kc @ yc / ss
    rng = np.random.default_rng(seed)
    nl = np.array([rng.permutation(kc) @ yc / ss for _ in range(n_perm)])
    return float(o), float((np.sum(np.abs(nl) >= abs(o) - 1e-15) + 1) / (n_perm + 1))

def signatures(runs, n_perm=2000):
    """Per run: slope of effective heads and of top share on k (% of mean per step),
    on all measured items and on items the model answered correctly."""
    rows = []
    for r in runs:
        ok_meas = np.array([x["measured"] for x in r["items"]])
        C = r["C"][ok_meas]; its = [x for x in r["items"] if x["measured"]]
        if len(its) < 10: continue
        k = np.array([x["k"] for x in its]); cor = np.array([x["correct"] for x in its])
        top, ent, flag = item_stats(C, r["L"], r["H"])
        keep = ~flag
        row = {"model": r["model"], "mode": r["mode"], "family": r["family"], "n": int(keep.sum()), "flagged": int(flag.sum())}
        for name, y in (("eff_heads", ent), ("top1", top)):
            for sub, sel in (("all", keep), ("correct", keep & cor)):
                if sel.sum() >= 10 and len(set(k[sel])) >= 2:
                    s, p = slope_p(y[sel], k[sel], n_perm)
                    row[f"{name}_{sub}"] = (100 * s / y[sel].mean(), p)
        rows.append(row)
    return rows

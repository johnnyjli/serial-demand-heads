#!/usr/bin/env python3
"""
Serial-depth task suites with task-agnostic item metadata.

Every probe downstream (sweep_probe, ablation_probe, signature_analysis) reads
depth, answer set and echo from the items written here instead of parsing a
prompt template, so any family below can go through the same pipeline.

A suite is one JSON file

    {"name", "family", "wording", "setting", "chance", "items": [item, ...]}

and an item is

    id          "<suite name>/<k>/<i>", unique and stable given the seed
    family      pointer | arith | binding | relational | fixed_digit | ladder
    wording     w0 | w1 (paraphrase frames, WORDINGS)
    k           serial depth (fixed_digit: effective depth, see below)
    label       "D{k}N{i:02d}", for the existing parsers
    text        the prompt; the answer follows as " <digit>"
    gold        the answer digit
    answer_set  admissible answers; competitors = answer_set - gold
    k_written   whether the numeral of k appears in the prompt
    echo        that numeral, if written; echoing it is never correct
    split       calib (about 30%, stratified by k) | test
    setting     item-level generator parameters (start, table, ...)

Families (FAMILIES documents each, WORDINGS holds the templates the generator
formats, so the two cannot drift apart):

    pointer      random n-cycle, k written ("Take k steps"); w0 is the
                 serial_depth_fs template.  Worked examples are drawn per item
                 with balanced answers (shot_mode="per_item") or shared by all
                 items (shot_mode="shared", reproducing serial_depth_fs).
    arith        add d, k times, mod `modulus` (default 10); k written.
    binding      variable chains rooted in digit literals, one link longer than
                 the deepest query (no queried variable is a chain's leaf); k is
                 the hop distance of the queried variable.  Numeral-free.
    relational   "The boss of Q is M" chains rooted in digit bosses, built like
                 binding; k is the number of hops from the queried person to
                 the top boss.  Numeral-free.
    fixed_digit  the numeral control: written step count fixed, rule table
                 varied (one c-cycle through the start plus fixed points), so
                 k = distinct states traversed changes while the numeral does
                 not.

Within a family, items are drawn from one seeded stream that no template
consumes, so w0 and w1 suites built with the same seed and settings hold the
same instances (tables, starts, answers, split): wordings are compared on
paired items.  Different families never share instances: pointer draws from
Random(seed) (the serial_depth_fs reproduction needs it), every other family
from Random("<family>:<seed>").  seed must be an int.  Per k, gold digits are
balanced over the admissible answers (counts differ by at most one); the calib
split takes about calib_frac of every depth, round-robin over gold digits, and
keeps a matched family's body (shared across k) on one side.

Final problems are drawn without replacement within each depth and gold digit,
so a depth cell repeats a problem only when it holds more items than the
settings have problems (e.g. arith modulus=7: 36 per depth), and then as evenly
as possible; identical final problems always fall on one side of the split, so
calibration stays held out.  No worked example is the item's own query: at the
item's depth and gold an example could otherwise be the query with its answer
shown, which a copying model solves for free, and only at the shallow depths
the examples use.

Worked examples (shot_mode="per_item") come in a random depth order, and at
every depth and example position their answers are balanced over the gold's own
admissible set: never the item's written numeral (echo), never the example's own
step count, distinct within a prompt as far as possible.  Copying any one
example is then right 1/(n-1) of the time at every k (n admissible states), so a
copy bias cannot make a depth effect.

Guarantees, checked on every suite built or loaded (abort, never write a wrong
item): the gold answer, depth, answer set, written numeral (echo, k_written)
and start/table/query metadata are recomputed from the prompt text alone by an
independent parser (solve_text), including every worked example; gold is in the
answer set and never equals the echo; chance = 1/|answer set|; ids and labels
agree and are unique; the suite has items; every table, increment and chain
has the family's form (pointer: one n-cycle; fixed_digit: one cycle through
the start; arith: k*d != 0 mod m).  check_suite reports the rest: per
tokenizer, constant token count across k and deep_probe.align_answer on every
item; per k, answer balance, worked-example balance per position and the
accuracy of copying each example, identical prompts and worked examples equal
to the item's own query (both failures); distractor digits in every prompt;
and, as warnings, design limits the analysis must handle (shortest lookup path
below k, arith offsets fixed by k, repeated final problems, a queried
definition whose distance to the query trends with k, calib split too small
for the min-correct rule, prompts above a tokenizer's model_max_length).

    python3 serial_tasks.py --family pointer arith binding relational fixed_digit \\
        --wording w0 w1 --out suites [--k 1 6] [--n-per 40] [--seed 0] \\
        [--set shots=5 --set shot_mode=per_item] [--tokenizer gpt2 Qwen/Qwen2.5-0.5B]
    python3 serial_tasks.py --ladder --out suites
    python3 serial_tasks.py --check suites/*.json --tokenizer gpt2 Qwen/Qwen2.5-0.5B
"""
import argparse, collections, hashlib, json, math, operator, random, re, sys
from pathlib import Path

DIGITS = "0123456789"
# Bumped whenever the generator draws different items for the same arguments; it
# enters every automatic suite name, so a regenerated suite never takes the name
# (and hence the item ids and sweep_probe output file) of an older version.
GENERATOR_VERSION = 3
# Re-draws of a worked example that came out as the item's own query before its
# depth-answer pair is taken to have no other problem (then another answer).
_TRIES = 64
# Every entity name follows a space, and a space plus one letter is one token in
# the gpt2 and Qwen2.5 tokenizers (check_suite verifies this per suite), so
# swapping which entity is queried cannot change the token count.  "a"/"A" and
# "i"/"I" are dropped: they read as English words.
LOWER = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in "ai"]
UPPER = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "AI"]
EXAMPLE_END = " {ans}.\n\n"          # a solved instance + this = one worked example
ITEM_KEYS = ("id", "family", "wording", "k", "label", "text", "gold", "answer_set",
             "k_written", "echo", "split", "setting")
SUITE_KEYS = ("name", "family", "wording", "setting", "chance", "items")


# ────────────────────────────── templates ───────────────────────────────────
# Placeholders: a,b rule source/target  s start  k written step count
# d increment  m modulus  x,y statement subject/object  v queried entity.
_POINTER_WORDINGS = {
    "w0": {"head": "Rules: ", "rule": "{a} goes to {b}.",
           "query": "Start at {s}. Take {k} steps. You end at"},
    "w1": {"head": "Map: ", "rule": "{a} leads to {b}.",
           "query": "From {s}, make {k} moves. You finish at"},
}
WORDINGS = {
    "pointer": _POINTER_WORDINGS,
    # same frames as pointer: the control must differ from it only in the table
    "fixed_digit": _POINTER_WORDINGS,
    # "query" is the mod-10 frame; "query_mod" is used for any other modulus
    "arith": {
        "w0": {"query": "Rule: add {d} and keep only the last digit. Start at {s}. "
                        "Apply the rule {k} times. You end at",
               "query_mod": "Rule: add {d} and keep the remainder after dividing by {m}. "
                            "Start at {s}. Apply the rule {k} times. You end at"},
        "w1": {"query": "x = {s}. Repeat {k} times: x = (x + {d}) mod 10. Now x =",
               "query_mod": "x = {s}. Repeat {k} times: x = (x + {d}) mod {m}. Now x ="},
    },
    "binding": {
        "w0": {"head": "Assignments: ", "stmt": "{x} = {y}.", "query": "Therefore {v} ="},
        "w1": {"head": "Facts: ", "stmt": "Let {x} be {y}.", "query": "The value of {v} is"},
    },
    "relational": {
        "w0": {"head": "Chart: ", "stmt": "The boss of {x} is {y}.",
               "query": "The top boss above {v} is"},
        "w1": {"head": "Chart: ", "stmt": "{x} reports to {y}.",
               "query": "{v} ultimately reports to"},
    },
    # the original single-prompt ladder (active_subgraph_dot serial_depth)
    "ladder": {"w0": _POINTER_WORDINGS["w0"]},
}

_COMMON = {"shots": 3, "shot_mode": "per_item", "shot_depths": [1, 2, 3], "calib_frac": 0.3}
FAMILIES = {
    "pointer": {
        "doc": "Pointer chase on a random single n-cycle over states 0..n_states-1 with a "
               "random start; each item draws its own cycle.  Gold never equals k.",
        "k": "number of steps, written as a numeral", "k_written": True, "matched": False,
        "answer_set": "all states 0..n_states-1 (chance 1/n_states)",
        "balance": "per k, gold uniform over the states other than k",
        "defaults": dict(_COMMON, n_states=7, balance=True),
        "notes": "Seeded from Random(seed) (the other families from Random('<family>:"
                 "<seed>')).  shot_mode='shared' draws one worked-example prefix exactly as "
                 "_gen_serial_depth_rep does; with balance=False (unbalanced legacy item "
                 "draw) and seed 0 it reproduces serial_depth_fs byte for byte.  Llama-3.1-70B "
                 "copied the shared prefix's answer '6' (2 of 3 examples) on 119/240 items, "
                 "hence per_item examples by default: random depth order, and per depth and "
                 "example position answers balanced over the states other than k (never the "
                 "example's own step count either), so copying any example scores "
                 "1/(n_states-1) at every k.  On one n-cycle, k steps forward land where "
                 "n_states-k steps backward do, so the shortest lookup path is "
                 "min(k, n_states-k) (1,2,3,3,2,1 for the 7-state default); it is recorded "
                 "per item as setting.min_hops and check_suite warns.  Use it as a covariate, "
                 "restrict to k <= n_states//2, or raise n_states (e.g. n_states=10, k 1..5).  "
                 "With balance=True no cycle repeats within a (k, gold) cell until all "
                 "(n_states-1)! are used, and no final problem equals a shared example.",
    },
    "arith": {
        "doc": "Composed arithmetic: start digit s, add d (from increments) k times, keep "
               "the remainder mod `modulus` (default 10: 'keep only the last digit').  "
               "Gold = (s + k*d) mod modulus.",
        "k": "number of additions, written as a numeral", "k_written": True, "matched": False,
        "answer_set": "0..modulus-1 (chance 1/modulus; 0..9 and 0.1 by default)",
        "balance": "per k, gold uniform over 0..modulus-1 minus k; s is solved for",
        "defaults": dict(_COMMON, modulus=10, increments=list(range(1, 10))),
        "notes": "d with k*d = 0 mod modulus is never used, so the answer never equals the "
                 "start digit (copying the start would otherwise score at k=5 for 4 of 9 d).  "
                 "Mod 10 leaks: at k=5 every allowed d gives offset 5 (gold = start+5 without "
                 "composing anything) and even k keep the start's parity; check_suite reports "
                 "the offsets per k and warns.  A prime modulus above max k removes both: "
                 "modulus=7 (increments default to 1..modulus-1) gives offsets uniform over "
                 "1..6 at every k in 1..6, chance 1/7.  Per (k, gold) the increment cycles "
                 "through the allowed ones, so a final problem repeats only when a gold has "
                 "more items than allowed increments: modulus=7 holds 36 problems per depth "
                 "(6 golds x 6 increments), so n_per above 36 repeats some (kept on one side "
                 "of the split; check_suite warns, and fails identical prompts, i.e. shots=0).",
    },
    "binding": {
        "doc": "n_chains chains of single-letter variables, each rooted in a distinct digit "
               "literal (x = 3. y = x. ...), all of max(k_range)+1 links, in one body.  The "
               "same body is queried at every depth.  The last link of a chain is never "
               "queried, so every queried variable is also the value of the next link: it "
               "is mentioned twice at every k, never as a leaf.",
        "k": "hop distance of the queried variable to its literal (1 = assigned the digit)",
        "k_written": False, "matched": True,
        "answer_set": "the n_chains root digits (chance 1/n_chains)",
        "balance": "gold uniform over 0..9 across bodies (identical at every k)",
        "defaults": dict(_COMMON, n_chains=None, order="shuffled"),
        "notes": "Numeral-free.  Items with the same i share their body and worked "
                 "examples, so prompts at different k differ only in the queried letter.  "
                 "Names: n_chains x (max k + 1) of 24 letters; n_chains=None takes 4 where "
                 "that fits (max k <= 5), else as many as fit (3 for k up to 6).  "
                 "order='shuffled' (default) permutes all statements, with the queried "
                 "chain's statement order balanced across bodies (every link takes every "
                 "rank equally often), so the distance from the queried definition to the "
                 "query (setting.def_distance, statements, 1 = the last) does not trend with "
                 "k.  order='interleaved' keeps each chain in order (definition before use), "
                 "so deeper links sit later and def_distance falls with k (about 22 -> 4 for "
                 "k 1..6): a recency covariate confounded with depth; check_suite warns.  "
                 "setting.query_refs counts the statements that use the queried entity as a "
                 "value (1 at every k).  binding and relational draw from separate seeded "
                 "streams, so the two families never hold the same instances (RQ1 counts "
                 "them as two families).",
    },
    "relational": {
        "doc": "n_chains reporting chains (The boss of Q is M.) of capital-letter people, "
               "each topped by a distinct digit boss, all of max(k_range)+1 links (the last "
               "never queried, as in binding); query 'The top boss above X is'.",
        "k": "hops from the queried person to the top boss", "k_written": False,
        "matched": True,
        "answer_set": "the n_chains top-boss digits; these are the only digits in the "
                      "prompt, so 'root digits' and 'all digits used' coincide "
                      "(chance 1/n_chains)",
        "balance": "gold uniform over 0..9 across bodies (identical at every k)",
        "defaults": dict(_COMMON, n_chains=None, order="shuffled"),
        "notes": "Numeral-free and matched, with the same chain sizes, orders and "
                 "def_distance / query_refs records as binding.  Non-root people are "
                 "letters, not digits, because ten digit labels cannot hold chains of depth "
                 "6 plus distractor chains.",
    },
    "fixed_digit": {
        "doc": "Numeral control on n_states-state tables: the written step count is fixed "
               "(written_k); the table is one c-cycle through the start plus fixed points, "
               "c = k+1, so the walk traverses k distinct new states (k=0: identity, "
               "1: transposition, ..., n_states-1: the full cycle).",
        "k": "effective depth = distinct states visited after the start "
             "(len(set(s_0..s_K)) - 1)",
        "k_written": True, "matched": False,
        "answer_set": "all states 0..n_states-1 (chance 1/n_states)",
        "balance": "per k, gold uniform over the states other than written_k",
        "defaults": dict(_COMMON, n_states=7, written_k=None),
        "notes": "echo = written_k for every item.  When (k+1) divides written_k the walk "
                 "ends on its start, so copying the start is correct there (unavoidable with "
                 "permutation tables, as in the original); flagged per item as "
                 "setting.gold_eq_start.  written_k=None picks the numeral >= max(k_range) "
                 "with the fewest such cells: 7 for k 1..6 (only k=6), where 6 would give "
                 "three (k=1, 2, 5).  Worked examples are ordinary pointer examples.  The "
                 "original's identity cell is k=0 and is opt-in: pass k_range=(0, 6) to keep "
                 "it (the shared default k_range=(1, 6) leaves it out; k=0 is a pure "
                 "copy-the-start cell).  setting.min_hops = the shortest lookup path from "
                 "start to gold, forward or backward (1,1,1,2,1,0 for k 1..6 with "
                 "written_k 7): k counts states traversed, not lookups needed.  Tables are "
                 "drawn without replacement per (k, gold): k=1 has n_states-1 "
                 "transpositions per gold (42 problems per depth on 7 states, so n_per up to "
                 "42 without repeats), the k=0 identity cell one problem per gold (with "
                 "shots=0 its prompts repeat exactly, which check_suite fails).",
    },
    "ladder": {
        "doc": "The original single-prompt ladder TASK_SUITES['serial_depth'] (fixed "
               "5-cycle, start 0, k=1..6), wrapped by ladder_suite().",
        "k": "number of steps, written", "k_written": True, "matched": False,
        "answer_set": "the 5 states (chance 0.2)", "balance": "one item per k",
        "defaults": {}, "notes": "split 'test' only; label D{k}N00 (original D{k}W1 kept "
                                 "in setting.orig_label).",
    },
}


# ────────────────────────────── primitives ──────────────────────────────────
def _rand_cycle(rng, n):
    """Random single n-cycle on 0..n-1 -> dict.  Same draws as
    active_subgraph_dot._rand_cycle, which serial_depth_fs reproduction needs."""
    order = list(range(n))
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % n] for i in range(n)}


def _walk(tbl, s, k):
    for _ in range(k):
        s = tbl[s]
    return s


def _back(tbl, g, k):
    """The state k steps before g (tables are permutations)."""
    inv = {b: a for a, b in tbl.items()}
    for _ in range(k):
        g = inv[g]
    return g


def _cycles(tbl):
    """The cycles of a permutation table, each as a list of states."""
    seen, out = set(), []
    for x in sorted(tbl):
        if x not in seen:
            cyc = [x]
            seen.add(x)
            while tbl[cyc[-1]] != x:
                cyc.append(tbl[cyc[-1]])
                seen.add(cyc[-1])
            out.append(cyc)
    return out


def _fresh(draw, key, used, space, clash=None):
    """draw() without replacement within a cell and away from worked examples.

    Returns a draw whose key is not in `used` (a set, then updated) and that
    does not `clash` (reproduce a worked example), as far as possible.  `space`
    is the number of distinct keys draw() can return: once all are used a new
    round starts, so a cell repeats a problem only when it holds more items
    than the settings have problems, and then as evenly as possible.  Rejection
    sampling with 50 tries per possible key finds a free problem whenever one
    exists (the spaces that can run out are small); failing that it takes a
    repeat, and a clash only when every problem clashes (degenerate settings
    such as a 2-state pointer; check_suite fails the suite then).
    """
    if len(used) >= space:
        used.clear()
    tries = int(min(50 * space, 20000))
    for fresh_only in (True, False):
        for _ in range(tries):
            x = draw()
            if not (clash and clash(x)) and not (fresh_only and key(x) in used):
                used.add(key(x))
                return x
        used.clear()                              # nothing new clear of the examples: repeat
    x = draw()
    used.add(key(x))
    return x


def _example_blocks(prefix):
    """The worked examples of a prefix without their answers: the blocks no
    final problem may equal."""
    return {re.sub(r" \d\.$", "", b) for b in prefix.split("\n\n")[:-1]}


def _build_avoiding(build, kd, a, forbid):
    """build(kd, a), re-drawn while it is one of the `forbid` blocks; None when
    every try is (that depth-answer pair then has no other problem)."""
    for _ in range(_TRIES if forbid else 1):
        t = build(kd, a)
        if t not in forbid:
            return t
    return None


def _min_hops(tbl, s, g):
    """Fewest lookups from s to g walking the table forward or backward: on a
    cycle of length c with g f steps ahead, min(f, c - f).  (A 7-cycle's k=6 is
    one backward lookup.)"""
    ring, x = [s], tbl[s]
    while x != s:
        ring.append(x)
        x = tbl[x]
    f = ring.index(g)
    return min(f, len(ring) - f)


def _balanced(pool, n, rng):
    """n draws from pool with counts differing by at most one, in random order."""
    pool = list(pool)
    if not pool:
        raise ValueError("no admissible answers to balance over")
    out = []
    while len(out) < n:
        blk = pool[:]
        rng.shuffle(blk)
        out += blk
    out = out[:n]
    rng.shuffle(out)
    return out


def _pointer_body(W, tbl, s, k):
    return (W["head"] + " ".join(W["rule"].format(a=a, b=tbl[a]) for a in sorted(tbl))
            + " " + W["query"].format(s=s, k=k))


def _chain_body_text(W, stmts, v):
    return (W["head"] + " ".join(W["stmt"].format(x=x, y=y) for x, y in stmts)
            + " " + W["query"].format(v=v))


def _new_tally():
    """Worked-example answer counts: one Counter per example position, plus
    the pooled count used to break ties."""
    return {"slot": [], "all": collections.Counter()}


def _max_distinct(cands, taken):
    """How many of the positions (candidate lists) can get pairwise distinct
    answers outside `taken`: a maximum bipartite matching (at most ~10 x 10)."""
    owner = {}

    def augment(j, seen):
        for d in cands[j]:
            if d in taken or d in seen:
                continue
            seen.add(d)
            if d not in owner or augment(owner[d], seen):
                owner[d] = j
                return True
        return False

    return sum(augment(j, set()) for j in range(len(cands)))


def _examples(build, pool, S, rng, tally, avoid=(), own=True, shuffle=True, forbid=()):
    """Worked-example prefix with balanced answers.

    The example depths (shot_depths cycled to `shots`) come in random order when
    `shuffle`, so no position is tied to one depth.  An example's answer is never
    in `avoid` (the item's written numeral: the gold's own exclusion) nor, with
    `own`, the example's own step count.  Answers are distinct within the prefix
    as far as the exclusions allow (a maximum matching is kept reachable), and
    each is the least-used digit so far at its position (tally["slot"][j]), then
    overall (tally["all"]), ties at random.  One tally per depth makes every
    position balanced over the gold's admissible set at every k, so copying any
    example scores the same at every k.  build(kd, answer) returns the solved
    instance's text.

    No example is one of the `forbid` blocks (the item's own final problem):
    at the item's depth and gold an example could otherwise be the query with
    its answer shown.  Such an example is re-drawn with the same answer (the
    balance is kept); only when that depth-answer pair has no other problem
    (e.g. a single allowed increment) is another answer taken, and only when
    every answer does (degenerate settings) is the clash kept, for check_suite
    to fail.
    """
    depths = [S["shot_depths"][j % len(S["shot_depths"])] for j in range(S["shots"])]
    if shuffle:
        rng.shuffle(depths)
    avoid = {str(a) for a in avoid}
    cands = [[d for d in pool if str(d) not in avoid and not (own and str(d) == str(kd))]
             for kd in depths]
    if any(not c for c in cands):
        raise ValueError("a worked example has no admissible answer")
    while len(tally["slot"]) < len(cands):
        tally["slot"].append(collections.Counter())
    goal = _max_distinct(cands, set())
    used, answers, parts = set(), [], []
    for j, (kd, cand) in enumerate(zip(depths, cands)):
        rest = cands[j + 1:]
        fresh = [d for d in cand if d not in used
                 and len(used) + 1 + _max_distinct(rest, used | {d}) >= goal]
        opts = fresh or cand          # no fresh digit fits: a repeat cannot cost distinctness
        cnt = tally["slot"][j]

        def least_used(live):
            best = min((cnt[d], tally["all"][d]) for d in live)
            return rng.choice([d for d in live if (cnt[d], tally["all"][d]) == best])

        a = text = None
        for tier in (opts, [d for d in cand if d not in opts]):
            live = list(tier)
            while live and text is None:
                a = least_used(live)
                text = _build_avoiding(build, kd, a, forbid)
                live.remove(a)
            if text is not None:
                break
        if text is None:              # every answer at this depth is the query itself
            a = least_used(opts)
            text = build(kd, a)
        cnt[a] += 1
        tally["all"][a] += 1
        used.add(a)
        answers.append(a)
        parts.append(text + EXAMPLE_END.format(ans=a))
    return "".join(parts), answers, depths


# ────────────────────────────── generators ──────────────────────────────────
# Each returns a list of partial items: k, i, text, gold, answer_set, echo, setting.

def _gen_pointer(W, ks, n_per, rng, S):
    n = S["n_states"]
    states = list(range(n))

    def build(kd, a):
        tbl = _rand_cycle(rng, n)
        return _pointer_body(W, tbl, _back(tbl, a, kd), kd)

    shared = S["shot_mode"] == "shared"
    ex_blocks = set()
    if shared:
        # _gen_serial_depth_rep's prefix, draw for draw (answers not balanced:
        # this mode exists to reproduce that condition)
        parts, sh_ans, sh_k = [], [], []
        for j in range(S["shots"]):
            kd = S["shot_depths"][j % len(S["shot_depths"])]
            tbl, s = _rand_cycle(rng, n), rng.randrange(n)
            a = _walk(tbl, s, kd)
            parts.append(_pointer_body(W, tbl, s, kd) + EXAMPLE_END.format(ans=a))
            sh_ans.append(a); sh_k.append(kd)
        prefix = "".join(parts)
        ex_blocks = _example_blocks(prefix)
    items = []
    key = lambda t: tuple(t[d] for d in states)   # noqa: E731
    for k in ks:
        tally = _new_tally()                       # example answers, balanced within k
        targets = (_balanced([d for d in states if d != k], n_per, rng)
                   if S["balance"] else None)
        used = collections.defaultdict(set)        # cycles drawn per gold: no repeats
        for i in range(n_per):
            if targets is not None:
                g = targets[i]
                clash = ((lambda t: _pointer_body(W, t, _back(t, g, k), k) in ex_blocks)
                         if ex_blocks else None)
                tbl = _fresh(lambda: _rand_cycle(rng, n), key, used[g],
                             math.factorial(n - 1), clash)
                s = _back(tbl, g, k)
            else:                                  # legacy draw: resample gold == k
                while True:
                    tbl, s = _rand_cycle(rng, n), rng.randrange(n)
                    if _walk(tbl, s, k) != k:
                        break
            gold = _walk(tbl, s, k)
            final = _pointer_body(W, tbl, s, k)
            if not shared:                         # drawn after the final: never equal to it
                prefix, sh_ans, sh_k = _examples(build, states, S, rng, tally, avoid={k},
                                                 forbid={final})
            items.append(dict(k=k, i=i, text=prefix + final,
                              gold=str(gold), answer_set=[str(d) for d in states],
                              echo=str(k),
                              setting={"start": s, "table": [tbl[d] for d in states],
                                       "min_hops": _min_hops(tbl, s, gold),
                                       "shot_answers": list(sh_ans), "shot_ks": list(sh_k)}))
    return items


def _gen_fixed_digit(W, ks, n_per, rng, S):
    n, K = S["n_states"], S["written_k"]
    states = list(range(n))

    def build(kd, a):                              # ordinary pointer examples
        tbl = _rand_cycle(rng, n)
        return _pointer_body(W, tbl, _back(tbl, a, kd), kd)

    shared = S["shot_mode"] == "shared"
    ex_blocks = set()
    if shared:
        prefix, sh_ans, sh_k = _examples(build, states, S, rng, _new_tally(), avoid={K},
                                         shuffle=False)
        ex_blocks = _example_blocks(prefix)
    items = []
    key = lambda t: tuple(t[d] for d in states)   # noqa: E731
    for e in ks:
        tally = _new_tally()
        targets = _balanced([d for d in states if str(d) != str(K)], n_per, rng)
        c = e + 1                                  # cycle length through the start
        # c-cycles through a given gold: the other c-1 members in every cyclic order
        space = math.comb(n - 1, c - 1) * math.factorial(c - 1)
        used = collections.defaultdict(set)        # tables drawn per gold: no repeats
        for i in range(n_per):
            g = targets[i]

            def draw():
                members = rng.sample([d for d in states if d != g], c - 1) + [g]
                rng.shuffle(members)
                tbl = {d: d for d in states}
                for j in range(c):
                    tbl[members[j]] = members[(j + 1) % c]
                return tbl
            clash = ((lambda t: _pointer_body(W, t, _back(t, g, K), K) in ex_blocks)
                     if ex_blocks else None)
            tbl = _fresh(draw, key, used[g], space, clash)
            s = _back(tbl, g, K)
            final = _pointer_body(W, tbl, s, K)
            if not shared:                         # drawn after the final: never equal to it
                prefix, sh_ans, sh_k = _examples(build, states, S, rng, tally, avoid={K},
                                                 forbid={final})
            items.append(dict(k=e, i=i, text=prefix + final,
                              gold=str(g), answer_set=[str(d) for d in states], echo=str(K),
                              setting={"start": s, "table": [tbl[d] for d in states],
                                       "written_k": K, "cycle_len": c, "gold_eq_start": s == g,
                                       "min_hops": _min_hops(tbl, s, g),
                                       "shot_answers": list(sh_ans), "shot_ks": list(sh_k)}))
    return items


def _arith_query(W, m):
    """The frame for modulus m: the mod-10 template, else the general one."""
    return W["query"] if m == 10 else W["query_mod"]


def _gen_arith(W, ks, n_per, rng, S):
    incs, m = S["increments"], S["modulus"]
    Q = _arith_query(W, m)

    def allowed(k):                                # never back to the start: k*d != 0 mod m
        return sorted({d for d in incs if (k * d) % m})

    def build(kd, a):
        d = rng.choice(allowed(kd))
        return Q.format(s=(a - kd * d) % m, d=d, k=kd, m=m)

    digits = list(range(m))
    shared = S["shot_mode"] == "shared"
    ex_blocks = set()
    if shared:                                     # one prefix for every k: no echo to avoid
        prefix, sh_ans, sh_k = _examples(build, digits, S, rng, _new_tally(), shuffle=False)
        ex_blocks = _example_blocks(prefix)
    items = []
    for k in ks:
        tally = _new_tally()
        targets = _balanced([d for d in digits if d != k], n_per, rng)
        ds = allowed(k)
        used = collections.defaultdict(set)        # increments drawn per gold: no repeats
        for i in range(n_per):
            g = targets[i]
            clash = ((lambda d: Q.format(s=(g - k * d) % m, d=d, k=k, m=m) in ex_blocks)
                     if ex_blocks else None)
            d = _fresh(lambda: rng.choice(ds), lambda d: d, used[g], len(ds), clash)
            s = (g - k * d) % m
            final = Q.format(s=s, d=d, k=k, m=m)
            if not shared:                         # drawn after the final: never equal to it
                prefix, sh_ans, sh_k = _examples(build, digits, S, rng, tally, avoid={k},
                                                 forbid={final})
            items.append(dict(k=k, i=i, text=prefix + final,
                              gold=str(targets[i]), answer_set=[str(x) for x in digits],
                              echo=str(k),
                              setting={"start": s, "increment": d,
                                       "shot_answers": list(sh_ans), "shot_ks": list(sh_k)}))
    return items


def _chain_body(rng, target, C, H, pool, order, ranks=None):
    """C chains of H names, each rooted in a distinct digit; the target chain
    (root = target) sits at a random position.  Statement slots are dealt to the
    chains at random; order='interleaved' fills each chain's slots in chain
    order (definition before use), 'shuffled' in a random order, except that
    `ranks` (ranks[j] = rank of link j among its chain's statements) fixes the
    target chain's order, which _gen_chain balances across bodies."""
    roots = [target] + rng.sample([d for d in range(10) if d != target], C - 1)
    t = rng.randrange(C)
    roots[0], roots[t] = roots[t], roots[0]
    names = rng.sample(pool, C * H)
    chains = [names[c * H:(c + 1) * H] for c in range(C)]
    per = [[(ch[0], str(r))] + [(ch[j], ch[j - 1]) for j in range(1, H)]
           for ch, r in zip(chains, roots)]
    seq = [c for c in range(C) for _ in range(H)]  # which chain fills each slot
    rng.shuffle(seq)
    orders = []
    for c in range(C):
        if order == "interleaved":
            orders.append(list(range(H)))
        elif c == t and ranks is not None:
            orders.append(sorted(range(H), key=ranks.__getitem__))
        else:
            o = list(range(H))
            rng.shuffle(o)
            orders.append(o)
    pos, stmts = [0] * C, []
    for c in seq:
        stmts.append(per[c][orders[c][pos[c]]]); pos[c] += 1
    return {"stmts": stmts, "chains": chains, "roots": roots, "target": t}


def _gen_chain(family, W, ks, n_per, rng, S):
    # Chains one link longer than the deepest query: the last link is never
    # queried, so every queried entity is also the next link's value (mentioned
    # twice) at every k.  With chains exactly max(k) long the deepest cell alone
    # queried a leaf (mentioned once), a surface property confounded with depth.
    C, H, order = S["n_chains"], max(ks) + 1, S["order"]
    pool = LOWER if family == "binding" else UPPER

    def build(kd, a):
        b = _chain_body(rng, a, C, H, pool, order)
        return _chain_body_text(W, b["stmts"], b["chains"][b["target"]][kd - 1])

    def finals(b):                                # the body's final block at every k
        return [_chain_body_text(W, b["stmts"], b["chains"][b["target"]][k - 1]) for k in ks]

    digits = list(range(10))
    tally = _new_tally()                          # bodies are shared across k
    shared = S["shot_mode"] == "shared"
    ex_blocks = set()
    if shared:                                    # no numerals: nothing to avoid
        prefix, sh_ans, sh_k = _examples(build, digits, S, rng, tally, own=False,
                                         shuffle=False)
        ex_blocks = _example_blocks(prefix)
    targets = _balanced(digits, n_per, rng)
    bodies, used, ranks = [], set(), None
    for i in range(n_per):
        if order == "shuffled":
            # Within each block of H bodies link j takes every rank once among
            # the target chain's statements (a random permutation composed with a
            # rotation, so each body's order is still uniform): the queried
            # definition's distance to the query does not trend with k.
            if i % H == 0:
                perm = list(range(H))
                rng.shuffle(perm)
            ranks = [perm[(j + i) % H] for j in range(H)]
        clash = (lambda b: any(f in ex_blocks for f in finals(b))) if ex_blocks else None
        b = _fresh(lambda: _chain_body(rng, targets[i], C, H, pool, order, ranks),
                   lambda b: tuple(b["stmts"]), used, math.inf, clash)
        if not shared:                            # drawn after the body: never equal to it
            prefix, sh_ans, sh_k = _examples(build, digits, S, rng, tally, own=False,
                                             forbid=set(finals(b)))
        bodies.append((prefix, list(sh_ans), list(sh_k), b))
    items = []
    for k in ks:
        for i, (pre, sa, sk, b) in enumerate(bodies):
            v = b["chains"][b["target"]][k - 1]
            xs = [x for x, _ in b["stmts"]]
            items.append(dict(k=k, i=i, text=pre + _chain_body_text(W, b["stmts"], v),
                              gold=str(targets[i]),
                              answer_set=sorted(str(r) for r in b["roots"]), echo=None,
                              setting={"body": i, "query": v, "roots": list(b["roots"]),
                                       "target_chain": b["target"],
                                       "def_distance": len(xs) - xs.index(v),
                                       "query_refs": sum(y == v for _, y in b["stmts"]),
                                       "shot_answers": sa, "shot_ks": sk}))
    return items


_GEN = {
    "pointer": _gen_pointer,
    "fixed_digit": _gen_fixed_digit,
    "arith": _gen_arith,
    "binding": lambda W, ks, n, rng, S: _gen_chain("binding", W, ks, n, rng, S),
    "relational": lambda W, ks, n, rng, S: _gen_chain("relational", W, ks, n, rng, S),
}


# ────────────────────────────── settings ────────────────────────────────────
def _settings(family, lo, hi, given):
    """Merge defaults, reject unknown keys, validate ranges."""
    D = FAMILIES[family]["defaults"]
    unknown = set(given) - set(D)
    if unknown:
        raise ValueError(f"{family}: unknown setting(s) {sorted(unknown)}; "
                         f"known: {sorted(D)}")
    S = json.loads(json.dumps(D))                 # deep copy, JSON types only
    try:                                          # tuples -> lists, so save/load round-trips
        S.update(json.loads(json.dumps(_plain(given))))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{family}: settings must be JSON values (numbers, strings, "
                         f"lists): {e}") from None
    if not isinstance(S["shots"], int) or S["shots"] < 0:
        raise ValueError("shots must be an int >= 0")
    if S["shot_mode"] not in ("per_item", "shared"):
        raise ValueError("shot_mode must be 'per_item' or 'shared'")
    if not 0 <= S["calib_frac"] < 1:
        raise ValueError("calib_frac must be in [0, 1)")
    if lo > hi:
        raise ValueError(f"k_range {lo, hi}: low above high")
    if lo < (0 if family == "fixed_digit" else 1):
        raise ValueError(f"{family}: k must be >= {0 if family == 'fixed_digit' else 1}")

    if family in ("pointer", "fixed_digit"):
        n = S["n_states"]
        if not 2 <= n <= 10:
            raise ValueError("n_states must be 2..10 (single-digit states)")
        dmax = n - 1                              # worked examples are n-cycle walks
        if family == "pointer":
            if hi >= n:
                raise ValueError(f"pointer: k must stay below n_states ({hi} >= {n}); a "
                                 f"longer walk revisits states")
        else:
            if S["written_k"] is None:
                S["written_k"] = _auto_written_k(lo, hi)
            K = S["written_k"]
            if not isinstance(K, int) or not 1 <= K <= 9:
                raise ValueError("written_k must be a single digit 1..9")
            if hi > min(K, n - 1):
                raise ValueError(f"fixed_digit: effective depth {hi} exceeds min(written_k, "
                                 f"n_states-1) = {min(K, n - 1)}")
    elif family == "arith":
        if hi > 9:
            raise ValueError("arith: k must be a single digit")
        m = S["modulus"]
        if isinstance(m, bool) or not isinstance(m, int) or not 2 <= m <= 10:
            raise ValueError("modulus must be an int in 2..10 (single-digit answers)")
        if "increments" not in given:
            S["increments"] = list(range(1, m))
        incs = S["increments"]
        if not incs or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d < m
                           for d in incs):
            raise ValueError(f"increments must be ints in 1..{m - 1} (modulus {m})")
        dmax = 9
    else:
        pool = LOWER if family == "binding" else UPPER
        links = hi + 1                            # the last link is never queried
        fit = len(pool) // links
        if S["n_chains"] is None:                 # 4 where they fit, else as many as fit
            if fit < 2:
                raise ValueError(f"{family}: k up to {hi} needs chains of {links} names; "
                                 f"{len(pool)} single-letter names hold fewer than 2 such "
                                 f"chains (max k is {len(pool) // 2 - 1})")
            S["n_chains"] = min(4, fit)
        C = S["n_chains"]
        if isinstance(C, bool) or not isinstance(C, int) or not 2 <= C <= 10:
            raise ValueError("n_chains must be an int 2..10 (distinct root digits, >= 1 "
                             "distractor), or None for the most that fit up to 4")
        if C * links > len(pool):
            raise ValueError(f"{family}: {C} chains of {links} names (max k {hi} plus the "
                             f"unqueried last link) need {C * links}; only {len(pool)} "
                             f"single-letter names (n_chains <= {fit} fits, or lower max k)")
        if S["order"] not in ("interleaved", "shuffled"):
            raise ValueError("order must be 'interleaved' or 'shuffled'")
        dmax = hi                                 # examples use the same chain length

    if "shot_depths" in given:
        bad = [d for d in S["shot_depths"] if not isinstance(d, int) or not 1 <= d <= dmax]
        if bad or not S["shot_depths"]:
            raise ValueError(f"{family}: shot_depths must be ints in 1..{dmax}")
    else:                                         # defaults: keep those that fit
        fits = (lambda d: d <= dmax and any((d * i) % S["modulus"] for i in S["increments"])
                ) if family == "arith" else (lambda d: d <= dmax)
        S["shot_depths"] = [d for d in S["shot_depths"] if fits(d)] or [max(lo, 1)]
    if family == "arith":
        # an increment with k*d = 0 mod m returns to the start; none may remain
        m = S["modulus"]
        for k in list(range(lo, hi + 1)) + (S["shot_depths"] if S["shots"] else []):
            if not any((k * d) % m for d in S["increments"]):
                raise ValueError(f"arith: every increment gives k*d = 0 mod {m} at k={k}")
    if family == "pointer" and not isinstance(S["balance"], bool):
        raise ValueError("balance must be a bool")
    return S


def _plain(v):
    """A setting value in plain JSON types: numpy scalars and arrays (a grid
    built with np.arange) become the equal int/float/list, tuples lists, so
    the name, equality and save/load all see one value."""
    if v is None or isinstance(v, (bool, str, int, float)):
        return v
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "tolist"):                      # numpy scalar or array
        return _plain(v.tolist())
    return v


def _copy_start_depths(lo, hi, K):
    """Effective depths at which a K-step walk ends on its start: the cycle
    through the start has length e+1, and the walk wraps home when e+1 divides K."""
    return [e for e in range(lo, hi + 1) if e < K and K % (e + 1) == 0]


def _auto_written_k(lo, hi):
    """The single-digit numeral >= hi with the fewest copy-the-start cells
    (ties: the smallest).  For k 1..6 that is 7 (only e=6 wraps home); 6 would
    make e=1, 2 and 5 all answerable by copying the start."""
    return min(range(max(hi, 1), 10), key=lambda K: (len(_copy_start_depths(lo, hi, K)), K))


def _auto_name(family, wording, lo, hi, n_per, seed, S):
    """Readable main knobs plus a hash of the full setting, so every point of a
    settings grid gets its own file name."""
    knob = {"pointer": lambda: f"_st{S['n_states']}",
            "fixed_digit": lambda: f"_st{S['n_states']}K{S['written_k']}",
            "arith": lambda: "" if S["modulus"] == 10 else f"_m{S['modulus']}",
            "binding": lambda: f"_ch{S['n_chains']}",
            "relational": lambda: f"_ch{S['n_chains']}"}[family]()
    canon = json.dumps({"family": family, "wording": wording, "k_range": [lo, hi],
                        "n_per": n_per, "seed": seed, "generator": GENERATOR_VERSION, **S},
                       sort_keys=True)
    h = hashlib.sha256(canon.encode()).hexdigest()[:6]
    mode = "p" if S["shot_mode"] == "per_item" else "s"
    return (f"{family}_{wording}_k{lo}-{hi}{knob}_sh{S['shots']}{mode}"
            f"_n{n_per}_seed{seed}_{h}")


# ────────────────────────────── split ───────────────────────────────────────
def _stratified_pick(golds, sizes, n_pick, rng):
    """Indices of the units (groups of sizes[j] items, one gold each) that make
    up n_pick items, spread over gold digits as the singleton rule would: items
    taken round-robin across golds, larger golds first, give each gold a quota,
    so the calib and test parts both stay answer-balanced.  Units are indivisible;
    when their sizes cannot meet the quotas exactly, the total nearest n_pick
    and then the counts nearest the quotas win."""
    groups = {}
    for idx, g in enumerate(golds):
        groups.setdefault(g, []).append(idx)
    for v in groups.values():
        rng.shuffle(v)
    keys = sorted(groups)
    rng.shuffle(keys)
    tot = {g: sum(sizes[j] for j in groups[g]) for g in keys}
    keys.sort(key=lambda g: -tot[g])              # stable: ties stay in random order
    quota, left = dict.fromkeys(keys, 0), n_pick
    while left > 0 and any(quota[g] < tot[g] for g in keys):
        for g in keys:
            if left > 0 and quota[g] < tot[g]:
                quota[g] += 1
                left -= 1
    # per gold, one subset of its units for every reachable item count (the first
    # found in the shuffled order: for single-item units a random subset)
    reach = {}
    for g in keys:
        r = {0: []}
        for j in groups[g]:
            for s, sub in list(r.items()):
                r.setdefault(s + sizes[j], sub + [j])
        reach[g] = r
    best = {0: (0, [])}                           # total -> (deviation from quotas, choice)
    for g in keys:
        nxt = {}
        for t, (dev, ch) in best.items():
            for s in reach[g]:
                d = dev + abs(s - quota[g])
                if t + s not in nxt or d < nxt[t + s][0]:
                    nxt[t + s] = (d, ch + [(g, s)])
        best = nxt
    t = min(best, key=lambda t: (abs(t - n_pick), best[t][0], t))
    return {j for g, s in best[t][1] for j in reach[g][s]}


def _calib_members(members, key, gold, frac, rng):
    """The members for calib: round(frac * len(members)), answer-balanced, with
    members of one key (the same final problem) kept together on one side."""
    units = {}
    for m in members:
        units.setdefault(key(m), []).append(m)
    units = list(units.values())
    pick = _stratified_pick([gold(u[0]) for u in units], [len(u) for u in units],
                            round(frac * len(members)), rng)
    return {m for j in pick for m in units[j]}


def _assign_split(items, frac, seed, family, matched):
    """calib / test per item.  Identical problems go to one side whole: the same
    final problem in both would let calibration (E1) see test items (E2)."""
    rng = random.Random(f"split:{family}:{seed}")  # own stream: split never moves items
    last = lambda it: it["text"].split("\n\n")[-1]  # noqa: E731
    if matched:
        # a body is shared across k, so the whole body goes to one side
        first = {}
        for it in items:
            first.setdefault(it["i"], it)
        calib = _calib_members(sorted(first), lambda b: last(first[b]),
                               lambda b: first[b]["gold"], frac, rng)
        for it in items:
            it["split"] = "calib" if it["i"] in calib else "test"
        return
    for k in sorted({it["k"] for it in items}):
        at = [j for j, it in enumerate(items) if it["k"] == k]
        calib = _calib_members(at, lambda j: last(items[j]), lambda j: items[j]["gold"],
                               frac, rng)
        for j in at:
            items[j]["split"] = "calib" if j in calib else "test"


# ────────────────────────────── public API ──────────────────────────────────
class _DefaultInt(int):
    """An int default that make_suite can tell apart from the same value passed
    explicitly (the ladder takes no arguments; a grid loop must not slip past)."""


class _DefaultRange(tuple):
    """The tuple counterpart of _DefaultInt."""


_K_RANGE, _N_PER, _SEED = _DefaultRange((1, 6)), _DefaultInt(40), _DefaultInt(0)


def make_suite(family, wording="w0", k_range=_K_RANGE, n_per=_N_PER, seed=_SEED, **setting):
    """Build a suite dict (see the module docstring).  k_range is inclusive
    (default (1, 6)), n_per items per k (default 40), seed an int (default 0;
    anything else, None included, raises: a suite name promises reproducible
    items).  Settings not given take FAMILIES[family]['defaults']; unknown ones
    raise; `name=` overrides the automatic name.  family="ladder" returns
    ladder_suite() and raises if given any other argument, since the original
    ladder has no knobs."""
    if family == "ladder":
        given = [n for n, v, d in (("k_range", k_range, _K_RANGE), ("n_per", n_per, _N_PER),
                                   ("seed", seed, _SEED)) if v is not d] + sorted(setting)
        if wording != "w0" or given:
            raise ValueError(f"ladder: the original single-prompt ladder is fixed (w0, k 1..6, "
                             f"one item per k) and takes no {given or ['wording']}; call "
                             f"ladder_suite(), or use family 'pointer' for a many-item ladder")
        return ladder_suite()
    if family not in _GEN:
        raise ValueError(f"unknown family {family!r}; one of {sorted(_GEN)}")
    if wording not in WORDINGS[family]:
        raise ValueError(f"{family}: unknown wording {wording!r}; one of "
                         f"{sorted(WORDINGS[family])}")
    try:
        seed = operator.index(seed)               # ints, numpy ints; not None, floats, str
    except TypeError:
        raise ValueError(f"seed must be an int, got {seed!r}: a random seed would give "
                         f"different items under one name and one set of ids") from None
    name = setting.pop("name", None)
    try:                                          # (lo, hi); a list of depths is not one
        lo, hi = (operator.index(x) for x in k_range)
    except (TypeError, ValueError):
        raise ValueError(f"k_range must be an inclusive (lo, hi) pair of ints, got "
                         f"{k_range!r}") from None
    if int(n_per) < 1:
        raise ValueError("n_per must be >= 1")
    n_per = int(n_per)
    S = _settings(family, lo, hi, setting)
    name = name or _auto_name(family, wording, lo, hi, n_per, seed, S)
    if "/" in name:
        raise ValueError("suite name must not contain '/' (it is part of every item id)")

    # pointer keeps Random(seed) (serial_depth_fs is drawn from it); every other
    # family has its own stream, so no two families share instances
    rng = random.Random(seed if family == "pointer" else f"{family}:{seed}")
    raw = _GEN[family](WORDINGS[family][wording], list(range(lo, hi + 1)), n_per, rng, S)
    _assign_split(raw, S["calib_frac"], seed, family, FAMILIES[family]["matched"])
    items = [{"id": f"{name}/{r['k']}/{r['i']}", "family": family, "wording": wording,
              "k": r["k"], "label": f"D{r['k']}N{r['i']:02d}", "text": r["text"],
              "gold": r["gold"], "answer_set": r["answer_set"],
              "k_written": FAMILIES[family]["k_written"], "echo": r["echo"],
              "split": r["split"], "setting": r["setting"]} for r in raw]
    sizes = {len(it["answer_set"]) for it in items}
    suite = {"name": name, "family": family, "wording": wording,
             "setting": {"k_range": [lo, hi], "n_per": n_per, "seed": seed, **S},
             "chance": 1 / max(sizes), "items": items}
    verify_suite(suite)
    return suite


def ladder_suite():
    """The original single-prompt ladder, active_subgraph_dot.TASK_SUITES
    ['serial_depth'], as a suite (texts and answers unchanged)."""
    from active_subgraph_dot import TASK_SUITES, _PERM
    src = TASK_SUITES["serial_depth"]
    n = len(_PERM)
    items = []
    for text, label, gold in zip(src["tasks"], src["labels"], src["_answers"]):
        k = int(re.match(r"D(\d+)", label).group(1))
        _, info = solve_text("ladder", "w0", text)
        items.append({"id": f"ladder/{k}/0", "family": "ladder", "wording": "w0", "k": k,
                      "label": f"D{k}N00", "text": text, "gold": str(gold),
                      "answer_set": [str(d) for d in range(n)], "k_written": True,
                      "echo": str(k), "split": "test",
                      "setting": {"start": info["start"],
                                  "table": [_PERM[d] for d in range(n)],
                                  "min_hops": info["min_hops"], "orig_label": label}})
    ks = [it["k"] for it in items]
    suite = {"name": "ladder", "family": "ladder", "wording": "w0",
             "setting": {"source": "active_subgraph_dot.TASK_SUITES['serial_depth']",
                         "k_range": [min(ks), max(ks)], "n_per": 1, "n_states": n},
             "chance": 1 / n, "items": items}
    verify_suite(suite)
    return suite


def save_suite(suite, path):
    """Verify, then write atomically (a Colab disconnect must not leave half a file)."""
    verify_suite(suite)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(suite, indent=1))
    tmp.replace(path)
    return path


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    verify_suite(suite)
    return suite


# ────────────────────────────── verification ────────────────────────────────
_PH = {"a": r"\d", "b": r"\d", "s": r"\d", "k": r"\d", "d": r"\d", "m": r"\d",
       "x": r"(?<![A-Za-z0-9])[A-Za-z]", "v": r"(?<![A-Za-z0-9])[A-Za-z]",
       "y": r"(?<![A-Za-z0-9])[A-Za-z0-9]"}


def _tmpl_re(t):
    out, pos = "", 0
    for m in re.finditer(r"\{(\w+)\}", t):
        out += re.escape(t[pos:m.start()]) + f"(?P<{m.group(1)}>{_PH[m.group(1)]})"
        pos = m.end()
    return out + re.escape(t[pos:])


def solve_text(family, wording, text):
    """Answer the final query of `text` from the text alone.

    Independent of the generators: parses the last block with the WORDINGS
    templates, checks that re-rendering the parse gives the block back exactly,
    and computes the answer.  Returns (answer digit, info) where info has
    'depth' (the k the item should carry), 'answer_set' (the admissible answers
    the text defines: its states for pointer-like and arith, the digit literals
    of the final block for binding/relational), 'written' if the text writes a
    step count, and what can be read off the text (start, table, min_hops,
    increment, modulus, query).
    """
    W = WORDINGS[family][wording]
    block = text.split("\n\n")[-1]
    if family == "arith":
        for key, m in (("query", 10), ("query_mod", None)):
            q = re.fullmatch(_tmpl_re(W[key]), block)
            if q is not None:
                break
        else:
            raise ValueError(f"arith/{wording}: query not found in {block[-80:]!r}")
        m = m or int(q["m"])
        s, d, K = int(q["s"]), int(q["d"]), int(q["k"])
        if not (2 <= m <= 10 and 0 <= s < m and 1 <= d < m):
            raise ValueError(f"arith/{wording}: start {s} / increment {d} outside mod {m}: "
                             f"{block!r}")
        return str((s + K * d) % m), {"depth": K, "written": K, "start": s, "increment": d,
                                      "modulus": m,
                                      "answer_set": [str(x) for x in range(m)]}
    q = re.search((" " if "head" in W else "") + _tmpl_re(W["query"]) + "$", block)
    if q is None or not block.startswith(W.get("head", "")):
        raise ValueError(f"{family}/{wording}: query not found in {block[-80:]!r}")
    mid = block[len(W.get("head", "")):q.start()]
    if family in ("pointer", "fixed_digit", "ladder"):
        tbl = {int(m["a"]): int(m["b"]) for m in re.finditer(_tmpl_re(W["rule"]), mid)}
        s, K = int(q["s"]), int(q["k"])
        if (sorted(tbl) != list(range(len(tbl))) or sorted(tbl.values()) != sorted(tbl)
                or s not in tbl or _pointer_body(W, tbl, s, K) != block):
            raise ValueError(f"{family}/{wording}: rule table does not parse as a "
                             f"permutation: {block[:120]!r}")
        seen = [s]
        for _ in range(K):
            seen.append(tbl[seen[-1]])
        eff = len(set(seen)) - 1
        return str(seen[-1]), {"depth": eff if family == "fixed_digit" else K,
                               "written": K, "effective": eff, "start": s,
                               "table": [tbl[x] for x in range(len(tbl))],
                               "min_hops": _min_hops(tbl, s, seen[-1]),
                               "answer_set": [str(x) for x in range(len(tbl))]}
    stmts = [(m["x"], m["y"]) for m in re.finditer(_tmpl_re(W["stmt"]), mid)]
    mp = dict(stmts)
    if len(mp) != len(stmts) or _chain_body_text(W, stmts, q["v"]) != block:
        raise ValueError(f"{family}/{wording}: statements do not parse: {block[:120]!r}")
    v, hops = q["v"], 0
    while not v.isdigit():
        if v not in mp or hops > len(mp):
            raise ValueError(f"{family}/{wording}: {q['v']} has no root")
        v, hops = mp[v], hops + 1
    xs = [x for x, _ in stmts]
    return v, {"depth": hops, "query": q["v"],
               "def_distance": len(xs) - xs.index(q["v"]),
               "query_refs": sum(y == q["v"] for _, y in stmts),
               "answer_set": sorted({y for _, y in stmts if y.isdigit()})}


def _is_digit_str(x):
    return isinstance(x, str) and len(x) == 1 and x in DIGITS


def verify_suite(suite):
    """Schema and answer checks; raises ValueError on the first failing suite
    rather than let a wrong gold, answer set, echo or chance reach a
    measurement.  Everything an item claims about its prompt is recomputed from
    the text by solve_text: gold, depth, answer set (competitors downstream),
    whether and which numeral is written (echo, k_written), start / table /
    min_hops / increment / query, and every worked example; chance must be
    1/|answer set|; the suite must have items."""
    probs = []
    miss = [k for k in SUITE_KEYS if k not in suite]
    if miss:
        raise ValueError(f"suite lacks {miss}")
    fam, wd = suite["family"], suite["wording"]
    if fam not in WORDINGS or wd not in WORDINGS[fam]:
        raise ValueError(f"unknown family/wording {fam}/{wd}")
    if not isinstance(suite["items"], list) or not suite["items"]:
        raise ValueError(f"suite {suite['name']} has no items")
    spec = FAMILIES[fam]
    ss = suite["setting"] if isinstance(suite["setting"], dict) else {}
    ids, sizes = set(), []
    for it in suite["items"]:
        miss = [k for k in ITEM_KEYS if k not in it]
        if miss:
            probs.append(f"{it.get('id')}: lacks {miss}"); continue
        tag = it["id"]
        if it["id"] in ids:
            probs.append(f"{tag}: duplicate id")
        ids.add(it["id"])
        m_id = re.fullmatch(re.escape(suite["name"]) + rf"/{it['k']}/(\d+)", str(it["id"]))
        if not m_id:
            probs.append(f"{tag}: id is not <name>/<k>/<i>")
        elif it["label"] != f"D{it['k']}N{int(m_id.group(1)):02d}":
            probs.append(f"{tag}: label {it['label']!r} does not carry the id's k and i")
        if it["family"] != fam or it["wording"] != wd:
            probs.append(f"{tag}: family/wording differ from the suite's")
        if it["split"] not in ("calib", "test"):
            probs.append(f"{tag}: split {it['split']!r}")
        A, g = it["answer_set"], it["gold"]
        if (not isinstance(A, list) or not A or not all(map(_is_digit_str, A))
                or len(set(A)) != len(A)):
            probs.append(f"{tag}: answer_set {A!r} is not a list of distinct digits")
        else:
            sizes.append(len(A))
        if not _is_digit_str(g) or not isinstance(A, list) or g not in A:
            probs.append(f"{tag}: gold {g!r} is not a digit in the answer set")
        if it["k_written"] is not spec["k_written"]:
            probs.append(f"{tag}: k_written {it['k_written']!r}, but {fam} "
                         f"{'writes' if spec['k_written'] else 'never writes'} its step count")
        if it["k_written"] != (it["echo"] is not None):
            probs.append(f"{tag}: echo {it['echo']!r} inconsistent with k_written")
        if it["echo"] is not None and it["echo"] == g:
            probs.append(f"{tag}: gold equals the echoed numeral {g}")
        if not isinstance(it["setting"], dict):
            probs.append(f"{tag}: setting is not a dict")
        st = it["setting"] if isinstance(it["setting"], dict) else {}
        try:
            ans, info = solve_text(fam, wd, it["text"])
            if ans != g:
                probs.append(f"{tag}: text solves to {ans}, gold says {g}")
            if info["depth"] != it["k"]:
                probs.append(f"{tag}: text has depth {info['depth']}, item says k={it['k']}")
            if ("written" in info) != bool(it["k_written"]):
                probs.append(f"{tag}: the text {'writes' if 'written' in info else 'writes no'} "
                             f"step count, item says k_written={it['k_written']!r}")
            if "written" in info and str(info["written"]) != it["echo"]:
                probs.append(f"{tag}: written numeral {info['written']} != echo {it['echo']!r}")
            if A != info["answer_set"]:
                probs.append(f"{tag}: answer_set {A!r}, but the text admits "
                             f"{info['answer_set']}")
            for key in ("start", "table", "min_hops", "increment", "query", "def_distance",
                        "query_refs"):
                if key in st and key in info and st[key] != info[key]:
                    probs.append(f"{tag}: setting.{key} {st[key]!r}, the text gives "
                                 f"{info[key]!r}")
            # the family's form, without which k is not a serial depth
            if fam in ("pointer", "ladder") and len(_cycles(dict(enumerate(info["table"])))) != 1:
                probs.append(f"{tag}: the rule table is not one {len(info['table'])}-cycle")
            if fam == "fixed_digit":
                moved = [c for c in _cycles(dict(enumerate(info["table"]))) if len(c) > 1]
                if len(moved) > 1 or (moved and info["start"] not in moved[0]):
                    probs.append(f"{tag}: the rule table is not one cycle through the start "
                                 f"plus fixed points")
            if fam == "arith" and (info["depth"] * info["increment"]) % info["modulus"] == 0:
                probs.append(f"{tag}: k*d = 0 mod {info['modulus']}: the walk returns to its "
                             f"start, so k is not a serial depth")
            if "written_k" in st and st["written_k"] != info.get("written"):
                probs.append(f"{tag}: setting.written_k {st['written_k']!r} != written "
                             f"{info.get('written')!r}")
            if "gold_eq_start" in st and st["gold_eq_start"] != (g == str(info.get("start"))):
                probs.append(f"{tag}: setting.gold_eq_start {st['gold_eq_start']!r} is wrong")
            if "roots" in st and sorted(str(r) for r in st["roots"]) != info["answer_set"]:
                probs.append(f"{tag}: setting.roots {st['roots']!r} are not the text's roots")
            if fam == "arith" and ss.get("modulus", 10) != info["modulus"]:
                probs.append(f"{tag}: text is mod {info['modulus']}, suite says "
                             f"{ss.get('modulus', 10)}")
            n_key = "n_chains" if fam in ("binding", "relational") else "n_states"
            if n_key in ss and fam != "arith" and len(info["answer_set"]) != ss[n_key]:
                probs.append(f"{tag}: {len(info['answer_set'])} admissible answers, suite "
                             f"{n_key} = {ss[n_key]}")
            # every worked example must be solved correctly as well
            efam = "pointer" if fam in ("fixed_digit", "ladder") else fam
            for b in it["text"].split("\n\n")[:-1]:
                m = re.search(r" (\d)\.$", b)
                if m is None or solve_text(efam, wd, b[:m.start()])[0] != m.group(1):
                    probs.append(f"{tag}: a worked example is wrong: {b[-60:]!r}")
        except ValueError as e:
            probs.append(f"{tag}: {e}")
        if len(probs) > 20:
            break
    c = suite["chance"]
    if sizes and (isinstance(c, bool) or not isinstance(c, (int, float))
                  or abs(c - 1 / max(sizes)) > 1e-9):
        probs.append(f"chance {c!r} != 1/{max(sizes)} (the largest answer set)")
    if probs:
        raise ValueError(f"suite {suite['name']} failed verification "
                         f"({len(probs)} problem(s)):\n  " + "\n  ".join(probs[:20]))
    return True


# ────────────────────────────── check_suite ─────────────────────────────────
def _named_tokenizers(tokenizers):
    """[(name, tokenizer)] from a dict, one tokenizer, a list of them, or
    tokenizer names (a name stays a str here; check_suite loads it as the CLI
    does)."""
    if tokenizers is None:
        return []
    if isinstance(tokenizers, str) or hasattr(tokenizers, "convert_ids_to_tokens"):
        tokenizers = [tokenizers]
    if isinstance(tokenizers, dict):
        return list(tokenizers.items())
    return [(t if isinstance(t, str) else getattr(t, "name_or_path", None) or f"tokenizer{j}", t)
            for j, t in enumerate(tokenizers)]


def _example_answers(text):
    """Worked-example answers of a prompt, in order (from the text)."""
    out = []
    for b in text.split("\n\n")[:-1]:
        m = re.search(r" (\d)\.$", b)
        if m:
            out.append(m.group(1))
    return out


def _query_among_examples(text):
    """Is the final problem one of the prompt's own worked examples (answer shown)?"""
    blocks = text.split("\n\n")
    return blocks[-1] in {re.sub(r" \d\.$", "", b) for b in blocks[:-1]}


def _corr(x, y):
    """Pearson correlation; 0 when either side is constant."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def check_suite(suite, tokenizers, min_correct=20):
    """Report (never raise) everything a run needs to trust the suite.

    Tokenizer-independent: verification (each failing item and why), per-k
    answer balance over the admissible digits (answer sets minus the echo;
    balanced = counts differ by at most one), fraction of gold == echo (must be
    0), fraction of items with a digit other than gold in the prompt (must be
    1), split counts per k, and how often gold equals the start state where
    there is one.  Per k, distinct final problems (rep["distinct_problems"]),
    identical prompts (rep["duplicate_prompts"]; must be 0: one measurement
    would count as several items) and items whose final problem is one of their
    own worked examples (rep["example_is_query"]; must be 0: copying solves
    them).
    Worked examples (rep["examples"][k]): per example position the answer
    counts, copy_expected (accuracy of always copying that example if it were
    independent of gold: sum_d P(gold=d) P(answer=d)) and copy_realised, plus
    echo_frac (items whose examples answer the item's written numeral; must be
    0 for shot_mode="per_item").
    Per tokenizer (rep["tokenizers"][name]; tokenizer objects, a dict of them,
    or names to load): token counts per k and whether one count holds for the
    whole suite (constant, n_tokens); the same for the fed length after
    deep_probe.align_answer (feed_counts, feed_constant); whether align_answer
    succeeds on every item (align_ok, align_failures); the appended token forms;
    for matched families, whether prompts of one body have one length and differ
    in at most one position across k (the queried entity); plus a copy of the
    tokenizer-independent checks above.
    `ok` is True only when every check passes; `problems` lists what failed.
    `warnings` lists design limits the analysis must handle, without failing:
    shortest lookup path below k (rep["min_hops"]), arith offsets that k fixes
    (rep["arith_offsets"]), repeated final problems (more items per depth than
    the settings have problems), a queried definition whose distance to the
    query trends with k (rep["def_distance"], binding/relational) or a queried
    entity whose use as a value varies with k (rep["query_refs"]), copy accuracy
    that varies with k (depths with >= 20 items only, below that it is noise), a
    calib split with fewer than `min_correct` items per depth (then no depth can
    pass eligible_depths), and prompts longer than a tokenizer's
    model_max_length (e.g. GPT-2's 1024 positions).
    """
    rep = {"name": None, "family": None, "wording": None, "n_items": 0, "n_per_k": {},
           "verified": False, "tokenizers": {}, "problems": [], "warnings": []}
    try:
        _check_suite(suite, tokenizers, min_correct, rep)
    except Exception as e:                        # a malformed suite: report it, never raise
        rep["problems"].append(f"check_suite could not finish on this suite: "
                               f"{type(e).__name__}: {e}")
    rep["ok"] = not rep["problems"]
    return rep


def _check_suite(suite, tokenizers, min_correct, rep):
    from deep_probe import align_answer
    P, Wn = rep["problems"], rep["warnings"]
    rep.update(name=suite.get("name"), family=suite.get("family"),
               wording=suite.get("wording"))
    items = suite.get("items") or []
    rep["n_items"] = len(items)
    if not items:                                 # e.g. a split filter left nothing
        P.append("suite has no items")
        return
    try:
        verify_suite(suite)
        rep["verified"] = True
    except ValueError as e:                       # keep which item failed, and why
        lines = str(e).splitlines()
        P.append(lines[0])
        P.extend(ln.strip() for ln in lines[1:])
    ks = sorted({it["k"] for it in items})
    by_k = {k: [it for it in items if it["k"] == k] for k in ks}
    rep["n_per_k"] = {k: len(v) for k, v in by_k.items()}

    bal = {}
    for k, its in by_k.items():
        echoes = {it["echo"] for it in its}
        adm = sorted(set().union(*(it["answer_set"] for it in its))
                     - (echoes if len(echoes) == 1 else set()))
        cnt = {d: sum(it["gold"] == d for it in its) for d in adm}
        spread = max(cnt.values()) - min(cnt.values()) if cnt else 0
        bal[k] = {"counts": cnt, "spread": spread, "balanced": spread <= 1}
    rep["answer_balance"] = bal
    rep["balanced"] = all(b["balanced"] for b in bal.values())
    if not rep["balanced"]:
        P.append("answer imbalance at k=" + ",".join(str(k) for k, b in bal.items()
                                                     if not b["balanced"]))
    rep["gold_eq_echo"] = sum(it["echo"] == it["gold"] for it in items) / len(items)
    if rep["gold_eq_echo"]:
        P.append(f"gold equals echo on {rep['gold_eq_echo']:.1%} of items")
    rep["distractor_frac"] = sum(bool(set(re.findall(r"\d", it["text"])) - {it["gold"]})
                                 for it in items) / len(items)
    if rep["distractor_frac"] < 1:
        P.append(f"gold is the only digit in {1 - rep['distractor_frac']:.1%} of prompts")
    rep["split"] = {k: {s: sum(it["split"] == s for it in its) for s in ("calib", "test")}
                    for k, its in by_k.items()}
    rep["gold_eq_start"] = {k: sum(str(it["setting"].get("start")) == it["gold"]
                                   for it in its) / len(its)
                            for k, its in by_k.items() if "start" in its[0]["setting"]}
    ss = suite.get("setting") or {}

    # repeated problems: one final problem is one problem, one prompt one measurement
    rep["distinct_problems"] = {k: len({it["text"].split("\n\n")[-1] for it in its})
                                for k, its in by_k.items()}
    rep["duplicate_prompts"] = {k: len(its) - len({it["text"] for it in its})
                                for k, its in by_k.items()}
    dup = [f"k={k}: {n}" for k, n in rep["duplicate_prompts"].items() if n]
    if dup:
        P.append(f"identical prompts ({', '.join(dup)} duplicate(s)): one measurement would "
                 f"count as several items; lower n_per to at most "
                 f"{min(rep['distinct_problems'].values())} or add worked examples")
    rpt = [f"k={k}: {n}/{len(by_k[k])}" for k, n in rep["distinct_problems"].items()
           if n < len(by_k[k])]
    if rpt:
        Wn.append(f"final problems repeat (distinct/items {', '.join(rpt)}): the settings hold "
                  f"fewer problems per depth than items, so repeats are not independent items "
                  f"(they stay on one side of the split)")
    # a worked example that is the item's own query hands a copying model the answer
    rep["example_is_query"] = {k: sum(_query_among_examples(it["text"]) for it in its)
                               for k, its in by_k.items()}
    selfq = [f"k={k}: {n}" for k, n in rep["example_is_query"].items() if n]
    if selfq:
        P.append(f"a worked example is the item's own query, answer shown ({', '.join(selfq)} "
                 f"item(s)): copying it solves the item")

    # worked examples: per position balance and what copying one would score
    ex, spread = {}, collections.defaultdict(list)
    for k, its in by_k.items():
        rows = [_example_answers(it["text"]) for it in its]
        slots = []
        for j in range(max(map(len, rows))):
            pairs = [(r[j], it["gold"]) for it, r in zip(its, rows) if len(r) > j]
            n = len(pairs)
            cnt = collections.Counter(a for a, _ in pairs)
            gcnt = collections.Counter(g for _, g in pairs)
            exp = sum(cnt[d] * gcnt[d] for d in cnt) / n / n
            slots.append({"counts": dict(sorted(cnt.items())), "copy_expected": round(exp, 4),
                          "copy_realised": round(sum(a == g for a, g in pairs) / n, 4)})
            if n >= 20:
                spread[j].append(exp)
        echo_frac = sum(it["echo"] is not None and it["echo"] in r
                        for it, r in zip(its, rows)) / len(its)
        if slots:
            ex[k] = {"slots": slots, "echo_frac": echo_frac}
    rep["examples"] = ex
    bad = [k for k, e in ex.items() if e["echo_frac"]]
    if bad and ss.get("shot_mode") == "per_item":
        P.append("worked examples answer the item's written numeral at k="
                 + ",".join(map(str, bad)))
    for j, v in sorted(spread.items()):
        if len(v) == len(by_k) > 1 and max(v) - min(v) > 0.03:
            Wn.append(f"copying worked example {j} would score {min(v):.3f}-{max(v):.3f} "
                      f"across k, so a copy bias would look like a depth effect")

    # shortest lookup path: k steps on an n-cycle = n-k steps backward
    if all("min_hops" in it["setting"] for it in items):
        rep["min_hops"] = {k: dict(sorted(collections.Counter(
            it["setting"]["min_hops"] for it in its).items())) for k, its in by_k.items()}
        short = [k for k, its in by_k.items() if any(it["setting"]["min_hops"] < k for it in its)]
        if short:
            Wn.append("the shortest lookup path (forward or backward) is below k at k="
                      + ",".join(map(str, short)) + "; use setting.min_hops as a covariate "
                      "or restrict k")

    # matched families: where the queried entity sits and how it is used, read
    # from the text (suites written before these were recorded are checked too)
    if suite.get("family") in ("binding", "relational"):
        try:
            info = {it["id"]: solve_text(suite["family"], suite["wording"], it["text"])[1]
                    for it in items}
        except ValueError:                        # verification has already failed
            info = None
        if info:
            dd = {k: [info[it["id"]]["def_distance"] for it in its] for k, its in by_k.items()}
            rep["def_distance"] = {k: round(sum(v) / len(v), 2) for k, v in dd.items()}
            r = _corr([it["k"] for it in items], [info[it["id"]]["def_distance"] for it in items])
            rep["def_distance_corr"] = round(r, 3)
            if abs(r) >= 0.2:
                # 'shuffled' balances the queried link's rank only per block of
                # max(k)+1 bodies and the other chains' slots stay random, so at
                # small n_per a trend is sampling noise; recommending 'shuffled'
                # there would send the user to the setting already in use
                fix = ("use setting.def_distance as a covariate, or raise n_per (the "
                       "shuffled order balances the queried link's rank per block of "
                       "max(k)+1 bodies; the rest is sampling noise)"
                       if ss.get("order", "shuffled") == "shuffled" else
                       "use order='shuffled' or setting.def_distance as a covariate")
                Wn.append(f"the queried definition's distance to the query (mean statements "
                          f"per k {rep['def_distance']}) trends with k (r = {r:.2f}): recency "
                          f"is confounded with depth; {fix}")
            rep["query_refs"] = {k: dict(sorted(collections.Counter(
                info[it["id"]]["query_refs"] for it in its).items())) for k, its in by_k.items()}
            if len({frozenset(v) for v in rep["query_refs"].values()}) > 1:
                Wn.append(f"how often the queried entity is used as a value varies with k "
                          f"({rep['query_refs']}): at some depth it is a chain's leaf, a "
                          f"surface property confounded with depth")

    # arith: which offsets (gold - start) a depth allows
    if suite.get("family") == "arith":
        m = ss.get("modulus", 10)
        incs = ss.get("increments")
        rep["arith_offsets"] = {
            k: (sorted({(k * d) % m for d in incs} - {0}) if incs else
                sorted({(int(it["gold"]) - it["setting"]["start"]) % m for it in its}))
            for k, its in by_k.items()}
        sizes = {k: len(o) for k, o in rep["arith_offsets"].items()}
        if len(set(sizes.values())) > 1 or 1 in sizes.values():
            Wn.append(f"arith mod {m}: the number of possible offsets start->gold varies "
                      f"with k ({sizes}); where it is 1 the answer needs no composition "
                      f"(a prime modulus above max k, e.g. modulus=7, avoids this)")

    # calibration size against the correct-item rule
    calib = {k: v["calib"] for k, v in rep["split"].items()}
    if any(calib.values()) and min(calib.values()) < min_correct:
        frac = ss.get("calib_frac")
        need = f"; calibration suites need n_per >= {math.ceil(min_correct / frac)}" if frac else ""
        Wn.append(f"the calib split has {min(calib.values())} items at some depth, fewer than "
                  f"the {min_correct} correct items eligible_depths needs{need}")

    matched = FAMILIES.get(suite["family"], {}).get("matched", False)
    for tname, tok in _named_tokenizers(tokenizers):
        if isinstance(tok, str):                  # a name, as the CLI takes them
            try:
                tok = _load_tokenizers([tok])[tok]
            except Exception as e:
                P.append(f"{tname}: tokenizer could not be loaded ({type(e).__name__}: {e})")
                continue
        r = {"token_counts": {}, "feed_counts": {}, "align_failures": [], "appended": []}
        ids_of = {}
        for k, its in by_k.items():
            n_tok, n_feed = set(), set()
            for it in its:
                ids = tok(it["text"])["input_ids"]
                ids_of[it["id"]] = ids
                n_tok.add(len(ids))
                try:
                    feed, _, app = align_answer(tok, it["text"])
                    n_feed.add(len(feed))
                    if app not in r["appended"]:
                        r["appended"].append(app)
                except SystemExit as e:           # align_answer aborts via SystemExit
                    r["align_failures"].append([it["id"], str(e)])
            r["token_counts"][k] = sorted(n_tok)
            r["feed_counts"][k] = sorted(n_feed)
        alln = set().union(*map(set, r["token_counts"].values()))
        allf = set().union(*map(set, r["feed_counts"].values()))
        r["constant"] = len(alln) == 1
        r["n_tokens"] = min(alln) if r["constant"] else None
        r["feed_constant"] = len(allf) == 1
        r["align_ok"] = not r["align_failures"]
        if not r["constant"]:
            P.append(f"{tname}: token count varies {sorted(alln)}")
        if not r["feed_constant"] and r["align_ok"]:
            P.append(f"{tname}: measured position varies {sorted(allf)}")
        if not r["align_ok"]:
            P.append(f"{tname}: align_answer fails on {len(r['align_failures'])} item(s)")
        # HF sets model_max_length to a huge sentinel when a tokenizer has no limit
        mml = getattr(tok, "model_max_length", None)
        r["model_max_length"] = mml if isinstance(mml, int) and mml < 10 ** 7 else None
        if r["model_max_length"] and max(alln) > r["model_max_length"]:
            Wn.append(f"{tname}: prompts reach {max(alln)} tokens, above the tokenizer's "
                      f"model_max_length {r['model_max_length']}: a model with that context "
                      f"cannot run this suite (use fewer worked examples)")
        if matched:
            worst, same_len, bodies = 0, True, {}
            for it in items:
                bodies.setdefault(it["setting"].get("body"), []).append(ids_of[it["id"]])
            for seqs in bodies.values():
                for s in seqs[1:]:
                    if len(s) != len(seqs[0]):
                        same_len = False
                    else:
                        worst = max(worst, sum(a != b for a, b in zip(s, seqs[0])))
            r["matched_same_length"] = same_len
            r["matched_max_diff"] = worst
            r["matched_ok"] = same_len and worst <= 1
            if not r["matched_ok"]:
                P.append(f"{tname}: prompts of one body differ across k beyond the queried "
                         f"entity (same length {same_len}, up to {worst} positions)")
        # the tokenizer-independent checks, repeated so each entry stands alone
        for key in ("answer_balance", "balanced", "gold_eq_echo", "distractor_frac"):
            r[key] = rep[key]
        rep["tokenizers"][tname] = r


def _summary(rep):
    lines = [f"{rep['name']}: {rep['n_items']} items, k={sorted(rep['n_per_k'])}, "
             f"{'OK' if rep['ok'] else 'FAILED'}"]
    for t, r in rep["tokenizers"].items():
        lines.append(f"    {t:<24} tokens {r['n_tokens'] if r['constant'] else r['token_counts']}"
                     f"   align {'ok' if r['align_ok'] else 'FAILS'}   appended {r['appended']}"
                     + (f"   matched diff {r['matched_max_diff']}" if "matched_ok" in r else ""))
    for p in rep["problems"]:
        lines.append(f"    ! {p}")
    for w in rep.get("warnings", []):
        lines.append(f"    ~ {w}")
    return "\n".join(lines)


# ────────────────────────────── CLI ─────────────────────────────────────────
def _parse_set(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set expects KEY=VALUE, got {p!r}")
        key, val = p.split("=", 1)
        try:
            out[key] = json.loads(val)
        except json.JSONDecodeError:
            out[key] = val
    return out


def _load_tokenizers(names):
    import os
    from transformers import AutoTokenizer
    tok = os.environ.get("HF_TOKEN") or None
    return {n: AutoTokenizer.from_pretrained(n, token=tok) for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", nargs="+", choices=sorted(_GEN))
    ap.add_argument("--wording", nargs="+", default=["w0"])
    ap.add_argument("--k", nargs=2, type=int, default=[1, 6], metavar=("LO", "HI"))
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="family setting, JSON value (e.g. shots=5, shot_mode=shared)")
    ap.add_argument("--ladder", action="store_true", help="write the original ladder suite")
    ap.add_argument("--out", default="suites")
    ap.add_argument("--check", nargs="+", metavar="JSON", help="check existing suite files")
    ap.add_argument("--tokenizer", nargs="+", default=[],
                    help="run check_suite with these tokenizers")
    ap.add_argument("--show", type=int, default=0, help="print the first N item texts")
    a = ap.parse_args()
    if not (a.family or a.ladder or a.check):
        ap.error("--family, --ladder or --check required")
    fams, words = list(dict.fromkeys(a.family or [])), list(dict.fromkeys(a.wording))
    sets = _parse_set(a.set)
    if "name" in sets and len(fams) * len(words) > 1:
        # one name for several suites would write them all to one file (and one set of ids)
        ap.error(f"--set name=... names one suite, but {len(fams) * len(words)} "
                 f"(family, wording) pairs were requested")
    toks = _load_tokenizers(a.tokenizer) if a.tokenizer else {}
    suites, written = [], {}

    def _write(s, path):
        key = str(path.resolve())
        if key in written:
            raise SystemExit(f"{path}: already written in this run by suite {written[key]!r}; "
                             f"refusing to overwrite it with {s['name']!r}")
        written[key] = s["name"]
        print(f"  -> {save_suite(s, path)}")

    if a.check:
        suites += [load_suite(p) for p in a.check]
    for fam in fams:
        for wd in words:
            s = make_suite(fam, wd, tuple(a.k), a.n_per, a.seed, **sets)
            _write(s, Path(a.out) / (s["name"] + ".json"))
            suites.append(s)
    if a.ladder:
        s = ladder_suite()
        _write(s, Path(a.out) / "ladder.json")
        suites.append(s)
    failed = 0
    for s in suites:
        for it in s["items"][:a.show]:
            print(f"\n[{it['id']}] gold {it['gold']}  echo {it['echo']}\n{it['text']}")
        if toks:
            rep = check_suite(s, toks)
            print(_summary(rep))
            failed += not rep["ok"]
    sys.exit(1 if failed else 0)

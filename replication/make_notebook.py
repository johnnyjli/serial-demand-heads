#!/usr/bin/env python3
"""Write E2b_accuracy_first.ipynb next to this file, embedding e2b.py byte for byte.

    python3 colab_e2b/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.strip("\n").splitlines(True)}

cells = [
md('''
# E2b · Serial-depth tasks, accuracy first

**What changed from E2.** E2's models were near chance past k = 2, so its activation signatures
mostly described failing models. E2b tries two ways to raise accuracy on the **same items**:

| mode | prompt | where activity is measured |
|---|---|---|
| **direct** | 8 worked examples covering every depth k (each on its own table, never showing the query's answer); the query ends where the answer goes | the token whose next token is the answer digit (E2's measurement) |
| **stepwise** | the same worked examples, each writing out its intermediate states, e.g. `Path: 4 -> 9 -> 1 -> 7. You end at 7.` The model generates its own trace, then its answer | the token whose next token is the model's **final** answer digit, with its own trace in context |

In direct mode the model also generates a short continuation, so an answer given later is recorded.

**Tasks** (one wording each, 30 items per depth): pointer chase on a 10-cycle (k = 1–5),
addition mod 7 (k = 1–6), variable binding and relational lookup (3 chains of 6, k = 1–6).
Answers are balanced per depth, never equal the written k, and every item is re-solved from
its text by an independent parser before use.

**Activity:** per head, ‖z_h W_O^h‖ at the measured position (E2's definition), checked
against the attention output projection on every item.

**Runtime:** A100. The default model list takes about 6 hours on an A100 80 GB. On an A100
40 GB, Mistral-Small-24B is skipped automatically (it needs 80 GB) and the rest takes about 4.5 hours.
Everything is saved to Drive per model, mode and task, so after a disconnect re-run from the
top; finished files are skipped. A time budget (`TIME_BUDGET_H`) stops starting new work before
the session ends.
'''),
md("## 1 · Setup"),
code('''
!nvidia-smi -L
!pip install -q -U transformers accelerate
import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)
'''),
code("%%writefile e2b.py\n" + (HERE / "e2b.py").read_text()),
code('''
from google.colab import drive
drive.mount("/content/drive")
import os
OUT = "/content/drive/MyDrive/e2b"
os.makedirs(OUT, exist_ok=True); print("outputs ->", OUT)
'''),
md("Hugging Face token (Llama needs its licence accepted on huggingface.co). It stays in memory only."),
code('''
from getpass import getpass
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Hugging Face token: ").strip()
'''),
md("## 2 · Configuration"),
code('''
# Models in priority order: the earliest ones finish first if the budget runs out.
# (name, parameters in billions)
MODELS = [
    ("Qwen/Qwen2.5-7B", 7.6),
    ("meta-llama/Llama-3.1-8B", 8.0),
    ("Qwen/Qwen2.5-7B-Instruct", 7.6),
    ("meta-llama/Llama-3.1-8B-Instruct", 8.0),
    ("allenai/OLMo-2-1124-7B", 7.3),
    ("allenai/OLMo-2-1124-7B-Instruct", 7.3),
    ("Qwen/Qwen2.5-14B", 14.8),
    ("mistralai/Mistral-Small-24B-Base-2501", 23.6),   # needs an 80 GB GPU; skipped otherwise
]
N_PER, SHOTS, SEED = 30, 8, 0          # 30 items per depth; 8 worked examples per prompt
MODES = ("direct", "stepwise")
TIME_BUDGET_H = 7.5                     # stop starting new work after this many hours
'''),
md("## 3 · Items\nBuilt and verified here (fast); saved to Drive for reference. One example of each family and mode is printed."),
code('''
import json, e2b
ITEMS = e2b.make_all(n_per=N_PER, shots=SHOTS, seed=SEED)
json.dump({f: its for f, its in ITEMS.items()}, open(f"{OUT}/items.json", "w"))
print({f: len(v) for f, v in ITEMS.items()}, "items; all verified")
for fam, its in ITEMS.items():
    it = next(x for x in its if x["k"] == 3)
    print(f"\\n==== {fam}, k = 3, answer {it['gold']} ====")
    print("[stepwise, first worked example]\\n" + it["text_stepwise"].split("\\n\\n")[0])
    print("[stepwise, query]\\n" + it["text_stepwise"].split("\\n\\n")[-1])
    print("[direct, query]\\n" + it["text_direct"].split("\\n\\n")[-1])
'''),
md('''
## 4 · Smoke test (≈ 3 minutes)
Qwen2.5-0.5B on 2 items per depth, both modes, into a separate folder. *Check:* no errors, the
stepwise generations look like `4 -> 9 -> 1 -> 7. You end at 7.`, and the split error is small
(< 0.05).
'''),
code('''
import gc, time
small = e2b.make_all(n_per=2, shots=SHOTS, seed=99)
m, tok = e2b.load("Qwen/Qwen2.5-0.5B", token=os.environ.get("HF_TOKEN"))
e2b.run(m, tok, "Qwen/Qwen2.5-0.5B", small, f"{OUT}/_smoke")
for r in e2b.load_results(f"{OUT}/_smoke"):
    if r["mode"] == "stepwise":
        x = r["items"][2]; print(f"{r['family']:<10} k={x['k']} gold={x['gold']}  gen: {x['gen'][:90]!r}")
del m; gc.collect(); torch.cuda.empty_cache()
'''),
md('''
## 5 · Main run
One model at a time: load, run both modes on all four tasks, save, free the GPU, delete the
downloaded weights. A model too large for this GPU is skipped with a message.
'''),
code('''
import shutil
T0 = time.time(); DEADLINE = T0 + TIME_BUDGET_H * 3600
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
def done(name):
    return all(e2b.out_path(OUT, name, mo, f).exists() for mo in MODES for f in ITEMS)
for name, size_b in MODELS:
    if done(name):
        print(f"== {name}: done already"); continue
    if size_b * 2 + 8 > GPU_GB:
        print(f"== {name}: needs ~{size_b*2+8:.0f} GB, this GPU has {GPU_GB:.0f} GB; skipped"); continue
    if time.time() > DEADLINE:
        print(f"== {name}: time budget reached; not started"); continue
    print(f"== {name}  ({(time.time()-T0)/3600:.2f} h elapsed)")
    t = time.time()
    model, tok = e2b.load(name, token=os.environ.get("HF_TOKEN"))
    print(f"   loaded in {time.time()-t:.0f}s")
    e2b.run(model, tok, name, ITEMS, OUT, modes=MODES, deadline=DEADLINE)
    del model; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}"), ignore_errors=True)
    print(f"   finished in {(time.time()-t)/60:.0f} min")
print(f"total {(time.time()-T0)/3600:.2f} h")
'''),
md('''
## 6 · Results
### Accuracy
Lines: direct (blue) and stepwise (orange) accuracy by depth, grey = chance. The table gives
mean accuracy per task, and for stepwise how often the model's own trace was exactly right.
'''),
code('''
import numpy as np, pandas as pd, matplotlib.pyplot as plt
RUNS = [r for r in e2b.load_results(OUT)]
ACC = e2b.accuracy_table(RUNS)
models = [m for m, _ in MODELS if any(r["model"] == m for r in RUNS)]
FAMS = list(ITEMS)
fig, axes = plt.subplots(len(models), len(FAMS), figsize=(3.2 * len(FAMS), 2.3 * len(models)), sharey=True, squeeze=False)
for i, m in enumerate(models):
    for j, f in enumerate(FAMS):
        ax = axes[i, j]
        ch = ITEMS[f][0]["chance"]; ax.axhline(ch, color="#898781", lw=0.8)
        for mode, col in (("direct", "#2a78d6"), ("stepwise", "#eb6834")):
            v = ACC.get((m, mode, f))
            if v is not None: ax.plot(range(1, len(v) + 1), v, marker="o", color=col, label=mode)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
        if i == 0: ax.set_title(f)
        if j == 0: ax.set_ylabel(m.split("/")[-1], fontsize=8)
        if i == len(models) - 1: ax.set_xlabel("serial depth k")
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/accuracy.png", dpi=150); plt.show()
rows = []
for r in RUNS:
    its = r["items"]
    row = {"model": r["model"].split("/")[-1], "mode": r["mode"], "task": r["family"],
           "accuracy": np.mean([x["correct"] for x in its]),
           "accuracy k>=3": np.mean([x["correct"] for x in its if x["k"] >= 3])}
    if r["mode"] == "stepwise":
        row["answer parsed"] = np.mean([x["parsed"] for x in its])
        row["trace exact"] = np.mean([x["trace_exact"] for x in its])
    rows.append(row)
T = pd.DataFrame(rows).pivot_table(index=["model", "task"], columns="mode", values=["accuracy", "accuracy k>=3", "trace exact", "answer parsed"])
T.to_csv(f"{OUT}/accuracy_summary.csv"); display(T.round(2))
'''),
md('''
### Head-activity signature
For each model, task and mode: the effective number of heads (entropy-based, / H) and the
top-head share, averaged over 40–80% depth, and their slope on k (% of the mean per step),
on all measured items and on correct items. Negative effective-heads slope = activity
concentrates; positive = it spreads (top share: the opposite). OLMo-2's rare items where one
head outputs > 50× its layer's median are excluded.
In stepwise mode the measured position has the model's trace in context, so the context
grows with k.
'''),
code('''
SIG = pd.DataFrame(e2b.signatures(RUNS))
def fmt(v): return "" if not isinstance(v, tuple) else f"{v[0]:+.2f}{'*' if v[1] < .05 else ''}"
for c in [c for c in SIG.columns if c.startswith(("eff_heads", "top1"))]: SIG[c] = SIG[c].map(fmt)
SIG["model"] = SIG["model"].str.split("/").str[-1]
SIG.to_csv(f"{OUT}/signatures.csv", index=False)
display(SIG.set_index(["model", "family", "mode"]).sort_index())
print("* = permutation p < 0.05; values are % change of the window statistic per step of k")
'''),
code('''
# effective heads by depth, % change from k = 1: direct (blue) vs stepwise (orange)
fig, axes = plt.subplots(len(models), len(FAMS), figsize=(3.2 * len(FAMS), 2.3 * len(models)), sharey=True, squeeze=False)
for r in RUNS:
    if r["model"] not in models: continue
    i, j = models.index(r["model"]), FAMS.index(r["family"])
    meas = np.array([x["measured"] for x in r["items"]])
    if meas.sum() < 10: continue
    k = np.array([x["k"] for x in r["items"]])[meas]
    top, ent, flag = e2b.item_stats(r["C"][meas], r["L"], r["H"])
    ks = sorted(set(k)); v = [ent[(k == kk) & ~flag].mean() for kk in ks]
    axes[i, j].plot(ks, 100 * (np.array(v) / v[0] - 1), marker="o", color="#2a78d6" if r["mode"] == "direct" else "#eb6834", label=r["mode"])
for i, m in enumerate(models):
    axes[i, 0].set_ylabel(m.split("/")[-1], fontsize=8)
    for j in range(len(FAMS)):
        axes[i, j].axhline(0, color="#898781", lw=0.8); axes[i, j].grid(alpha=.3)
        if i == 0: axes[i, j].set_title(FAMS[j])
axes[0, 0].legend(fontsize=7)
fig.suptitle("effective number of heads (40–80% depth), % change from k = 1  (up = spreads)", y=1.0)
plt.tight_layout(); plt.savefig(f"{OUT}/eff_heads_by_k.png", dpi=150); plt.show()
'''),
md('''
## Outputs on Drive (`MyDrive/e2b/`)
- `<model>/<mode>__<task>.npz`: `contribution` [items, layers, heads] (float16; NaN where no
  answer could be parsed), `digit_logits` [items, 10], and `meta` (JSON: every item's prompt,
  generation, prediction, correctness, trace score, split-check errors).
- `items.json`, `accuracy.png`, `accuracy_summary.csv`, `signatures.csv`, `eff_heads_by_k.png`.

Download the model folders and the E2 analysis scripts can read the arrays; `e2b.load_results`
does it here.
'''),
]

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": cells}
out = HERE / "E2b_accuracy_first.ipynb"
out.write_text(json.dumps(nb, indent=1)); print("wrote", out)

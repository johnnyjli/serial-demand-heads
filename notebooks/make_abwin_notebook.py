#!/usr/bin/env python3
"""Build the E7 notebooks (top-10% head ablation in fixed depth bands, the paper's 17 models,
ablation_window.py), reusing E4's setup cells:

  colab/E7_ablation_17models.ipynb      all 17 models in one runtime
  colab/E7_ablation_17models_A.ipynb    } the 17 models split in two halves of about equal
  colab/E7_ablation_17models_B.ipynb    } download size: run them in two runtimes at once

    python3 colab/make_abwin_notebook.py
"""
import copy, hashlib, json, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tasks_and_capture"   # the modules
NBDIR = Path(__file__).resolve().parent
SRC = NBDIR / "E4_ablation.ipynb"
ALL = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
       "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
       "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
       "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B",
       "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
       "allenai/OLMo-2-1124-7B-Instruct",
       "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-Small-24B-Base-2501"]
# about 118 GB of bf16 weights each
HALF_A = ["mistralai/Mistral-Small-24B-Base-2501", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-7B",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B",
          "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
HALF_B = [m for m in ALL if m not in HALF_A]
assert sorted(HALF_A + HALF_B) == sorted(ALL) and len(ALL) == 17
VARIANTS = {"E7_ablation_17models.ipynb": (ALL, None),
            "E7_ablation_17models_A.ipynb": (HALF_A, "A"),
            "E7_ablation_17models_B.ipynb": (HALF_B, "B")}


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n").splitlines(True)}


def text(cell):
    return "".join(cell["source"])


src_nb = json.load(open(SRC))
assert text(src_nb["cells"][0]).startswith("# E4"), "unexpected E4 notebook layout"
mod_src = (ROOT / "ablation_window.py").read_text()
mod_sha = hashlib.sha256(mod_src.encode()).hexdigest()
# the embedded modules of the E4 notebook must be the local, tested ones
for c in src_nb["cells"]:
    t = text(c)
    if t.startswith("%%writefile "):
        f = t.split("\n", 1)[0].split()[1]
        assert t.split("\n", 1)[1] == (ROOT / f).read_text(), f"{f}: E4 notebook copy differs from the local file"


def build(fname, models, half):
    nb = copy.deepcopy(src_nb)
    cells = nb["cells"]
    other = {"A": "E7_ablation_17models_B.ipynb", "B": "E7_ablation_17models_A.ipynb"}.get(half)
    HALF_NOTE = (f"\n    > **This notebook runs half {half} ({len(models)} models).** Run `{other}` in a second "
                 f"runtime at the same time; both write to the same Drive folder (different files). Run "
                 f"step 4 (analysis) at the end of whichever finishes last.\n" if half else "")
    INTRO = md(f"""
    # E7 · Top-10% head ablation in fixed depth bands, 17 models{f' · half {half}' if half else ''}
{HALF_NOTE}
    Does a model whose head activity is more concentrated (higher top-10% share) depend more on its most
    active heads? E4 compared one pair of 70B models; this runs the paper's 17 models with
    `ablation_window.py`, on the four serial tasks (fixed suites, wording w0, test split).

    **Design, fixed before running** (module docstring):
    * Items halved by k as in E4 (same seed): a selection half and a measurement half.
    * **Selection half:** per-head activity ‖z_h W_O^h‖ at the answer position on every layer, the
      heads' mean outputs (mean-ablation values), heads ranked per layer by mean activity, and each
      layer's top-10% share.
    * **Heads:** q = ⌈0.1 H⌉ per layer (the paper's top-10% group); random baseline: q mid-ranked heads
      per layer (25th–75th percentile of activity, top heads excluded).
    * **Layer sets, independent of how layers respond to k:** the 40–80% window (E4's), and five depth
      bands of 20%. All layers of a set are ablated together.
    * **Conditions:** clean; per set top-q mean-ablation, top-q zero-ablation, random-q (5 draws for the
      window, 2 per band); for the window also top-1 and top-3 (E4's sizes). 30 conditions, run in one
      batched forward per item.
    * **Outcome:** gold-minus-competitor logit difference at the answer position (E4's), and every E4
      verification (per-head split, intended-input and o_proj delta checks on the first item, KV-cache
      checks, repeat check).

    **Analysis (step 4):** per model and task, the top-10% share of the window against the *excess drop*
    (top-q minus random, fraction of clean, clean-correct items, depth-balanced). Within each task across
    the 17 models (Spearman, permutation null shuffling models within task), and within each model across
    its four tasks.

    **Runtime:** H100 80GB (A100 80GB works; 40 GB is enough for all but Mistral-Small-24B), standard RAM.
    Computation is ~1–3 min per suite; downloading ~240 GB of weights dominates, so the next model is
    downloaded while the current one runs. All 17 models about 1.5–2 h in one runtime, or about 1 h with
    the two half notebooks in two runtimes.
    **Outputs (Drive):** `next_runs/E7_ablation_window/bf16_<kernel>/…__abwin.npz`, `…/report/`, `logs/`.
    """)

    cells = [INTRO] + copy.deepcopy(cells[1:16])
    c01 = next(c for c in cells if text(c).startswith("# 0.1 Code and packages"))
    t = text(c01).replace('"head_alignment_probe.py"]', '"head_alignment_probe.py", "ablation_window.py"]')
    assert "ablation_window.py" in t
    c01["source"] = t.splitlines(True)
    i_last = max(i for i, c in enumerate(cells) if text(c).startswith("%%writefile"))
    _wf = code("x"); _wf["source"] = ("%%writefile ablation_window.py\n" + mod_src).splitlines(True)
    cells.insert(i_last + 1, _wf)
    c03 = next(c for c in cells if text(c).startswith("# 0.3 Check the written modules"))
    t = text(c03)
    key = '"head_alignment_probe.py": "e7a62b1988cf8927288010e2d43f78b7e89d7e97cead8620b3216a82a2e3b9f1"'
    assert key in t
    c03["source"] = t.replace(key, key + f',\n    "ablation_window.py": "{mod_sha}"').splitlines(True)
    fixed = next(c for c in src_nb["cells"] if text(c).startswith("# Fixed task suites"))
    pin = next(c for c in src_nb["cells"] if text(c).startswith("# Pin the load path"))

    cells += [
        md("""
    ## Configuration
    Everything you may want to change is in this cell.
    """),
        code("""
    # CONFIG ─ E7
    MODELS = __MODELS__
    KERNEL = "sdpa"               # E3: bf16 sdpa is closest to fp32; "eager" matches E2's captures
    KERNEL_PER_MODEL = {}         # e.g. {"Qwen/Qwen2.5-7B": "eager"} to override one model
    PRECISION, DEVICE = "bf16", "cuda"
    FAMILIES = ["pointer", "arith", "binding", "relational"]   # add "fixed_digit" for the numeral control
    WORDINGS = ["w0"]
    SPLIT, SEED = "test", 0
    DRAWS_WIN, DRAWS_BAND = 5, 2
    BATCH_CONDITIONS = None       # None: all conditions (30) in one forward
    SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"   # a few items first (None to skip)
    PREFETCH = True               # download the next model while the current one runs
    GROUP = "all"

    E7_ROOT = f"{DRIVE_ROOT}/E7_ablation_window"
    E7_REPORT = f"{E7_ROOT}/report"
    os.makedirs(E7_REPORT, exist_ok=True)
    kernel_of = lambda m: KERNEL_PER_MODEL.get(m, KERNEL)
    out_dir = lambda m: f"{E7_ROOT}/{PRECISION}_{kernel_of(m)}"
    print(len(MODELS), "models ->", E7_ROOT)
    """),
        md("""
    ### Task suites and load path
    The fixed suites (built and saved to Drive if missing; the same files as E2 and E4). *Check:* the
    four families with wording w0.
    """),
        copy.deepcopy(fixed),
        copy.deepcopy(pin),
        md("""
    ## 1 · Plan
    Suites per model and which outputs exist already (a finished model × suite is skipped).
    """),
        code("""
    import ablation_window as aw
    E7_SUITES = chosen_suites(GROUP, FAMILIES, WORDINGS)
    SUITE_PATHS = [p for p, _ in E7_SUITES]
    for p, s in E7_SUITES:
        print(f"  {s['name']}  ({s['n_test']} test items)")
    def todo_suites(m):
        return [p for p, s in E7_SUITES if not aw.output_path(out_dir(m), m, s["name"]).exists()]
    TODO = [m for m in MODELS if todo_suites(m)]
    print(f"models with suites left: {len(TODO)} of {len(MODELS)}")
    for m in TODO:
        print(f"  {m:42s} kernel {kernel_of(m)}  suites left {len(todo_suites(m))}")
    if TODO:
        _g, _r, _d = hardware()
        print(f"GPU {_g:.0f} GB | RAM {_r:.0f} GB | free disk {_d:.0f} GB")
        big = [m for m in TODO if not fits_gpu(m, _g)]
        if big:
            print("WARNING: these need a larger GPU and will fail here:", big)
    """),
        md("""
    ## 2 · Smoke test
    `SMOKE_MODEL` on one suite with 4 items per depth into a temporary folder: loading, the KV-cache
    path, the ablation checks and writing the output, in about a minute.
    """),
        code("""
    if SMOKE_MODEL and TODO:
        check_access([SMOKE_MODEL])
        _tmp = tempfile.mkdtemp(prefix="e7smoke_")
        try:
            with model_weights(SMOKE_MODEL):
                run_cmd(["ablation_window.py", "--model", SMOKE_MODEL, "--suites", SUITE_PATHS[0], "--out", _tmp,
                         "--device", DEVICE, "--attn-impl", kernel_of(SMOKE_MODEL), "--precision", PRECISION,
                         "--split", SPLIT, "--seed", SEED, "--max-items-per-k", 4,
                         "--draws-win", 1, "--draws-band", 1], f"E7_smoke__{safe(SMOKE_MODEL)}.log")
        finally:
            shutil.rmtree(_tmp, ignore_errors=True)
        print("smoke test ok")
    else:
        print("smoke test skipped")
    """),
        md("""
    ## 3 · Ablation — **H100 / A100 80GB**
    One process per model (it loads once for all its suites). While a model runs, the next one is
    downloaded in the background (`PREFETCH`); each model's weights are deleted when its runs end. A
    finished model × suite is skipped on re-running. *Check* per suite (the `->` line): clean-correct
    items, the window's top-10% share, the drops of top-q (mean, zero) and random heads, and the checks
    (split, delta, cache, repeat) all small.
    """),
        code("""
    import threading
    from huggingface_hub import snapshot_download
    _PATTERNS = ["*.json", "model*.safetensors", "tokenizer*", "*.model", "*.tiktoken", "*.txt"]
    _prefetch = {}

    def prefetch(model):
        if not PREFETCH or model in _prefetch:
            return
        def job():
            try:
                snapshot_download(model, allow_patterns=_PATTERNS, token=os.environ.get("HF_TOKEN"))
            except Exception as e:            # the run downloads what is missing itself
                print(f"(prefetch of {model} failed: {type(e).__name__}; it will download when it runs)")
        th = threading.Thread(target=job, daemon=True); th.start(); _prefetch[model] = th

    if TODO:
        check_access(TODO)

    def e7_model(model):
        i = TODO.index(model)
        if i + 1 < len(TODO):
            prefetch(TODO[i + 1])
        th = _prefetch.get(model)
        if th is not None and th.is_alive():
            print(f"waiting for {model}'s download ...", flush=True); th.join()
        with model_weights(model):
            run_cmd(["ablation_window.py", "--model", model, "--suites", *todo_suites(model),
                     "--out", out_dir(model), "--device", DEVICE, "--attn-impl", kernel_of(model),
                     "--precision", PRECISION, "--split", SPLIT, "--seed", SEED,
                     "--draws-win", DRAWS_WIN, "--draws-band", DRAWS_BAND,
                     *(["--batch-conditions", BATCH_CONDITIONS] if BATCH_CONDITIONS else [])],
                    f"E7__{safe(model)}.log")

    t_all = time.time()
    run_models(TODO, e7_model)
    print(f"done in {(time.time() - t_all) / 60:.0f} min")
    """),
        md("""
    ## 4 · Analyse
    Every E7 output on Drive (both halves, if run in two runtimes). *Read:* within each task, the
    Spearman correlation across models between the window's top-10% share and the excess drop (top-q
    minus random), with a permutation p; and within each model, the same correlation across its four
    tasks. The rows table has every model × task.
    """),
        code("""
    files = sorted(glob.glob(f"{E7_ROOT}/*/*__abwin.npz"))
    print(len(files), "outputs")
    if files:
        run_cmd(["ablation_window.py", "--analyse", *files, "--out", E7_REPORT], "E7__analyse.log")
        display(Markdown(open(f"{E7_REPORT}/abwin_report.md").read()))
    """),
        md("""
    ## Outputs
    Everything under `E7_ablation_window/`. The Hugging Face token is removed from this VM.
    """),
        code("""
    list_outputs(E7_ROOT)
    forget_token()
    """),
    ]
    for c in cells:
        t = text(c)
        if t.startswith("# CONFIG"):
            c["source"] = t.replace("__MODELS__", "[\n    " + ",\n    ".join(repr(m) for m in models) + ",\n]").splitlines(True)
    assert not any("__MODELS__" in text(c) for c in cells)
    nb["cells"] = cells
    out = NBDIR / fname
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out.name} ({len(cells)} cells; {len(models)} models)")


for _f, (_m, _h) in VARIANTS.items():
    build(_f, _m, _h)
print(f"ablation_window.py sha256 {mod_sha[:12]}")

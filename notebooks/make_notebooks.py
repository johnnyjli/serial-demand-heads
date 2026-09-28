#!/usr/bin/env python3
"""
Write the five Colab notebooks for the next runs (E1-E6 of
paper_manuscript/research_outline.tex) and colab/README.md, then validate them.

  E3_kernel_reference.ipynb        Qwen2.5-7B on the original ladder in fp32 (CPU)
                                   and bf16 eager / sdpa (GPU): the kernel to pin
  E1_tasks_and_calibration.ipynb   every family x wording x a settings grid,
                                   check_suite for every tokenizer family,
                                   accuracy-only calibration, one setting per
                                   family and comparison group
  E2_pinned_sweep.ipynb            sweep_probe over models x fixed suites (test
                                   split, --pos0 --generate), eager attention,
                                   plus the kernel check under the other kernel
  E4_ablation.ipynb                ablation_probe on the matched 70B pair, then
                                   --analyse on the pair
  E5_E6_analysis.ipynb             signature_analysis over every capture on Drive

Each notebook stands alone on Colab.  It clones the GitHub repository (which
has active_subgraph_dot.py, deep_probe.py and path_analyzer.py), writes the
modules the repository does not have yet (EMBED) with %%writefile cells whose
text is read here from this checkout, so they are the tested files byte for
byte, and stops unless the written files have the SHA-256 recorded here and the
clone's dependencies are the versions they were tested with.  It mounts Drive
and writes every output under one folder there (PATHS: the same cell in every
notebook), skips work already done, asks for a Hugging Face token only with
getpass into os.environ["HF_TOKEN"] (never printed or saved), and pins the load
path in os.environ so every subprocess inherits it.

No notebook depends on another's output except E5/E6, which analyses the
captures of E2 (and E3, E4).  E2 and E4 build the same fixed recommended suites
themselves (SETTINGS in their suite cell) and use sdpa attention unless
KERNEL_OVERRIDE is set; E1 (calibration) and E3 (kernel reference) are checks
whose results are reported, not read.

Validation (after writing, or alone with --check): every notebook parses as
JSON and, when nbformat is installed, passes nbformat.validate; every
%%writefile body equals its local file byte for byte, both as cell text and as
IPython runs it (when IPython is installed); every code cell compiles once
IPython magics and ! lines are replaced by `pass` (and after IPython's own
transform); the paths cell is identical in every notebook, every P[...] key a
notebook uses is defined, and every hand-off file has its writer and readers;
no cell prints the token.  Generation also warns when a dependency differs from
the local git HEAD or HEAD is not on origin (the clone would then not match).

    python3 colab/make_notebooks.py            # write, then validate
    python3 colab/make_notebooks.py --check    # validate the files on disk
"""
import argparse, ast, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "tasks_and_capture"
# written into the clone by %%writefile (not on GitHub yet); sweep_probe and
# ablation_probe import head_alignment_probe, so it is embedded too
EMBED = ["serial_tasks.py", "sweep_probe.py", "signature_analysis.py", "ablation_probe.py",
         "head_alignment_probe.py"]
# on GitHub; the embedded modules import them (active_subgraph_dot imports path_analyzer)
DEPS = ["deep_probe.py", "active_subgraph_dot.py", "path_analyzer.py"]
REPO_URL = "https://github.com/JohnnyPunchinello/LLM_path_analysis.git"
REPO_DIR = "/content/LLM_path_analysis"
DRIVE_ROOT = "/content/drive/MyDrive/next_runs"
HF_HOME = "/content/hf"
NB = {"E1": "E1_tasks_and_calibration.ipynb", "E2": "E2_pinned_sweep.ipynb",
      "E3": "E3_kernel_reference.ipynb", "E4": "E4_ablation.ipynb",
      "E5": "E5_E6_analysis.ipynb"}
# (accelerator, Colab gpuType, High-RAM) per notebook
RUNTIME = {"E1": ("GPU", "A100", True), "E2": ("GPU", "A100", True),
           "E3": ("GPU", "L4", True), "E4": ("GPU", "A100", True), "E5": (None, None, False)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    """(commit of the local HEAD, warnings): the clone must hold the dependency
    versions the embedded modules were tested against."""
    def g(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    rc, head = g("rev-parse", "HEAD")
    if rc:
        return None, ["not a git checkout: the notebooks cannot name a tested commit"]
    warn = []
    rc, _ = g("diff", "--quiet", "HEAD", "--", *DEPS)
    if rc:
        warn.append(f"{DEPS} differ from HEAD here: commit and push them, or the notebooks "
                    f"stop at the dependency check (the clone has the pushed versions)")
    rc, out = g("branch", "-r", "--contains", "HEAD")
    if rc or not out:
        warn.append("HEAD is not on any remote branch (as of the last fetch): push it, or "
                    "the clone will not have these dependency versions")
    return head, warn


# ───────────────────────────── cell builders ────────────────────────────────
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, **subs):
    text = text.strip("\n")
    for k, v in subs.items():
        text = text.replace(f"@{k}@", v)
    if re.search(r"@[A-Z_]+@", text):
        raise ValueError(f"unfilled placeholder in cell: {re.search(r'@[A-Z_]+@', text).group(0)}")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(True)}


def writefile(name):
    # the file's own text after the magic line, unstripped: IPython hands the
    # cell body to %%writefile as it is (it only adds a final newline if missing)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": (f"%%writefile {name}\n" + (ROOT / name).read_text(encoding="utf-8"))
            .splitlines(True)}


def notebook(cells, key):
    acc, gpu, hm = RUNTIME[key]
    colab = {"provenance": [], "toc_visible": True}
    meta = {"colab": colab, "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}
    if acc:
        meta["accelerator"] = acc
        colab["gpuType"] = gpu
    if hm:
        colab["machine_shape"] = "hm"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


# ───────────────────────────── shared cells ─────────────────────────────────
SETUP_MD = r'''
## 0 · Setup

Run these cells in order at the start of every session (after a disconnect too:
finished work is skipped further down).

- **0.1** clones the code (or pulls it if this runtime already has it), installs the
  packages, and records the clone's commit in `NEXT_RUNS_COMMIT`. If pip says numpy
  was upgraded and asks for a restart, restart the runtime and run again from the top.
- **0.2** writes the modules that are not on GitHub yet. They are generated from the
  tested files by `colab/make_notebooks.py`; do not edit them here.
- **0.3** stops unless the written modules are exactly the embedded ones and the
  clone's `deep_probe.py`, `active_subgraph_dot.py` and `path_analyzer.py` are the
  versions they were tested with, then appends the module hashes to
  `NEXT_RUNS_COMMIT` (every output records it). *Check:* the printed versions.
'''

SETUP = r'''
# 0.1 Code and packages
import os, subprocess, sys
REPO_URL = "@REPO_URL@"
REPO_DIR = "@REPO_DIR@"
PIP_PACKAGES = "transformers accelerate networkx scipy statsmodels"   # pin here if needed, e.g. "transformers==5.12.1 ..."
EMBEDDED = @EMBED_LIST@

def git(*args):
    r = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout.strip()

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    # 0.2 rewrites the embedded modules; remove this session's copies first so a pull
    # never stops on them (untracked now, perhaps tracked upstream later)
    for f in EMBEDDED:
        if os.path.exists(os.path.join(REPO_DIR, f)):
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=REPO_DIR,
                                     capture_output=True).returncode == 0
            git("checkout", "--", f) if tracked else os.remove(os.path.join(REPO_DIR, f))
    git("pull", "-q", "--ff-only")
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
!pip install -q -U {PIP_PACKAGES}
if globals().get("_exit_code"):         # IPython's exit status of the last ! command
    raise RuntimeError("pip install failed (read its output above); nothing further would run "
                       "with the intended package versions")
CLONE_COMMIT = git("rev-parse", "HEAD")
os.environ["NEXT_RUNS_COMMIT"] = CLONE_COMMIT          # 0.3 appends the module hashes
# weights, and the login token load_model stores, stay on this VM's disk, never on Drive
os.environ["HF_HOME"] = "@HF_HOME@"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("code:", REPO_DIR, "at commit", CLONE_COMMIT)
'''

WRITEFILE_MD = r'''
### 0.2 Modules not yet on GitHub
Generated from the tested local files; the next cell checks them byte for byte.
'''

VERIFY = r'''
# 0.3 Check the written modules and the clone, then record the code version
import hashlib, importlib.metadata as _im

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

EXPECTED_SHA256 = @EXPECTED@
TESTED_COMMIT = "@TESTED_COMMIT@"      # the repository commit the modules were tested with
TESTED_DEPS = @TESTED_DEPS@
STRICT_DEPS = True       # False: run with the clone's versions anyway (outputs record the hashes)

bad = [f for f, h in EXPECTED_SHA256.items() if not os.path.exists(f) or sha256_file(f) != h]
if bad:
    raise RuntimeError(f"{bad}: not the embedded module; re-run its %%writefile cell unedited")
changed = [f for f, h in TESTED_DEPS.items() if not os.path.exists(f) or sha256_file(f) != h]
if changed:
    msg = (f"{changed} in the clone differ from the versions the modules were tested with "
           f"(commit {TESTED_COMMIT[:12]}); to use those: !git checkout {TESTED_COMMIT} -- "
           + " ".join(changed))
    if STRICT_DEPS:
        raise RuntimeError(msg + "   (or set STRICT_DEPS = False)")
    print("WARNING:", msg)
os.environ["NEXT_RUNS_COMMIT"] = CLONE_COMMIT + "+" + ",".join(
    f"{f[:-3]}:{sha256_file(f)[:12]}" for f in EXPECTED_SHA256)
print("NEXT_RUNS_COMMIT =", os.environ["NEXT_RUNS_COMMIT"])

def _version(pkg):
    try:
        return _im.version(pkg)
    except _im.PackageNotFoundError:
        return None
print({p: _version(p) for p in ("torch", "transformers", "accelerate", "numpy", "scipy",
                                "statsmodels")})
'''

DRIVE_MD = r'''
### 0.4 Google Drive
Every output of E1–E6 is written under `MyDrive/next_runs` (the same paths cell in
every notebook). Change `DRIVE_ROOT` here, identically in all five notebooks, to use
another folder.
'''

PATHS = r'''
# 0.4 Google Drive: every output goes under DRIVE_ROOT
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "@DRIVE_ROOT@"
P = {
    "suites_grid":     f"{DRIVE_ROOT}/E1_suites/grid",          # E1: family x wording x setting
    "suites_chosen":   f"{DRIVE_ROOT}/E1_suites/chosen",        # E1's choice (informational)
    "suites_fixed":    f"{DRIVE_ROOT}/fixed_suites",            # E2, E4: the fixed recommended suites
    "chosen_manifest": f"{DRIVE_ROOT}/E1_suites/chosen_suites.json",
    "e1_checks":       f"{DRIVE_ROOT}/E1_suites/check_suite.json",
    "e1_calib":        f"{DRIVE_ROOT}/E1_calib",                # accuracy-only calibration captures
    "e1_choice":       f"{DRIVE_ROOT}/E1_choice",               # groups.json, choose_setting.md/.json
    "e3":              f"{DRIVE_ROOT}/E3_kernel_ref",           # ladder captures: fp32 / bf16 eager / bf16 sdpa
    "kernel_choice":   f"{DRIVE_ROOT}/E3_kernel_ref/kernel_choice.json",   # E3's report (not read)
    "e2":              f"{DRIVE_ROOT}/E2_sweep",                # one folder per <precision>_<kernel>
    "e4":              f"{DRIVE_ROOT}/E4_ablation",             # one folder per <precision>_<kernel>, smoke/
    "e4_report":       f"{DRIVE_ROOT}/E4_ablation/report",
    "report":          f"{DRIVE_ROOT}/E5_E6_report",
    "logs":            f"{DRIVE_ROOT}/logs",                    # full output of every run
}
for _k in ("suites_grid", "suites_chosen", "suites_fixed", "e1_calib", "e1_choice", "e3", "e2", "e4",
           "e4_report", "report", "logs"):
    os.makedirs(P[_k], exist_ok=True)
print("outputs ->", DRIVE_ROOT)
'''

TOKEN_MD = r'''
### 0.5 Hugging Face token
Llama 3.x and Gemma 2 are gated: accept their licences on huggingface.co first. The token
is typed here, kept only in this session's environment (`HF_TOKEN`) and passed to each
run; it is never printed or written to Drive. (`load_model` logs in with it, which stores
a copy under `/content/hf` on this VM's local disk; the last cell deletes it.) Press
Enter to skip if you only run ungated models.
'''

TOKEN = r'''
# 0.5 Hugging Face token (getpass only; never printed or saved)
from getpass import getpass
if not os.environ.get("HF_TOKEN"):
    _t = getpass("Hugging Face token (Enter to skip): ").strip()
    if _t:
        os.environ["HF_TOKEN"] = _t
    del _t
_have_token = bool(os.environ.get("HF_TOKEN"))
print("token:", "set for this session" if _have_token else "not set (gated models will fail)")
'''

HELPERS_MD = r'''
### 0.6 Helpers
`run_cmd` runs each module's command line in a fresh process (its memory is freed when
it exits), streams its output with per-item progress thinned to every 25th item plus
the rate and time left, and appends the full output to `logs/` on Drive. `run_models`
moves on to the next model when one fails and lists every failure at the end. Before a
long loop the run cells call `check_access` (every model that still has work can be
downloaded with this token: a licence not yet accepted stops the cell at once, not hours
in) and, before a 70B-class cell, `preflight` (a few items of the smallest model with the
same flags). A model's weights are deleted from local disk when its runs end, except after
an interrupt, so re-running the cell reuses them.
'''

HELPERS = r'''
# 0.6 Helpers shared by the run cells
import contextlib, gc, glob, json, re, shlex, shutil, tempfile, time
import pandas as pd
from IPython.display import Markdown, display
import sweep_probe as sp

DROP_WEIGHTS = True          # delete each model's weights from local disk after its runs
_PROGRESS = re.compile(r"^\s*\[(?:(select|measure) )?(\d+)/(\d+)\]")
_BAR = re.compile(r"\d+%\|")                 # tqdm bars (downloads, shard loading)

def safe(name):
    return str(name).replace("/", "_")

def run_cmd(args, log_name):
    """Run `python <args>` in the clone with this notebook's environment; raise if it fails."""
    cmd = [sys.executable, *map(str, args)]
    print("$", " ".join(shlex.quote(c) for c in cmd[1:]), flush=True)
    t0, phase, last_bar = time.time(), None, 0.0
    with open(os.path.join(P["logs"], log_name), "a") as log:
        log.write(f"\n$ {' '.join(cmd[1:])}   [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        # unbuffered: a child writing to a pipe otherwise holds ~8 kB of output (an hour of
        # 70B progress lines, which the s/item estimate below would misread) and loses it
        # if it is killed (out of memory); cwd: the modules are in the clone whatever the
        # notebook's working folder is now
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=REPO_DIR, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            for line in proc.stdout:
                log.write(line)
                m = _PROGRESS.match(line)
                if m:
                    i, n = int(m.group(2)), int(m.group(3))
                    if i == 1 or m.group(1) != phase:
                        t0, phase = time.time(), m.group(1)
                    if i == 1 or i % 25 == 0 or i == n:
                        rate = (time.time() - t0) / max(i - 1, 1)
                        print(f"{line.rstrip()}   ({rate:.1f} s/item, about "
                              f"{rate * (n - i) / 60:.0f} min left in this pass)", flush=True)
                    continue
                if _BAR.search(line) and "100%" not in line and time.time() - last_bar < 30:
                    continue
                last_bar = time.time() if _BAR.search(line) else last_bar
                print(line, end="", flush=True)
            rc = proc.wait()
        except BaseException:              # interrupted: do not leave the run behind
            proc.kill()
            proc.wait()
            raise
    if rc:
        raise RuntimeError(f"exit code {rc}; read the output above (full log: "
                           f"{P['logs']}/{log_name})")

def run_models(models, one):
    """one(model) for every model; a failure is reported and the next model runs, and all
    failures are raised together at the end."""
    failed = {}
    for m in models:
        print(f"\n===== {m} =====", flush=True)
        try:
            one(m)
        except Exception as e:
            failed[m] = f"{type(e).__name__}: {e}"
            print(f"\n!!! {m} FAILED: {failed[m]}\n", flush=True)
        gc.collect()
    if failed:
        raise RuntimeError("failed models (read their output above):\n" +
                           "\n".join(f"  {m}: {why}" for m, why in failed.items()))

def drop_weights(model, force=False):
    """Delete a model's downloaded weights (Colab's local disk holds one 70B at a time)."""
    if DROP_WEIGHTS or force:
        d = os.path.join(os.environ["HF_HOME"], "hub", "models--" + model.replace("/", "--"))
        shutil.rmtree(d, ignore_errors=True)

@contextlib.contextmanager
def model_weights(model):
    """Around one model's runs: delete its weights when they end, after a failure too (the
    next model needs the disk), but not after an interrupt, so re-running the cell reuses
    them instead of downloading a 70B again."""
    try:
        yield
    except KeyboardInterrupt:
        print(f"\ninterrupted: {model}'s weights stay in {os.environ['HF_HOME']} for the next "
              f"run (drop_weights({model!r}, force=True) frees the disk)", flush=True)
        raise
    except BaseException:
        drop_weights(model)
        raise
    drop_weights(model)

def cached_models():
    """{model: GB} of the weights in HF_HOME (blobs only; snapshots are links to them)."""
    out = {}
    for d in glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--*")):
        b = os.path.join(d, "blobs")
        gb = sum(os.path.getsize(os.path.join(b, f)) for f in os.listdir(b)) / 1e9 \
            if os.path.isdir(b) else 0.0
        out[os.path.basename(d)[len("models--"):].replace("--", "/")] = round(gb, 1)
    return out

def check_access(models):
    """Stop before a long loop, not hours into it, when a model cannot be downloaded with
    this session's token: a licence not accepted (Llama 3.1 and 3.2 and Gemma 2 are
    separate), no token, a typo.  Local folders and offline sessions are not checked; a
    network error only warns."""
    models = [m for m in dict.fromkeys(models) if not os.path.isdir(m)]
    if not models or os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        return
    from huggingface_hub import HfApi
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError
    api, bad = HfApi(), {}
    for m in models:
        try:
            api.auth_check(m)
        except RepositoryNotFoundError as e:           # includes gated without access
            bad[m] = type(e).__name__
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                bad[m] = f"HTTP {status}"
            else:
                print(f"WARNING: could not check {m} ({type(e).__name__}); continuing")
    if bad:
        raise RuntimeError("these models cannot be downloaded in this session (accept the licence "
                           "on huggingface.co with the account of the token from 0.5, re-run 0.5, "
                           "or remove them from the list):\n" +
                           "\n".join(f"  {m}: {why}" for m, why in bad.items()))
    print(f"access ok for {len(models)} model(s) with work left", flush=True)

def preflight(args, model, log_name="preflight.log"):
    """`python args[0] --model model args[1:]` into a temporary folder that is then deleted:
    a few items of a small model with the flags of the long run that follows, so a broken
    environment (a package upgrade, CUDA) costs a minute, not a 70B download and load."""
    tmp = tempfile.mkdtemp(prefix="preflight_")
    try:
        with model_weights(model):
            run_cmd([args[0], "--model", model, *args[1:], "--out", tmp], log_name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"preflight ok on {model}; the long run starts now\n", flush=True)

def size_b(model):
    """Parameters in billions, from the model name (Qwen2.5-0.5B -> 0.5); None if absent."""
    m = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", str(model).split("/")[-1])
    return float(m[-1]) if m else None

def need_gb(model):
    s = size_b(model)
    return None if s is None else 2.15 * s + 2          # bf16 weights plus activations

def hardware():
    import psutil, torch
    gpu = (torch.cuda.get_device_properties(0).total_memory / 1e9
           if torch.cuda.is_available() else 0.0)
    return gpu, psutil.virtual_memory().total / 1e9, shutil.disk_usage(os.environ["HF_HOME"]).free / 1e9

def fits_gpu(model, gpu_gb):
    need = need_gb(model)
    return need is None or need <= 0.95 * gpu_gb

def require_big_runtime(models=()):
    """Stop unless this runtime can hold a 70B-class model: GPU, GPU + RAM, and disk for the
    weights (weights of `models` already on disk, kept after an interrupt, count as free)."""
    gpu, ram, disk = hardware()
    held = cached_models()
    have = disk + sum(held.get(m, 0.0) for m in models)
    print(f"GPU {gpu:.0f} GB | RAM {ram:.0f} GB | free disk {disk:.0f} GB"
          + (f" (+{have - disk:.0f} GB of weights already here)" if have > disk else ""))
    if not (gpu >= 75 and gpu + ram >= 150):
        raise RuntimeError("70B-class runs need an 80 GB GPU (A100 80GB or H100) with High-RAM: "
                           "GPU + RAM above ~150 GB (what does not fit on the GPU is offloaded "
                           "to RAM) and ~150 GB free disk for one model's weights")
    if have < 150:
        raise RuntimeError(f"{disk:.0f} GB free disk; one 70B-class model needs ~150 GB. Weights "
                           f"on this VM: {held or 'none'}; free them with "
                           f"drop_weights('<model>', force=True)")

def read_meta(path):
    """The JSON metadata of a sweep_probe / ablation_probe output, without its arrays."""
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))

def list_outputs(root, pattern="**/*", show=8):
    """Files under root per folder: count, size, and the first few names."""
    by_dir = {}
    for f in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        if os.path.isfile(f):
            by_dir.setdefault(os.path.dirname(f), []).append(f)
    for d, fs in by_dir.items():
        mb = sum(os.path.getsize(f) for f in fs) / 1e6
        print(f"{os.path.relpath(d, DRIVE_ROOT)}/  {len(fs)} files, {mb:.1f} MB")
        for f in fs[:show]:
            print("    ", os.path.basename(f))
        if len(fs) > show:
            print(f"     ... {len(fs) - show} more")
    if not by_dir:
        print(f"(nothing under {root})")

def forget_token():
    """Remove the Hugging Face token from this session and from the VM's disk."""
    os.environ.pop("HF_TOKEN", None)
    for f in ("token", "stored_tokens"):
        p = os.path.join(os.environ["HF_HOME"], f)
        if os.path.exists(p):
            os.remove(p)
    print("Hugging Face token removed from this session and this VM; re-run 0.5 before "
          "running more models.")
'''

PIN_MD = r'''
### Pin the load path
Set in `os.environ`, so every run (`!` commands and `run_cmd`) inherits it: native hooks
(`ASD_FORCE_STRATEGY=B`), the precision, and the attention kernel. `sweep_probe` and
`ablation_probe` also set these from their own arguments and record the load settings
in every output. *Check:* the printed values.
'''

PIN = r'''
# Pin the load path; every subprocess inherits os.environ
os.environ["ASD_FORCE_STRATEGY"] = "B"                                  # native HF hooks
os.environ["ASD_FORCE_PRECISION"] = "bf16" if PRECISION == "bf16" else ""  # empty = fp32 (CPU only)
os.environ["ASD_ATTN_IMPL"] = KERNEL
print({k: os.environ[k] for k in ("ASD_FORCE_STRATEGY", "ASD_FORCE_PRECISION", "ASD_ATTN_IMPL",
                                  "NEXT_RUNS_COMMIT")})
'''

KERNEL_RESOLVE = r'''
# The attention kernel: sdpa, unless overridden.  E3 (Qwen2.5-7B, Sep 24) found bf16
# eager far from fp32 (per-head scores r = 0.88, answers changed) and bf16 sdpa matching
# it (r = 0.9999).  No other notebook is read.
KERNEL = KERNEL_OVERRIDE or "sdpa"
_why = "KERNEL_OVERRIDE" if KERNEL_OVERRIDE else "default (sdpa: closest to fp32 in E3)"
if KERNEL not in ("eager", "sdpa"):
    raise RuntimeError(f"kernel {KERNEL!r}: expected eager or sdpa")
OTHER = "sdpa" if KERNEL == "eager" else "eager"
print(f"kernel: {KERNEL}  (from {_why})")
'''

MODELS = r'''
# Models, smallest first.  The 70B-class pair runs in its own cell (A100 80GB + High-RAM).
MODELS_SMALL = [
    "Qwen/Qwen2.5-0.5B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-SFT",
    "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B",                        # needs a 40 GB GPU
    "mistralai/Mistral-Small-24B-Base-2501",   # needs an 80 GB GPU
    "google/gemma-2-27b",                      # needs an 80 GB GPU
]
MODELS_BIG = [
    "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-72B",
]
'''

MANIFEST_READ = r'''
# Fixed task suites (no E1 needed): one recommended setting per family, the same for every
# model, built here and saved to Drive. To change them, set SETTINGS (and N_PER_FIXED,
# SEED_FIXED, WORDINGS_FIXED) in the configuration cell; otherwise these defaults are used.
import serial_tasks as st
_DEFAULT_SETTINGS = {
    # a 10-cycle with k = 1..5: the fewest lookups equal k (on an n-cycle k steps
    # forward land where n - k steps back do, so k must stay <= n/2)
    "pointer":     dict(k_range=(1, 5), n_states=10, shots=3),
    "arith":       dict(k_range=(1, 6), modulus=7, shots=3),     # mod 7: no k makes the answer trivial
    "binding":     dict(k_range=(1, 6), n_chains=3, shots=3),
    "relational":  dict(k_range=(1, 6), n_chains=3, shots=3),
    "fixed_digit": dict(k_range=(1, 6), n_states=7, shots=3),    # the numeral control
}
SETTINGS = globals().get("SETTINGS") or _DEFAULT_SETTINGS
N_PER_FIXED = globals().get("N_PER_FIXED", 60)          # 60 per depth: 42 test items
SEED_FIXED = globals().get("SEED_FIXED", 0)
WORDINGS_FIXED = tuple(globals().get("WORDINGS_FIXED", ("w0", "w1")))
_fixed = {}
for _fam, _kw in SETTINGS.items():
    for _w in WORDINGS_FIXED:
        _s = st.make_suite(_fam, wording=_w, n_per=N_PER_FIXED, seed=SEED_FIXED, **_kw)
        _dst = f"{P['suites_fixed']}/{_s['name']}.json"
        if not os.path.exists(_dst):
            st.save_suite(_s, _dst)
        elif st.load_suite(_dst) != _s:
            raise RuntimeError(f"{_dst} exists with other items (settings or code changed); "
                               f"delete it or change SEED_FIXED")
        _fixed[(_fam, _w)] = (_dst, {"file": f"{_s['name']}.json", "name": _s["name"], "wording": _w,
                                     "family": _fam, "status": "fixed", "setting": dict(_kw),
                                     "n_test": sum(it["split"] == "test" for it in _s["items"]),
                                     "k_range": list(_kw["k_range"])})

def chosen_suites(group, families, wordings):
    # [(path, entry)] of the fixed suites, in family then wording order; group is
    # ignored: every model runs the same items
    return [(p, dict(e, group=group)) for f in families for w in wordings
            if (f, w) in _fixed for p, e in [_fixed[(f, w)]]]

_all = [m for _n in ("MODELS_SMALL", "MODELS_BIG", "MODELS") for m in globals().get(_n, [])]
GROUPS = {"all": list(dict.fromkeys(_all))}
manifest = {"chosen": {}}
for (_f, _w), (_p, _e) in _fixed.items():
    print(f"  {_f:12s} {_w}  {_e['n_test']:4d} test items  {_e['setting']}")
'''


# ───────────────────────────── E3 ───────────────────────────────────────────
E3_TITLE = r'''
# E3 · Kernel reference: which bf16 attention kernel to pin

The attention kernel changed Qwen2.5-7B's result on the step ladder (outline, decision
points: eager and the default kernel disagree). This notebook decides which bf16 kernel
is closer to exact. Qwen2.5-7B runs the original single-prompt ladder (six prompts,
k = 1–6, `serial_tasks.ladder_suite()`) three times through `sweep_probe.py`, measured at
the position whose next token is the answer digit:

| run | precision | device | attention |
|---|---|---|---|
| `fp32_eager` | float32 | CPU | eager: the reference (`load_model` loads fp32 only on the CPU) |
| `bf16_eager` | bfloat16 | GPU | eager |
| `bf16_sdpa` | bfloat16 | GPU | sdpa (the library default) |

Each bf16 run is compared with fp32 (`signature_analysis.kernel_compare`), and the kernel
closer to fp32 is written to `E3_kernel_ref/kernel_choice.json` on Drive for reference; no
other notebook reads it (they use eager unless `KERNEL_OVERRIDE` is set).

**Run order:** independent of the other notebooks.

**Runtime:** a GPU runtime **with High-RAM** (L4 or A100): the fp32 run holds Qwen2.5-7B
in system RAM (about 31 GB). About 30–45 minutes: a 15 GB download, the fp32 run on the
CPU (10–20 min), each bf16 run a few minutes. Qwen is not gated, so no token is needed
(press Enter at the prompt).

**Outputs (Drive):** `next_runs/E3_kernel_ref/` — `ladder.json`, one folder per run with
the capture, `kernel_choice.json`, and `report/` (signature_analysis on the three runs).
'''

E3_CONFIG = r'''
# CONFIG ─ E3
MODEL = "Qwen/Qwen2.5-7B"
RUNS = {                          # name: (precision, device, attention kernel)
    "fp32_eager": ("fp32", "cpu", "eager"),     # the reference; fp32 loads only on the CPU
    "bf16_eager": ("bf16", "cuda", "eager"),
    "bf16_sdpa":  ("bf16", "cuda", "sdpa"),
}
REFERENCE = "fp32_eager"
FLAGS = ["--generate"]            # identical for every run (--pos0 needs eager, so it is left out)
TIE_MARGIN = 0.10                 # median |bf16 - fp32| within 10% of each other: a tie -> eager
N_PERM = 5000
KERNEL, PRECISION = "eager", "bf16"   # defaults for the environment; each run below sets its own
'''

E3_STEP1_MD = r'''
## 1 · The ladder suite
The six prompts of `active_subgraph_dot.TASK_SUITES["serial_depth"]` as a suite (texts and
answers unchanged), saved once to Drive. *Check:* six items, k = 1–6, one gold digit each.
'''

E3_STEP1 = r'''
import serial_tasks as st
LADDER = f"{P['e3']}/ladder.json"
lad = st.ladder_suite()
if os.path.exists(LADDER):
    if st.load_suite(LADDER) != lad:
        raise RuntimeError(f"{LADDER} differs from serial_tasks.ladder_suite(); move it (and "
                           f"E3's runs) away to rebuild")
else:
    st.save_suite(lad, LADDER)
for it in lad["items"]:
    print(f"k={it['k']}  gold {it['gold']}  ...{it['text'][-60:]!r}")
'''

E3_STEP2_MD = r'''
## 2 · The three runs
Each run is a fresh process, so memory is released between them; finished runs are
skipped. *Check* in each run's output: `load_info` shows strategy B with the precision
of its row, `attn_impl` is the row's kernel, and the `checks (max relative error)` line
has no WARNING (the run aborts rather than write wrong numbers).
'''

E3_STEP2 = r'''
gpu, ram, disk = hardware()
print(f"GPU {gpu:.0f} GB | RAM {ram:.0f} GB | free disk {disk:.0f} GB")
if ram < 40:
    print("WARNING: the fp32 run needs about 35 GB of RAM: turn on High-RAM")
if gpu == 0 and any(d == "cuda" for _, d, _ in RUNS.values()):
    print("WARNING: no GPU here: only the fp32 run can finish; its capture is kept, so run this "
          "cell again on a GPU runtime for the bf16 runs")

def e3_run(name):
    prec, device, attn = RUNS[name]
    out = f"{P['e3']}/{name}"
    os.environ["ASD_FORCE_PRECISION"] = "bf16" if prec == "bf16" else ""     # empty = fp32
    os.environ["ASD_ATTN_IMPL"] = attn
    try:
        run_cmd(["sweep_probe.py", "--model", MODEL, "--suite", LADDER, "--out", out,
                 "--device", device, "--attn-impl", attn, "--precision", prec, *FLAGS],
                f"E3__{name}.log")
    finally:
        os.environ["ASD_FORCE_PRECISION"] = "bf16" if PRECISION == "bf16" else ""
        os.environ["ASD_ATTN_IMPL"] = KERNEL

run_models(list(RUNS), e3_run)
drop_weights(MODEL)
'''

E3_STEP3_MD = r'''
## 3 · Compare with fp32 and choose the kernel
Per bf16 kernel against fp32, on the same six items: correlation of all per-head scores
(`score_r`), median relative difference over all layers and over the final 20% of layers,
correlation of the per-layer slopes on k (top-head share and PR/H) and how many layers
keep their sign, and the all-items signature under each.

**Rule (fixed before running):** if exactly one kernel gives fp32's all-items signature
(`agree`: same sign and class), pin it; otherwise pin the kernel with the smaller median
relative difference to fp32, and call it a tie, which goes to eager (the outline's
default), when the two medians are within `TIE_MARGIN` of each other.

*Check:* the fp32 run's precision reads `fp32`. If both kernels agree closely with fp32
(score r above 0.99, same signature) the kernel matters little at 7B; if they disagree,
the pinned one is used for every later run. `KERNEL TO PIN` is printed last.
'''

E3_STEP3 = r'''
import signature_analysis as sa, pandas as pd
paths = {n: sp.output_path(f"{P['e3']}/{n}", MODEL, lad["name"]) for n in RUNS}
missing = [n for n, f in paths.items() if not os.path.exists(f)]
if missing:
    raise RuntimeError(f"no capture yet for {missing}: finish step 2 first")
runs = {n: sa.load_run(str(f)) for n, f in paths.items()}
ref = runs[REFERENCE]
if ref["precision"] != "fp32":
    raise RuntimeError(f"the reference run's precision is {ref['precision']}, not fp32")
rows = {}
for n, r in runs.items():
    if n == REFERENCE:
        continue
    x = sa.kernel_compare(r, ref, n_perm=N_PERM, rng=0)
    ts, pr = x["per_layer"]["top_share"], x["per_layer"]["pr_h"]
    rows[n] = {"attn_impl": r["attn_impl"], "precision": r["precision"], "n_items": x["n_items"],
               "score_r": x["score_r"], "rel_diff_median": x["rel_diff_median"],
               "late_rel_diff_median": x["late_rel_diff_median"],
               "top_share_slope_r": ts["r"], "top_share_same_sign": f"{ts['same_sign']}/{ts['L']}",
               "pr_h_slope_r": pr["r"], "pr_h_same_sign": f"{pr['same_sign']}/{pr['L']}",
               "signature_bf16": x["signature_a"]["classification_all_items"],
               "signature_fp32": x["signature_b"]["classification_all_items"],
               "agree": x["agree"]}
display(pd.DataFrame(rows).T)

def choose(rows):
    agree = [n for n, v in rows.items() if v["agree"]]
    if len(agree) == 1:
        return agree[0], f"only {agree[0]} gives fp32's all-items signature"
    a, b = sorted(rows, key=lambda n: rows[n]["rel_diff_median"])
    da, db = rows[a]["rel_diff_median"], rows[b]["rel_diff_median"]
    if db - da <= TIE_MARGIN * db:
        e = next(n for n in rows if rows[n]["attn_impl"] == "eager")
        return e, (f"tie: median relative differences {da:.2e} and {db:.2e} are within "
                   f"{TIE_MARGIN:.0%}; eager is the outline's default")
    return a, f"{a} is closer to fp32 (median relative difference {da:.2e} vs {db:.2e})"

best, reason = choose(rows)
choice = {"attn_impl": rows[best]["attn_impl"], "precision": "bf16", "run": best,
          "reason": reason, "model": MODEL, "suite": lad["name"], "reference": REFERENCE,
          "rule": "one kernel agreeing with fp32's all-items signature wins; else the smaller "
                  f"median relative difference, ties within {TIE_MARGIN:.0%} to eager",
          "metrics": rows, "next_runs_commit": os.environ["NEXT_RUNS_COMMIT"],
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(P["kernel_choice"] + ".tmp", "w") as f:
    json.dump(choice, f, indent=1, default=str)
os.replace(P["kernel_choice"] + ".tmp", P["kernel_choice"])
print(f"\nKERNEL TO PIN: {choice['attn_impl']}   ({reason})\n  -> {P['kernel_choice']}")
'''

E3_STEP4_MD = r'''
## 4 · Report on the three runs
`signature_analysis.py` over E3's captures: the three runs, their signatures and the
kernel checks (fp32 as the reference). The ladder has one item per depth, so the
correct-item analysis is missing by design and the ladder is reported beside RQ1, never
in it.
'''

E3_STEP4 = r'''
run_cmd(["signature_analysis.py", "--runs", f"{P['e3']}/*/*.npz", "--out", f"{P['e3']}/report",
         "--n-perm", N_PERM, "--no-mixedlm"], "E3__report.log")
from IPython.display import Markdown
display(Markdown(open(f"{P['e3']}/report/report.md").read()))
'''

E3_FINAL = r'''
# Outputs and the key table
list_outputs(P["e3"])
kc = json.load(open(P["kernel_choice"]))
display(pd.DataFrame(kc["metrics"]).T)
print(f"\nKERNEL TO PIN: {kc['attn_impl']}  ({kc['reason']})")
forget_token()
'''


# ───────────────────────────── E1 ───────────────────────────────────────────
E1_TITLE = r'''
# E1 · Task suites and difficulty calibration

1. **Suites.** Every task family — pointer chase, composed arithmetic, variable binding,
   multi-hop relational lookup, and the numeral control `fixed_digit` — in two wordings,
   at each setting of a difficulty grid (`serial_tasks.make_suite`), saved to Drive.
2. **Checks.** `check_suite` with one tokenizer per model family: constant token count
   across k, the answer position found on every item, answers balanced per depth, the
   gold never equal to the written numeral, other digits in every prompt.
3. **Calibration.** Accuracy only (no activations) on each suite's held-out calibration
   split (30% of items, stratified by k) for every model: small models in one cell, the
   70B-class pair in another.
4. **Choice.** Per task family and comparison group (the matched pair, the Qwen, Llama and
   OLMo ladders), the setting under which the group's weakest model has the most depths
   passing the correct-item rule on the *test* split (≥ 20 correct items and accuracy ≥
   2c/(1+c)); at least three such depths are needed; ties go to more serial depth
   (`signature_analysis.choose_setting`).
5. **Report.** The chosen suites and `chosen_suites.json` go to Drive for reference. E2 and
   E4 do not read them: they use fixed recommended settings (edit `SETTINGS` there to use
   a setting chosen here).

**Run order:** independent of the other notebooks.

**Runtime:** A100 with High-RAM. Steps 1–2: ~10 min. Step 3 (small models): the 12 models
of the comparison groups, up to Qwen2.5-14B (all fit a 40 GB A100), a few minutes each after
the download (~2,300 calibration items per model). The choice reads only group models, so
the Instruct, Mistral and Gemma models are not calibrated (`CALIB_ONLY_GROUPS = False`
adds them for their accuracy tables; the 24B and 27B then need an 80 GB GPU). Step 4
(70B-class): **A100 80GB + High-RAM**, about 5–7 h per model (~2,300 items at ~8 s under
CPU offload) plus ~30 min download and load; `CALIB_LIMIT_BIG` shortens it. Step 4 can run
in its own session (run setup, configuration, steps 1–2 and the calibration-command cell
first). Every suite is saved as it finishes; after a disconnect re-run from the top and
finished work is skipped.

**Outputs (Drive, `next_runs/`):** `E1_suites/grid/` (all suites), `E1_suites/check_suite.json`,
`E1_calib/` (one accuracy-only capture per model and suite), `E1_choice/`
(`groups.json`, `choose_setting.md/.json`), `E1_suites/chosen/` and
`E1_suites/chosen_suites.json` (read by E2 and E4), `logs/`.
'''

E1_CONFIG = r'''
# CONFIG ─ E1
KERNEL, PRECISION, DEVICE = "sdpa", "bf16", "cuda"    # bf16 eager changed Qwen2.5-7B's answers in E3; sdpa matched fp32
FAMILIES = ["pointer", "arith", "binding", "relational", "fixed_digit"]
WORDINGS = ["w0", "w1"]
N_PER, SEED = 60, 0        # items per depth per suite: 18 calibration + 42 test (E2 measures the test split)
# difficulty grid: each setting is built in both wordings and calibrated on every model
GRID = {
    # On an n-cycle, k steps forward land where n - k steps back do, so only
    # k <= n/2 is a clean depth (min_hops == k): a 10-cycle to k = 5, a 7-cycle to k = 3.
    "pointer":     [dict(k_range=(1, 5), n_states=10, shots=3),
                    dict(k_range=(1, 5), n_states=10, shots=8),
                    dict(k_range=(1, 3), n_states=7, shots=8)],
    "arith":       [dict(k_range=(1, 6), modulus=7, shots=3),    # prime modulus: no offset k can fix
                    dict(k_range=(1, 6), modulus=7, shots=8)],
    "binding":     [dict(k_range=(1, 6), n_chains=3, shots=3),
                    dict(k_range=(1, 6), n_chains=2, shots=3)],
    "relational":  [dict(k_range=(1, 6), n_chains=3, shots=3),
                    dict(k_range=(1, 6), n_chains=2, shots=3)],
    # numeral control: depth = distinct states visited; its min_hops is reported alongside
    "fixed_digit": [dict(k_range=(1, 6), n_states=7, shots=3),
                    dict(k_range=(1, 6), n_states=7, shots=8)],
}
@MODELS@
# comparison groups: one setting per family is chosen per group, decided by its weakest model
GROUPS = {
    "pair":         ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"],
    "qwen_ladder":  ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
                     "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"],
    "llama_ladder": ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B",
                     "meta-llama/Llama-3.1-70B"],
    "olmo_ladder":  ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT",
                     "allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-7B-Instruct"],
}
# one tokenizer per model family for check_suite (Instruct variants share their base's)
TOKENIZERS = {
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "llama-3": "meta-llama/Llama-3.2-1B",
    "olmo-2": "allenai/OLMo-2-1124-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-small": "mistralai/Mistral-Small-24B-Base-2501",
    "gemma-2": "google/gemma-2-27b",
}
CALIB_ONLY_GROUPS = True       # calibrate only models in GROUPS (the choice reads nothing else; the
                               # Instruct, Mistral and Gemma models run the groups' suites in E2);
                               # False: every model, for their accuracy tables only
CALIB_LIMIT_SMALL = None       # items per suite (round-robin over k); None = the whole calibration split
CALIB_LIMIT_BIG = None         # e.g. 60 (10 per depth): about half the 70B time, noisier accuracy
PREFLIGHT = True               # before the 70B cell: 6 items of MODELS_SMALL[0] with the same flags
SEC_PER_ITEM_SMALL, SEC_PER_ITEM_BIG = 0.2, 8    # time estimates only
ALLOW_CHECK_FAILURES = False   # True: continue although check_suite failed (the report is saved)
FALLBACK_WHEN_FAILING = True   # no setting passes for a group's weakest model: take the best, marked "fallback"
CHOICE_OVERRIDE = {}           # by hand, e.g. {"pair/pointer": {"n_states": 7, "shots": 8}}
REPLACE_MANIFEST = False       # True: a new choice may replace a different earlier one
DROP_WEIGHTS = True
'''

E1_STEP1_MD = r'''
## 1 · Build the suites
Every family × wording × grid setting, `N_PER` items per depth, saved to
`E1_suites/grid/`. A suite's name holds its main knobs and a hash of the full setting, so
every grid point has its own file; a file already on Drive must equal the rebuilt suite
(else the cell stops). *Check:* the number of suites, the calibration items per model, and
the example prompts (one per family and wording, at the deepest k).
'''

E1_STEP1 = r'''
import serial_tasks as st
GRID_FILES, SUITES = [], {}
for fam in FAMILIES:
    for g in GRID[fam]:
        g = dict(g)
        kr = tuple(g.pop("k_range"))
        for w in WORDINGS:
            s = st.make_suite(fam, w, kr, N_PER, SEED, **g)
            path = f"{P['suites_grid']}/{s['name']}.json"
            if os.path.exists(path):
                if st.load_suite(path) != s:
                    raise RuntimeError(f"{path} exists with other items (another serial_tasks "
                                       f"version?); move it away to rebuild")
            else:
                st.save_suite(s, path)
            GRID_FILES.append(path)
            SUITES[path] = s
n_calib = {p: sum(it["split"] == "calib" for it in s["items"]) for p, s in SUITES.items()}
per_model = lambda lim: sum(min(n, lim or n) for n in n_calib.values())
print(f"{len(GRID_FILES)} suites in {P['suites_grid']}")
print(f"calibration items per model: {per_model(CALIB_LIMIT_SMALL)} (small, ~"
      f"{per_model(CALIB_LIMIT_SMALL) * SEC_PER_ITEM_SMALL / 60:.0f} min each), "
      f"{per_model(CALIB_LIMIT_BIG)} (70B-class, ~{per_model(CALIB_LIMIT_BIG) * SEC_PER_ITEM_BIG / 3600:.1f} h each)")
shown = set()
for p, s in SUITES.items():
    if (s["family"], s["wording"]) in shown:
        continue
    shown.add((s["family"], s["wording"]))
    it = max(s["items"], key=lambda it: it["k"])
    print(f"\n--- {s['name']}: k={it['k']}, gold {it['gold']}, answer set {it['answer_set']} ---")
    print(it["text"][-500:])
'''

E1_STEP2_MD = r'''
## 2 · Check the suites with every tokenizer family
`serial_tasks.check_suite` per suite and tokenizer; results are cached in
`E1_suites/check_suite.json`, so only new suite–tokenizer pairs are checked again. Gated
tokenizers (Llama, Gemma) need the token from 0.5.

*Check:* every row `ok`. A failure (token count changing with k, the answer position not
found, unbalanced answers, ...) stops the notebook unless `ALLOW_CHECK_FAILURES = True`.
Warnings are design limits the analysis handles (e.g. the shortest lookup path below k on
a cycle, repeated final problems when a depth holds more items than the setting has
problems); they are counted, not fatal.
'''

E1_STEP2 = r'''
import pandas as pd
from transformers import AutoTokenizer
checks = json.load(open(P["e1_checks"])) if os.path.exists(P["e1_checks"]) else {}
toks = {}
for fam_name, repo in TOKENIZERS.items():
    try:
        toks[repo] = AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        raise RuntimeError(f"could not load the {repo} tokenizer ({type(e).__name__}: {e}); "
                           f"gated models need the token (0.5) and an accepted licence, or "
                           f"drop the family from TOKENIZERS") from None
t0 = time.time()
for path in GRID_FILES:
    s = SUITES[path]
    for repo, tok in toks.items():
        if checks.get(s["name"], {}).get(repo, {}).get("ok"):
            continue
        rep = st.check_suite(s, {repo: tok})
        r = rep["tokenizers"].get(repo, {})
        checks.setdefault(s["name"], {})[repo] = {
            "ok": rep["ok"], "n_tokens": r.get("n_tokens"), "constant": r.get("constant"),
            "align_ok": r.get("align_ok"), "appended": r.get("appended"),
            "problems": rep["problems"], "warnings": rep["warnings"]}
with open(P["e1_checks"] + ".tmp", "w") as f:
    json.dump(checks, f, indent=1, default=str)
os.replace(P["e1_checks"] + ".tmp", P["e1_checks"])
print(f"checked in {time.time() - t0:.0f} s -> {P['e1_checks']}")
rows, failed, warns = [], [], {}
for path in GRID_FILES:
    name = SUITES[path]["name"]
    for repo in toks:
        c = checks[name][repo]
        rows.append({"suite": name, "tokenizer": repo, "ok": c["ok"], "tokens": c["n_tokens"],
                     "align": c["align_ok"], "warnings": len(c["warnings"])})
        if not c["ok"]:
            failed.append((name, repo, c["problems"]))
        for w in c["warnings"]:
            warns[w.split(":")[0]] = warns.get(w.split(":")[0], 0) + 1
df = pd.DataFrame(rows)
display(df.pivot(index="suite", columns="tokenizer", values="tokens"))
print("warnings (counted over suites x tokenizers):")
for w, n in sorted(warns.items(), key=lambda kv: -kv[1]):
    print(f"  {n:4d}  {w}")
for name, repo, probs in failed:
    print(f"FAILED {name} / {repo}:\n    " + "\n    ".join(probs))
if failed and not ALLOW_CHECK_FAILURES:
    raise RuntimeError(f"{len(failed)} suite/tokenizer checks failed (listed above)")
print("all checks ok" if not failed else "continuing despite failures (ALLOW_CHECK_FAILURES)")
'''

E1_RUN_FN_MD = r'''
### Calibration command (steps 3 and 4)
Run this cell before step 3 or step 4, also when you run step 4 alone in its own session.
It lists the models to calibrate (with `CALIB_ONLY_GROUPS`, only the models of `GROUPS`).
'''

E1_RUN_FN = r'''
_in_group = lambda m: any(m in ms for ms in GROUPS.values())
CALIB_SMALL = [m for m in MODELS_SMALL if _in_group(m) or not CALIB_ONLY_GROUPS]
CALIB_BIG = [m for m in MODELS_BIG if _in_group(m) or not CALIB_ONLY_GROUPS]

def calib_pending(model):
    """Grid suites this model has no calibration output for yet."""
    return [p for p in GRID_FILES
            if not os.path.exists(sp.output_path(P["e1_calib"], model, SUITES[p]["name"]))]

def calib_args(limit, suites=None):
    return ["--suite", *(suites or GRID_FILES), "--device", DEVICE, "--attn-impl", KERNEL,
            "--precision", PRECISION, "--accuracy-only", "--split", "calib",
            *(["--limit", limit] if limit else [])]

def calibrate(model, limit):
    with model_weights(model):
        run_cmd(["sweep_probe.py", "--model", model, *calib_args(limit), "--out", P["e1_calib"]],
                f"E1_calib__{safe(model)}.log")

print("small models:", ", ".join(m.split("/")[-1] for m in CALIB_SMALL) or "none")
print("70B-class:   ", ", ".join(m.split("/")[-1] for m in CALIB_BIG) or "none")
if CALIB_ONLY_GROUPS:
    print("not calibrated (in no group):",
          ", ".join(m.split("/")[-1] for m in MODELS_SMALL + MODELS_BIG if not _in_group(m)) or "none")
'''

E1_STEP3_MD = r'''
## 3 · Calibration: small models
`sweep_probe.py --accuracy-only --split calib` over every grid suite: logits and predictions
at the measured position, no activations. One output per model and suite in `E1_calib/`;
finished suites are skipped (the module checks that an existing output was written with the
same settings and code version, and stops otherwise). Weights are deleted after each model.
With `CALIB_ONLY_GROUPS` every model here fits a 40 GB A100.

*Check:* per suite, the `constrained by k` line (correct / items per depth) and the
measured-token line (the space before the digit for Llama/Qwen/OLMo/Mistral/Gemma). A model
that does not fit this GPU is skipped with a message: run it on an 80 GB GPU.
'''

E1_STEP3 = r'''
gpu, ram, disk = hardware()
print(f"GPU {gpu:.0f} GB | RAM {ram:.0f} GB | free disk {disk:.0f} GB")
if DEVICE == "cuda" and gpu == 0:
    raise RuntimeError("no GPU in this runtime: Runtime > Change runtime type > A100 with High-RAM")

def calibrate_small(model):
    if DEVICE == "cuda" and calib_pending(model) and not fits_gpu(model, gpu):
        print(f"SKIP {model}: needs about {need_gb(model):.0f} GB of GPU memory, this GPU has "
              f"{gpu:.0f} GB; run this cell again on an 80 GB GPU")
        return
    calibrate(model, CALIB_LIMIT_SMALL)

check_access([m for m in CALIB_SMALL if calib_pending(m)
              and (DEVICE != "cuda" or fits_gpu(m, gpu))])
run_models(CALIB_SMALL, calibrate_small)
'''

E1_STEP4_MD = r'''
## 4 · Calibration: 70B-class pair — **A100 80GB + High-RAM**
Same as step 3 for Llama-3.1-70B and Qwen2.5-72B. The model does not fit on the GPU, so
`load_model` offloads the rest to CPU RAM (GPU + RAM must exceed ~150 GB) and every forward
streams those weights: about 8 s per item. While work is left, the cell first checks the
runtime (it stops at once on a smaller one), that both models can be downloaded, and runs
the preflight (6 items of `MODELS_SMALL[0]`, about a minute). Each model downloads ~140 GB,
deleted afterwards. Finished suites are skipped, so you can run one model per session
(also at the same time in two runtimes: set `MODELS_BIG` to a different model in each,
never the same model twice).
'''

E1_STEP4 = r'''
BIG_TODO = [m for m in CALIB_BIG if calib_pending(m)]
if BIG_TODO:
    if DEVICE == "cuda":
        require_big_runtime(BIG_TODO)
    check_access(BIG_TODO)
    if PREFLIGHT and MODELS_SMALL:
        preflight(["sweep_probe.py", *calib_args(6, GRID_FILES[:1])], MODELS_SMALL[0],
                  "E1_preflight.log")
else:
    print("every 70B-class calibration output exists")
run_models(CALIB_BIG, lambda m: calibrate(m, CALIB_LIMIT_BIG))
'''

E1_STEP5_MD = r'''
## 5 · Choose one setting per family and comparison group
`signature_analysis.py --calib` judges every (setting, family, model) on the calibration
split, projected to the test split's items per depth, and picks per family and group the
setting under which the weakest model has the most eligible depths (≥ 3 needed; a wording
without a run counts as 0). Then this cell copies the chosen suites to `E1_suites/chosen/`
and writes `E1_suites/chosen_suites.json`, which E2 and E4 read.

Statuses: `ok` (passes), `fallback` (no setting gives the weakest model 3 eligible depths;
the best one is used and the family's correct-item analysis may be missing — the outline's
"calibration fails" decision point), `override` (`CHOICE_OVERRIDE`), `missing` (a group
model has no calibration run yet: run steps 3–4 for it). The manifest is not silently
replaced: if a choice changes after E2 has started, set `REPLACE_MANIFEST = True`
knowingly (the old manifest is kept as a backup).

*Check:* the table below; per the outline, if the matched pair cannot share a passing
setting on at least 3 families, the invariance claims are restricted to what passes.
'''

E1_STEP5 = r'''
missing_runs = sorted({(m, g) for g, ms in GROUPS.items() for m in ms
                       if not glob.glob(f"{P['e1_calib']}/{safe(m)}__*.npz")})
if missing_runs:
    print("no calibration yet (their groups come out 'missing'):",
          ", ".join(f"{m} [{g}]" for m, g in missing_runs))
groups_path = f"{P['e1_choice']}/groups.json"
with open(groups_path, "w") as f:
    json.dump(GROUPS, f, indent=1)
run_cmd(["signature_analysis.py", "--calib", f"{P['e1_calib']}/*.npz", "--groups", groups_path,
         "--out", P["e1_choice"]], "E1_choose.log")
res = json.load(open(f"{P['e1_choice']}/choose_setting.json"))

by_setting = {}                     # (family, setting key) -> {wording: suite}
for p, s in SUITES.items():
    by_setting.setdefault((s["family"], json.dumps(s["setting"], sort_keys=True)), {})[s["wording"]] = s

def override_key(fam, want):
    want = {k: list(v) if isinstance(v, tuple) else v for k, v in want.items()}
    hits = sorted({key for (f, key) in by_setting if f == fam
                   and all(json.loads(key).get(k) == v for k, v in want.items())})
    if len(hits) != 1:
        raise ValueError(f"CHOICE_OVERRIDE {fam} {want}: matches {len(hits)} grid settings, need 1")
    return hits[0]

chosen = {}
for g in GROUPS:
    for fam in FAMILIES:
        c = res["choices"].get(fam, {}).get(g) or {
            "status": "missing", "setting": None, "reason": "no calibration run of this family"}
        e = {"status": c["status"], "setting": c.get("setting"), "weakest": c.get("weakest"),
             "n_eligible": c.get("n_eligible"), "reason": c.get("reason"), "suites": []}
        if f"{g}/{fam}" in CHOICE_OVERRIDE:
            e.update(status="override", setting=override_key(fam, CHOICE_OVERRIDE[f"{g}/{fam}"]),
                     reason=f"CHOICE_OVERRIDE {CHOICE_OVERRIDE[f'{g}/{fam}']}")
        elif c["status"] == "fails" and FALLBACK_WHEN_FAILING and c.get("candidates"):
            best = c["candidates"][0]          # sorted best first by choose_setting
            e.update(status="fallback", setting=best["setting"], weakest=best["weakest"],
                     n_eligible=best["n_eligible"])
        if e["setting"] is not None:
            files = by_setting.get((fam, e["setting"]), {})
            if sorted(files) != sorted(WORDINGS):
                raise RuntimeError(f"{g}/{fam}: the chosen setting has suites for {sorted(files)}, "
                                   f"not every wording {WORDINGS}: {e['setting']}")
            for w in WORDINGS:
                s = files[w]
                dst = f"{P['suites_chosen']}/{s['name']}.json"
                if not os.path.exists(dst):
                    st.save_suite(s, dst)
                elif st.load_suite(dst) != s:
                    raise RuntimeError(f"{dst} exists with other items")
                e["suites"].append({"file": f"{s['name']}.json", "name": s["name"], "wording": w,
                                    "n_test": sum(it["split"] == "test" for it in s["items"]),
                                    "k_range": s["setting"]["k_range"]})
        chosen.setdefault(g, {})[fam] = e

manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "next_runs_commit": os.environ["NEXT_RUNS_COMMIT"], "groups": GROUPS,
            "families": FAMILIES, "wordings": WORDINGS, "n_per": N_PER, "seed": SEED,
            "suite_dir": P["suites_chosen"], "chosen": chosen}
if os.path.exists(P["chosen_manifest"]):
    old = json.load(open(P["chosen_manifest"]))
    changed = [f"{g}/{f}" for g, fs in old.get("chosen", {}).items() for f, e in fs.items()
               if e.get("setting") is not None
               and chosen.get(g, {}).get(f, {}).get("setting") != e["setting"]]
    if changed and not REPLACE_MANIFEST:
        raise RuntimeError(f"the choice changed for {changed} since the manifest of "
                           f"{old.get('created')}; E2/E4 may already have run on the old suites. "
                           f"Set REPLACE_MANIFEST = True to replace it (a backup is kept).")
    if changed:
        shutil.copy2(P["chosen_manifest"], P["chosen_manifest"].replace(
            ".json", f".{time.strftime('%Y%m%d-%H%M%S')}.bak.json"))
with open(P["chosen_manifest"] + ".tmp", "w") as f:
    json.dump(manifest, f, indent=1)
os.replace(P["chosen_manifest"] + ".tmp", P["chosen_manifest"])
print("->", P["chosen_manifest"])
from IPython.display import Markdown
display(Markdown(open(f"{P['e1_choice']}/choose_setting.md").read()))
'''

E1_FINAL = r'''
# Outputs and the key table: the chosen setting per group and family
list_outputs(DRIVE_ROOT, "E1_*/**/*")
_short = lambda key: ", ".join(f"{k}={v}" for k, v in json.loads(key).items()
                               if k in ("k_range", "n_states", "modulus", "n_chains", "shots",
                                        "shot_mode", "written_k")) if key else "-"
m = json.load(open(P["chosen_manifest"]))
display(pd.DataFrame([{"group": g, "family": f, "status": e["status"], "weakest": e["weakest"],
                       "eligible depths": e["n_eligible"], "setting": _short(e["setting"]),
                       "suites": len(e["suites"])}
                      for g, fs in m["chosen"].items() for f, e in fs.items()]))
forget_token()
'''


# ───────────────────────────── E2 ───────────────────────────────────────────
E2_TITLE = r'''
# E2 · The pinned sweep (decides RQ1)

`sweep_probe.py` over every model × the fixed recommended suites (built in this notebook), on
the **test** split, with `--generate` (strict correctness: the first generated digit must
also be the answer) and `--pos0` (attention to position 0, for the sink control; needs
eager, so only in the eager kernel-check runs), under sdpa attention (`KERNEL_OVERRIDE` to change). Per item and head it records the activity
`contribution` = ‖z_h W_O^h‖ at the position whose next token is the answer digit, and
for OLMo-2 / Gemma-2 also the post-norm write `write_norm`; every linearisation is checked
on every item and a run aborts rather than write wrong numbers.

**Kernel check:** the matched pair and one model per ladder re-run one calibrated family
under the other kernel (`KCHECK_*`); E5 reports a signature only where both kernels agree.

**Needs:** nothing from the other notebooks.

**Runtime:** A100 with High-RAM for the small-model cell (≤ 14B fit 40 GB; the 24B and 27B
models need 80 GB and are skipped with a message otherwise): about 10–30 min per model,
several hours in all. The 70B-class cell needs **A100 80GB + High-RAM**: about 20 s per
item under CPU offload, ~12–15 h per model for 10 suites; split it across sessions — each
suite is saved when it finishes and finished suites are skipped.

**Outputs (Drive):** `next_runs/E2_sweep/bf16_<kernel>/<model>__<suite>.npz` (the pinned
sweep) and `E2_sweep/bf16_<other kernel>/` (the kernel check), `logs/`.
'''

E2_CONFIG = r'''
# CONFIG ─ E2
KERNEL_OVERRIDE = None        # "sdpa" (default) or "eager"
PRECISION, DEVICE = "bf16", "cuda"
@MODELS@
# models outside E1's comparison groups run the suites chosen for these groups, i.e. the same
# items as the models they are compared with
EXTRA_GROUPS = {
    "meta-llama/Llama-3.2-3B-Instruct": ["llama_ladder"],
    "meta-llama/Llama-3.1-8B-Instruct": ["llama_ladder"],
    "Qwen/Qwen2.5-7B-Instruct": ["qwen_ladder"],
    "mistralai/Mistral-7B-v0.1": ["pair"],
    "mistralai/Mistral-Small-24B-Base-2501": ["pair"],
    "google/gemma-2-27b": ["pair"],
}
FAMILIES = ["pointer", "binding", "arith", "relational", "fixed_digit"]   # run in this order
WORDINGS = ["w0", "w1"]
USE_STATUSES = ["ok", "override", "fallback"]    # which E1 choices to run
LIMIT_SMALL = None            # items per suite (round-robin over k); None = the whole test split
LIMIT_BIG = None
BIG_FAMILIES = FAMILIES       # e.g. drop "fixed_digit" to save ~2 h per 70B model
BIG_WORDINGS = WORDINGS
SEC_PER_ITEM_SMALL, SEC_PER_ITEM_BIG = 0.5, 20   # time estimates only
# kernel check: the matched pair and one model per ladder, one calibrated family, the other kernel
KCHECK_MODELS = ["Qwen/Qwen2.5-7B", "allenai/OLMo-2-1124-7B", "meta-llama/Llama-3.1-8B",
                 "meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"]
KCHECK_FAMILY, KCHECK_WORDING, KCHECK_LIMIT = "pointer", "w0", None
PREFLIGHT = True              # before the 70B cell: 6 items of MODELS_SMALL[0] with the same flags
DROP_WEIGHTS = True
'''

E2_PLAN_MD = r'''
## 1 · The plan
Per model: its groups, the suites it runs (union over its groups, in family order), the
items, and a time estimate; plus the kernel-check suite. With the eager kernel pinned the
sweep records `--pos0`; with sdpa it cannot (sdpa returns no attention weights), and the
sink control then comes from the eager kernel-check runs only.

*Check:* every model has suites; groups whose choice is `missing` in E1 are listed.
'''

E2_PLAN = r'''
MAIN_DIR = f"{P['e2']}/{PRECISION}_{KERNEL}"
CHECK_DIR = f"{P['e2']}/{PRECISION}_{OTHER}"
POS0 = KERNEL == "eager"

def groups_of(model):
    return [g for g, ms in GROUPS.items() if model in ms] + [
        g for g in EXTRA_GROUPS.get(model, []) if model not in GROUPS.get(g, [])]

def suites_for(model, families, wordings):
    out, seen = [], set()
    for g in groups_of(model):
        for path, s in chosen_suites(g, families, wordings):
            if path not in seen:
                seen.add(path)
                out.append((path, s))
    return out

def kcheck_suite(model):
    for g in groups_of(model):
        hit = chosen_suites(g, [KCHECK_FAMILY], [KCHECK_WORDING])
        if hit:
            return hit[0]
    return None

import pandas as pd
plan = []
for m, fams, words, lim, sec in [(m, FAMILIES, WORDINGS, LIMIT_SMALL, SEC_PER_ITEM_SMALL) for m in MODELS_SMALL] + \
                                [(m, BIG_FAMILIES, BIG_WORDINGS, LIMIT_BIG, SEC_PER_ITEM_BIG) for m in MODELS_BIG]:
    ss = suites_for(m, fams, words)
    n = sum(min(s["n_test"], lim or s["n_test"]) for _, s in ss)
    kc = kcheck_suite(m) if m in KCHECK_MODELS else None
    plan.append({"model": m, "groups": ",".join(groups_of(m)) or "NONE", "suites": len(ss),
                 "items": n, "hours (est.)": round(n * sec / 3600, 1),
                 "kernel check": kc[1]["name"] if kc else "-"})
display(pd.DataFrame(plan))
not_run = [f"{g}/{f} ({e['status']})" for g, fs in manifest["chosen"].items() for f, e in fs.items()
           if e["status"] not in USE_STATUSES]
if not_run:
    print("E1 choices not run:", ", ".join(not_run))
empty = [p["model"] for p in plan if not p["suites"]]
if empty:
    print("WARNING: no chosen suites, so nothing runs, for:", ", ".join(empty),
          "\n  (their groups' E1 status is not in USE_STATUSES: finish E1's calibration and step 5)")
print(f"pinned sweep -> {MAIN_DIR}\nkernel check -> {CHECK_DIR}   (--pos0 in the pinned sweep: {POS0})")
'''

E2_RUN_FN = r'''
def e2_pending(model, families, wordings):
    """Outputs of this model not on Drive yet: its suites, and its kernel-check suite."""
    outs = [sp.output_path(MAIN_DIR, model, s["name"]) for _, s in suites_for(model, families, wordings)]
    kc = kcheck_suite(model) if model in KCHECK_MODELS else None
    if kc:
        outs.append(sp.output_path(CHECK_DIR, model, kc[1]["name"]))
    return [o for o in outs if not os.path.exists(o)]

def sweep_args(suites, limit):
    return ["--suite", *suites, "--device", DEVICE, "--attn-impl", KERNEL, "--precision", PRECISION,
            "--split", "test", "--generate", *(["--pos0"] if POS0 else []),
            *(["--limit", limit] if limit else [])]

def e2_model(model, families, wordings, limit):
    ss = suites_for(model, families, wordings)
    if not ss:
        print(f"{model}: no chosen suites (groups {groups_of(model)}); nothing to run")
        return
    with model_weights(model):
        run_cmd(["sweep_probe.py", "--model", model, *sweep_args([p for p, _ in ss], limit),
                 "--out", MAIN_DIR], f"E2__{safe(model)}.log")
        kc = kcheck_suite(model) if model in KCHECK_MODELS else None
        if kc:
            os.environ["ASD_ATTN_IMPL"] = OTHER
            try:
                run_cmd(["sweep_probe.py", "--model", model, "--suite", kc[0], "--out", CHECK_DIR,
                         "--device", DEVICE, "--attn-impl", OTHER, "--precision", PRECISION,
                         "--split", "test", "--generate", *(["--pos0"] if OTHER == "eager" else []),
                         *(["--limit", KCHECK_LIMIT] if KCHECK_LIMIT else [])],
                        f"E2_kcheck__{safe(model)}.log")
            finally:
                os.environ["ASD_ATTN_IMPL"] = KERNEL
'''

E2_SMALL_MD = r'''
## 2 · Small models (≤ 27B), smallest first
One process per model loads it once for all its suites; finished suites are skipped (and
an existing output written with other settings or another code version stops the run
instead of mixing versions). Models that do not fit this GPU are skipped with a message.

*Check* per suite: `checks (max relative error)` with no WARNING, `constrained by k` and
`strict by k` accuracy, the measured token (the space before the digit for byte-level and
SentencePiece tokenizers), `BOS n/n` for Llama/OLMo/Gemma/Mistral and `0/n` for Qwen.
'''

E2_SMALL = r'''
gpu, ram, disk = hardware()
print(f"GPU {gpu:.0f} GB | RAM {ram:.0f} GB | free disk {disk:.0f} GB")
if DEVICE == "cuda" and gpu == 0:
    raise RuntimeError("no GPU in this runtime: Runtime > Change runtime type > A100 with High-RAM")

def e2_small(model):
    if DEVICE == "cuda" and not fits_gpu(model, gpu):
        print(f"SKIP {model}: needs about {need_gb(model):.0f} GB of GPU memory, this GPU has "
              f"{gpu:.0f} GB; run this cell again on an 80 GB GPU")
        return
    e2_model(model, FAMILIES, WORDINGS, LIMIT_SMALL)

check_access([m for m in MODELS_SMALL if e2_pending(m, FAMILIES, WORDINGS)
              and (DEVICE != "cuda" or fits_gpu(m, gpu))])
run_models(MODELS_SMALL, e2_small)
'''

E2_BIG_MD = r'''
## 3 · 70B-class pair — **A100 80GB + High-RAM**
Llama-3.1-70B and Qwen2.5-72B, offloaded to CPU RAM (~20 s per item). Suites run in the
order of the plan (the pair's suites first), each saved when it finishes, so a
disconnect costs at most the suite in progress: re-run from the top and this cell
continues. While work is left, the cell first checks the runtime (it stops at once on a
smaller one), that the models can be downloaded, and runs the preflight (6 items of
`MODELS_SMALL[0]` with the same flags, about a minute). The two models can run at the same
time in two runtimes: set `MODELS_BIG` to a different model in each, never the same model
twice (both would write the same files).
'''

E2_BIG = r'''
BIG_TODO = [m for m in MODELS_BIG if e2_pending(m, BIG_FAMILIES, BIG_WORDINGS)]
if BIG_TODO:
    if DEVICE == "cuda":
        require_big_runtime(BIG_TODO)
    check_access(BIG_TODO)
    if PREFLIGHT and MODELS_SMALL:
        _first = suites_for(BIG_TODO[0], BIG_FAMILIES, BIG_WORDINGS)[0][0]
        preflight(["sweep_probe.py", *sweep_args([_first], 6)], MODELS_SMALL[0], "E2_preflight.log")
else:
    print("every 70B-class output exists")
run_models(MODELS_BIG, lambda m: e2_model(m, BIG_FAMILIES, BIG_WORDINGS, LIMIT_BIG))
'''

E2_FINAL = r'''
# Outputs and the key table: accuracy per model and suite (constrained / strict)
list_outputs(P["e2"])
rows = []
for f in sorted(glob.glob(f"{P['e2']}/*/*.npz")):
    mt = read_meta(f)
    its = mt["items"]
    acc_c = sum(r["correct_constrained"] for r in its) / len(its)
    acc_s = (sum(r["correct_strict"] for r in its) / len(its)) if mt["generate"] else None
    rows.append({"model": mt["model"].split("/")[-1], "kernel": mt["attn_impl"],
                 "cell": f"{mt['family']}/{mt['wording']}", "suite": mt["suite"], "n": len(its),
                 "chance": round(mt["chance"], 3),
                 "acc": f"{acc_c:.2f}" + (f" / {acc_s:.2f}" if acc_s is not None else ""),
                 "split warnings": mt["checks"].get("split_warn", 0)})
if rows:
    df = pd.DataFrame(rows)
    # a model in two groups (the 70B pair and its ladder) can run one family and wording
    # under two settings: name those columns by suite rather than keep only one of them
    _dup = df.duplicated(["model", "kernel", "cell"], keep=False)
    df.loc[_dup, "cell"] = df.loc[_dup, "suite"]
    display(df.pivot_table(index=["model", "kernel"], columns="cell", values="acc", aggfunc="first"))
    warn = df[df["split warnings"] > 0]
    if len(warn):
        print("outputs with per-head split warnings (read their logs):")
        display(warn)
forget_token()
'''


# ───────────────────────────── E4 ───────────────────────────────────────────
E4_TITLE = r'''
# E4 · Head ablation on the matched pair (RQ2)

`ablation_probe.py` on Llama-3.1-70B and Qwen2.5-72B, on the fixed recommended suites
(test split, built in this notebook), with sdpa attention. Design, fixed before running (outline E4):
the items are split 50/50 by k into a selection half and a measurement half. On the
selection half: per-head activity, the heads' mean outputs (the mean-ablation values), and
the window layers (40–80% depth) classified as spreading or funnelling
(`signature_analysis.classify_layers`). On the measurement half, at the measured position
only: top-1 and top-3 heads per layer ablated jointly across each set, by mean and by zero;
a random baseline from mid-ranked heads (mean only); with `PER_LAYER`, top-1 one layer at a
time; and clean. Outcome: gold-minus-competitor logit difference as a fraction of clean, per
k, on clean-correct items. `--analyse` then gives per model the drop by k with bootstrap CIs,
sub-additivity top3 / (3 top1), the excess over random, the slope on k, and for the pair the
ablation × k × model interaction with a 90% CI and the equivalence rule (a null only if the
CI lies within ±10% of the clean logit difference).

**Needs:** nothing from the other notebooks. Optionally read E5's RQ1 gate first: if the
signature is task-dependent, keep in `FAMILIES` the families where the pair's signatures
differ.

**Runtime:** **A100 80GB + High-RAM**. About 2 h per suite per model (126 selection items at
~10 s, 126 measurement items at ~50 s with 16 conditions per forward); the defaults (4
families, one wording, 2 models) are ~16 h. Every suite is checkpointed as it runs
(`<output>.ckpt` next to the output), so a disconnected session resumes where it stopped.

**Outputs (Drive):** `next_runs/E4_ablation/bf16_<kernel>/<model>__<suite>__ablation.npz`,
`E4_ablation/smoke/`, `E4_ablation/report/ablation_report.md/.json`, `logs/`.
'''

E4_CONFIG = r'''
# CONFIG ─ E4
KERNEL_OVERRIDE = None        # "sdpa" (default) or "eager"
PRECISION, DEVICE = "bf16", "cuda"
MODELS = ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"]
GROUP = "pair"                # label only: every model runs the same fixed suites
FAMILIES = ["pointer", "binding", "arith", "relational"]   # in priority order
WORDINGS = ["w0"]
USE_STATUSES = ["ok", "override", "fallback"]
SPLIT = "test"
WINDOW = (0.4, 0.8)           # target layers floor(0.4(L-1))..floor(0.8(L-1))
RANDOM_DRAWS, SEED, PER_LAYER = 5, 0, True
MAX_ITEMS_PER_K = None        # e.g. 30 to shorten (items per depth before the 50/50 split)
BATCH_CONDITIONS = None       # conditions per forward; None = the module's default (16 on cuda)
N_PERM, N_BOOT = 5000, 2000
EQUIV = 0.10                  # equivalence margin, as a fraction of the clean logit difference
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"    # a few items on a small model first (None to skip)
SEC_PER_SELECTION_ITEM, SEC_PER_MEASURED_ITEM = 10, 50   # time estimates only
DROP_WEIGHTS = True
'''

E4_PLAN_MD = r'''
## 1 · The plan
The pair's chosen suites (family order as in `FAMILIES`), items per suite and a time
estimate. *Check:* each family you want is listed with status `ok` (a `fallback` suite
runs, but its eligible depths may be too few: the analysis then falls back to the depths
solved above chance and labels them exploratory, as the outline prescribes).
'''

E4_PLAN = r'''
import ablation_probe as ap, pandas as pd
E4_RUNS = f"{P['e4']}/{PRECISION}_{KERNEL}"
SMOKE_DIR = f"{P['e4']}/smoke"
E4_SUITES = chosen_suites(GROUP, FAMILIES, WORDINGS)
if not E4_SUITES:
    raise RuntimeError(f"no chosen suites for group {GROUP!r} and families {FAMILIES}; see E1 step 5")
rows = []
for path, s in E4_SUITES:
    n = s["n_test"] if not MAX_ITEMS_PER_K else min(s["n_test"], MAX_ITEMS_PER_K * (s["k_range"][1] - s["k_range"][0] + 1))
    rows.append({"suite": s["name"], "status": s["status"], "items": n,
                 "hours per model (est.)": round((n / 2 * SEC_PER_SELECTION_ITEM + n / 2 * SEC_PER_MEASURED_ITEM) / 3600, 1)})
display(pd.DataFrame(rows))
E4_TODO = [m for m in MODELS if any(not os.path.exists(ap.output_path(E4_RUNS, m, s["name"]))
                                    for _, s in E4_SUITES)]
print(f"outputs -> {E4_RUNS}\nmodels with suites left: {E4_TODO or 'none'}")
if E4_TODO and DEVICE == "cuda":
    _g, _r, _ = hardware()
    if not (_g >= 75 and _g + _r >= 150):
        print(f"WARNING: GPU {_g:.0f} GB + RAM {_r:.0f} GB: step 3 needs an 80 GB GPU with "
              f"High-RAM and will stop there; switch the runtime before the smoke test")
'''

E4_SMOKE_MD = r'''
## 2 · Smoke test on a small model
`SMOKE_MODEL` on the first suite with 4 items per depth, one random draw and few
permutations, into `E4_ablation/smoke/`: checks loading, the KV-cache path (the cached
last token must reproduce a full forward), the ablation bookkeeping and the output on this
runtime in a few minutes. With so few items the layer classification usually finds no
spreading or funnelling layer, so only the clean condition may run; that is expected here.
It runs afresh in every session while ablation work is left (a previous smoke output is
replaced), and is skipped once every ablation output exists.
'''

E4_SMOKE = r'''
if SMOKE_MODEL and E4_TODO:
    # always a fresh run: it checks this runtime, and an old smoke output written with another
    # kernel or code version would otherwise stop the notebook here
    _out = ap.output_path(SMOKE_DIR, SMOKE_MODEL, E4_SUITES[0][1]["name"])
    for _f in (_out, ap.checkpoint_path(_out)):
        if os.path.exists(_f):
            os.remove(_f)
    check_access([SMOKE_MODEL])
    with model_weights(SMOKE_MODEL):
        run_cmd(["ablation_probe.py", "--model", SMOKE_MODEL, "--suite", E4_SUITES[0][0], "--out", SMOKE_DIR,
                 "--device", DEVICE, "--attn-impl", KERNEL, "--precision", PRECISION, "--split", SPLIT,
                 "--max-items-per-k", 4, "--random-draws", 1, "--seed", SEED, "--n-perm", 200],
                f"E4_smoke__{safe(SMOKE_MODEL)}.log")
else:
    print("smoke test skipped:", "SMOKE_MODEL is None" if not SMOKE_MODEL else "no ablation left to run")
'''

E4_RUN_MD = r'''
## 3 · Ablation on the pair — **A100 80GB + High-RAM**
Model by model (each model's weights are downloaded once and deleted after its suites),
one process per suite (`ablation_probe` takes one suite, so each suite loads the model
from local disk again: ~10 min for a 70B). A finished suite is skipped; an interrupted one
resumes from its checkpoint when the settings and code version match (the module stops
otherwise). While work is left the cell first checks the runtime and that the models can be
downloaded. The two models can run at the same time in two runtimes: set `MODELS` to a
different model in each, never the same model twice (both would write the same checkpoint).

*Check* per suite: the window layers classified as spreading / funnelling on the selection
half (an empty set is skipped and reported), `checks: cache ...` below its abort limit, and
the clean accuracy by k of the measurement half.
'''

E4_RUN = r'''
if E4_TODO:
    if DEVICE == "cuda":
        require_big_runtime(E4_TODO)
    check_access(E4_TODO)

def e4_model(model):
    with model_weights(model):
        for path, s in E4_SUITES:
            run_cmd(["ablation_probe.py", "--model", model, "--suite", path, "--out", E4_RUNS,
                     "--device", DEVICE, "--attn-impl", KERNEL, "--precision", PRECISION,
                     "--window", *WINDOW, "--split", SPLIT, "--random-draws", RANDOM_DRAWS,
                     "--seed", SEED, "--n-perm", N_PERM,
                     *(["--per-layer"] if PER_LAYER else []),
                     *(["--max-items-per-k", MAX_ITEMS_PER_K] if MAX_ITEMS_PER_K else []),
                     *(["--batch-conditions", BATCH_CONDITIONS] if BATCH_CONDITIONS else [])],
                    f"E4__{safe(model)}.log")

run_models(MODELS, e4_model)
'''

E4_ANALYSE_MD = r'''
## 4 · Analyse the pair
`ablation_probe.py --analyse` over every ablation output of this configuration (a quick CPU
step; it runs again whenever new outputs exist). Only two different models on the same suite
are compared. *Read:* per model and condition the mean drop and its slope on k
(specialisation predicts Qwen's top-1 drop grows with k; consensus predicts Llama's top-3 drop
is sub-additive and flat in k), and per pair the interaction verdict: `null` only if the 90% CI
lies within ±`EQUIV`, `difference` if it excludes 0, otherwise `inconclusive` (never read as a
null). A `†` marks a fractional effect without an absolute one.
'''

E4_ANALYSE = r'''
from IPython.display import Markdown
files = sorted(glob.glob(f"{E4_RUNS}/*__ablation.npz"))
if not files:
    print("no ablation outputs yet in", E4_RUNS)
else:
    run_cmd(["ablation_probe.py", "--analyse", *files, "--out", P["e4_report"], "--n-boot", N_BOOT,
             "--n-perm", N_PERM, "--seed", SEED, "--equiv", EQUIV], "E4__analyse.log")
    display(Markdown(open(f"{P['e4_report']}/ablation_report.md").read()))
'''

E4_FINAL = r'''
# Outputs and the key tables: mean drop per condition, and the pair's interaction verdicts
list_outputs(P["e4"])
rep_path = f"{P['e4_report']}/ablation_report.json"
if os.path.exists(rep_path):
    rep = json.load(open(rep_path))
    _r = lambda x: None if x is None else round(x, 3)
    rows = []
    for r in rep["runs"]:
        for g, e in r.get("groups", {}).items():
            if "/L" in g:                      # per-layer conditions: see the report
                continue
            sl = e.get("slope") or {}
            rows.append({"model": r["model"].split("/")[-1], "suite": r["suite"], "condition": g,
                         "mean drop": _r(e.get("mean")),
                         "95% CI": [_r(x) for x in (e.get("ci95") or [None, None])],
                         "slope on k": sl.get("slope"), "p": sl.get("p")})
    display(pd.DataFrame(rows))
    pairs = [{"a": x["a"].split("/")[-1], "b": x["b"].split("/")[-1], "suite": x["suite_a"],
              "pairing": key, "interaction": v["interaction"], "90% CI": v["ci90"],
              "verdict": v["verdict"] + (" †" if v.get("fractional_only") else "")}
             for x in rep["pairs"] if x.get("compared") for key, v in x.get("groups", {}).items()]
    if pairs:
        display(pd.DataFrame(pairs))
forget_token()
'''


# ───────────────────────────── E5/E6 ────────────────────────────────────────
E5_TITLE = r'''
# E5 / E6 · Analysis: signatures, RQ1 invariance, layer maps, ladders

`signature_analysis.py` over every capture on Drive (E2's pinned sweep and kernel check,
and E3's ladder runs): per model and suite the signature (spread / funnel / flat / mixed /
inconclusive / missing) on all items and on correct items at eligible depths; **RQ1**
invariance of each model's pre-registered sign across the four task families and two
wordings (Holm across models; the numeral control and the ladder are reported beside it);
**E5** layer maps (spreading / funnelling layers under the joint k-permutation null, BH
within model and task) and PR bands; **E6** ladders — scale (Qwen2.5 0.5B–72B, Llama 1B–70B)
and post-training (OLMo-2 base → SFT → DPO → Instruct, and the base/Instruct pairs) under
one joint relabelling of k — plus kernel checks, the attention-sink control and power
targets (items per depth needed). The E4 report is shown too when it exists.

Pre-registered signs (outline RQ1): Qwen2.5 base **funnel**, Llama base **spread**,
post-trained OLMo-2 **spread**. OLMo-2 base, the Instruct models of Llama and Qwen, and the
further families (Gemma-2, Mistral) are not pre-registered: their direction is recorded
first and compared afterwards.

**Runtime:** CPU (no GPU needed; the standard runtime is enough). About 10–30 minutes with
the full permutation counts; `QUICK = True` for a first look. Run it after E2 (it reads the
RQ1 gate that decides how E4 is framed) and again after E4.

**Outputs (Drive):** `next_runs/E5_E6_report/report.md`, `report.json`, `inputs.json` (what
was analysed), `ladders.png`.
'''

E5_CONFIG = r'''
# CONFIG ─ E5/E6
KERNEL_OVERRIDE = None        # the pinned kernel for RQ1; default sdpa
PRECISION = "bf16"
PREREG = ["qwen2.5=funnel", "llama=spread", "OLMo-2-1124-7B-SFT=spread",
          "OLMo-2-1124-7B-DPO=spread", "OLMo-2-1124-7B-Instruct=spread"]
PAIRS = ["Llama-3.1-70B:Qwen2.5-72B"]   # paired tests B minus A; base/Instruct pairs are found automatically
INCLUDE_E1_CALIB = False      # accuracy-only calibration captures carry no activity
QUICK = False                 # True: 1000 permutations, 500 bootstraps, no MixedLM (a first look)
N_PERM, N_BOOT, ALPHA, Q, SEED = 5000, 2000, 0.05, 0.05, 0
MIXEDLM = True                # the secondary MixedLM fit in the invariance test
MIN_FAMILIES = 2              # RQ1 criterion (b): families that must be present
'''

E5_INV_MD = r'''
## 1 · Inputs
The capture files found (E2 pinned sweep and kernel check, E3), per folder and model.
*Check:* every model you ran is there, under the pinned kernel.
'''

E5_INV = r'''
import pandas as pd
if QUICK:
    N_PERM, N_BOOT, MIXEDLM = 1000, 500, False
GLOBS = [f"{P['e2']}/**/*.npz", f"{P['e3']}/**/*.npz"] + ([f"{P['e1_calib']}/*.npz"] if INCLUDE_E1_CALIB else [])
files = sorted({f for g in GLOBS for f in glob.glob(g, recursive=True)})
rows = [{"folder": os.path.relpath(os.path.dirname(f), DRIVE_ROOT),
         "model": os.path.basename(f).split("__")[0]} for f in files]
if not rows:
    raise RuntimeError("no capture files yet: run E2 (and E3) first")
display(pd.DataFrame(rows).value_counts().rename("files").reset_index())
'''

E5_RUN_MD = r'''
## 2 · Run the analysis
Skipped when the inputs (paths, sizes, times) and settings are unchanged since the last
report. Files that cannot be analysed without wrong numbers are listed as rejected; a
capture superseded by a better one of the same model, suite and configuration is listed as
superseded.
'''

E5_RUN = r'''
from IPython.display import Markdown
settings = {"kernel": KERNEL, "precision": PRECISION, "prereg": PREREG, "pairs": PAIRS,
            "n_perm": N_PERM, "n_boot": N_BOOT, "alpha": ALPHA, "q": Q, "seed": SEED,
            "mixedlm": MIXEDLM, "min_families": MIN_FAMILIES,
            "code": os.environ["NEXT_RUNS_COMMIT"]}
inputs = {"settings": settings,
          "files": [[f, os.path.getsize(f), int(os.path.getmtime(f))] for f in files]}
stamp = f"{P['report']}/inputs.json"
if (os.path.exists(stamp) and os.path.exists(f"{P['report']}/report.json")
        and json.load(open(stamp)) == json.loads(json.dumps(inputs))):
    print("report is up to date:", P["report"])
else:
    run_cmd(["signature_analysis.py", "--runs", *GLOBS, "--out", P["report"],
             "--preregistered", *PREREG, "--pair", *PAIRS, "--pinned", KERNEL, PRECISION,
             "--n-perm", N_PERM, "--n-boot", N_BOOT, "--alpha", ALPHA, "--q", Q, "--seed", SEED,
             "--min-families", MIN_FAMILIES, *([] if MIXEDLM else ["--no-mixedlm"])],
            "E5__signature_analysis.log")
    with open(stamp, "w") as f:
        json.dump(inputs, f, indent=1)
R = json.load(open(f"{P['report']}/report.json"))
print(f"{len(R['runs'])} runs analysed, {len(R['rejected'])} rejected, "
      f"{len(R.get('superseded', []))} superseded, {len(R['errors'])} analyses refused")
'''

E5_REPORT_MD = r'''
## 3 · The report
Sections: runs; 1 signatures; 2 invariance (RQ1); 3 layer maps (RQ3); 4 bands; 5 ladders
and joint relabelling (RQ4); 6 kernel checks; 7 power targets; 8 sink control; 9 mass
centroid. *Read first:* §2's verdict per model after Holm across models (`invariant` /
`task-dependent` / `inconclusive`) on correct items and on all items, and whether the two
disagree — this is the gate for E4–E6 (outline, decision points).
'''

E5_REPORT = r'''
display(Markdown(open(f"{P['report']}/report.md").read()))
'''

E5_E6_MD = r'''
## 4 · E6 ladders at a glance
Top-head share change from k = 1 to k = max (all items, 40–80% depth) against model size,
per suite and model family (from the report's §5), and the post-training pairs under one
joint relabelling of k (instruct minus base: a negative difference means post-training moved
the model toward spreading). *Read with* §5 of the report, which gives the slopes and p-values.
'''

E5_E6 = r'''
import matplotlib.pyplot as plt
_sized = lambda m: (isinstance(m.get("size_b"), (int, float)) and m["size_b"] == m["size_b"]
                    and isinstance(m.get("pct_change"), (int, float)))
lad = [l for l in R.get("ladders", []) if l["attn_impl"] == KERNEL and l["precision"] == PRECISION
       and sum(map(_sized, l["models"])) >= 2]
if lad:
    fams = sorted({l["family"] for l in lad})
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 3.6), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for l in [l for l in lad if l["family"] == fam]:
            ms = [m for m in l["models"] if _sized(m)]
            ax.plot([m["size_b"] for m in ms], [m["pct_change"] for m in ms], marker="o",
                    label=l["suite"][:28])
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{fam} ladder")
        ax.set_xlabel("parameters (B)")
    axes[0][0].set_ylabel("top-head share, % change k=1 -> max")
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{P['report']}/ladders.png", dpi=150)
    plt.show()
else:
    print("no ladder with two or more sized models under the pinned configuration yet")
pairs = [{"suite": x["suite"], "pair": f"{d['b']} - {d['a']}", "diff": d["diff"], "p": d["p"],
          "n items": x["n_items"]} for x in R.get("pairs", []) for d in x.get("diffs", [])]
if pairs:
    display(pd.DataFrame(pairs))
'''

E5_CHECKS_MD = r'''
## 5 · Kernel checks, sink control, power
Kernel checks: a signature counts only where both kernels agree (`agree`). Sink control:
the signature with heads whose attention to position 0 exceeds 0.9 removed; a sign change
means the result depends on sink heads. Power: items per depth needed to detect a 2% change
of the primary statistic between k = 1 and k = max (if above the suites' 42 test items per
depth, raise `N_PER` in E1 for a follow-up run, per the outline's "inconclusive" decision
point).
'''

E5_CHECKS = r'''
if R["kernel"]:
    display(pd.DataFrame([{"model": x["model"].split("/")[-1], "suite": x["suite"],
                           "a": f"{x['a']['attn_impl']} {x['a']['precision']}",
                           "b": f"{x['b']['attn_impl']} {x['b']['precision']}", "n": x["n_items"],
                           "score_r": x["score_r"], "agree": x["agree"]} for x in R["kernel"]]))
if R["sink"]:
    display(pd.DataFrame([{"model": x["model"].split("/")[-1], "suite": x["suite"],
                           "removed (window)": x.get("frac_removed_window"),
                           "sign agree": x.get("sign_agree"), "class agree": x.get("class_agree")}
                          for x in R["sink"] if x.get("status") == "ok"]))
if R["power"]:
    display(pd.DataFrame(R["power"]))
e4_md = f"{P['e4_report']}/ablation_report.md"
if os.path.exists(e4_md):
    display(Markdown("## E4 ablation report\n\n" + open(e4_md).read()))
'''

E5_FINAL = r'''
# Outputs and the key tables: RQ1 verdicts and the signature of every run
list_outputs(P["report"])
if R["invariance"]:
    display(pd.DataFrame([{"model": x["model"].split("/")[-1],
                           "pre-registered": "funnel" if x["preregistered_sign"] > 0 else "spread",
                           "correct items": x.get("verdict_holm"),
                           "all items": x.get("verdict_all_items_holm"),
                           "pooled slope": x["correct_items"].get("pooled"),
                           "disagree": x.get("disagree")} for x in R["invariance"]]))
sig = pd.DataFrame([{"model": s["model"].split("/")[-1], "config": f"{s['precision']} {s['attn_impl']}",
                     "cell": f"{s['family']}/{s['wording']}", "suite": s["suite"],
                     "signature": s["classification"]}
                    for s in R["signatures"] if s["metric"] == "contribution"])
if len(sig):
    # one family and wording under two settings (a model in two groups): columns by suite
    _dup = sig.duplicated(["model", "config", "cell"], keep=False)
    sig.loc[_dup, "cell"] = sig.loc[_dup, "suite"]
    display(sig.pivot_table(index=["model", "config"], columns="cell", values="signature", aggfunc="first"))
'''


# ───────────────────────────── assembly ─────────────────────────────────────
def common_setup(key, token=True):
    head, _ = git_state()
    subs = {"REPO_URL": REPO_URL, "REPO_DIR": REPO_DIR, "HF_HOME": HF_HOME,
            "EMBED_LIST": json.dumps(EMBED)}
    cells = [md(SETUP_MD), code(SETUP, **subs), md(WRITEFILE_MD)]
    cells += [writefile(n) for n in EMBED]
    cells.append(code(VERIFY,
                      EXPECTED=json.dumps({n: sha256(ROOT / n) for n in EMBED}, indent=4),
                      TESTED_COMMIT=head or "unknown",
                      TESTED_DEPS=json.dumps({n: sha256(ROOT / n) for n in DEPS}, indent=4)))
    cells += [md(DRIVE_MD), code(PATHS, DRIVE_ROOT=DRIVE_ROOT)]
    if token:
        cells += [md(TOKEN_MD), code(TOKEN)]
    cells += [md(HELPERS_MD), code(HELPERS)]
    return cells


def config_md(extra=""):
    return md("## Configuration\nEverything you may want to change is in this cell." + extra)


def build():
    nbs = {}
    nbs["E3"] = notebook([
        md(E3_TITLE), *common_setup("E3"),
        config_md(), code(E3_CONFIG), md(PIN_MD), code(PIN),
        md(E3_STEP1_MD), code(E3_STEP1), md(E3_STEP2_MD), code(E3_STEP2),
        md(E3_STEP3_MD), code(E3_STEP3), md(E3_STEP4_MD), code(E3_STEP4),
        md("## Outputs\nWhat E3 wrote to Drive, the comparison, and the kernel to pin. "
           "The Hugging Face token (if any) is removed from this VM."),
        code(E3_FINAL)], "E3")
    nbs["E1"] = notebook([
        md(E1_TITLE), *common_setup("E1"),
        config_md(" The grid, the models and the comparison groups decide what E2 runs."),
        code(E1_CONFIG, MODELS=MODELS.strip("\n")), md(PIN_MD), code(PIN),
        md(E1_STEP1_MD), code(E1_STEP1), md(E1_STEP2_MD), code(E1_STEP2),
        md(E1_RUN_FN_MD), code(E1_RUN_FN),
        md(E1_STEP3_MD), code(E1_STEP3), md(E1_STEP4_MD), code(E1_STEP4),
        md(E1_STEP5_MD), code(E1_STEP5),
        md("## Outputs\nEverything E1 wrote to Drive, and per group and family the chosen "
           "setting, its status, the weakest model and its eligible depths. The Hugging Face "
           "token is removed from this VM."),
        code(E1_FINAL)], "E1")
    nbs["E2"] = notebook([
        md(E2_TITLE), *common_setup("E2"),
        config_md(" `MODELS_SMALL` / `MODELS_BIG` are the same lists as in E1; a model runs "
                  "the same fixed suites."),
        code(E2_CONFIG, MODELS=MODELS.strip("\n")),
        md("### Kernel and task suites\nEager attention (or `KERNEL_OVERRIDE`) and the fixed "
           "recommended suites, built here. *Check:* the kernel and the suites listed."),
        code(KERNEL_RESOLVE), code(MANIFEST_READ), md(PIN_MD), code(PIN),
        md(E2_PLAN_MD), code(E2_PLAN), code(E2_RUN_FN),
        md(E2_SMALL_MD), code(E2_SMALL), md(E2_BIG_MD), code(E2_BIG),
        md("## Outputs\nEverything under `E2_sweep/`, and accuracy per model and cell "
           "(constrained / strict) for both kernels, with any per-head split warnings. "
           "The Hugging Face token is removed from this VM."),
        code(E2_FINAL)], "E2")
    nbs["E4"] = notebook([
        md(E4_TITLE), *common_setup("E4"), config_md(), code(E4_CONFIG),
        md("### Kernel and task suites\n*Check:* the kernel and the suites listed."),
        code(KERNEL_RESOLVE), code(MANIFEST_READ), md(PIN_MD), code(PIN),
        md(E4_PLAN_MD), code(E4_PLAN), md(E4_SMOKE_MD), code(E4_SMOKE),
        md(E4_RUN_MD), code(E4_RUN), md(E4_ANALYSE_MD), code(E4_ANALYSE),
        md("## Outputs\nEverything under `E4_ablation/`, the mean drop per model, suite and "
           "condition, and the pair's interaction verdicts. The Hugging Face token is "
           "removed from this VM."),
        code(E4_FINAL)], "E4")
    nbs["E5"] = notebook([
        md(E5_TITLE), *common_setup("E5", token=False),
        config_md(" No model is loaded here, so no Hugging Face token is needed."),
        code(E5_CONFIG),
        md("### Kernel\nRQ1 uses one configuration per model: sdpa by default (or "
           "`KERNEL_OVERRIDE`) at `PRECISION`."),
        code(KERNEL_RESOLVE), md(PIN_MD), code(PIN),
        md(E5_INV_MD), code(E5_INV), md(E5_RUN_MD), code(E5_RUN),
        md(E5_REPORT_MD), code(E5_REPORT), md(E5_E6_MD), code(E5_E6),
        md(E5_CHECKS_MD), code(E5_CHECKS),
        md("## Outputs\nThe report files, the RQ1 verdicts and every run's signature "
           "(pinned configuration and kernel checks)."),
        code(E5_FINAL)], "E5")
    return nbs


# ───────────────────────────── README ───────────────────────────────────────
README = r'''
# Colab notebooks for the next runs (E1–E6)

Generated by `make_notebooks.py` from the tested modules in this checkout (see
`paper_manuscript/research_outline.tex`, Sections 4, 6 and 9). Do not edit the notebooks by
hand: change a module or the generator and run

    python3 colab/make_notebooks.py            # write the notebooks and this README, then validate
    python3 colab/make_notebooks.py --check    # validate what is on disk

Each notebook is self-contained: it clones `@REPO_URL@` (for `active_subgraph_dot.py`,
`deep_probe.py`, `path_analyzer.py`), installs `transformers accelerate networkx scipy
statsmodels`, writes the five modules GitHub does not have yet (`serial_tasks.py`,
`sweep_probe.py`, `signature_analysis.py`, `ablation_probe.py`, `head_alignment_probe.py`)
with `%%writefile` cells that are byte-identical to the local files, and stops unless their
SHA-256 matches and the clone's `deep_probe.py`, `active_subgraph_dot.py` and
`path_analyzer.py` are the versions they were tested with (commit `@COMMIT@`). Every output
records `NEXT_RUNS_COMMIT` (the clone's commit plus the module hashes). The modules were
tested locally with transformers 5.12.1 and torch 2.11.0; set `PIP_PACKAGES` in cell 0.1
to pin versions.

To open a notebook: in Colab, File > Upload notebook (or open it from Drive / GitHub), then
Runtime > Change runtime type as its first cell says, and run the cells in order.

## Run order

| step | notebook | what it does | needs | runtime | time (estimate) |
|---|---|---|---|---|---|
| 1 | `E3_kernel_reference.ipynb` | Qwen2.5-7B on the original ladder in fp32 (CPU) and bf16 eager / sdpa (GPU); writes the kernel to pin | – | GPU + High-RAM (L4 or A100; ≥ 40 GB RAM for fp32 7B) | 30–45 min |
| 1 | `E1_tasks_and_calibration.ipynb` | suites for 5 families × 2 wordings × 11 settings, `check_suite` for 6 tokenizer families, accuracy-only calibration of the 14 comparison-group models, one setting per family and group | – | A100 + High-RAM (40 GB is enough); the 70B cell **A100 80GB + High-RAM** | small models 2–4 h in all; 70B cell 5–7 h per model |
| 2 | `E2_pinned_sweep.ipynb` | `sweep_probe` (test split, `--pos0 --generate`) over models × the fixed suites under eager attention, plus the kernel check | – | A100 + High-RAM (80 GB for 24–27B); the 70B cell **A100 80GB + High-RAM** | small models 6–10 h in all; 70B cell 12–15 h per model per 10 suites (more if its ladder's chosen suites differ from the pair's) |
| 3 | `E4_ablation.ipynb` | head ablation on Llama-3.1-70B and Qwen2.5-72B, then `--analyse` on the pair | – (optionally E5 to pick families) | **A100 80GB + High-RAM** | ~2 h per suite per model; defaults ~16 h |
| 4 | `E5_E6_analysis.ipynb` | `signature_analysis` over all captures: signatures, RQ1 invariance, layer maps, bands, ladders, kernel checks, sink control, power; shows the E4 report | E2 (E3) | CPU | 10–30 min |

E1, E2, E3 and E4 are independent and can run in any order, in separate runtimes. E2 and E4
use fixed recommended task settings (a 10-cycle pointer chase with k = 1–5, mod-7
arithmetic, 3-chain binding and relational lookup, the numeral control; 3 worked examples;
60 items per depth, 42 in the test split) and sdpa attention (E3 on Qwen2.5-7B: bf16
sdpa matches fp32, bf16 eager does not); E1 and E3 report whether those choices hold up. E5/E6 analyses whatever E2 (and E3,
E4) have written; run it after E2 and again after E4.

Times are rough: a 70B-class model does not fit on one 80 GB GPU, so `load_model` offloads
the rest to CPU RAM and every forward streams it (~8 s per accuracy-only item, ~20 s per
capture item with generation, ~50 s per ablation measurement item). `run_cmd` prints the
measured s/item and the time left as it goes. Long cells are resumable: after a disconnect,
re-run the notebook from the top; finished outputs are skipped (`sweep_probe` per suite,
`ablation_probe` per suite plus an in-suite checkpoint), and an existing output written
with other settings or another code version stops the run rather than mixing versions.
Before a long loop each run cell checks that every model with work left can be downloaded
with your token (a licence not accepted stops it at once), and the 70B-class cells first
check the runtime and run a one-minute preflight on the smallest model with the same
flags. A model's weights are kept on the VM after an interrupt (re-running reuses them)
and deleted otherwise. The two 70B-class models can run at the same time in two runtimes
(set `MODELS_BIG`, or `MODELS` in E4, to a different model in each; never the same model
in two sessions: both would write the same files).

## Models

Small-model cells, smallest first: Qwen2.5-0.5B, Llama-3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B,
Llama-3.2-3B (+Instruct), Mistral-7B-v0.1, OLMo-2-1124-7B base / SFT / DPO / Instruct,
Qwen2.5-7B (+Instruct), Llama-3.1-8B (+Instruct), Qwen2.5-14B (40 GB GPU),
Mistral-Small-24B-Base-2501 and Gemma-2-27B (80 GB GPU). 70B cell: Llama-3.1-70B,
Qwen2.5-72B. E1's comparison groups: the matched pair; the Qwen2.5 ladder (0.5B–72B); the
Llama ladder (3.2-1B, 3.2-3B, 3.1-8B, 3.1-70B); the OLMo-2 post-training ladder. E1
calibrates only these group models (`CALIB_ONLY_GROUPS`), since the choice reads nothing
else. In E2 the Instruct models run their base ladder's suites and the further families
(Mistral, Gemma) the pair's (`EXTRA_GROUPS`), so every comparison is on shared items.

Before you start: accept the licences of Llama 3.1 / 3.2 and Gemma 2 on huggingface.co.
The token is typed at a `getpass` prompt into `os.environ["HF_TOKEN"]` only; it is never
printed or saved to Drive (`load_model` logs in with it, which stores a copy in
`/content/hf` on the VM's local disk; the last cell of each notebook that asks for it
deletes it). Weights also go to `/content/hf` and are deleted after each model.

## Where the outputs go (Google Drive)

    MyDrive/next_runs/
      E1_suites/grid/*.json              every family x wording x grid setting (E1)
      E1_suites/check_suite.json         check_suite per suite and tokenizer (E1)
      E1_calib/<model>__<suite>.npz      accuracy-only calibration captures (E1)
      E1_choice/groups.json, choose_setting.md/.json   the setting choice (E1)
      E1_suites/chosen/*.json            the setting E1 would choose (reference only)
      fixed_suites/*.json                the fixed suites E2 and E4 run (test split)
      E1_suites/chosen_suites.json       manifest: group -> family -> status, setting, suites
      E3_kernel_ref/ladder.json, fp32_eager/, bf16_eager/, bf16_sdpa/, report/
      E3_kernel_ref/kernel_choice.json   the kernel closer to fp32 (reference only)
      E2_sweep/bf16_<kernel>/<model>__<suite>.npz        the pinned sweep (E2)
      E2_sweep/bf16_<other kernel>/<model>__<suite>.npz  the kernel check (E2)
      E4_ablation/bf16_<kernel>/<model>__<suite>__ablation.npz, smoke/, report/
      E5_E6_report/report.md, report.json, inputs.json, ladders.png
      logs/                              the full output of every run

Every notebook has the same paths cell (`DRIVE_ROOT`); change it identically in all five to
use another folder. (The interim notebooks in `colab_now/` write to `next_runs/E3_kernel` and
`next_runs/pointer_per_item`; these notebooks do not read or write those folders.)

## What to check

Each step's markdown cell says what to look for in its output. The essentials:
E3 prints `KERNEL TO PIN`; E1's check table must be all `ok` and step 5 shows the status per
group and family (`ok`, `fallback`, `override`, `missing`); every capture prints
`checks (max relative error)` with no WARNING and accuracy by k; E5's §2 gives the RQ1
verdict per model (correct items and all items, after Holm across models).

Decision points (outline): if the matched pair cannot share a passing setting on at least 3
families, restrict the invariance claims to what passes; if RQ1 is task-dependent, run E4 on
the families where the pair's signatures differ; if inconclusive, raise items per depth to
the power target (E5 §7) before re-scoping; an E4 null is claimed only if the 90% interval of
the interaction lies within ±10% of the clean logit difference.

## Known limits

- The prepended-`<|endoftext|>` re-run of Qwen2.5 (outline, BOS control) is not in these
  notebooks: `sweep_probe.py` has no option to prepend a token.
- With sdpa pinned (the default), the pinned sweep cannot record attention to position 0
  (sdpa returns no attention weights); the sink control then covers only the eager
  kernel-check runs, whose bf16 numbers are unreliable for Qwen (E3).
- `head_alignment_probe.py` is written only because `sweep_probe` and `ablation_probe` import
  it; its own CLI reads the old single-prompt suites, so the post-norm write for OLMo-2 and
  Gemma-2 comes from `sweep_probe`'s `write_norm`.

@OLD@
'''


def readme():
    head, _ = git_state()
    old = sorted(p.name for p in HERE.glob("*.ipynb") if p.name not in NB.values())
    old_txt = ("## Other notebooks in this folder\n\n" + ", ".join(f"`{n}`" for n in old) +
               " come from an earlier version of this generator (two or three notebooks for "
               "E1 + E2). They are superseded by the five above and are not regenerated.\n"
               if old else "")
    return (README.strip("\n").replace("@REPO_URL@", REPO_URL)
            .replace("@COMMIT@", (head or "unknown")[:12]).replace("@OLD@", old_txt).rstrip()
            + "\n")


# ───────────────────────────── validation ───────────────────────────────────
_MAGIC = re.compile(r"^(\s*)[!%]")


def _strip_magics(src):
    """IPython magics and shell escapes -> `pass` at the same indentation, so the rest
    of the cell compiles as plain Python."""
    return "".join((m.group(1) + "pass\n") if (m := _MAGIC.match(line)) else line
                   for line in src.splitlines(True))


def _writefile_body(src):
    """(file name, the text IPython passes to %%writefile) for a %%writefile cell."""
    first, _, body = src.partition("\n")
    name = first.split(None, 1)[1].strip()
    return name, (body if body.endswith("\n") else body + "\n")


def validate(nbs_text, log=print):
    """Check the notebooks (name -> JSON text).  Returns a list of problems."""
    problems = []
    try:
        import nbformat
    except ImportError:
        nbformat = None
        log("  nbformat not installed: schema validation skipped")
    try:
        from IPython.core.inputtransformer2 import TransformerManager
        tm = TransformerManager()
    except ImportError:
        tm = None
        log("  IPython not installed: IPython-transform checks skipped")
    local = {n: (ROOT / n).read_bytes() for n in EMBED}
    paths_src, keys_defined, uses = {}, None, {}
    for name, text in nbs_text.items():
        try:
            nb = json.loads(text)
        except json.JSONDecodeError as e:
            problems.append(f"{name}: not JSON ({e})")
            continue
        if nbformat is not None:
            try:
                nbformat.validate(nbformat.reads(text, as_version=4))
            except Exception as e:
                problems.append(f"{name}: nbformat.validate: {type(e).__name__}: {e}")
        key = next(k for k, v in NB.items() if v == name)
        acc = RUNTIME[key][0]
        if nb["metadata"].get("accelerator") != acc:
            problems.append(f"{name}: accelerator {nb['metadata'].get('accelerator')!r}, want {acc!r}")
        written = {}
        all_src = []
        for i, c in enumerate(nb["cells"]):
            src = "".join(c["source"])
            if c["cell_type"] != "code":
                continue
            where = f"{name} cell {i}"
            if not src.startswith("%%writefile"):
                all_src.append(src)          # the notebook's own code, not the embedded modules
            if src.startswith("%%writefile"):
                fname, body = _writefile_body(src)
                written[fname] = written.get(fname, 0) + 1
                if fname not in local:
                    problems.append(f"{where}: writes {fname}, which is not an embedded module")
                elif body.encode("utf-8") != local[fname]:
                    problems.append(f"{where}: %%writefile {fname} differs from the local file")
                if tm is not None:
                    out = tm.transform_cell(src)
                    call = ast.parse(out).body[0].value
                    if not (isinstance(call, ast.Call) and len(call.args) == 3
                            and ast.literal_eval(call.args[0]) == "writefile"
                            and ast.literal_eval(call.args[1]) == fname
                            and ast.literal_eval(call.args[2]).encode("utf-8") == local.get(fname)):
                        problems.append(f"{where}: IPython would not write {fname} byte for byte")
                try:
                    compile(body, f"{fname} (embedded)", "exec")
                except SyntaxError as e:
                    problems.append(f"{where}: embedded {fname} does not compile: {e}")
                continue
            try:
                compile(_strip_magics(src), where, "exec")
            except SyntaxError as e:
                problems.append(f"{where}: does not compile after stripping magics: {e}\n{src[:300]}")
            if tm is not None:
                try:
                    compile(tm.transform_cell(src), where + " (IPython)", "exec")
                except SyntaxError as e:
                    problems.append(f"{where}: does not compile after IPython's transform: {e}")
            if src.lstrip().startswith("# 0.4 Google Drive"):
                paths_src[name] = src
            for line in src.splitlines():
                if re.search(r"\bprint\(.*HF_TOKEN", line) or re.search(r"\bprint\(.*\b_t\b", line):
                    problems.append(f"{where}: a print may show the token: {line.strip()}")
            if "HF_TOKEN" in src and "getpass" not in src and "forget_token" not in src \
                    and "os.environ.pop(\"HF_TOKEN\"" not in src:
                problems.append(f"{where}: uses HF_TOKEN outside the getpass cell and forget_token")
        for fname in EMBED:
            if written.get(fname) != 1:
                problems.append(f"{name}: {written.get(fname, 0)} %%writefile cells for {fname}, want 1")
        joined = "\n".join(all_src)
        uses[name] = (set(re.findall(r"""P\[["'](\w+)["']\]""", joined)), joined)
        for need in ("NEXT_RUNS_COMMIT", "ASD_FORCE_STRATEGY", "ASD_FORCE_PRECISION", "ASD_ATTN_IMPL",
                     "drive.mount", REPO_URL, "pip install -q -U"):
            if need not in joined:
                problems.append(f"{name}: never mentions {need!r}")
        if key != "E5" and "getpass(" not in joined:
            problems.append(f"{name}: no getpass token prompt")
    # one paths cell, identical everywhere, defining every key used
    if len(paths_src) != len(nbs_text):
        problems.append(f"paths cell found in {sorted(paths_src)} only")
    if len(set(paths_src.values())) > 1:
        problems.append("the paths cell differs between notebooks")
    if paths_src:
        keys_defined = set(re.findall(r'^\s*"(\w+)":\s*f"', next(iter(paths_src.values())), re.M))
        for name, (used, _) in uses.items():
            extra = used - keys_defined
            if extra:
                problems.append(f"{name}: uses P keys {sorted(extra)} not defined in the paths cell")
    # hand-offs: writer and readers of each file
    flows = {"chosen_manifest": ("E1", []), "suites_chosen": ("E1", []),
             "suites_fixed": ("E2", ["E4"]), "kernel_choice": ("E3", []), "e2": ("E2", ["E5"]),
             "e3": ("E3", ["E5"]), "e4_report": ("E4", ["E5"])}
    for k, (w, rs) in flows.items():
        for who in [w] + rs:
            nm = NB[who]
            if nm in uses and k not in uses[nm][0]:
                problems.append(f"{nm}: does not use P[{k!r}] (hand-off {w} -> {rs})")
    for who, pat in (("E1", r"json\.dump\(manifest"), ("E3", r"json\.dump\(choice")):
        if NB[who] in uses and not re.search(pat, uses[NB[who]][1]):
            problems.append(f"{NB[who]}: does not write its hand-off file")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate the notebooks on disk only")
    a = ap.parse_args(argv)
    if a.check:
        texts = {n: (HERE / n).read_text(encoding="utf-8") for n in NB.values()}
    else:
        head, warn = git_state()
        for w in warn:
            print("  WARNING:", w)
        texts = {NB[k]: json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
                 for k, nb in build().items()}
        problems = validate(texts)
        if problems:
            print("\n".join(f"  PROBLEM: {p}" for p in problems))
            raise SystemExit(f"{len(problems)} problem(s); nothing written")
        for n, t in texts.items():
            (HERE / n).write_text(t, encoding="utf-8")
            print(f"  -> {HERE / n}  ({len(t.encode()) / 1e3:.0f} kB)")
        (HERE / "README.md").write_text(readme(), encoding="utf-8")
        print(f"  -> {HERE / 'README.md'}")
    problems = validate(texts)
    for p in problems:
        print("  PROBLEM:", p)
    print(f"validated {len(texts)} notebooks: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

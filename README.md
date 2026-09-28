# How Language Models Differ in Redistributing Attention-Head Activity Under Serial Demand

Code for the paper by Johnny Jingze Li, Abdulla Kuleib, Kalyan Basu and Gabriel A. Silva.

The way a model distributes activity over each layer's attention heads offers a coarse view of how it
routes information through depth; how this changes with the task is part of what a mechanistic account
must explain. Holding prompt length fixed, we vary how many serial steps a task demands and measure, in
every layer of 17 open-weight models, whether activity concentrates on a few heads or spreads across many
as demand rises. Both occur: in most models, layers just before mid-depth concentrate activity and later
layers spread it. Models differ in where and how strongly this happens. The Qwen2.5 base models from 0.5B
to 7B, for example, spread less than the average model in every task and concentrate activity in parts of
their second half, where Llama models from 1B to 8B and OLMo-2 spread; the contrast largely holds between
Llama-3.1-70B and Qwen2.5-72B, which have the same number of layers and heads. These differences are
reproducible, and post-trained models keep much of their base model's pattern. An ablation study suggests
that, within a task, models whose activity is more concentrated on their top heads also depend more on
those heads for the answer. Concentration and spreading across layers thus offer a new way to compare
models, by how they route information through depth.

## Citation

```bibtex
@article{li2026serialdemand,
  title   = {How Language Models Differ in Redistributing Attention-Head Activity Under Serial Demand},
  author  = {Li, Johnny Jingze and Kuleib, Abdulla and Basu, Kalyan and Silva, Gabriel A.},
  journal = {arXiv preprint (to appear)},
  year    = {2026}
}
```

License: MIT (see `LICENSE`).

## Repository layout

| folder | contents |
|---|---|
| `tasks_and_capture/` | task generators (`serial_tasks.py`), per-head capture (`sweep_probe.py`), head ablation (`ablation_probe.py`, `ablation_window.py`), write-norm alignment (`head_alignment_probe.py`), model/tokenizer compatibility check (`check_compat.py`), and their dependencies (`deep_probe.py`, `active_subgraph_dot.py`, `path_analyzer.py`) |
| `notebooks/` | Colab notebooks for the GPU runs and the generators that write them (`make_notebooks.py` for E1–E4, `make_abwin_notebook.py` for E7); `notebooks/README.md` is the run guide written by the generator |
| `replication/` | replication on fresh items (E2b): `e2b.py`, its notebook and generator |
| `analysis/` | main analysis pipeline: per-layer response, profiles and signatures, decomposition, robustness, shared component, post-training and size trends (`trends/`), vetting (`vet3/`) and review checks (`verify*/`) |
| `analysis/replication/` | analyses of the replication (`appendix/` produces its tables and figure) |
| `analysis/ablation/` | 17-model ablation (E7): statistics and figure |
| `analysis/accuracy/` | accuracy and correctness appendix: item table (`setup/`), independent verification (`verify_setup/`), correctness analyses (`correctness/`) |
| `figures/` | figure scripts that are not part of `analysis/` |
| `checks/` | 70B matched pair and prompt-position checks |
| `paths.py` | the single place where data locations are set (see below) |

## Requirements

Python 3.10 or newer (analysis was run with Python 3.12).

* GPU runs (notebooks): `torch` 2.11.0 and `transformers` 5.12.1 (the versions the capture modules were
  tested with; see `notebooks/README.md`), plus `accelerate`, `networkx`, `scipy`, `statsmodels`.
  Llama 3.1/3.2 need an accepted licence on the Hugging Face Hub; the notebooks ask for a token at a
  `getpass` prompt and never store it.
* Analysis (CPU): `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`.

The Colab notebooks clone `https://github.com/JohnnyPunchinello/LLM_path_analysis.git` for
`deep_probe.py`, `active_subgraph_dot.py` and `path_analyzer.py` (copies of the tested versions are in
`tasks_and_capture/`), write the other capture modules into the clone with `%%writefile` cells that are
byte-identical to `tasks_and_capture/`, and stop unless the SHA-256 of every module matches.

## Data

Raw captures are **not** included (tens of GB of `.npz` files). All scripts find them through `paths.py`,
which reads the environment variable `SDH_DATA` (default `./data`):

```
$SDH_DATA/
  E2_sweep/bf16_eager/<model>__<suite>.npz        main run (E2), eager attention, 17 models x 5 families x 2 wordings
  E2_sweep/bf16_sdpa/<model>__<suite>.npz         main run under SDPA (kernel check; Qwen2.5-7B-Instruct's primary run)
  E4_ablation/bf16_sdpa/<model>__<suite>__ablation.npz   Llama-3.1-70B / Qwen2.5-72B ablation (E4)
  E7_ablation_window/bf16_sdpa/<model>__<suite>__abwin.npz  17-model ablation (E7)
  e2b/items.json, e2b/<model>/<mode>__<family>.npz          replication on fresh items (E2b)
```

These are the folders the notebooks write under `MyDrive/next_runs/` (E2, E4, E7) and `MyDrive/e2b/` (E2b)
on Google Drive; copy them into `$SDH_DATA` with the names above.

Scripts write their outputs (JSON, CSV, PDF) next to themselves, as in the original pipeline, where later
steps read the outputs of earlier ones from the same folder. These outputs are ignored by git.

## Pipeline

Commands are run from the repository root unless stated otherwise. `analysis` as the first argument is
the pipeline folder (scripts import `e2lib.py`, `model_diff.py` and `vet3/v2lib.py` from it and write
their outputs there).

### 1. Capture (GPU, Colab)

| notebook | run | paper |
|---|---|---|
| `notebooks/E3_kernel_reference.ipynb` | Qwen2.5-7B in fp32 and bf16 eager / SDPA: choice of attention kernel | kernel note (Section 3.3, Appendix on robustness) |
| `notebooks/E1_tasks_and_calibration.ipynb` | task suites, tokenizer checks, accuracy-only calibration | task settings (Section 3.1, Appendix A) |
| `notebooks/E2_pinned_sweep.ipynb` | main run: per-head activity at the answer position, 17 models, fixed suites, eager (+ SDPA check) | all main results |
| `notebooks/E4_ablation.ipynb` | head ablation on Llama-3.1-70B and Qwen2.5-72B; its unablated passes give the 70B profiles | Appendix on the 70B pair (Table 6, Figure 6) |
| `notebooks/E7_ablation_17models.ipynb` (or the `_A` / `_B` halves in two runtimes) | top-10% head ablation at 40–80% of depth, 17 models | Appendix on ablation (Table 8, Figure 5) |
| `replication/E2b_accuracy_first.ipynb` | replication on fresh items, eight models, direct and stepwise modes | Appendix on replication (Tables 4–5, Figures 4 and 10) |

To regenerate the notebooks after changing a module: `python3 notebooks/make_notebooks.py` (E1–E4; it
also writes the analysis notebook E5, which the paper does not use and which is not included here) and
`python3 notebooks/make_abwin_notebook.py` (E7, built from E4); `python3 replication/make_notebook.py` for E2b.

### 2. Main analysis (CPU)

Set `E2_CLEAN=1` (OLMo-2 item exclusion) for every step of the main run.

```bash
export E2_CLEAN=1
# per-layer response z of each statistic, wording-averaged items (Qwen2.5-7B-Instruct under SDPA)
python3 analysis/consist.py analysis analysis/consist.json q10,q25,q50,entropy,top1
python3 analysis/align_null.py analysis analysis/consist.json analysis/align_null.json
E2_STAT=q10 VQ=.10 PYTHONHASHSEED=0 python3 analysis/backhalf.py analysis analysis/backhalf_q10.json
# per-cell caches and per-wording tables (run inside analysis/vet3; VPRIM=y10 selects the top-10% share)
(cd analysis/vet3 && export VPRIM=y10 && python3 build.py && python3 repro.py && python3 layertab.py \
   && python3 sens.py && python3 splits.py && python3 altexp.py)
# consistently responding layers, shared profile, numeral control
python3 analysis/conc_q10.py analysis
# model signatures, associations, robustness, review checks
python3 analysis/model_diff.py analysis
python3 analysis/model_diff2.py analysis
python3 analysis/model_diff3.py analysis
python3 analysis/model_diff4.py analysis
python3 analysis/model_diff5.py analysis
# post-training and other trends
python3 analysis/trends/posttrain_trend.py
python3 analysis/trends/other_trends.py
```

The remaining scripts in `analysis/` (`layers.py`, `decomp.py`, `prep.py`, `pairs.py`, `outliers.py`,
`mech.py`, `conc.py`, `ident.py`, `hops.py`, `cw_layers.py`, `decode*.py`, `regions.py`) are called as
`python3 analysis/<script>.py analysis <output.json>` and give supporting numbers (see each docstring).
`analysis/vet3/` (effect sizes, item-level model, depth regression, mass/floor alternative, dependence),
`analysis/trends/indep/`, `analysis/trends/verify_other/`, `analysis/trends/review_*` and
`analysis/verify*/` are independent re-derivations and review checks of numbers in the paper; several of
them are run from inside their own folder.

### 3. Replication, ablation, accuracy

```bash
# replication (E2b): tables and figure of the appendix, then the supporting analyses
python3 analysis/replication/appendix/replication_numbers.py
python3 analysis/replication/appendix/replication_features.py
python3 analysis/replication/appendix/replication_extra.py
python3 analysis/replication/appendix/verify_design.py
python3 analysis/replication/appendix/fig_replication.py
python3 analysis/replication/e2b_direct_replication.py        # and the other analysis/replication/*.py
# 17-model ablation (E7)
python3 analysis/ablation/e7_analysis.py && python3 analysis/ablation/e7_fig.py
# 70B matched pair
python3 checks/large_pair.py
# accuracy and correctness
python3 analysis/accuracy/setup/load_items.py && python3 analysis/accuracy/setup/setup.py
python3 analysis/accuracy/verify_setup/v_load.py              # then v1_prompts.py ... v8_copy_test.py
(cd analysis/accuracy/correctness && python3 build_cache.py && for s in cw cw_sens cw_bydepth decode decode_parts \
   altexp_check subsets kge2_ref interact setup_checks assemble; do python3 $s.py; done)
```

### 4. Figures

```bash
python3 figures/prepare_fig_data.py                  # copies consist.json (q10) and model_diff.json into figures/
python3 figures/make_fig1_relational_contrast.py
python3 figures/make_figM2_signatures.py
python3 figures/figS_data/large_pair_layers.py && python3 figures/make_figS_spreading_window.py
python3 figures/make_figA_accuracy_e2b.py
python3 analysis/model_figs.py analysis figures      # figM1, figM3, figP (and an earlier figM2)
PAPER=1 E2_STAT=q10 python3 analysis/plots4.py analysis figures   # fig17 layer maps
```

## Paper figures and tables

| paper | label | produced by |
|---|---|---|
| Figure 1 (relational contrast) | `fig:contrast` | `figures/make_fig1_relational_contrast.py` (reads `figures/fig1_data/consist_q10.json` and the E2 captures) |
| Figure 2 (signatures) | `fig:signatures` | `figures/make_figM2_signatures.py` (`analysis/model_diff.py` via `figures/prepare_fig_data.py`) |
| Figure 3 (axes of difference) | `fig:axes` | `analysis/model_figs.py` (`figM3_axes.pdf`; `model_diff.py`, `model_diff2.py`) |
| Figure 4 (replication signatures) | `fig:replication` | `analysis/replication/appendix/fig_replication.py` (after `replication_numbers.py`) |
| Figure 5 (ablation) | `fig:ablation` | `analysis/ablation/e7_fig.py` (after `e7_analysis.py`) |
| Figure 6 (spreading window) | `fig:window` | `figures/make_figS_spreading_window.py` (after `figures/figS_data/large_pair_layers.py`) |
| Figure 7 (model profiles) | `fig:profiles` | `analysis/model_figs.py` (`figM1_model_profiles.pdf`) |
| Figure 8 (layer-by-task maps) | `fig:layermaps` | `analysis/plots4.py` with `PAPER=1 E2_STAT=q10` (`fig17_layer_maps_by_task.pdf`) |
| Figure 9 (post-training) | `fig:posttrain` | `analysis/model_figs.py` (`figP_posttraining.pdf`; needs `analysis/trends/posttrain_trend.json`) |
| Figure 10 (accuracy on fresh items) | `fig:acc` | `figures/make_figA_accuracy_e2b.py` |
| Table 1 (task families) | `tab:tasks` | `tasks_and_capture/serial_tasks.py`; answers re-derived by `analysis/solve.py` and `analysis/accuracy/verify_setup/v1_prompts.py` (`v_parse.py`) |
| Table 2 (models) | `tab:models` | model lists in the notebooks and `analysis/e2lib.py`; L and H are read from the captures |
| Table 3 (accuracy) | `tab:acc` | `analysis/accuracy/setup/setup.py` (after `load_items.py`); checked by `analysis/accuracy/verify_setup/v3_acc.py` |
| Table 4 (replication) | `tab:replication` | `analysis/replication/appendix/replication_numbers.py` |
| Table 5 (replication features) | `tab:replication-features` | `analysis/replication/appendix/replication_features.py`, `replication_extra.py` |
| Table 6 (70B matched pair) | `tab:large` | `checks/large_pair.py` |
| Table 7 (robustness) | `tab:robust` | `analysis/model_diff4.py` (inputs: `consist.json`, `vet3/repro.py`, `vet3/sens.py`, `model_diff3.py` / `vet3/altexp.py`); adjusted rows re-derived by `analysis/accuracy/correctness/altexp_check.py` |
| Table 8 (ablation variants) | `tab:ablation` | `analysis/ablation/e7_analysis.py` |
| Table 9 (consistent layers) | `tab:layers` | `analysis/conc_q10.py` (`consist.json`, `vet3/layertab.csv`) |

Numbers in the text: signatures, decomposition, label permutation, split-half and similarity
(`analysis/model_diff.py`); item-split reliability and task-pair agreement (`analysis/model_diff5.py`);
associations with accuracy, entropy and head count (`analysis/model_diff2.py`, `analysis/trends/`);
shared profile and per-step size (`analysis/conc_q10.py`, `analysis/vet3/effsize.py`,
`analysis/vet3/itemlevel.py`, `analysis/vet3/altexp.py`); post-training (`analysis/trends/posttrain_trend.py`,
`analysis/pairs.py`); correctness at fixed depth (`analysis/accuracy/correctness/`); replication
details (`analysis/replication/`); prompt positions (`checks/positions.py`, `checks/positions2.py`).

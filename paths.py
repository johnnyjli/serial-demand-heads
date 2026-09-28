"""Data locations shared by every analysis, check and figure script in this repository.

The raw captures are not part of the repository.  Point SDH_DATA at the folder that holds them
(default: ./data next to this file); the expected layout is described in README.md:

    $SDH_DATA/E2_sweep/bf16_eager/*.npz        main run (E2), eager attention
    $SDH_DATA/E2_sweep/bf16_sdpa/*.npz         main run, SDPA attention (kernel check)
    $SDH_DATA/E4_ablation/bf16_sdpa/*.npz      70B-pair ablation (E4); its unablated passes give the 70B profiles
    $SDH_DATA/E7_ablation_window/bf16_sdpa/*.npz   17-model ablation (E7)
    $SDH_DATA/e2b/                             replication on fresh items (items.json + <model>/<mode>__<family>.npz)

Analysis scripts write their outputs (JSON / CSV / PDF) next to themselves, as in the original
pipeline, where later scripts read the outputs of earlier ones from the same folder; these
outputs are ignored by git.  SDH_RESULTS is read here for convenience only (default ./results).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SDH_DATA", REPO / "data")).expanduser().resolve()
RESULTS = Path(os.environ.get("SDH_RESULTS", REPO / "results")).expanduser().resolve()

CAPTURE = REPO / "tasks_and_capture"        # serial_tasks.py, sweep_probe.py, ablation_window.py, ...
ANALYSIS = REPO / "analysis"                # main analysis pipeline (imports e2lib, model_diff, vet3/v2lib)
REPLICATION = REPO / "replication"          # e2b.py (replication run and its loader)

E2_SWEEP = DATA / "E2_sweep"
E4_ABLATION = DATA / "E4_ablation"
E7_ABLATION = DATA / "E7_ablation_window"
E2B = DATA / "e2b"

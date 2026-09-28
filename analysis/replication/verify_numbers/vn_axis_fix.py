"""Spearman with average ranks for ties (heads per layer), and PC1 vs the paper's accuracy feature (model_diff2.json)."""
import json, numpy as np
from vn_lib import *
J = json.load(open(f"{HERE}/vn_axis_post.json")); MD2 = json.load(open(f"{SP8}/model_diff2.json"))["features"]
out = {}
for tag, models in (("M8", M8), ("M6", M6)):
    pB = [J[tag]["proj"][m] for m in models]; paper = [J[tag]["paper"][m] for m in models]
    H = [CACHE[(m, "direct", "pointer")]["H"] for m in models]
    out[tag] = {"rho_proj_H": exact_perm_corr(pB, H, spearman=True), "rho_proj_acc": exact_perm_corr(pB, [J[tag]["E2b_strict_acc"][m] for m in models], spearman=True),
                "rho_paperPC1_paperacc": exact_perm_corr(paper, [MD2["acc"][m] for m in models], spearman=True),
                "rho_paperPC1_myE2acc": exact_perm_corr(paper, [J[tag]["E2_acc"][m] for m in models], spearman=True),
                "spearman_proj_paper": exact_perm_corr(pB, paper, spearman=True)}
    print(tag, {k: np.round(v, 3).tolist() for k, v in out[tag].items()})
json.dump(out, open(f"{HERE}/vn_axis_fix.json", "w"), indent=1)

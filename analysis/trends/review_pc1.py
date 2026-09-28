"""PC1 circularity check: refit signature PCA on the 11 base models only, project post-trained models."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP); import model_diff as md; md.SP = SP
from model_diff import ORDER, from_consist, decompose
MD = json.load(open(f"{SP}/model_diff.json")); Z = from_consist("q10"); a = decompose(Z)[1]
bi = [i for i, m in enumerate(ORDER) if not any(s in m for s in ("Instruct", "SFT", "DPO"))]
ab = a[bi]; U, s, Vt = np.linalg.svd(ab - ab.mean(0), full_matrices=False); v = Vt[0]
v = v if np.corrcoef(v, MD["pca"]["loadings"][0])[0, 1] > 0 else -v
pr = {m: float((a[i] - ab.mean(0)) @ v) for i, m in enumerate(ORDER)}
P = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
     ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
print("base-only PC1 shift", [round(pr[p] - pr[b], 2) for b, p in P], "loading r with all-17 PC1", round(float(np.corrcoef(v, MD["pca"]["loadings"][0])[0, 1]), 3))

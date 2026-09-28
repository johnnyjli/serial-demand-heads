import sys, json, numpy as np
sys.path.insert(0, "..")
import e2lib
idx = e2lib.index("eager")
f = idx[("Llama-3.2-1B","pointer/w0")]
z = np.load(f, allow_pickle=False)
m = json.loads(str(z["meta"]))
for key in ["setting","load_info","dependency_sha256","head_layout","checks","constrained_ties","measured_tokens","appended_forms","pos0_tokens","n_tokens_by_k","timing","cli","post_norm_kind"]:
    print(key, json.dumps(m[key])[:1500])

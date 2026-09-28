"""Chance baseline for '11 of 14 E2b consistently spreading layers are also consistent in E2'.
Uniform: E2b layer placed at random among the model's L layers; back-half: at random among back-half layers."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(OUT, "..", "e2b_direct_layers.json")))
exp_u = exp_b = 0.0; obs = 0; rows = {}
for m, v in d.items():
    if m == "_totals": continue
    L = v["L"]; e2 = [int(s.split()[0]) for s in v["E2_spread"]]; eb = [int(s.split()[0]) for s in v["E2b_spread"]]
    back = [l for l in range(L) if l / (L - 1) >= 0.5]; e2back = [l for l in e2 if l in back]
    exp_u += len(eb) * len(e2) / L; exp_b += len(eb) * len(e2back) / len(back)
    obs += len(set(eb) & set(e2))
    rows[m] = {"E2": e2, "E2b": eb, "recur": sorted(set(eb) & set(e2)), "E2_kept_frac": f"{len(set(eb) & set(e2))}/{len(e2)}"}
out = {"observed": obs, "expected_uniform": round(exp_u, 2), "expected_backhalf": round(exp_b, 2), "per_model": rows}
json.dump(out, open(os.path.join(OUT, "skeptic_layer_overlap.json"), "w"), indent=1); print(json.dumps(out, indent=1))

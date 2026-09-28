import json, numpy as np, sys
J = json.load(open("stepwise_vs_direct.json"))
S = J["summary"]; P = J["paired"]
stats = sys.argv[1:] or ["q10"]
for stat in stats:
  for sub in [("direct","all"),("stepwise","all"),("direct","matched"),("direct","correct"),("stepwise","correct"),("stepwise","trace_exact")]:
    for sname in ("all8","no_qwen7"):
        s = S[f"{stat}|{sub[0]}|{sub[1]}|{sname}"]
        if s is None: print(stat, sub, sname, "none"); continue
        print(f"[{stat} {sub[0]:8s} {sub[1]:11s} {sname:8s}] cells {s['n_cells']} back<0 {s['n_back_neg']} sigspread {s['n_back_sig_spread']} sigconc {s['n_back_sig_conc']} "
              f"median z {s['median_back_z']:+.2f} mean z {s['mean_back_z']:+.2f} | med slope {s['median_back_slope_pp']:+.3f}pp mean {s['mean_back_slope_pp']:+.3f}pp | "
              f"layers sig spread {s['frac_back_layers_sig_spread']:.2f} conc {s['frac_back_layers_sig_conc']:.2f}")
        print("     deciles(models):", " ".join(f"{v:+.2f}" for v in s["decile_mean_over_models"]))
        print("     CI lo          :", " ".join(f"{c[0]:+.2f}" for c in s["decile_ci_models"]))
        print("     CI hi          :", " ".join(f"{c[1]:+.2f}" for c in s["decile_ci_models"]))
        print("     deciles(lineag):", " ".join(f"{v:+.2f}" for v in s["decile_mean_over_lineages"]))
        print(f"     dec4>dec5 {s['n_models_dec4_gt_dec5']}/{len(s['per_model_decile'])} dec4>0 {s['n_models_dec4_pos']} 50-80<0 {s['n_models_50_80_neg']} mean50-80 {s['mean_50_80']:+.2f} CI {s['ci_50_80_models'][0]:+.2f},{s['ci_50_80_models'][1]:+.2f}")
        print("     per family mean back z:", {f: round(v,2) for f,v in s["per_family_mean_back_z"].items()}, "neg", s["per_family_n_back_neg"], "sig", s["per_family_n_sig_spread"])
        print("     per model back z:", {m[:14]: round(v,2) for m,v in s["per_model_back_z"].items()})

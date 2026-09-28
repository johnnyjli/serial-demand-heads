"""Copy the pipeline outputs that the figure scripts read into figures/fig1_data and figures/fig2_data.

    fig1_data/consist_q10.json       the stat == "q10" records of analysis/consist.json   (Figures 1 and 6)
    fig2_data/model_diff_primary.json  primary Z and mu, and the signature similarity R, of analysis/model_diff.json (Figure 2)

Run after analysis/consist.py and analysis/model_diff.py:  python3 figures/prepare_fig_data.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(os.path.dirname(HERE), "analysis")
os.makedirs(f"{HERE}/fig1_data", exist_ok=True); os.makedirs(f"{HERE}/fig2_data", exist_ok=True)
C = [r for r in json.load(open(f"{AN}/consist.json")) if r["stat"] == "q10"]
json.dump(C, open(f"{HERE}/fig1_data/consist_q10.json", "w"))
MD = json.load(open(f"{AN}/model_diff.json"))
json.dump({"primary": {"Z": MD["primary"]["Z"], "mu": MD["primary"]["mu"]}, "similarity": {"R": MD["similarity"]["R"]}},
          open(f"{HERE}/fig2_data/model_diff_primary.json", "w"), indent=1)
print(len(C), "q10 records;", len(MD["primary"]["Z"]), "models")

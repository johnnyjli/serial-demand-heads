"""Worked-example copy: observed vs expected with a per-item Poisson-binomial z (items treated as independent)
and a per-model count of excess."""
import os, pickle, numpy as np, json
HERE = os.path.dirname(os.path.abspath(__file__))
M = pickle.load(open(os.path.join(HERE, "v_main.pkl"), "rb")); SH = pickle.load(open(os.path.join(HERE, "v_shot_answers.pkl"), "rb"))
R = {}
for fam in ["pointer", "arith", "binding", "relational"]:
    T = M[(M.fam == fam) & (M.k >= 2) & (M.pc != M.gold)]
    o, e, mm = [], [], []
    for c, i, p, g, a, m in zip(T.cell, T.iid, T.pc, T.gold, T.aset, T.model):
        sh = set(SH[(c, i)]); oth = [x for x in a if x != g]
        o.append(p in sh); e.append(np.mean([x in sh for x in oth])); mm.append(m)
    o, e, mm = np.array(o, float), np.array(e), np.array(mm)
    z = (o.sum() - e.sum()) / np.sqrt((e * (1 - e)).sum())
    per = {m: o[mm == m].mean() - e[mm == m].mean() for m in set(mm)}
    R[fam] = dict(obs=round(o.mean(), 4), exp=round(e.mean(), 4), z_indep=round(float(z), 2),
                  models_with_excess=int(sum(v > 0 for v in per.values())), n_models=len(per))
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(HERE, "v8_copy_test.json"), "w"), indent=1)

"""Setup checks bearing on correctness labels (read-only on the raw captures):
(1) ties at the maximum of the admissible-digit logits (bf16): how often the gold digit is tied at the maximum
    but the item is scored incorrect (tie broken against gold), per task family;
(2) are wrong answers copies of worked-example answers more often than uniform guessing over the admissible
    non-gold digits would give?  Writes setup_checks.json."""
import glob, json, os, sys, numpy as np
from clib import *
from clib import _path
sys.path.insert(0, OLD)
from solve import blocks, solve_block
res = {}
for f in FAMS:
    n = tie = gold_tied = gold_tied_wrong = strict_wrong = 0; exp_copy = []; obs_copy = []; per_model = {}
    for m in ORDER:
        for w in ("w0", "w1"):
            p = _path(m, f, w, attn_of(m)); z = np.load(p, allow_pickle=False); items = json.loads(str(z["meta"]))["items"]
            lg = z["digit_logits"].astype(np.float64); am = z["answer_mask"]; gold = z["gold"].astype(int)
            lgm = np.where(am, lg, -np.inf); mx = lgm.max(1, keepdims=True); tied = (lgm == mx)
            gt = tied[np.arange(len(gold)), gold] & (tied.sum(1) > 1)
            cs = z["correct_strict"].astype(bool); cc = z["correct_constrained"].astype(bool)
            n += len(gold); tie += int((tied.sum(1) > 1).sum()); gold_tied += int(gt.sum()); gold_tied_wrong += int((gt & ~cs).sum()); strict_wrong += int((~cs).sum())
            pm = per_model.setdefault(m, [0, 0, 0]); pm[0] += len(gold); pm[1] += int((gt & ~cs).sum()); pm[2] += int((~cs).sum())
            pred = z["pred_constrained"].astype(int)
            for i, t in enumerate(items):
                if cs[i]: continue
                shown = {solve_block(f, w, b)[1] for b in blocks(t["text"])[:-1]}
                adm = set(np.nonzero(am[i])[0].tolist()) - {gold[i]}
                if not adm: continue
                exp_copy.append(len(shown & adm) / len(adm)); obs_copy.append(int(pred[i] in shown))
    res[f] = {"n_items": n, "frac_items_with_tie_at_max": tie / n, "frac_items_gold_tied_at_max": gold_tied / n,
              "frac_wrong_items_where_gold_tied_at_max": gold_tied_wrong / strict_wrong,
              "max_model_frac_wrong_gold_tied": max(v[1] / v[2] for v in per_model.values() if v[2]),
              "worst_model": max(per_model, key=lambda k: per_model[k][1] / max(per_model[k][2], 1)),
              "wrong_pred_is_worked_example_answer": float(np.mean(obs_copy)), "expected_under_uniform_guessing": float(np.mean(exp_copy)), "n_wrong": len(obs_copy)}
    print(f, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[f].items()}))
json.dump(res, open("setup_checks.json", "w"), indent=1)
# (3) items scored wrong with gold tied at the admissible maximum AND first generated digit = gold: accuracy if counted correct
tie2 = {}
for f in FAMS:
    n = add = 0; acc0 = 0; addk = {}; nk = {}
    for m in ORDER:
        for w in ("w0", "w1"):
            z = np.load(_path(m, f, w, attn_of(m)), allow_pickle=False)
            lg = np.where(z["answer_mask"], z["digit_logits"].astype(np.float64), -np.inf); gold = z["gold"].astype(int); k = z["k"].astype(int)
            tied = lg == lg.max(1, keepdims=True); gt = tied[np.arange(len(gold)), gold] & (tied.sum(1) > 1)
            cs = z["correct_strict"].astype(bool); fd = z["pred_first_digit"].astype(int)
            a = gt & ~cs & (fd == gold); n += len(gold); add += int(a.sum()); acc0 += int(cs.sum())
            for kk in np.unique(k): addk[int(kk)] = addk.get(int(kk), 0) + int(a[k == kk].sum()); nk[int(kk)] = nk.get(int(kk), 0) + int((k == kk).sum())
    tie2[f] = {"acc_strict": acc0 / n, "acc_if_gold_ties_with_gold_first_digit_counted": (acc0 + add) / n, "n_items_affected": add,
               "gain_by_k": {kk: addk[kk] / nk[kk] for kk in sorted(addk)}}
    print(f, json.dumps(tie2[f]))
res["ties_counted_correct"] = tie2
json.dump(res, open("setup_checks.json", "w"), indent=1)

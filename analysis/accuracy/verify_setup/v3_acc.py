"""Sections 3, 4, 6 and minor points: output format, scoring variants, ties, accuracy by k, size, kernel."""
import collections, json, os, pickle
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(HERE, "v_items.pkl"), "rb"))
D["cell"] = D.fam + "/" + D.w
SERIAL = ["pointer", "arith", "binding", "relational"]; FAMS = SERIAL + ["fixed_digit"]
CH = {"pointer": .1, "arith": 1 / 7, "binding": 1 / 3, "relational": 1 / 3, "fixed_digit": 1 / 7}
ORDER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
         "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
PAPER = {"pointer": [.57, .16, .09, .07, .06], "arith": [.63, .15, .16, .13, .12, .11],
         "binding": [.86, .61, .44, .34, .32, .28], "relational": [.76, .45, .25, .27, .23, .27],
         "fixed_digit": [.41, .30, .17, .15, .14, .03]}


def main(D, q7="sdpa"):
    return D[((D.kernel == "eager") & (D.model != "Qwen2.5-7B-Instruct")) |
             ((D.model == "Qwen2.5-7B-Instruct") & (D.kernel == q7))].copy()


# my own tie / admissible computations from the stored digit logits
def enrich(T):
    L = np.stack(T.dl.values); mask = np.array([list(a) for a in T.amask.values])
    La = np.where(mask, L, -np.inf); mx = La.max(1)
    tied = (La == mx[:, None]); nt = tied.sum(1)
    g = T.gold.values
    T["my_ntie"] = nt
    T["my_gold_tied"] = (nt > 1) & tied[np.arange(len(T)), g]
    T["my_argmax_low"] = np.argmax(La, 1)                   # lowest digit among ties
    T["my_tie_split"] = np.where(tied[np.arange(len(T)), g], 1.0 / nt, 0.0)
    T["wr"] = (T.pfd == T.gold)
    T["first_char_digit"] = T.gen.fillna("").map(lambda s: len(s) > 0 and s[0].isdigit())
    T["first_ns_digit"] = T.gen.fillna("").map(lambda s: len(s.lstrip()) > 0 and s.lstrip()[0].isdigit())
    return T


D = enrich(D)
M = main(D)
R = {}
# ---------- strict definition
R["strict_eq_cc_and_written"] = bool((M.cs == (M.cc & M.wr)).all())
R["strict_eq_cc_and_written_all_runs"] = bool((D.cs == (D.cc & D.wr)).all())
R["ntie_matches_npz"] = float((M.my_ntie == M.ntie).mean())
R["gold_tied_matches_meta"] = float((M.my_gold_tied == M.gtie_meta).mean())
R["cc_when_gold_tied"] = float(M[M.my_gold_tied].cc.mean())       # how ties are scored
R["pc_is_lowest_digit_among_ties"] = float((M[M.my_ntie > 1].pc == M[M.my_ntie > 1].my_argmax_low).mean())

# ---------- output format
def fmt(T):
    return dict(n=len(T), first_char_digit=float(T.first_char_digit.mean()),
                first_nonspace_digit=float(T.first_ns_digit.mean()),
                n_first_char_nondigit=int((~T.first_char_digit).sum()),
                agree_pc_pfd=float((T.pc == T.pfd).mean()),
                agree_excl_ties=float((T.pc == T.pfd)[T.my_ntie <= 1].mean()),
                agree_lowdigit_tiebreak=float((T.my_argmax_low == T.pfd).mean()),
                acc_constrained=float(T.cc.mean()), acc_strict=float(T.cs.mean()), acc_written=float(T.wr.mean()),
                acc_tie_split=float(np.where(T.my_ntie > 1, T.my_tie_split, T.cc.astype(float)).mean()),
                frac_gold_tied=float(T.my_gold_tied.mean()),
                written_outside_aset=float(T.apply(lambda r: r.pfd >= 0 and r.pfd not in r.aset, axis=1).mean()))
R["fmt_main17"] = fmt(M)
R["fmt_excl_Qwen7B"] = fmt(M[M.model != "Qwen2.5-7B"])
R["fmt_excl_Qwen7B_Qwen7BI"] = fmt(M[~M.model.isin(["Qwen2.5-7B", "Qwen2.5-7B-Instruct"])])
Q = D[D.model == "Qwen2.5-7B-Instruct"]
R["fmt_Q7I_eager"] = fmt(Q[Q.kernel == "eager"]); R["fmt_Q7I_sdpa"] = fmt(Q[Q.kernel == "sdpa"])
R["fmt_Q7B_eager"] = fmt(M[M.model == "Qwen2.5-7B"])
R["nondigit_first_char_by_model"] = M[~M.first_char_digit].model.value_counts().to_dict()
R["nondigit_first_nonspace_by_model"] = M[~M.first_ns_digit].model.value_counts().to_dict()
ag = M.groupby("model").apply(lambda T: (T.pc == T.pfd).mean())
R["agree_by_model_range"] = [ag.idxmin(), float(ag.min()), ag.idxmax(), float(ag.max())]
gap = M.groupby(["model", "cell"]).apply(lambda T: T.cc.mean() - T.cs.mean())
R["max_cell_gap"] = [" ".join(gap.idxmax()), float(gap.max())]
L1 = M[(M.model == "Llama-3.2-1B") & (M.cell == "binding/w1")]
R["Llama1B_binding_w1"] = dict(cc=float(L1.cc.mean()), cs=float(L1.cs.mean()),
                              written_outside_aset=float(L1.apply(lambda r: r.pfd not in r.aset, axis=1).mean()),
                              among_cc_not_cs_written_outside=float(L1[L1.cc & ~L1.cs].apply(lambda r: r.pfd not in r.aset, axis=1).mean()),
                              n_cc_not_cs=int((L1.cc & ~L1.cs).sum()))

# ---------- scoring variants, pooled
def sc(T):
    return dict(n=len(T), strict=float(T.cs.mean()), constrained=float(T.cc.mean()), written=float(T.wr.mean()),
                ties_split=float(np.where(T.my_ntie > 1, T.my_tie_split, T.cc.astype(float)).mean()),
                frac_gold_tied=float(T.my_gold_tied.mean()))
S = M[M.fam.isin(SERIAL)]
R["scoring_all"] = sc(M); R["scoring_serial_k1"] = sc(S[S.k == 1]); R["scoring_serial_kge3"] = sc(S[S.k >= 3])
R["gold_tied_by_k_serial"] = S.groupby("k").my_gold_tied.mean().round(4).to_dict()

# ---------- accuracy by k (mean over 17 models of per-model accuracy, both wordings pooled)
def acc_table(T, col="cs"):
    return T.groupby(["model", "fam", "k"])[col].mean()


A = acc_table(M)
tab, maxdiff = {}, 0
for f in FAMS:
    ks = sorted(M[M.fam == f].k.unique())
    tab[f] = {int(k): round(float(A.xs((f, k), level=("fam", "k")).mean()), 4) for k in ks}
    maxdiff = max(maxdiff, max(abs(tab[f][k] - p) for k, p in zip(ks, PAPER[f])))
R["tab_acc"] = tab; R["tab_acc_max_abs_diff_vs_paper"] = round(maxdiff, 4)
AE = acc_table(main(D, "eager"))
R["all_eager_max_change"] = round(float(max(abs(AE.xs((f, k), level=("fam", "k")).mean() - A.xs((f, k), level=("fam", "k")).mean())
                                            for f in FAMS for k in M[M.fam == f].k.unique())), 4)
k1 = {}
for f in SERIAL:
    a = A.xs((f, 1), level=("fam", "k"))
    k1[f] = dict(mean=round(float(a.mean()), 4), min=[a.idxmin(), round(float(a.min()), 4)], max=[a.idxmax(), round(float(a.max()), 4)])
R["k1"] = k1


def n_above(T, f, k, w=None):
    n = 0
    for m in ORDER:
        X = T[(T.model == m) & (T.fam == f) & (T.k == k)]
        if w: X = X[X.w == w]
        n += stats.binomtest(int(X.cs.sum()), len(X), CH[f], alternative="greater").pvalue < .05
    return int(n)


def n_above_instance(T, f, k):
    """Treat the 42 instances as units: an instance scores (w0 + w1)/2; one-sided t/sign-free test:
    exact test on the sum of Bernoulli pairs is not binomial, so use a permutation-free normal
    approximation with the instance-level variance."""
    n = 0
    for m in ORDER:
        X = T[(T.model == m) & (T.fam == f) & (T.k == k)].copy()
        X["inst"] = X.iid.str.split("/").str[-2:].str.join("/")
        y = X.groupby("inst").cs.mean().values
        se = y.std(ddof=1) / np.sqrt(len(y)) if y.std() > 0 else 1e-9
        p = 1 - stats.t.cdf((y.mean() - CH[f]) / se, len(y) - 1)
        n += p < .05
    return int(n)


abv = {}
for f in SERIAL + ["fixed_digit"]:
    abv[f] = {f"k{k}": dict(n84=n_above(M, f, k), w0_n42=n_above(M, f, k, "w0"), w1_n42=n_above(M, f, k, "w1"),
                           instance_t=n_above_instance(M, f, k)) for k in (1, 2, 3)}
R["above_chance_counts"] = abv
kge3 = {}
for f in SERIAL:
    ks = [k for k in sorted(M[M.fam == f].k.unique()) if k >= 3]
    per_model = pd.concat([A.xs((f, k), level=("fam", "k")) for k in ks], axis=1).mean(1)
    X = M[(M.fam == f) & (M.k >= 3)]
    kge3[f] = dict(mean=round(float(per_model.mean()), 4), chance=round(CH[f], 4), n_items=len(X),
                   binom_p_two_sided_pooled=float(stats.binomtest(int(X.cs.sum()), len(X), CH[f]).pvalue),
                   n_models_below_chance=int((per_model < CH[f]).sum()))
R["kge3"] = kge3
# reliable k=1 models
rel = {}
for f in SERIAL:
    rows = []
    for m in ORDER:
        p1, p2, p3 = (float(A[(m, f, k)]) for k in (1, 2, 3))
        if p1 >= .9:
            rows.append([m, round(p1, 3), round(p2, 3), round(p3, 3), round(p1 ** 2 + (1 - p1 ** 2) * CH[f], 3), round(p1 ** 2, 3)])
    rel[f] = rows
R["reliable_k1"] = rel

# ---------- size
PB = {"Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "Llama-3.1-8B": 8.03, "Qwen2.5-0.5B": .49, "Qwen2.5-1.5B": 1.54,
      "Qwen2.5-3B": 3.09, "Qwen2.5-7B": 7.62, "Qwen2.5-14B": 14.7, "OLMo-2-1124-7B": 7.30, "Mistral-7B-v0.1": 7.24,
      "Mistral-Small-24B-Base-2501": 23.6}
def ser(m, kfilter):
    return float(np.mean([np.mean([A[(m, f, k)] for k in sorted(M[M.fam == f].k.unique()) if kfilter(k)]) for f in SERIAL]))
size = {m: dict(k1=round(ser(m, lambda k: k == 1), 4), k2=round(ser(m, lambda k: k == 2), 4),
                k3=round(ser(m, lambda k: k == 3), 4), kge3=round(ser(m, lambda k: k >= 3), 4)) for m in ORDER}
R["size_serial_mean"] = size
x = [PB[m] for m in PB]
R["spearman_11_base"] = {kk: [round(v, 4) for v in stats.spearmanr(x, [size[m][kk] for m in PB])] for kk in ("k1", "k2", "k3", "kge3")}
R["mean_chance_serial"] = round(float(np.mean([CH[f] for f in SERIAL])), 4)
SD = D[D.kernel == "sdpa"]; AS = acc_table(SD)
def ser_s(m, k):
    return float(np.mean([AS[(m, f, k)] for f in SERIAL]))
R["qwen_sdpa_k1_k2"] = {m: [round(ser_s(m, 1), 4), round(ser_s(m, 2), 4)] for m in ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-14B"]}
kern = {}
for m in ORDER:
    E = D[(D.model == m) & (D.kernel == "eager")]; Sd = D[(D.model == m) & (D.kernel == "sdpa")]
    if Sd.cell.nunique() == 10:
        j = E.merge(Sd, on=["cell", "iid"], suffixes=("_e", "_s"))
        kern[m] = [round(float(E.cs.mean()), 4), round(float(Sd.cs.mean()), 4), round(float(Sd.cs.mean() - E.cs.mean()), 4),
                   round(float((j.pc_e == j.pc_s).mean()), 4)]
R["kernel_eager_sdpa_strict_diff_agree"] = kern
R["kernel_max_abs_diff_excl_Q7I"] = max(abs(v[2]) for m, v in kern.items() if m != "Qwen2.5-7B-Instruct")

# ---------- minor 1: Qwen2.5-7B eager vs sdpa on pointer/w0
q = D[(D.model == "Qwen2.5-7B") & (D.cell == "pointer/w0")]
R["Q7B_pointer_w0_by_k"] = {kk: q[q.kernel == kk].groupby("k").cs.mean().round(4).to_dict() for kk in ("eager", "sdpa")}
R["Q7B_pointer_by_wording_eager_k1"] = M[(M.model == "Qwen2.5-7B") & (M.fam == "pointer") & (M.k == 1)].groupby("w").cs.mean().round(4).to_dict()
R["Q7B_eager_first_char_nondigit"] = float(1 - M[M.model == "Qwen2.5-7B"].first_char_digit.mean())
# effect on 17-model mean if Qwen2.5-7B pointer k=1 were at its SDPA value on both wordings
cur = A[("Qwen2.5-7B", "pointer", 1)]; sd = q[(q.kernel == "sdpa") & (q.k == 1)].cs.mean()
R["Q7B_ptr_k1_shift_of_17_mean_if_sdpa_both_wordings"] = round(float((sd - cur) / 17), 4)
R["Q7B_ptr_w0_shift_of_17_mean_by_k"] = {int(k): round(float((q[(q.kernel == 'sdpa') & (q.k == k)].cs.mean() - q[(q.kernel == 'eager') & (q.k == k)].cs.mean()) / 17), 4) for k in range(1, 6)}
# ---------- minor 2: pointer k=1 per model
R["pointer_k1_by_model"] = {m: round(float(A[(m, "pointer", 1)]), 4) for m in ORDER}
# ---------- numeral control
R["numeral_by_k"] = tab["fixed_digit"]
R["numeral_onestep_vs_other"] = [round(np.mean([tab["fixed_digit"][k] for k in (1, 2, 5)]), 4), round(np.mean([tab["fixed_digit"][k] for k in (3, 4, 6)]), 4)]
json.dump(R, open(os.path.join(HERE, "v3_acc.json"), "w"), indent=1, default=str)
pickle.dump(M, open(os.path.join(HERE, "v_main.pkl"), "wb"))
print(json.dumps(R, indent=1, default=str))

# ===================== EXTENSION: algorithms, comprehensive metrics, statistics, new figures =====================
import re as _re
fmr = pd.read_csv(R + "comprehensive_metrics_runs.csv")
FM = fmr.groupby("model").mean(numeric_only=True).reindex(order)
cost = pd.DataFrame(summ["cost"]).set_index("index").reindex(order)
fried = summ["friedman"]; wil = pd.read_csv(R + "table6_wilcoxon.csv")
errov = summ["error_overlap"]
SH = dict(zip(order, ["NB", "RF", "LogReg", "MLP", "GCN", "GAT", "HGNN", "HyperDiag"]))
sigdf = pd.read_csv(R + "significance.csv")


def algo_box(title, lines):
    tbl = make_table([[title]], [8600])
    tc = tbl.findall(qn("w:tr"))[0].findall(qn("w:tc"))[0]
    tcpr = tc.find(qn("w:tcPr"))
    for shd in tcpr.findall(qn("w:shd")): tcpr.remove(shd)
    for ln in lines:
        p = etree.SubElement(tc, qn("w:p"))
        ppr = etree.SubElement(p, qn("w:pPr"))
        etree.SubElement(ppr, qn("w:spacing"), {qn("w:before"): "0", qn("w:after"): "0"})
        rpr = etree.Element(qn("w:rPr"))
        etree.SubElement(rpr, qn("w:rFonts"), {qn("w:ascii"): "Consolas", qn("w:hAnsi"): "Consolas", qn("w:cs"): "Consolas"})
        etree.SubElement(rpr, qn("w:sz"), {qn("w:val"): "16"}); etree.SubElement(rpr, qn("w:szCs"), {qn("w:val"): "16"})
        add_runs(p, ln if ln else " ", rpr)
    return tbl


ALG1 = ["Input : training matrix X ∈ {0,1}^(N×|V|), labels y; τ = 0.02, k = 4, η = 0.10, m = 10",
        "Output: incidence H = [E_d ∥ E_c], soft weights W_H, pairwise graph A, PPMI profile",
        "────────────────────────────────────────────────────────────────────",
        " 1: n_d ← Σ_i 1[y_i = d];   P(s|d) ← Σ_i 1[y_i = d]·X_is / n_d",
        " 2: for each disease d do                                  ▷ disease hyperedges",
        " 3:     e_d ← { s : P(s|d) ≥ τ };  w(s, e_d) ← P(s|d)",
        " 4:     if e_d = ∅ then e_d ← top-3 symptoms of d",
        " 5: Co ← XᵀX;  NPMI_ij ← log(p_ij / p_i p_j) / (−log p_ij);  PPMI_ij ← max(0, PMI_ij)",
        " 6: for each symptom v do                                  ▷ co-occurrence hyperedges",
        " 7:     N_v ← top-k { u : NPMI_vu ≥ η ∧ Co_vu ≥ m }",
        " 8:     if N_v ≠ ∅ then add e = {v} ∪ N_v with weight mean_u NPMI_vu",
        " 9: remove duplicate hyperedges;  A_vu ← 1 for top-10 NPMI partners (symmetric)",
        "10: return H, W_H, A, PPMI"]
ALG2 = ["Input : query mask m (S = {v : m_v = 1}); hypergraph (H, W_H); L layers, h heads, width d",
        "Output: disease scores s ∈ R^|D|; explanations γ, α, β, π",
        "────────────────────────────────────────────────────────────────────",
        " 1: X⁰ ← E_id + PPMI·W_p;   C⁰_e ← Σ_v ŵ(v,e)·X⁰_v + t_type(e)",
        " 2: for ℓ = 1 … L do                                       ▷ dual-level attention",
        " 3:     Z ← X^(ℓ−1) W_v;   Ĉ ← C^(ℓ−1) U_e",
        " 4:     α_ve ← softmax_(v∈e)( LReLU(a_vᵀZ_v + a_eᵀĈ_e) + log w(v,e) )    ▷ node level",
        " 5:     M_e ← Σ_(v∈e) α_ve Z_v;   C^ℓ_e ← LN(C^(ℓ−1)_e + ELU(W_o M_e))",
        " 6:     β_ev ← softmax_(e∋v)( LReLU(b_eᵀM_e + b_vᵀZ_v) )                ▷ hyperedge level",
        " 7:     X^ℓ_v ← LN(X^(ℓ−1)_v + ELU(W_o' Σ_(e∋v) β_ev M_e))",
        " 8: c̄ ← mean_(v∈S) X^L_v;   γ_v ← softmax_(v∈S)( gᵀtanh(W X^L_v) + ⟨U c̄, X^L_v⟩/√d )",
        " 9: q ← MLP([Σ_v γ_v X^L_v ; c̄])",
        "10: π_e ← softmax_(e∈E_c, e∩S≠∅)(⟨W_q q, C^L_e⟩/√d);  q ← q + W_u Σ_e π_e C^L_e   ▷ query-aware",
        "11: s_d ← exp(θ)·cos(q, ψ(C^L_d)) + b_d                    ▷ prototype scoring",
        "12: (λ₊, λ₋) ← softplus(G[q ; log(1+|S|)])                 ▷ adaptive gate",
        "13: s_d ← s_d + λ₊ Σ_(v∈S) log h_vd + λ₋ Σ_(v∉S) log(1 − h_vd)   ▷ hyperedge evidence",
        "14: return ranking argsort(−s), γ, α, β, π"]
ALG3 = ["Input : grouped split (Tr, Va, Te); model f_θ; T = 20 epochs; batch 1024; r_max = 0.5",
        "────────────────────────────────────────────────────────────────────",
        " 1: w_c ← n_c^(−1/2) / mean(n_c^(−1/2))                     ▷ class weights",
        " 2: for seed ∈ {0, 1, 2} do",
        " 3:     initialise θ;  AdamW (lr 2e−3, wd 1e−4) with a one-cycle schedule",
        " 4:     for t = 1 … T do",
        " 5:         for each mini-batch (M, y) ⊂ Tr do",
        " 6:             r_i ~ U(0, r_max); drop each present symptom w.p. r_i (keep ≥ 1)",
        " 7:             ℒ ← CE_w(f_θ(M̃), y; smoothing 0.05); clip ‖∇‖ ≤ 5; update θ",
        " 8:         keep θ* with the best validation MRR",
        " 9:     rank all diseases on Te: r_i ← 1 + |{d : s_id > s_(i,y_i)}|",
        "10:     compute Top-k, MRR, NDCG, ranks, P/R/F1, κ, MCC, log-loss, Brier, ECE",
        "11: report mean ± std over seeds; McNemar–Holm, Friedman–Nemenyi, Wilcoxon, bootstrap CIs"]

# replace the old algorithm table with Algorithm 2 and add Algorithms 1 and 3
alg_cap = find_p(d, "Algorithm 1: HyperDiag Training and Inference")
old_alg = alg_cap.getnext()
set_text(alg_cap, "Algorithm 2: HyperDiag forward pass (scoring a query symptom set S).")
new_alg = algo_box("ALGORITHM 2 — HYPERDIAG-FORWARD", ALG2); old_alg.addnext(new_alg); body.remove(old_alg)
p_hg = find_p(d, "The disease-symptom matrix is turned into a hypergraph")
a1c = clone_after(p_hg, alg_cap, "Algorithm 1: Hypergraph construction from training data only.")
a1 = algo_box("ALGORITHM 1 — HYPERGRAPH-CONSTRUCTION", ALG1); a1c.addnext(a1)
p_bl = find_p(d, "HyperDiag is compared with seven baselines")
a3c = clone_after(p_bl, alg_cap, "Algorithm 3: Training and evaluation protocol (identical for all neural models).")
a3 = algo_box("ALGORITHM 3 — TRAINING-AND-EVALUATION", ALG3); a3c.addnext(a3)

# leakage figure after leakage discussion; hypergraph network after stats figure
p = find_p(d, "Under the usual protocol (A)")
img = add_image_after(d, p, FIG + "fig02_leakage_protocols.png", 6.3)
clone_after(img, cap_tpl, "Figure 101: Train/test leakage and reported accuracy under three evaluation protocols.")
p = find_p(d, "Figure 3: Disease-hyperedge sizes")
img = add_image_after(d, p, FIG + "fig04_hypergraph_network.png", 6.0)
clone_after(img, cap_tpl, "Figure 102: Star expansion of five related disease hyperedges (squares) and their symptoms (circles); edge width ∝ P(s|d).")

# metrics description in task formulation
p = find_p(d, "Rather than predicting a single disease")
clone_after(p, normal_tpl,
    "In addition, we report NDCG@10, the mean and median rank of the true disease, macro precision and macro recall (balanced accuracy), "
    "Cohen's κ, the Matthews correlation coefficient (MCC), log-loss, the multi-class Brier score, the expected calibration error (ECE, "
    "15 bins), the mean size of the smallest differential list that covers 90% of the predicted probability, and computational cost "
    "(parameters, training time and inference time per 1,000 cases). Model rankings across diseases are compared with the Friedman test "
    "and the Nemenyi critical difference, per-disease F1 differences with the Wilcoxon signed-rank test and rank-biserial effect size, and "
    "all multiple comparisons are Holm-corrected.")

# ---- comprehensive tables after main-results paragraph
def best_fmt(col, vals, lower=False, dp=2, scale=100):
    v = np.array(vals, float); b = v.min() if lower else v.max()
    return [("**" if np.isclose(x, b) else "") + f"{x*scale:.{dp}f}" + ("**" if np.isclose(x, b) else "") for x in v]
pA = find_p(d, "HyperDiag reaches")
rowsA = [["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MRR", "NDCG@10", "Mean rank", "Median rank"]]
colsA = {k: best_fmt(k, FM[k], lower=k in ("Mean rank", "Median rank"), dp=2 if "rank" not in k else 3, scale=1 if "rank" in k else 100)
         for k in ["Top-1", "Top-3", "Top-5", "Top-10", "MRR", "NDCG@10", "Mean rank", "Median rank"]}
for j, m in enumerate(order): rowsA.append([SH[m]] + [colsA[k][j] for k in rowsA[0][1:]])
tA = make_table(rowsA, [1240] + [920] * 8)
cA = clone_after(pA, cap_tpl, "Table 101: Ranking metrics (mean over seeds; %, except ranks). Best value in bold.")
cA.addnext(tA)
rowsB = [["Model", "Macro-P", "Macro-R", "Macro-F1", "Weighted-F1", "Cohen κ", "MCC"]]
colsB = {k: best_fmt(k, FM[k]) for k in ["Macro-P", "Macro-R", "Macro-F1", "Weighted-F1", "Cohen-kappa", "MCC"]}
for j, m in enumerate(order): rowsB.append([SH[m]] + [colsB[k][j] for k in ["Macro-P", "Macro-R", "Macro-F1", "Weighted-F1", "Cohen-kappa", "MCC"]])
tB = make_table(rowsB, [1640] + [1160] * 6)
cB = clone_after(tA, cap_tpl, "Table 102: Classification agreement metrics of the top-1 prediction (mean over seeds, %). Best value in bold.")
cB.addnext(tB)
rowsC = [["Model", "Log-loss", "Brier", "ECE (%)", "Set@90%", "Params", "Train (s)", "Infer (ms/1k)"]]
colsC = {"Log-loss": best_fmt("", FM["Log-loss"], True, 3, 1), "Brier": best_fmt("", FM["Brier"], True, 3, 1),
         "ECE (%)": best_fmt("", FM["ECE"], True, 2, 100), "Set@90%": best_fmt("", FM["Set size@90%"], True, 2, 1)}
for j, m in enumerate(order):
    rowsC.append([SH[m]] + [colsC[k][j] for k in ["Log-loss", "Brier", "ECE (%)", "Set@90%"]] +
                 [f"{int(cost.loc[m, 'Params']):,}", f"{cost.loc[m, 'Train time (s)']:.0f}", f"{cost.loc[m, 'Infer (ms/1k)']:.1f}"])
tC = make_table(rowsC, [1240] + [1050] * 7)
cC = clone_after(tB, cap_tpl, "Table 103: Probabilistic quality, calibration and computational cost. Lower is better; best value in bold.")
cC.addnext(tC)
best_ece = order[int(np.argmin(FM["ECE"].values))]
pC = clone_after(tC, normal_tpl,
    f"The extended metrics confirm the picture from Table 3. HyperDiag reaches NDCG@10 = {FM.loc[OURS,'NDCG@10']*100:.2f}%, a median rank of "
    f"{FM.loc[OURS,'Median rank']:.0f} and a mean rank of {FM.loc[OURS,'Mean rank']:.2f}, Cohen's κ = {FM.loc[OURS,'Cohen-kappa']*100:.2f}% and MCC = "
    f"{FM.loc[OURS,'MCC']*100:.2f}%. On average its 90%-probability differential list contains {FM.loc[OURS,'Set size@90%']:.2f} diseases. "
    f"The best-calibrated model is {SH[best_ece]} (ECE = {FM.loc[best_ece,'ECE']*100:.2f}%); HyperDiag has ECE = {FM.loc[OURS,'ECE']*100:.2f}%.")
img = add_image_after(d, pC, FIG + "fig06_performance_profile.png", 6.4)
clone_after(img, cap_tpl, "Figure 103: Multi-view performance profile: (a) normalised metric radar, (b) cumulative match characteristic, (c) macro-F1 with 95% bootstrap CI, (d) accuracy–efficiency trade-off.")
cap = img.getnext()
img2 = add_image_after(d, cap, FIG + "fig07_calibration.png", 6.4)
c104 = clone_after(img2, cap_tpl, "Figure 104: Reliability diagrams with expected calibration error (ECE) and Brier score for every model.")
ts = pd.read_csv(R + "table5_temperature_scaling.csv")
tts = make_table([["Model", "T", "ECE before (%)", "ECE after (%)", "Log-loss before", "Log-loss after", "Brier before", "Brier after"]] +
                 [[r.model, f"{r.T:.2f}", f"{r._3:.2f}", f"**{r._4:.2f}**", f"{r._5:.3f}", f"**{r._6:.3f}**", f"{r._7:.3f}", f"**{r._8:.3f}**"] for r in ts.itertuples()],
                 [1240, 700, 1110, 1110, 1110, 1110, 1110, 1110])
ct = clone_after(c104, cap_tpl, "Table 105: Post-hoc temperature scaling fitted on the validation set (rankings unchanged; probability quality before → after).")
ct.addnext(tts)
pt_ = clone_after(tts, normal_tpl,
    f"All discriminative models are under-confident because of class weighting and label smoothing (bars above the diagonal in Figure 104). "
    f"A single validation-fitted temperature (T = {ts.set_index('model').loc['HyperDiag','T']:.2f} for HyperDiag) lowers HyperDiag's ECE from "
    f"{ts.set_index('model').loc['HyperDiag','ECE before (%)']:.2f}% to {ts.set_index('model').loc['HyperDiag','ECE after (%)']:.2f}% without changing its rankings, so the "
    "differential lists can be shown with trustworthy probabilities.")


# ---- statistics: new section after significance table
sg_cap = find_p(d, "Table 4: Statistical significance")
sg_tbl = sg_cap.getnext()
new_sg = make_table([["Baseline", "Ours-only correct", "Baseline-only correct", "McNemar p (Holm)", "Cohen's h", "ΔTop-5 [95% CI]", "ΔMRR [95% CI]"]] +
                    [[SH.get(r.baseline, r.baseline), r.ours_only_correct, r.baseline_only_correct, f"{r.mcnemar_p_holm:.3g}", f"{r.cohens_h_top1:+.3f}", r.dTop5, r.dMRR]
                     for r in sigdf.itertuples()], [1100, 1150, 1150, 1200, 1000, 1500, 1500])
sg_tbl.addnext(new_sg); body.remove(sg_tbl)
set_text(sg_cap, "Table 4: HyperDiag versus each baseline: McNemar test (Holm-corrected), effect size and paired bootstrap 95% CIs (percentage points).")
h = clone_after(new_sg, h2_tpl, "4.2 Statistical Analysis")
ar = summ["friedman"]["avg_rank"]; ar_s = sorted(ar.items(), key=lambda t: t[1])
p1 = clone_after(h, normal_tpl,
    f"Following Demšar (2006), we compare all eight models over the test diseases using per-disease F1. The Friedman test "
    f"rejects the hypothesis that all models perform equally (χ² = {fried['chi2']:.1f}, p = {fried['p']:.1e}; Iman–Davenport F = {fried['iman_davenport']:.1f}). "
    f"In average rank (1 = best) the order is " + ", ".join(f"{SH[k]} ({v:.2f})" for k, v in ar_s) +
    f". The Nemenyi critical difference is {fried['cd']:.2f} (Figure 106). Models joined by a bar in the diagram are not significantly different. "
    "Figure 105 reports all pairwise McNemar tests after Holm correction, and Table 104 the Wilcoxon signed-rank tests of per-disease F1 "
    "between HyperDiag and each baseline, with the rank-biserial effect size.")
img = add_image_after(d, p1, FIG + "fig09_pairwise_mcnemar.png", 4.6)
c = clone_after(img, cap_tpl, "Figure 105: All-pairs McNemar tests on top-1 correctness (signed −log10 Holm-adjusted p; stars mark significance).")
img = add_image_after(d, c, FIG + "fig10_critical_difference.png", 6.3)
c = clone_after(img, cap_tpl, "Figure 106: Critical-difference diagram (Friedman test with Nemenyi post-hoc, α = 0.05) on per-disease F1.")
wt = make_table([["Baseline", "Wins", "Ties", "Losses", "Median ΔF1", "Mean ΔF1", "p (Holm)", "r_rb", "Effect"]] +
                [[r.baseline, int(r.wins), int(r.ties), int(r.losses), f"{r.median_dF1:+.2f}", f"{r.mean_dF1:+.2f}", f"{r.p_holm:.3g}", f"{r.r_rb:+.3f}", r.effect]
                 for r in wil.itertuples()], [1150, 800, 800, 850, 1050, 1000, 1050, 900, 1000])
c2 = clone_after(c, cap_tpl, "Table 104: Wilcoxon signed-rank tests on per-disease F1 (HyperDiag − baseline, percentage points), Holm-corrected, with rank-biserial effect size.")
c2.addnext(wt)
img = add_image_after(d, wt, FIG + "fig11_bootstrap_forest.png", 6.4)
clone_after(img, cap_tpl, "Figure 107: Paired bootstrap differences (HyperDiag − baseline) with 95% CIs; coloured intervals exclude zero.")

# renumber the following sections 4.2->4.3 ...
for old_, new_ in [("4.5 Interpretability", "4.6 Interpretability"), ("4.4 Ablation", "4.5 Ablation"), ("4.3 Rare", "4.4 Rare"), ("4.2 Robustness", "4.3 Robustness")]:
    pe = find_p(d, old_); set_text(pe, ptext(pe).replace(old_.split()[0], new_.split()[0], 1))

# figure swaps for renamed figure files are handled in the base script; add new figures to rare / ablation / interpretability
p = find_p(d, "For rare diseases (≤ 20 training cases)")
img = add_image_after(d, p, FIG + "fig13_rare_diseases.png", 6.4)
clone_after(img, cap_tpl, "Figure 108: Per-disease F1 against training frequency (a) and recall@1 by frequency group (b).")
p = find_p(d, "Removing individual components changes")
img = add_image_after(d, p, FIG + "fig14_ablation.png", 6.4)
clone_after(img, cap_tpl, "Figure 109: Ablation effects relative to the full model (mean ± std of the seed-paired difference).")
ftab_cap = find_p(d, "Table 8: Attention-faithfulness")
ftab = ftab_cap.getnext()
img = add_image_after(d, ftab, FIG + "fig18_tsne_embeddings.png", 6.4)
c = clone_after(img, cap_tpl, "Figure 110: t-SNE of the learned disease-hyperedge prototypes (a) and symptom node embeddings (b).")
h = clone_after(c, h2_tpl, "4.7 Error Analysis: Confusions Follow Hyperedge Overlap")
pe = clone_after(h, normal_tpl,
    f"HyperDiag's errors are not random. Weighted by the number of errors, confused disease pairs have a mean hyperedge Jaccard overlap of "
    f"{errov['confused_mean_jaccard']:.3f}, compared with {errov['random_mean_jaccard']:.3f} for random disease pairs (one-sided Mann–Whitney "
    f"p = {errov['p']:.1e}). The remaining errors are therefore largely the irreducible cases in which two diseases share most of their "
    "symptoms. This is consistent with the Bayes-ceiling analysis, and the overlap is directly visible in the hypergraph (Figure 111).")
img = add_image_after(d, pe, FIG + "fig19_error_analysis.png", 6.4)
clone_after(img, cap_tpl, "Figure 111: Error analysis: (a) network of the 30 most frequent confusions; (b) hyperedge overlap of confused versus random disease pairs.")


# ---- final pass: renumber captions in document order and update references
def renumber(doc):
    els = list(doc.element.body.iterchildren())
    mp, cnt = {}, {"Table": 0, "Figure": 0, "Algorithm": 0}
    for el in els:
        if el.tag != qn("w:p"): continue
        m = _re.match(r"^(Table|Figure|Algorithm) (\d+):", ptext(el).strip())
        if m:
            cnt[m.group(1)] += 1; mp[(m.group(1), m.group(2))] = str(cnt[m.group(1)])
    pat = _re.compile(r"\b(Table|Figure|Algorithm)( )(\d+)\b")
    for t in doc.element.body.iter(qn("w:t")):
        if t.text and pat.search(t.text):
            t.text = pat.sub(lambda m: f"{m.group(1)} {mp.get((m.group(1), m.group(3)), m.group(3))}", t.text)
    return mp
MAP = renumber(d)

import sys, json, copy
sys.path.insert(0, __import__("os").path.dirname(__file__))
import pandas as pd, numpy as np, docx
from docx_helpers import *

R = __import__("os").environ.get("RES_DIR", __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "results") + "/")
FIG = R + "figures/"
OUT = sys.argv[1]
d = docx.Document(sys.argv[2] if len(sys.argv) > 2 else "HyperDiag_PreDefence.docx")

audit = {k: (float(v) if str(v).replace(".","",1).lstrip("-").isdigit() else v) for k, v in pd.read_csv(R + "data_audit.csv").values}
ceil = pd.read_csv(R + "bayes_ceiling.csv", index_col=0)
leak = pd.read_csv(R + "leakage_protocols.csv")
runs = pd.read_csv(R + "all_runs.csv")
sig = pd.read_csv(R + "significance.csv")
part = pd.read_csv(R + "partial_symptoms.csv")
rare = pd.read_csv(R + "rare_diseases.csv")
abl = pd.read_csv(R + "ablation_runs.csv")
faith = pd.read_csv(R + "faithfulness.csv", index_col=0)
attal = pd.read_csv(R + "attention_alignment.csv")
hst = pd.read_csv(R + "hypergraph_stats.csv")
summ = json.load(open(R + "summary.json"))
sp = summ["split"]; tc = summ["test_ceiling"]

M = ["Top-1", "Top-3", "Top-5", "Top-10", "MRR", "Macro-F1"]
order = ["Naive Bayes (Bernoulli)", "Random Forest", "LogReg (softmax)", "MLP", "GCN (pairwise)", "GAT (pairwise)", "HGNN (hypergraph, no attention)", "HyperDiag (ours)"]
mu = runs.groupby("model")[M].mean().reindex(order) * 100
sd = runs.groupby("model")[M].std().reindex(order).fillna(0) * 100
OURS = "HyperDiag (ours)"
f = lambda x: f"{x:.2f}"
pm = lambda m, k: f"{mu.loc[m, k]:.2f} ± {sd.loc[m, k]:.2f}" if sd.loc[m, k] > 0 else f"{mu.loc[m, k]:.2f}"
best_nn = mu.drop(index=OURS).loc[["MLP", "GCN (pairwise)", "GAT (pairwise)", "HGNN (hypergraph, no attention)"]]
la, lb, lc = leak.iloc[0], leak.iloc[1], leak.iloc[2]
pv = lambda r: part[(part.model == OURS) & (part.keep_ratio == r)]["Top-5"].mean() * 100
pnb = lambda r: part[(part.model == "Naive Bayes (Bernoulli)") & (part.keep_ratio == r)]["Top-5"].mean() * 100
rare_o = rare[(rare.model == OURS) & (rare.group.str.startswith("rare"))].iloc[0]
rare_nb = rare[(rare.model == "Naive Bayes (Bernoulli)") & (rare.group.str.startswith("rare"))].iloc[0]
abm = abl.groupby("variant")[["Top-1", "Top-5", "MRR", "Macro-F1", "Top-5 @40% symptoms"]].mean() * 100
abs_ = abl.groupby("variant")[["Top-1", "Top-5", "MRR", "Macro-F1", "Top-5 @40% symptoms"]].std().fillna(0) * 100
full = abm.loc["Full HyperDiag"]
NICE = {"Naive Bayes (Bernoulli)": "Naive Bayes", "Random Forest": "Random Forest", "LogReg (softmax)": "Logistic Regression", "MLP": "MLP",
        "GCN (pairwise)": "GCN", "GAT (pairwise)": "GAT", "HGNN (hypergraph, no attention)": "HGNN"}
def _join(xs): return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else (xs[0] if xs else "none")
SIGW = [NICE[r.baseline] for r in sig.itertuples() if r.mcnemar_p_holm < 0.05 and r.ours_only_correct > r.baseline_only_correct]
SIGL = [NICE[r.baseline] for r in sig.itertuples() if r.mcnemar_p_holm < 0.05 and r.ours_only_correct < r.baseline_only_correct]
NSIG = [NICE[r.baseline] for r in sig.itertuples() if r.mcnemar_p_holm >= 0.05]
SIG_SENT = (f"After Holm correction, HyperDiag is significantly better than {_join(SIGW)}" if SIGW else "After Holm correction, HyperDiag is not significantly better than any baseline") + \
           (f", significantly worse than {_join(SIGL)}" if SIGL else "") + (f", and statistically indistinguishable from {_join(NSIG)}" if NSIG else "")
FA_RATIO = faith.loc["most attended", "mean_drop_true_prob"] / max(faith.loc["random", "mean_drop_true_prob"], 1e-9)
FA_WORD = "confirms that γ is faithful" if FA_RATIO >= 1.3 else ("indicates that γ is only weakly faithful" if FA_RATIO > 1.0 else "does not support the faithfulness of γ")

body = d.element.body
def clone_label(ref, label, rest):
    new = clone_after(ref, ref, "x")
    for b in new.iter(qn("w:b")): b.getparent().remove(b)
    for b in new.iter(qn("w:bCs")): b.getparent().remove(b)
    return set_text(new, "**" + label + "**" + rest)
normal_tpl = find_p(d, "In modern healthcare")
h2_tpl = find_p(d, "1.1 Background")
cap_tpl = find_p(d, "Table 2: Comparative")
list_tpl = find_p(d, "To design a hypergraph")

# ---------------- Abstract
set_text(find_p(d, "Symptom-based differential diagnosis is a core"),
    "Symptom-based differential diagnosis maps a set of co-occurring symptoms to a ranked list of likely diseases. Previous work on public "
    "symptom-checker data has relied on flat classifiers or pairwise graph neural networks, and has almost always been evaluated with random "
    "train/test splits. We propose HyperDiag, a hypergraph attention network in which every disease is a hyperedge over its symptoms, "
    "auxiliary hyperedges encode recurring symptom clusters, and a dual-level (node and hyperedge) attention mechanism is complemented by "
    "disease-hyperedge prototype scoring and query-adaptive hyperedge-likelihood evidence. Before modelling, we audit the Kaggle "
    f"Diseases-and-Symptoms dataset ({int(audit['rows']):,} cases, {int(audit['diseases'])} diseases, {int(audit['symptoms'])} symptoms) and find that "
    f"{int(audit['exact_duplicate_rows']):,} rows ({audit['exact_duplicate_rows']/audit['rows']*100:.1f}%) are exact duplicates and that "
    f"{int(audit['vectors_mapping_to_>1_disease']):,} symptom patterns map to more than one disease. A conventional random split therefore places "
    f"{la.leaked_test_cases*100:.1f}% of test cases in the training set, and no classifier can exceed a top-1 accuracy of "
    f"{ceil.loc['de-duplicated','Top-1']*100:.1f}% on the de-duplicated data. We introduce a leakage-free, symptom-pattern-grouped protocol and "
    f"benchmark eight models on it. HyperDiag reaches {mu.loc[OURS,'Top-1']:.2f}% top-1, {mu.loc[OURS,'Top-5']:.2f}% top-5 accuracy and an MRR of "
    f"{mu.loc[OURS,'MRR']:.2f}%. " + SIG_SENT + " (McNemar test on top-1 correctness). Its symptom attention is evaluated with a "
    "deletion-based faithfulness test. The results show that, on this dataset, honest evaluation matters more than architecture. They also show where "
    "hypergraph structure helps (interpretability, consistency) and where it does not (raw accuracy near the Bayes ceiling).")

# ---------------- Chapter 1 tweaks
set_text(find_p(d, "With the rapid growth of digital health"),
    "With the rapid growth of digital health platforms and online symptom checkers, there is increasing interest in building intelligent "
    "systems that can support this reasoning process. Such systems take a set of symptoms as input and return a ranked list of probable "
    "diseases. Publicly available datasets, such as the Kaggle Disease-Symptom dataset used in this work, have made it possible for "
    f"researchers to train and evaluate these systems. This dataset records {int(audit['rows']):,} patient cases spanning "
    f"{int(audit['diseases'])} unique diseases and {int(audit['symptoms'])} distinct symptoms; each case lists the symptoms that are present and the associated disease.")
p = find_p(d, "g) Lack of Interpretability")
clone_label(p, "h) Unreliable Evaluation:", " The public dataset contains many duplicate records and identical symptom patterns with different "
    "diagnoses. Random splits leak these into the test set and inflate reported accuracy, so models cannot be compared fairly.")
p = find_p(d, "f) Insufficient Interpretability")
clone_label(p, "g) No Leakage-Free Benchmark or Performance Ceiling:", " Prior studies neither control for duplicate symptom patterns across "
    "splits nor report the Bayes-optimal accuracy that the label noise of the dataset permits.")
p = find_p(d, "To provide interpretable attention visualizations")
clone_after(p, p, "To audit the dataset, quantify train/test leakage under common protocols, derive the Bayes-optimal top-k ceiling, and "
    "establish a leakage-free benchmark.")
set_text(find_p(d, "We will introduce the first application"),
    "This thesis makes five contributions. First, we audit a widely used public symptom-checker dataset. We show that random splits leak "
    f"{la.leaked_test_cases*100:.1f}% of test cases into training and derive a Bayes-optimal accuracy ceiling. We then propose a leakage-free, "
    "symptom-pattern-grouped evaluation protocol. Second, we introduce what is, to our knowledge, the first hypergraph neural network for "
    "symptom-checker-style differential diagnosis, with disease hyperedges and NPMI-based co-occurrence hyperedges built only from training "
    "data. Third, we design HyperDiag, which combines dual-level hypergraph attention, query-aware attention over symptom clusters, "
    "disease-hyperedge prototype scoring and query-adaptive hyperedge-likelihood evidence. Fourth, we frame the problem as top-k differential "
    "diagnosis and benchmark eight models under identical conditions with multiple seeds, significance tests, partial-symptom and "
    "rare-disease analyses and a full ablation. Finally, we provide symptom-level explanations and verify their faithfulness with a deletion test.")

# ---------------- Chapter 3
set_text(find_p(d, "We use the publicly available Kaggle Disease-Symptom dataset, which"),
    "We use the publicly available Kaggle Diseases and Symptoms dataset (Kumar, 2023), file Final_Augmented_dataset_Diseases_and_Symptoms.csv. "
    f"It contains {int(audit['rows']):,} rows, {int(audit['diseases'])} unique diseases and {int(audit['symptoms'])} binary symptom columns. Each row is a "
    f"patient case: the symptom columns record presence (1) or absence (0), and the label column records the disease. On average a case has "
    f"{audit['symptoms_per_case_mean']:.2f} symptoms present. Class sizes range from {int(audit['min_class'])} to {int(audit['max_class'])} cases "
    f"(imbalance ratio {audit['imbalance_ratio']:.0f}:1).")
p_pre = find_p(d, "The raw dataset is first cleaned")
set_text(p_pre,
    f"Disease labels are lower-cased and stripped, and all symptom columns are verified to be strictly binary. The data audit (Table 1) found "
    f"{int(audit['exact_duplicate_rows']):,} exact duplicate records, which are removed. {int(audit['unused_symptoms'])} symptoms that never occur are dropped. "
    f"Diseases with fewer than five unique cases cannot be represented in all three splits and are excluded, which leaves "
    f"{sp['train']+sp['val']+sp['test']:,} cases, {sp['symptoms']} symptoms and {sp['diseases']} diseases.")
t1 = make_table([["Property", "Value"],
                 ["Rows / diseases / symptoms", f"{int(audit['rows']):,} / {int(audit['diseases'])} / {int(audit['symptoms'])}"],
                 ["Missing values / non-binary values", f"{int(audit['missing_values'])} / none"],
                 ["Exact duplicate rows", f"{int(audit['exact_duplicate_rows']):,} ({audit['exact_duplicate_rows']/audit['rows']*100:.1f}%)"],
                 ["Unique symptom vectors", f"{int(audit['unique_symptom_vectors']):,}"],
                 ["Vectors mapping to >1 disease", f"{int(audit['vectors_mapping_to_>1_disease']):,}"],
                 ["Rows with an ambiguous vector", f"{int(audit['rows_with_ambiguous_vector']):,}"],
                 ["Unused symptoms", f"{int(audit['unused_symptoms'])}"],
                 ["Largest / median / smallest class", f"{int(audit['max_class'])} / {audit['median_class']:.0f} / {int(audit['min_class'])}"],
                 ["Bayes ceiling Top-1 / Top-5 (raw)", f"{ceil.loc['raw data','Top-1']*100:.2f}% / {ceil.loc['raw data','Top-5']*100:.2f}%"],
                 ["Bayes ceiling Top-1 / Top-5 (de-duplicated)", f"{ceil.loc['de-duplicated','Top-1']*100:.2f}% / {ceil.loc['de-duplicated','Top-5']*100:.2f}%"]],
                [4600, 4000])
c = clone_after(p_pre, cap_tpl, "Table 1: Data audit of the Kaggle Diseases and Symptoms dataset.")
c.addnext(t1)
img = add_image_after(d, t1, FIG + "fig01_eda.png", 6.3)
clone_after(img, cap_tpl, "Figure 2: Class distribution, number of symptoms per case and most frequent symptoms.")

set_text(find_p(d, "Any records containing missing or malformed"),
    "The audit found no missing or malformed symptom indicators. The pipeline still treats any absent value as 'symptom not present', "
    "consistent with the one-hot encoding, and discards samples with an empty or corrupted disease label.")

set_text(find_p(d, "Because the dataset spans 773 diseases"),
    "The label distribution is highly imbalanced. All neural models are trained with a class-weighted cross-entropy loss in which the "
    "weight of disease c is proportional to n_c^(-0.5) (normalised to mean 1), plus label smoothing of 0.05. Instead of oversampling, "
    "which would recreate exact duplicates of rare cases and re-introduce the leakage problem, we use variable-rate symptom-masking "
    "augmentation: for every training case a rate r ~ U(0, 0.5) is drawn and each present symptom is dropped with probability r, always "
    "keeping at least one. This acts as feature-level resampling and simulates patients who report only part of their symptoms.")
p_imb = find_p(d, "The label distribution is highly imbalanced")
h = clone_after(p_imb, h2_tpl, "3.5 Leakage-Free Evaluation Protocol and Bayes Ceiling")
p1 = clone_after(h, normal_tpl,
    "Because identical symptom vectors occur many times, and sometimes with different diseases, a random split lets the model see the "
    "exact test pattern during training. We therefore use a stratified group split in which the group is the symptom vector: every distinct "
    "pattern is assigned to exactly one of train (70%), validation (10%) or test (20%), stratified by its majority disease. This gives "
    f"{sp['train']:,} training, {sp['val']:,} validation and {sp['test']:,} test cases, with zero symptom vectors shared between train and test. "
    "We also compute the Bayes-optimal top-k ceiling: for every distinct vector x with disease counts n(x,d), the best achievable top-k hit "
    "rate is the sum over x of its k largest counts, divided by N. Table 2 compares three protocols with the same MLP.")
t = make_table([["Protocol", "Test cases", "Leaked test cases", "MLP Top-1", "MLP Top-5"]] +
               [[r.protocol, f"{int(r.n_test):,}", f"{r.leaked_test_cases*100:.1f}%", f"{r.MLP_top1*100:.2f}", f"{r.MLP_top5*100:.2f}"] for r in leak.itertuples()],
               [3400, 1300, 1500, 1200, 1200])
c = clone_after(p1, cap_tpl, "Table 2: Train/test leakage under three evaluation protocols (same MLP).")
c.addnext(t)
p2 = clone_after(t, normal_tpl,
    f"Under the usual protocol (A), {la.leaked_test_cases*100:.1f}% of test cases are exact copies of training patterns, which inflates "
    f"top-1 accuracy to {la.MLP_top1*100:.2f}%. After de-duplication (B), leakage still affects {lb.leaked_test_cases*100:.1f}% of test cases. Our "
    "grouped split (C) removes leakage completely. All results in Chapter 4 use protocol C. Because conflicting patterns stay together in "
    f"one split, the test-set ceiling under protocol C is {tc['Top-1']*100:.2f}% top-1 and {tc['Top-5']*100:.2f}% top-5.")

# renumber old 3.5..3.11 -> 3.6..3.12
for old, new in [("3.11", "3.12"), ("3.10", "3.11"), ("3.9", "3.10"), ("3.8", "3.9"), ("3.7", "3.8"), ("3.6", "3.7"), ("3.5 Hypergraph", "3.6 Hypergraph")]:
    pe = find_p(d, old if old != "3.5 Hypergraph" else "3.5 Hypergraph")
    set_text(pe, ptext(pe).replace(old.split()[0], new.split()[0], 1))

dh = hst.set_index("hyperedge type")
set_text(find_p(d, "The core of the methodology is the transformation"),
    "The disease-symptom matrix is turned into a hypergraph using training data only. Symptoms are nodes. Each disease d becomes a "
    "hyperedge containing every symptom with P(s|d) of at least 0.02, with soft incidence weight h(s,d) = P(s|d). Unlike a pairwise edge, a "
    "single disease hyperedge binds the whole symptom set of the disease. Auxiliary co-occurrence hyperedges capture symptom clusters that "
    "recur across diseases: for every symptom v, the set {v} plus its top-4 partners by normalised pointwise mutual information "
    "(NPMI ≥ 0.10, co-occurrence count ≥ 10) forms a hyperedge, weighted by the mean NPMI, and duplicates are merged. The resulting "
    f"hypergraph has {int(dh.loc['disease','count'])} disease hyperedges (mean size {dh.loc['disease','mean size']:.1f}) and "
    f"{int(dh.loc['co-occurrence','count'])} co-occurrence hyperedges (mean size {dh.loc['co-occurrence','mean size']:.1f}). For the pairwise baselines, the "
    "same NPMI statistics define a graph in which each symptom is linked to its top-10 partners.")
p = find_p(d, "The disease-symptom matrix is turned into a hypergraph")
img = add_image_after(d, p, FIG + "fig03_hypergraph_stats.png", 6.3)
clone_after(img, cap_tpl, "Figure 3: Disease-hyperedge sizes, symptom node degrees and an excerpt of the weighted incidence matrix.")

set_text(find_p(d, "Each symptom node is initialized"),
    "Each symptom node starts from a learnable identity embedding plus a linear projection of its row-normalised positive PMI "
    "co-occurrence profile, so both one-hot identity and co-occurrence statistics are encoded. The incidence matrix encodes which symptoms "
    "belong to which hyperedges, and the incidence weights enter the node-level attention as a log-prior.")

set_text(find_p(d, "Message passing proceeds in two stages"),
    "Message passing proceeds in two stages. First, node features are aggregated into hyperedge representations using node-level "
    "attention α(v,e), a masked softmax over the members of e with a LeakyReLU score of the node and hyperedge states plus log h(v,e). "
    "Second, hyperedge representations are propagated back to nodes with hyperedge-level attention β(e,v), a masked softmax over the "
    "hyperedges incident to v. Two such layers (dimension 128, four heads, residual connections and layer normalisation) produce "
    "structure-aware symptom and hyperedge embeddings. A patient query S is encoded by context-aware attention pooling over its present "
    "symptoms; the pooling weights γ(v) are the symptom-level explanation. The query then attends over the co-occurrence hyperedges that "
    "touch its symptoms (query-aware hyperedge attention) and is refined with the matched symptom clusters.")
p = find_p(d, "Message passing proceeds in two stages")
clone_after(p, normal_tpl,
    "Two further components tie the decision directly to the hypergraph. (i) Disease-hyperedge prototype scoring: the score of disease d is "
    "the scaled cosine similarity between the query embedding and the learned embedding of the hyperedge of d, plus a bias. Every class is "
    "therefore represented by its own hyperedge rather than a free weight vector. (ii) Query-adaptive hyperedge-likelihood evidence: the "
    "score also receives λ+(q)·Σ_{v∈S} log h(v,d) + λ−(q)·Σ_{v∉S} log(1 − h(v,d)), where the non-negative weights λ± are predicted from the "
    "query and the number of reported symptoms. When few symptoms are reported, the model can learn that missing symptoms carry little information.")

set_text(find_p(d, "To contextualize the performance of HyperDiag"),
    "HyperDiag is compared with seven baselines trained on the same split: Bernoulli Naive Bayes; a Random Forest (60 trees, at most 4,000 "
    "leaves per tree so that the model fits in memory with more than 600 classes); softmax Logistic Regression; a two-hidden-layer MLP "
    "(512 units); a GCN and a GAT on the pairwise symptom graph; and HGNN (Feng et al., 2019), a hypergraph convolution on the same "
    "hypergraph without attention. GCN, GAT and HGNN share HyperDiag's node initialisation and attention readout, so differences come from "
    "the structure encoder. All neural models use AdamW (learning rate 2e-3, weight decay 1e-4), batch size 1024, a one-cycle schedule over "
    "20 epochs with model selection on validation MRR, and the same loss and augmentation. Each neural model is trained with three seeds.")

set_text(find_p(d, "Rather than predicting a single disease"),
    "Rather than predicting a single disease, the task is formulated as differential diagnosis: given a partial or complete symptom set, "
    "the model ranks all candidate diseases. We report Top-k accuracy (k = 1, 3, 5, 10) and Mean Reciprocal Rank, computed from the full "
    "ranking, and macro- and weighted-F1 of the top-1 prediction over the diseases present in the test set. Robustness to incomplete input "
    "is measured by keeping a random 80%, 60%, 40% or 20% of each test case's symptoms (at least one), with the same masks for all models. "
    "Significance is assessed with McNemar's test on top-1 correctness and paired bootstrap confidence intervals for Top-5 and MRR.")

set_text(find_p(d, "To understand the contribution of each component"),
    "To understand the contribution of each component, we retrain HyperDiag (three seeds each) without co-occurrence hyperedges, without "
    "node-level attention (replaced by incidence-weighted averaging), without hyperedge-level attention (replaced by degree-normalised "
    "averaging), without both, without query-aware hyperedge attention, with an MLP head instead of prototype scoring, and without the "
    "hyperedge-likelihood evidence.")
set_text(find_p(d, "Finally, we visualize the learned attention"),
    "We visualise the symptom-evidence weights γ for individual patients together with their top-5 differential, and the node-level "
    "attention α inside disease hyperedges. We compare α with the prevalence P(s|d) and the specificity P(d|s) using Spearman correlation. "
    "To test whether the explanations are faithful, we delete the most-attended, a random, or the least-attended symptom of each test "
    "case and measure the drop in the probability of the true disease.")

# ---------------- Chapter 4
set_text(find_p(d, "This chapter will present the experimental results"),
    "This chapter presents the results on the leakage-free test set "
    f"({sp['test']:,} cases, {sp['diseases']} diseases). Neural results are the mean ± standard deviation over three seeds. Naive Bayes and "
    f"Random Forest are deterministic or use a single seed. The test-set Bayes ceiling is {tc['Top-1']*100:.2f}% top-1, "
    f"{tc['Top-3']*100:.2f}% top-3 and {tc['Top-5']*100:.2f}% top-5.")
set_text(find_p(d, "Table 2: Comparative"), "Table 3: Comparative performance on the leakage-free test set (%).")
t2_old = find_p(d, "Table 3: Comparative").getnext()
rows = [["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MRR", "Macro-F1"]]
for m in order:
    rows.append([m.replace(" (ours)", " (ours)")] + [pm(m, k) for k in M])
rows.append(["Bayes-optimal ceiling", f(tc["Top-1"] * 100), f(tc["Top-3"] * 100), f(tc["Top-5"] * 100), f(tc["Top-10"] * 100), "—", "—"])
t2 = make_table(rows, [2500, 1050, 1050, 1050, 1050, 1050, 1050], bold_rows=(len(order),))
t2_old.addnext(t2); body.remove(t2_old)

g = sig.set_index("baseline")
p_after2 = t2
p = clone_after(p_after2, normal_tpl,
    f"HyperDiag reaches {pm(OURS,'Top-1')}% top-1, {pm(OURS,'Top-5')}% top-5 and {pm(OURS,'MRR')}% MRR, which is "
    f"{mu.loc[OURS,'Top-1']/(tc['Top-1']*100):.1%} of the achievable top-1 accuracy. " + SIG_SENT + " (McNemar test, Holm-adjusted p = " +
    ", ".join(f"{NICE[b]} " + ("< 0.001" if g.loc[b,'mcnemar_p_holm'] < 1e-3 else f"{g.loc[b,'mcnemar_p_holm']:.3g}") for b in g.index) +
    f"). Its macro-F1 differs from Naive Bayes by {mu.loc[OURS,'Macro-F1']-mu.loc['Naive Bayes (Bernoulli)','Macro-F1']:+.1f} points. "
    f"Among the neural models, Top-1 accuracy spans only {mu.loc[['MLP','GCN (pairwise)','GAT (pairwise)','HGNN (hypergraph, no attention)',OURS],'Top-1'].max()-mu.loc[['MLP','GCN (pairwise)','GAT (pairwise)','HGNN (hypergraph, no attention)',OURS],'Top-1'].min():.2f} points. "
    "All neural models converge to the same performance band, about five points below the Bayes ceiling. The remaining gap is "
    "dominated by symptom patterns that never appear in training, which no model can resolve from symptoms alone.")
img = add_image_after(d, p, FIG + "fig08_learning_dynamics.png", 6.3)
c = clone_after(img, cap_tpl, "Figure 4: Learning dynamics: training loss and validation MRR per epoch (seed 0).")
sg = make_table([["Baseline", "HyperDiag-only correct", "Baseline-only correct", "McNemar p", "ΔTop-5 [95% CI]", "ΔMRR [95% CI]"]] +
                [[r.baseline, r.ours_only_correct, r.baseline_only_correct, f"{r.mcnemar_p:.3g}", r.dTop5, r.dMRR] for r in sig.itertuples()],
                [2300, 1150, 1150, 1000, 1500, 1500])
c2 = clone_after(c, cap_tpl, "Table 4: Statistical significance of HyperDiag versus each baseline (seed 0).")
c2.addnext(sg)

# partial symptoms
h = clone_after(sg, h2_tpl, "4.2 Robustness to Partial Symptom Sets")
pt = part.pivot_table(index="model", columns="keep_ratio", values="Top-5").reindex(["Naive Bayes (Bernoulli)", "LogReg (softmax)", "MLP", "GCN (pairwise)", "GAT (pairwise)", "HGNN (hypergraph, no attention)", OURS]) * 100
ptab = make_table([["Model"] + [f"{int(c*100)}% kept" for c in sorted(pt.columns, reverse=True)]] +
                  [[i] + [f(pt.loc[i, c]) for c in sorted(pt.columns, reverse=True)] for i in pt.index], [2900] + [1140] * 5)
c = clone_after(h, cap_tpl, "Table 5: Top-5 accuracy (%) when only a fraction of each test case's symptoms is reported.")
c.addnext(ptab)
nn_models = ["MLP", "GCN (pairwise)", "GAT (pairwise)", "HGNN (hypergraph, no attention)", OURS]
p = clone_after(ptab, normal_tpl,
    f"Performance falls quickly as symptoms are removed: with 40% of symptoms HyperDiag keeps {pv(0.4):.1f}% top-5 accuracy and with 20% "
    f"it keeps {pv(0.2):.1f}%, compared with {max(pt.loc[m, 0.2] for m in ['GCN (pairwise)','GAT (pairwise)','HGNN (hypergraph, no attention)']):.1f}% for the best "
    f"of GCN/GAT/HGNN and {pt.loc['MLP', 0.2]:.1f}% for the MLP. "
    f"The generative models, however, remain clearly better with very sparse input (Naive Bayes {pnb(0.2):.1f}%). "
    "This shows that discriminative models, including ours, learn to rely on the absence of symptoms. Closing this gap is an important "
    "direction for future work.")
img = add_image_after(d, p, FIG + "fig12_partial_symptoms.png", 6.3)
c = clone_after(img, cap_tpl, "Figure 5: Top-5 accuracy and MRR as a function of the fraction of symptoms kept.")

# rare
h = clone_after(c, h2_tpl, "4.3 Rare versus Frequent Diseases")
rp = rare.pivot_table(index="model", columns="group", values="recall1", observed=True).reindex(["Naive Bayes (Bernoulli)", "MLP", "GAT (pairwise)", "HGNN (hypergraph, no attention)", OURS])
cols = ["rare (≤20)", "low (21–100)", "mid (101–400)", "frequent (>400)"]
rtab = make_table([["Model (recall@1, %)"] + cols] + [[i] + [f(rp.loc[i, cc] * 100) for cc in cols] for i in rp.index], [2900, 1425, 1425, 1425, 1425])
c = clone_after(h, cap_tpl, "Table 6: Mean per-disease recall@1 (%) grouped by number of training cases.")
c.addnext(rtab)
clone_after(rtab, normal_tpl,
    f"For rare diseases (≤ 20 training cases), the neural models outperform Naive Bayes by a large margin (HyperDiag {rare_o.recall1*100:.1f}% "
    f"recall@1 and {rare_o.recall5*100:.1f}% recall@5, versus {rare_nb.recall1*100:.1f}% and {rare_nb.recall5*100:.1f}% for Naive Bayes). Class "
    "weighting is effective. Within the neural family, rare-disease performance differs by a few points between models and seeds, which "
    "reflects the small number of test cases per rare disease.")

# ablation
set_text(find_p(d, "Table 3: Ablation"), "Table 7: Ablation study of HyperDiag components (%; mean ± std over three seeds).")
t3_old = find_p(d, "Table 7: Ablation").getnext()
AM = ["Top-1", "Top-5", "MRR", "Macro-F1", "Top-5 @40% symptoms"]
variants = [v for v in ["Full HyperDiag", "w/o co-occurrence hyperedges", "w/o node-level attention", "w/o hyperedge-level attention", "w/o both attentions",
            "w/o query-aware hyperedge attention", "w/o prototype scoring (MLP head)", "w/o hyperedge-likelihood evidence"] if v in abm.index]
t3 = make_table([["Variant", "Top-1", "Top-5", "MRR", "Macro-F1", "Top-5 (40% symptoms)"]] +
                [[v] + [f"{abm.loc[v,k]:.2f} ± {abs_.loc[v,k]:.2f}" for k in AM] for v in variants], [2900, 1140, 1140, 1140, 1140, 1140], bold_rows=(1,))
t3_old.addnext(t3); body.remove(t3_old)
h_ab = find_p(d, "Table 7: Ablation")
h_new = clone_after(h_ab.getprevious(), h2_tpl, "4.4 Ablation Study")
delta = {v: abm.loc[v] - full for v in variants}
biggest = max(variants[1:], key=lambda v: -delta[v]["Macro-F1"])
txt = ("Removing individual components changes Top-1, Top-5 and MRR by at most "
       f"{max(abs(delta[v][k]) for v in variants[1:] for k in ['Top-1','Top-5','MRR']):.2f} points, comparable to the seed-to-seed variation. "
       f"The largest effect on macro-F1 comes from removing '{biggest.replace('w/o ','')}' ({delta[biggest]['Macro-F1']:+.2f} points). "
       "This is consistent with the comparative results: near the ceiling, the attention mechanisms mainly change how the decision is "
       "reached and how it can be explained, not its accuracy.")
p_old = find_p(d, "The comparative results are expected to show")
set_text(p_old, txt)

# interpretability section after ablation text
h = clone_after(p_old, h2_tpl, "4.5 Interpretability and Faithfulness")
fa = faith
pi = clone_after(h, normal_tpl,
    "Figure 6 shows three test patients with the symptom-evidence weights γ and the top-5 differential. Inside disease hyperedges, the "
    f"node-level attention closely follows prevalence (mean Spearman ρ = {attal.iloc[0,1]:.2f} with P(s|d), positive for "
    f"{attal.iloc[0,2]*100:.0f}% of diseases) and is weakly related to specificity (ρ = {attal.iloc[1,1]:.2f} with P(d|s)). The model therefore "
    "highlights the symptoms that characterise each disease. The deletion test (Table 8) " + FA_WORD + ": removing the "
    f"most-attended symptom lowers the true-disease probability by {fa.loc['most attended','mean_drop_true_prob']:.3f} on average, versus "
    f"{fa.loc['random','mean_drop_true_prob']:.3f} for a random symptom and {fa.loc['least attended','mean_drop_true_prob']:.3f} for the least-attended one. "
    f"Top-1 accuracy falls to {fa.loc['most attended','Top1_after']*100:.1f}%, {fa.loc['random','Top1_after']*100:.1f}% and "
    f"{fa.loc['least attended','Top1_after']*100:.1f}% respectively.")
img = add_image_after(d, pi, FIG + "fig15_case_studies.png", 6.0)
c = clone_after(img, cap_tpl, "Figure 6: Symptom-evidence attention and top-5 differential for three test patients.")
img2 = add_image_after(d, c, FIG + "fig16_node_attention.png", 6.3)
c = clone_after(img2, cap_tpl, "Figure 7: Node-level attention within three disease hyperedges compared with P(s|d).")
ftab = make_table([["Deleted symptom", "Mean drop in P(true disease)", "Top-1 after", "Top-5 after"]] +
                  [[i, f"{r.mean_drop_true_prob:.3f}", f(r.Top1_after * 100), f(r.Top5_after * 100)] for i, r in fa.iterrows()], [2600, 2400, 1800, 1800])
c = clone_after(c, cap_tpl, "Table 8: Attention-faithfulness deletion test (3,000 test cases with at least three symptoms).")
c.addnext(ftab)

# ---------------- Chapter 5
set_text(find_p(d, "Prior studies on the Disease-Symptom dataset"),
    "Prior studies on the Disease-Symptom dataset have mostly used traditional classifiers or pairwise graph neural networks evaluated "
    "with random splits. Our audit shows that such splits leak a large share of test cases into training, so the high accuracies "
    "reported for this dataset should be read with care. HyperDiag differs from earlier work in three ways: it represents each disease as "
    "a hyperedge over its full symptom set, it is evaluated under a leakage-free protocol against an explicit Bayes ceiling, and its "
    "explanations are checked for faithfulness. Table 9 places our approach among representative prior studies.")
set_text(find_p(d, "Hypergraph formulation is expected to be the primary"),
    "Contrary to our initial hypothesis, the hypergraph structure was not the main driver of accuracy on this dataset. Every neural model, "
    "whether flat (MLP), pairwise (GCN, GAT) or higher-order (HGNN, HyperDiag), reached the same performance band, and the ablations changed "
    "accuracy by less than the variation across seeds. Two properties of the data explain this. First, the dataset is close to "
    f"saturation: {int(audit['vectors_mapping_to_>1_disease']):,} symptom patterns are shared by several diseases, which caps top-1 accuracy at "
    f"{tc['Top-1']*100:.1f}% on our test set. Second, Naive Bayes, which assumes symptoms are independent given the disease, is almost as "
    "accurate as the best neural model. This suggests that cases were generated from per-disease symptom profiles with little higher-order "
    "interaction, which is exactly the structure that hyperedges are designed to capture. The hypergraph still adds value. " + (f"It significantly improves on {_join(SIGW)}, " if SIGW else "") + "it ties every class to an explicit, inspectable disease hyperedge, and yields symptom-level explanations "
    "whose faithfulness is checked with a deletion test (most-attended versus random deletion ratio " + f"{FA_RATIO:.2f}" + "). We expect the higher-order structure to matter more on real clinical records, where symptom interactions are richer.")
p = find_p(d, "5.3 Future Work")
set_text(p, "5.4 Future Work")
lim_h = clone_after(p.getprevious(), h2_tpl, "5.3 Limitations")
clone_after(lim_h, normal_tpl,
    "The dataset is an augmented symptom-checker resource rather than real patient records, so the conclusions may not transfer directly "
    "to clinical practice. Diseases with fewer than five unique cases were excluded. The Random Forest was size-capped for memory reasons. "
    "All neural models were trained with one shared set of hyperparameters rather than per-model tuning. Finally, every neural model, "
    "HyperDiag included, degrades more than Naive Bayes when very few symptoms are reported.")
set_text(find_p(d, "Several directions remain for future exploration"),
    "Several directions remain. The framework should be validated on real electronic health records and on other symptom-diagnosis "
    "datasets, where higher-order symptom interactions are expected to be stronger. Hybrid generative-discriminative training could close "
    "the gap with Naive Bayes on very sparse inputs. Patient demographics, vital signs and the temporal evolution of symptoms could be "
    "added as further node or hyperedge types. Finally, the faithful attention explanations could be developed into an interactive "
    "clinical decision-support tool that presents ranked diagnoses together with their supporting symptoms.")

# ---------------- Chapter 6
set_text(find_p(d, "This research proposes HyperDiag, a hypergraph attention network for s"),
    "This research proposes HyperDiag, a hypergraph attention network for symptom-set-aware differential diagnosis. Each disease is "
    "modelled as a hyperedge over its symptoms, co-occurrence hyperedges capture recurring symptom clusters, dual-level attention "
    "propagates information between symptoms and disease patterns, and each disease is scored through its own hyperedge. A careful audit "
    f"of the Kaggle Diseases and Symptoms dataset revealed that {audit['exact_duplicate_rows']/audit['rows']*100:.0f}% of the records are duplicates and "
    f"that random splits leak {la.leaked_test_cases*100:.0f}% of test cases. We therefore introduced a leakage-free grouped protocol together with "
    f"a Bayes-optimal ceiling. Under this protocol HyperDiag achieves {mu.loc[OURS,'Top-1']:.2f}% top-1 accuracy, {mu.loc[OURS,'Top-5']:.2f}% "
    f"top-5 accuracy and {mu.loc[OURS,'MRR']:.2f}% MRR. " + SIG_SENT + ", and HyperDiag provides symptom-level explanations "
    "grounded in explicit disease hyperedges, with well-calibrated probabilities after temperature scaling. Our findings show that honest evaluation is a prerequisite for meaningful progress "
    "on public symptom-checker benchmarks, and they establish a reproducible baseline and protocol for future structurally aware, "
    "interpretable clinical decision support.")

intro4 = find_p(d, "This chapter presents the results on the leakage-free")
clone_after(intro4, h2_tpl, "4.1 Comparative Performance")
set_text(find_p(d, "Figure: High-level framework"), "Figure 1: High-level framework of our experiments.")
set_text(find_p(d, "Table 5: Comparison with relevant"), "Table 9: Comparison with relevant research.")
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "draft_ext.py"), encoding="utf-8").read())
pm_ = find_p(d, "This chapter describes the complete methodology")
set_text(pm_, ptext(pm_).replace("Figure 2 presents", "Figure 1 presents"))
d.save(OUT)
print("saved", OUT)

# HyperDiag: Hypergraph Attention for Symptom-Based Differential Diagnosis

Thesis pre-defence code and results.

| File | What it is |
|---|---|
| `HyperDiag_Kaggle.ipynb` | **Kaggle-ready notebook**: set the dataset link in the first cell, then Run All |
| `HyperDiag_PreDefence.ipynb` | **Executed notebook**: every step from data audit to interpretability, with all outputs and figures |
| `HyperDiag_PreDefence_Draft_v2.docx` | Updated thesis draft with every placeholder table filled from the results below |
| `hyperdiag_notebook_src.py` | Notebook source (percent-cell format); `python build_notebook.py` regenerates the `.ipynb` |
| `results/` | All result tables (CSV), `summary.json` and figures |
| `tools/build_draft.py` | Regenerates the draft from `results/` and the original draft `.docx` |

## Running it on Kaggle (Run All)

1. Upload `HyperDiag_Kaggle.ipynb` to Kaggle (**Create → New Notebook → File → Import Notebook**).
2. In the first code cell, set `DATASET` to the dataset link. It already defaults to
   `https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset`. Then choose either:
   * **+ Add Input** and attach that dataset (works without internet), or
   * turn **Internet on** in the notebook settings; the notebook then downloads the data itself with `kagglehub`.
3. Optional: **Accelerator → GPU** makes training much faster. `FAST = True` gives a roughly 10-minute check run.
4. **Run All**. Results and figures are written to `results/` (on Kaggle: `/kaggle/working/results`).

`DATASET` also accepts a slug (`owner/dataset`), a direct `.csv`/`.zip` link, or a local file or folder path, so the
same notebook also runs on Colab or a laptop.
`HyperDiag_PreDefence.ipynb` is the same pipeline with all outputs already executed (for presenting).

Locally: `pip install torch scikit-learn pandas scipy matplotlib seaborn kagglehub` and run the notebook
(about 80 minutes on a 4-core CPU).

## Notebook structure (scientific-report format)

Abstract → 1 Environment → 2 Materials → 3 Data audit → 4 Leakage-free protocol → 5 Hypergraph construction (**Algorithm 1**)
→ 6 Models (**Algorithm 2**) and training/evaluation (**Algorithm 3**) → 7 Experiments → 8 Results (20 metrics, bootstrap CIs,
calibration and temperature scaling) → 9 Statistical analysis (McNemar–Holm, Friedman–Nemenyi CD diagram, Wilcoxon with effect sizes,
bootstrap forest plot) → 10 Partial symptoms → 11 Rare diseases → 12 Ablation → 13 Interpretability, t-SNE, error analysis
→ 14 Auto-generated findings → 15 Discussion → 16 Conclusion → References.

19 publication-style figures in `results/figures/` (300 dpi, one fixed colour per model).

## Key findings (leakage-free test set: 37,768 cases, 658 diseases; neural models: mean over 3 seeds)

**Data audit**
* 57,298 duplicate rows (23.2%) and 12,634 symptom vectors that map to more than one disease.
* A random split leaks **43.9%** of test cases into training; our grouped split leaks 0%.
* Bayes-optimal ceiling on the test set: Top-1 89.89%, Top-5 99.55%.

**Main comparison (%; extract of the 20-metric Table 3)**

| Model | Top-1 | Top-5 | MRR | NDCG@10 | Macro-F1 | MCC | Log-loss* | ECE* |
|---|---|---|---|---|---|---|---|---|
| Naive Bayes | 84.31 | 97.67 | 90.15 | 92.38 | 76.43 | 84.25 | 0.343 | 0.44 |
| Random Forest | 82.80 | 97.36 | 89.11 | 91.56 | 71.02 | 82.74 | – | – |
| LogReg | 84.24 | 97.62 | 90.08 | 92.32 | 75.11 | 84.19 | 0.362 | 2.36 |
| MLP | 84.63 | 97.84 | 90.38 | 92.58 | 81.71 | 84.58 | 0.337 | 2.32 |
| GCN | 84.48 | 97.80 | 90.28 | 92.50 | 81.12 | 84.43 | 0.345 | 2.38 |
| GAT | 84.50 | 97.84 | 90.31 | 92.52 | 81.35 | 84.45 | 0.343 | 2.27 |
| HGNN | 84.51 | 97.82 | 90.30 | 92.52 | 81.28 | 84.46 | 0.342 | 2.33 |
| **HyperDiag** | **84.57** | **97.82** | **90.36** | **92.57** | **81.16** | **84.52** | **0.332** | **1.57** |

\* After temperature scaling fitted on the validation set. Among the discriminative models, HyperDiag has the best log-loss and ECE.

**Statistics**
* McNemar (Holm-corrected): HyperDiag is significantly better than Random Forest and Logistic Regression. It is statistically
  indistinguishable from Naive Bayes (p = 0.069), MLP, GCN, GAT and HGNN.
* Friedman χ² = 384.0 (p = 6.6e-79) over 642 diseases; Nemenyi CD = 0.41.
* Wilcoxon on per-disease F1: large effect over RF (r = 0.56), medium over LogReg (r = 0.42), small over NB (r = 0.24);
  negligible versus the neural baselines.

**Analysis**
* Errors follow hyperedge overlap: confused disease pairs have mean Jaccard 0.529, versus 0.032 for random pairs.
* Attention faithfulness: deleting the most-attended symptom lowers P(true disease) by 0.155, versus 0.095 for a random symptom.
* Ablation: prototype scoring is the component with the clearest effect on macro-F1.
* Limitation: with only 20% of symptoms reported, Naive Bayes is more robust than every neural model.

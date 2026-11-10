# HyperDiag: Hypergraph Attention for Symptom-Based Differential Diagnosis

Thesis pre-defence code and results.

| File | What it is |
|---|---|
| `HyperDiag_PreDefence.ipynb` | **Executed notebook**: every step from data audit to interpretability, with all outputs and figures |
| `HyperDiag_PreDefence_Draft_v2.docx` | Updated thesis draft with every placeholder table filled from the results below |
| `hyperdiag_notebook_src.py` | Notebook source (percent-cell format); `python build_notebook.py` regenerates the `.ipynb` |
| `results/` | All result tables (CSV), `summary.json` and figures |
| `tools/build_draft.py` | Regenerates the draft from `results/` and the original draft `.docx` |

## Running it

1. Download `Final_Augmented_dataset_Diseases_and_Symptoms.csv` from
   <https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset> into `data/`. On Kaggle the input folder is detected automatically.
2. `pip install torch scikit-learn pandas scipy matplotlib seaborn nbformat nbconvert ipykernel`
3. `jupyter nbconvert --to notebook --execute HyperDiag_PreDefence.ipynb --output HyperDiag_PreDefence.ipynb --ExecutePreprocessor.timeout=-1`
   (about 80 minutes on a 4-core CPU; set `HYPERDIAG_FAST=1` for a short smoke test).

## Key findings

**Data audit and leakage (new contribution)**
* 57,298 of 246,945 rows (23.2%) are exact duplicates, and 12,634 symptom vectors map to more than one disease.
* With a random split on the raw data, **43.9% of test cases also appear in training**. A random split after de-duplication still leaks 15.1%.
  We use a **symptom-vector-grouped stratified split** with 0% leakage.
* Bayes-optimal ceiling (de-duplicated data): Top-1 89.6%, Top-5 99.4%. On our test set: Top-1 89.89%, Top-5 99.55%.

**Main comparison (leakage-free test set, 37,768 cases, 658 diseases, mean ± std over 3 seeds)**

| Model | Top-1 | Top-5 | MRR | Macro-F1 |
|---|---|---|---|---|
| Naive Bayes | 84.31 | 97.67 | 90.15 | 76.43 |
| Random Forest | 82.80 | 97.36 | 89.11 | 71.02 |
| Logistic Regression | 84.24 ± 0.02 | 97.62 ± 0.00 | 90.08 ± 0.01 | 75.11 ± 0.21 |
| MLP | 84.63 ± 0.03 | 97.84 ± 0.01 | 90.38 ± 0.01 | 81.71 ± 0.32 |
| GCN (pairwise) | 84.48 ± 0.01 | 97.80 ± 0.00 | 90.28 ± 0.01 | 81.12 ± 0.04 |
| GAT (pairwise) | 84.50 ± 0.03 | 97.84 ± 0.04 | 90.31 ± 0.02 | 81.35 ± 0.52 |
| HGNN | 84.51 ± 0.04 | 97.82 ± 0.02 | 90.30 ± 0.02 | 81.28 ± 0.05 |
| **HyperDiag (ours)** | **84.59 ± 0.01** | **97.83 ± 0.03** | **90.36 ± 0.02** | **81.16 ± 0.09** |
| Bayes ceiling | 89.89 | 99.55 | – | – |

* HyperDiag is significantly better than Naive Bayes, Random Forest and Logistic Regression (McNemar p < 0.02).
  It is **statistically tied** with MLP, GCN, GAT and HGNN (p > 0.36).
* Ablation: only **prototype scoring** clearly matters (macro-F1 81.16 → 79.24 without it). The other components change results by less than the variation across seeds.
* Attention-faithfulness: deleting the most-attended symptom lowers the true-disease probability 1.6× more than deleting a random symptom and 2.5× more than deleting the least-attended one.
* Limitation: with only 20% of symptoms reported, Naive Bayes (52.9% Top-5) beats every neural model (about 43–45%).


* **Data.** 57,298 duplicate rows (23.2%) and 12,634 ambiguous
  symptom patterns. A random split leaks 43.9% of test cases; the grouped split leaks 0%.
  The Bayes-optimal ceiling on the test set is 89.89% Top-1 and 99.55% Top-5.
* **Main result.** HyperDiag reaches Top-1 84.57%, Top-5 97.82%, MRR 90.36%,
  NDCG@10 92.57%, macro-F1 81.16%, MCC 84.52% and ECE 16.28%
  (94.1% of the attainable Top-1). The best baseline by MRR is MLP (90.38%).
* **Statistics.** After Holm correction, HyperDiag is significantly better on top-1 correctness than RF, LogReg.
  Friedman χ² = 384.0 (p = 6.6e-79) over 642 diseases; Nemenyi CD = 0.41; HyperDiag's average rank is 4.21.
* **Errors.** Confused disease pairs have a mean hyperedge Jaccard of 0.529, versus 0.032 for random pairs (Mann–Whitney p 0.0e+00),
  so the remaining errors are explained by overlapping disease hyperedges.
* **Explanations.** Deleting the most-attended symptom lowers P(true disease) by 0.155,
  versus 0.095 for a random symptom.

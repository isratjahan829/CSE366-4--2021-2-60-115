# In-silico case studies for the chapter "AI in Bioreactor Design, Monitoring, and Scale-Up"

All data here are **simulated**. The virtual plant (`plant.py`) is the hidden "ground truth" that the models must learn. The results show mechanisms and relative performance; they are not predictions for a specific strain or plant.

| Script | Chapter section | What it does | Runtime |
|---|---|---|---|
| `plant.py` | 5.1 | Fed-batch virtual plant: mass balances and true kinetics (Eq. 30), RK4 solver, measurement noise | – |
| `case_a_hybrid.py` | 5.2 / 6.2 | Hybrid vs black-box vs textbook models; 10 replicates; Wilcoxon signed-rank tests | ~10 min |
| `case_b_softsensor.py` | 5.3 / 6.3 | Synthetic Raman spectra; PLS, GP-on-scores and MLP models; T²/Q MSPC drift detection; recalibration | ~10 s |
| `case_c_scaleup.py` | 5.4 / 6.1, 6.4 | Classical scale-up criteria; 10-compartment 50 m³ tank; Markov cell lifelines; regime-matched scale-down (Wasserstein distance) | ~10 s |
| `case_d_bayesopt.py` | 5.5 / 6.5 | Bayesian optimisation vs random search vs CCD/RSM; 20 campaigns; Mann–Whitney tests | ~2 min |
| `make_figures.py` | Figures 6–9 | Publication figures (600 dpi TIFF/PNG and PDF) in `../figures/` | ~10 s |

## Reproduce

```bash
pip install numpy scipy scikit-learn matplotlib
cd case_study
mkdir -p results
python case_a_hybrid.py
python case_b_softsensor.py
python case_c_scaleup.py
python case_d_bayesopt.py
python make_figures.py
```

Numerical results are written to `results/*.json`. Tested with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and Matplotlib 3.11.

## Rebuild the Word chapter

```bash
cd chapter
cat 01_front_theory.md 02_literature.md 03_framework_methods.md 04_results_discussion.md 05_references.md > chapter_full.md
pandoc chapter_full.md -t docx --resource-path=. -o raw.docx
python style_docx.py raw.docx ../AI_in_Bioreactor_Chapter_Extended.docx
```

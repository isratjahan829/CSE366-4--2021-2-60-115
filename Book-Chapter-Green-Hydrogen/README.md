# Book chapter: Generative AI scenarios + GA stochastic sizing for solar green hydrogen

Code and data for the chapter submitted to the Springer volume
*Generative AI and Mathematical Optimization for Green Hydrogen Systems: Operational Trends and Technical Challenges*.
It builds on the CSE366 genetic-algorithm and battery-aware robot assignments in this repository.

## What it does
1. `src/data_prep.py` – loads two open NREL solar datasets, makes hourly GHI / temperature,
   computes Ineichen clear-sky irradiance and the clearness index; writes `data/processed.csv` and `data/processed.xlsx`.
2. `src/cvae.py` – conditional variational autoencoder written from scratch in NumPy (manual backprop + Adam).
3. `src/scenarios.py` – CVAE sampling, latent AR(1) synthetic years, Gaussian / bootstrap baselines, energy score, CRPS, PICP.
4. `src/h2system.py` – PV (Erbs + isotropic POA) → PEM electrolyser → battery → H2 tank simulator with rule-based EMS and LCOH.
5. `src/ga.py` – real-coded GA (tournament, BLX-α, Gaussian mutation, elitism) and a random-search baseline.
6. `src/run_all.py` – full experiment; writes `results/*.csv`, `results/results.json` and `figures/*.png`.
7. `src/analysis.py` – deep statistical analysis (descriptive stats, Wilcoxon/bootstrap/PIT/KS tests of the generator,
   persistence, Monte Carlo CIs + McNemar tests, monthly shortfall, Latin-hypercube global sensitivity, variance
   decomposition, cost–reliability trade-off, GA seed robustness, SAA convergence); writes `results/analysis_*`.

Final documents are in `final/` (chapter DOCX + PDF, abstract email, dataset).

## Data sources (downloaded from the pvlib-python GitHub repository)
| File | Content |
| --- | --- |
| `data/nsrdb_psm3_read.csv` | NSRDB PSM3 2017, 40.53 N 108.54 W, 2168 m (semi-arid Colorado), 30-min — `pvlib-python/tests/data/test_read_psm3.csv` |
| `data/tmy3_723170.csv` | TMY3 station 723170, Greensboro NC (humid subtropical), hourly — `pvlib-python/pvlib/data/723170TYA.CSV` |

## Run
```bash
pip install numpy pandas scipy pvlib matplotlib openpyxl
cd src && python run_all.py      # ~10 min on one CPU core
python analysis.py               # ~15 min, run after run_all.py
```

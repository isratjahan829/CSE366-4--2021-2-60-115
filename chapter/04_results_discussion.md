# 6. Results of the case studies

## 6.1 Classical scale-up criteria (worked example)

Table 5 evaluates Eq. (7) for a 2 L to 50 m³ scale-up, with linear scale factor $\lambda\approx29$ (vessel diameter 0.137 m to 3.99 m). Holding P/V constant, the most common rule, keeps power input but increases the single-impeller mixing time about tenfold, from 2.1 s to 19.5 s. In tall multi-impeller vessels the increase is larger still (90 s in Case C). Holding mixing time constant would need almost 1,000 kW/m³ and a tip speed of 56 m/s, which is physically impossible. Holding Reynolds number constant would leave the tank essentially unmixed. No classical criterion preserves the environment the cells experience. This is the quantitative motivation for regime matching.

Table: **Table 5.** Consequences of classical scale-up criteria for a geometrically similar 2 L → 50 m³ scale-up (Rushton turbine, $N_P=5$, $D_i/T_v=1/3$; mixing time from Eq. 6; $k_La$ relative to the laboratory vessel from Eq. 5 at constant vvm).

| Scale and criterion | N (rpm) | P/V (kW/m³) | Tip speed (m/s) | Re (–) | θ₉₅ (s) | $k_La$ (relative) |
|---|---|---|---|---|---|---|
| 2 L laboratory (reference) | 800 | 1.16 | 1.9 | 2.8 × 10⁴ | 2.1 | 1.0 |
| 50 m³, constant P/V | 84 | 1.16 | 5.9 | 2.5 × 10⁶ | 19.5 | 5.4 |
| 50 m³, constant tip speed | 27 | 0.04 | 1.9 | 8.1 × 10⁵ | 60.0 | 1.4 |
| 50 m³, constant Re | 0.9 | ≈ 0 | 0.07 | 2.8 × 10⁴ | 1,755 | 0.02 |
| 50 m³, constant mixing time | 800 | 990 | 55.8 | 2.4 × 10⁷ | 2.1 | 80.5 |

## 6.2 Case A: hybrid models generalise best, but not everywhere

Table 6 and Figure 6 summarise the 10 replicates.

- **Interpolation.** The hybrid model halved the median error compared with both alternatives: 4.4% versus 8.9% for the black-box network and 8.2% for the textbook model. It won in all 10 replicates against each (Wilcoxon p = 0.002 for both comparisons).
- **Extrapolation in feed rate.** Here the mass balances carry the extrapolation. The hybrid model remained accurate (median 7.3%) and beat the black-box network in 8 of 10 replicates (p = 0.010) and the textbook model in 10 of 10 (p = 0.002). Figure 6b–c shows why. At an unseen high feed rate, the black-box network badly underestimated the exponential growth phase, predicting a biomass of about 36 g/L against 77 g/L in the virtual plant. The hybrid model followed the true trajectory closely.
- **Extrapolation in temperature.** Here the unknown kinetics must extrapolate, and *all* models failed badly (median errors 50–80%). The hybrid model was still significantly better than the alternatives (p ≤ 0.014). Even so, an error of 50% is unusable.

The lesson is precise. Physics in the model buys extrapolation along physically modelled directions. It does not create knowledge about biology that the data never contained. In practice, the experimental design must span the ranges of the variables that act through learned kinetics, and model uncertainty should flag excursions outside them.

Table: **Table 6.** Case A: prediction error (NRMSE, %, Eq. 31) over 10 replicates, as median [interquartile range]. The p-values are from two-sided Wilcoxon signed-rank tests of the hybrid model against each alternative; "wins" counts the replicates in which the hybrid model had the lower error.

| Test condition | Textbook mechanistic | Black-box NN | Hybrid (NN + mass balances) | Hybrid vs black-box | Hybrid vs textbook |
|---|---|---|---|---|---|
| Interpolation | 8.2 [8.0–9.4] | 8.9 [8.4–9.6] | **4.4 [3.2–4.5]** | p = 0.002 (10/10 wins) | p = 0.002 (10/10) |
| Extrapolation, feed rate | 11.7 [9.0–12.8] | 12.3 [10.4–13.4] | **7.3 [6.1–10.5]** | p = 0.010 (8/10) | p = 0.002 (10/10) |
| Extrapolation, temperature | 79.9 [54.4–107.9] | 67.1 [38.7–93.9] | **50.0 [36.3–84.8]** | p = 0.014 (8/10) | p = 0.002 (10/10) |

![**Figure 6.** Case A. (a) Distribution of prediction errors over 10 replicates (log scale) for interpolation and two types of extrapolation. (b, c) Example of feed-rate extrapolation (T = 34 °C, k = 0.31 h⁻¹, outside the training range): the black-box network fails, while the hybrid model follows the virtual plant. Simulated data.](../figures/Figure_6.png){width=16cm}

## 6.3 Case B: accuracy is easy; staying accurate is the real problem

Table 7 and Figure 7 summarise Case B. PLS needed only 3 latent variables for glucose and 4 for lactate. It achieved RMSEP = 0.26 g/L (R² = 0.985, RPD = 8.1) for glucose and RMSEP = 0.15 g/L (R² = 0.977, RPD = 6.6) for lactate. These values are in line with published in-line Raman performance for cell culture (Rafferty et al., 2020; Esmonde-White et al., 2022).

Placing a GP on the PLS scores added *calibrated* uncertainty at no loss of accuracy. For glucose, 95% of test values fell within the 95% interval (mean half-width 0.46 g/L). For lactate the intervals were slightly over-confident, with 85% coverage. The MLP, trained on the same 120 spectra, performed much worse (glucose RMSEP 1.46 g/L, R² = 0.51). This illustrates a robust lesson: with typical calibration-set sizes, well-regularised linear latent-variable models remain hard to beat, and deep networks pay off only with much larger datasets.

The monitoring run (Figure 7c–d) shows why maintenance matters more than initial accuracy. After the raw-material change, glucose RMSEP quadrupled from 0.22 to 0.93 g/L, even though the spectra still "looked normal" to the eye. The Q statistic crossed its 99% limit within two samples of the drift onset. It flagged 75% of the post-drift samples, with a false-alarm rate of 7.5% before the drift. Recalibration with just 12 spectra from the new lot restored the RMSEP to 0.28 g/L.

Table: **Table 7.** Case B: soft-sensor performance on an independent test set (n = 60) and behaviour during a raw-material drift.

| Analyte | Model | LVs | RMSEP (g/L) | R² | RPD | 95% PI coverage |
|---|---|---|---|---|---|---|
| Glucose | PLS (SG 1st derivative) | 3 | **0.26** | 0.985 | 8.1 | – |
| Glucose | GP on PLS scores | 3 | 0.26 | 0.984 | 8.1 | 95% |
| Glucose | MLP (64–16) | – | 1.46 | 0.514 | 1.4 | – |
| Lactate | PLS (SG 1st derivative) | 4 | **0.15** | 0.977 | 6.6 | – |
| Lactate | GP on PLS scores | 4 | 0.15 | 0.977 | 6.6 | 85% |
| Lactate | MLP (64–16) | – | 0.93 | 0.098 | 1.1 | – |
| **Drift test (glucose)** | Before drift / after drift / after recalibration | | 0.22 / 0.93 / 0.28 | | | First Q alarm at 2nd post-drift sample; detection rate 75%; false alarms 7.5% |

![**Figure 7.** Case B. (a) Simulated Raman spectra at low and high glucose (dotted lines: main glucose bands). (b) Predicted versus reference glucose on the test set with GP 95% intervals. (c) Q-statistic control chart during a monitoring run with a new raw-material lot from sample 40. (d) Glucose RMSEP before the drift, after the drift, and after recalibration with 12 new spectra. Simulated data.](../figures/Figure_7.png){width=16cm}

## 6.4 Case C: matching distributions, not numbers

The large tank had a Damköhler number of about 11 (Eq. 8). The steady glucose profile spanned two orders of magnitude, from 0.22 g/L at the feed point to 0.002 g/L at the bottom (Figure 8a). As a result, a cell spent 30% of its time starving, 30% in overflow or strong overflow, and only 40% in the limited and near-critical regimes that the laboratory reproduces (Table 8). The lifeline of a single cell (Figure 8b) shows about 148 overflow excursions per hour, with a mean duration of 7.2 s. The homogeneous laboratory vessel exposes cells to *none* of this: its Wasserstein distance to production is 0.30 g g⁻¹ h⁻¹, and it predicts zero acetate where production forms 27.7 mg g⁻¹ h⁻¹.

The common STR+PFR heuristic reduced the Wasserstein distance by only 13%. It produced a bimodal environment (90% limited, 10% strong overflow) with no starvation at all, and excursions twelve times longer than in production. Pulse feeding sized by the same heuristic did better (W₁ = 0.097). The **regime-matched** pulse design, a 45 s period with a 2% duty cycle found by solving Eq. (28), reduced the mismatch to W₁ = 0.044. This is an **85% reduction** relative to the laboratory vessel and a 55% reduction relative to the heuristic pulse design. It reproduced every regime fraction within 12 percentage points, the overflow excursion time within a factor of two (14 s versus 7 s), and 72% of the production acetate formation.

Two further results are instructive.

1. **A physical limit on the STR+PFR design.** When restricted to a practical PFR volume (≤ 30%), the STR+PFR topology could not reproduce starvation at any setting. A mass balance explains why: with all feed entering the PFR, the PFR can consume at most $\phi\,q_{S,max}X$, so the STR must take up the remainder and can never starve.
2. **Matching one outcome is not matching the environment.** The optimised STR+PFR design happened to match the production acetate rate (101%) while its regime distribution was badly wrong (JSD = 0.37, no starvation). Responses that depend on starvation, such as the stringent response, would therefore be missed. Agreement on one output variable is not evidence that a scale-down system is representative. This is a strong argument for distribution-based criteria such as Eq. (28).

Table: **Table 8.** Case C: environment experienced by cells in the 50 m³ production tank and in candidate laboratory scale-down systems (simulated). Regime fractions are % of time. W₁: Wasserstein distance to production (g g⁻¹ h⁻¹, Eq. 28); JSD: Jensen–Shannon divergence (Eq. 29).

| System | Starv. | Limited | Near-crit. | Overflow | Strong ovf. | W₁ | JSD | Overflow excursion (s) | Acetate (mg g⁻¹ h⁻¹) |
|---|---|---|---|---|---|---|---|---|---|
| 50 m³ production (target) | 30.2 | 20.0 | 19.9 | 19.9 | 9.9 | 0 | 0 | 7.2 | 27.7 |
| Lab, homogeneous | 0 | 0 | 100 | 0 | 0 | 0.302 | 0.611 | – | 0.0 |
| STR+PFR, heuristic (10%, τ = θ₉₅) | 0 | 90.0 | 0 | 0 | 10.0 | 0.263 | 0.524 | 89.9 | 24.0 |
| Pulse, heuristic (90 s, 50%) | 25.0 | 14.8 | 30.3 | 29.9 | 0 | 0.097 | 0.069 | 26.6 | 8.0 |
| STR+PFR, regime-matched (30%, 15 s) | 0 | 70.0 | 0 | 26.3 | 3.8 | 0.133 | 0.370 | 15.0 | 27.9 |
| **Pulse, regime-matched (45 s, 2%)** | **25.1** | **21.2** | **22.1** | **31.7** | **0** | **0.044** | **0.062** | **14.1** | **20.0** |

![**Figure 8.** Case C. (a) Quasi-steady glucose profile in the 10-compartment model of a 50 m³ tank (orange: overflow; teal: limited/near-critical; grey: starvation). (b) Glucose experienced by one simulated cell lifeline in the production tank, compared with the regime-matched pulse-feed design and the homogeneous lab vessel. (c) Time spent in each uptake-rate regime. (d) Wasserstein distance between each scale-down candidate and production. Simulated data.](../figures/Figure_8.png){width=16cm}

## 6.5 Case D: Bayesian optimisation finds better conditions with fewer runs, at a risk

The true optimum of the virtual plant lay at T = 32.8 °C and k = 0.32 h⁻¹. This is the fastest allowed feed, combined with a temperature *below* the growth optimum of 34 °C. The trade-off is a genuine one: the product pathway is heat-labile, so the best temperature for product is not the best temperature for growth. Every strategy had to discover it.

With 15 runs, BO's recommendations reached a median of 97.0% of the optimum, and 17 of 20 campaigns came within 5% of it (Table 9, Figure 9). This was significantly better than random search (median 67.5%; Mann–Whitney p = 1.5 × 10⁻⁵) and than the classical CCD + RSM design (median 89.3%; p = 0.001). The quadratic surface of RSM cannot represent the sharp ridge of the true landscape (Figure 9b).

BO was, however, less *reliable*. In one campaign, an unlucky initial design led the GP to an inferior region (worst case 4%). RSM never fell below 83%. A practical hybrid strategy follows: use a small space-filling or CCD design as BO's initial data, and constrain BO's proposals with the twin's uncertainty.

Table: **Table 9.** Case D: true value of each strategy's recommended condition after 15 runs, as % of the global optimum (20 campaigns each).

| Strategy | Mean ± SD | Median | Worst campaign | Campaigns within 5% of optimum | One-sided Mann–Whitney (BO greater) |
|---|---|---|---|---|---|
| **Bayesian optimisation (GP + EI)** | 90.1 ± 22.2 | **97.0** | 4.0 | **17/20** | – |
| DoE + RSM (face-centred CCD + 5 confirmation runs) | 88.9 ± 5.5 | 89.3 | 83.0 | 3/20 | p = 0.0012 |
| Random search | 66.0 ± 21.4 | 67.5 | 28.0 | 1/20 | p = 1.5 × 10⁻⁵ |

![**Figure 9.** Case D. (a) Best observed product mass versus number of bioreactor runs (median and interquartile range over 20 campaigns). (b) True product-mass landscape of the virtual plant with BO recommendations (circles) and the global optimum (star). Simulated data.](../figures/Figure_9.png){width=16cm}

# 7. Findings

Combining the literature review with the case-study evidence leads to six main findings.

1. **Hybrid models are the most reliable backbone, within limits.** The literature consistently reports lower data needs and better extrapolation for hybrid models (Narayanan et al., 2019; Bayer et al., 2020; Sokolov et al., 2021). Case A quantifies this: a twofold error reduction in interpolation and robust feed-rate extrapolation (p ≤ 0.01). It also shows the limit: extrapolation along directions governed by learned kinetics (temperature) remains unreliable for every model class.
2. **Soft sensors are mature; their maintenance is not.** Raman PLS models reach RMSEP values of about 0.2–0.3 g/L for glucose (Case B; Tulsyan et al., 2020). A single raw-material change can, however, quadruple the error invisibly. MSPC statistics detect this within samples and make recalibration targeted and cheap.
3. **Simple models often beat deep ones on small data.** With 120 spectra, PLS outperformed an MLP by a factor of five in RMSEP. Deep learning should be reserved for problems with large datasets, or used inside structured (hybrid or physics-informed) architectures.
4. **Classical scale-up criteria cannot preserve the cell's environment.** Table 5 shows that every single-parameter rule distorts some other quantity by one to three orders of magnitude. Damköhler numbers above 10 make gradients unavoidable.
5. **Regime matching is a better design principle for scale-down.** Distribution matching (Eq. 28) reduced environmental mismatch by 85%. It also showed that a scale-down system can match one output, such as acetate, while misrepresenting the environment.
6. **Sample-efficient optimisation pays off, but needs safeguards.** BO reached 97% of the optimum in 15 runs (median), against 89% for RSM, at the cost of a heavier lower tail. Uncertainty-aware safeguards and good initial designs are needed.

# 8. Discussion and recommendations

The literature and the case studies converge on one message. AI in bioreactors is no longer limited by whether algorithms can learn from bioprocess data; they clearly can. The real challenges are *integration*, *extrapolation* and *maintenance*.

A model that predicts titre in one laboratory is of limited value if it cannot be reused for monitoring, control and scale-up. A model that is accurate on day one is of limited value if no one notices when it stops being accurate. The DSDS loop addresses integration by making the hybrid digital twin the persistent asset. It addresses scale-up by redefining the goal as reproducing the *distribution* of a cell's experienced environment. And it addresses maintenance by building MSPC, uncertainty and version control into the trust layer.

The framework also has limitations. Hybrid models require mechanistic expertise that is scarce in many companies. CFD libraries for training surrogates are expensive to generate. Production-scale data are proprietary. The case studies are simulation-based: they demonstrate mechanisms and relative performance, not absolute numbers for a specific process. Their virtual plant, compartment model and acetate kinetics are deliberately simple, and real cells add memory effects, population heterogeneity and oxygen and CO₂ gradients that were not modelled here.

Based on these observations we recommend:

- **For industry:**
    - Start with hybrid models and Raman soft sensors, where the evidence is strongest.
    - Require uncertainty bands and T²/Q monitoring for every deployed model.
    - Invest early in contextualised data infrastructure, using language models to structure legacy batch records.
- **For process developers:**
    - Combine intensified DoE with Bayesian optimisation, seeding BO with a small space-filling design.
    - Design scale-down experiments by matching regime distributions (Eq. 28) rather than by mixing time alone.
    - Always span the ranges of variables that act through learned kinetics.
- **For academia:**
    - Publish benchmark datasets, ideally including pilot- and production-scale runs.
    - Report performance separately for interpolation and for physically and kinetically governed extrapolation, with replicated runs and statistical tests, as in Table 6.
- **For regulators and quality units:**
    - Adopt risk-based credibility assessment for AI models, building on FDA (2023) and ICH Q13.
    - Set clear expectations for change control of continuously learning models.
- **For educators:**
    - Train bioprocess engineers in statistics and ML, and data scientists in transport phenomena and cell physiology. Hybrid modelling sits exactly at their intersection.

# 9. Future study and conclusion

## 9.1 Future study

Six research directions follow from this chapter.

1. **End-to-end validation.** The DSDS loop should be validated on at least one industrial product, from mini-bioreactor development to commercial scale. Regime-matched scale-down experiments should be compared directly with production runs.
2. **Surrogate validation.** Neural-operator and graph-network surrogates need validation for gassed, multiphase, non-Newtonian and high-cell-density flows. Only then can $p_L(q)$ in Eq. (27) be computed for any candidate vessel.
3. **Richer regime definitions.** Equation (28) should be extended to joint distributions of substrate, oxygen and CO₂, and to the *dynamics* of excursions (spectra of lifeline signals). It should also be coupled with dynamic cell-response models that capture metabolic memory.
4. **Bioprocess foundation models.** Models pre-trained on spectra and time series from many processes, possibly using federated learning so that companies can share learning without sharing data, could drastically reduce data needs.
5. **Language models for knowledge capture.** Retrieval-augmented language models should be developed to turn unstructured batch records and deviation reports into structured, auditable training data.
6. **Certification of learning systems.** Continuously learning and RL-based controllers need certification methods, including conformal prediction (Eq. 25) and safe exploration, before autonomous bioreactors can be accepted in GMP environments.

## 9.2 Conclusion

AI is reshaping how bioreactors are designed, monitored and scaled up. This chapter has provided the mathematical foundations needed to understand both the engineering problem and the AI methods. It has critically reviewed the literature from 2020 to 2025 and proposed the Design–Sense–Decide–Scale loop: a single hybrid digital twin that learns throughout the lifecycle and enables regime-matched scale-down, formalised as a Wasserstein-distance optimisation.

Four reproducible in-silico case studies supplied quantitative evidence:

- hybrid models halve prediction error and extrapolate safely along physically modelled directions;
- MSPC-guarded soft sensors detect and recover from drift that would otherwise quadruple their error;
- distribution-based scale-down reduces environmental mismatch by 85%;
- Bayesian optimisation reaches 97% of the optimum in 15 runs.

Supported by sound data infrastructure, transparent uncertainty and risk-based regulation, this integrated approach can shorten development timelines, reduce failed scale-up campaigns, and make biomanufacturing more robust, affordable and sustainable.

# Declarations

**Data and code availability.** All simulated data, analysis scripts and figure-generation code are available in the supplementary repository (folder `case_study/`). Results can be regenerated with `python case_a_hybrid.py`, `case_b_softsensor.py`, `case_c_scaleup.py`, `case_d_bayesopt.py` and `make_figures.py`.

**Declaration of generative AI and AI-assisted technologies in the writing process** *(authors: adapt this statement to reflect your actual use before submission).* During the preparation of this work the author(s) used an AI writing assistant to help draft and edit text, write simulation code and prepare figures. After using this tool, the author(s) reviewed and edited the content, verified the references and results, and take full responsibility for the content of the publication.

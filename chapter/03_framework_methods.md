# 4. Conceptual framework: the Design–Sense–Decide–Scale (DSDS) loop

## 4.1 Structure of the framework

Building on the gaps identified above, we propose the DSDS loop (Figure 5). Its defining feature is that one **hybrid digital twin**, of the form of Eq. (16), is carried through the entire bioreactor lifecycle. Separate models are not built for each task.

1. **Design.** The hybrid model is trained on intensified-DoE and high-throughput mini-bioreactor data. Bayesian optimisation (Eq. 24) proposes each new experiment, and neural-operator surrogates of CFD screen vessel and impeller options.
2. **Sense.** The same model becomes the core of a state estimator (Eqs. 20–21) fed by Raman and conventional probes. Every soft-sensor prediction carries an uncertainty band (Eqs. 15, 25), and an MSPC monitor (Eqs. 10–11) triggers recalibration when spectra leave the validated region.
3. **Decide.** Hybrid-model MPC (Eq. 22) sets feeds and set points in real time. RL (Eq. 23), trained only inside the twin, proposes improved policies, which must pass a safety filter before use.
4. **Scale.** Surrogates compute the distribution of regimes that cells will meet in the target vessel. A scale-down simulator is designed to reproduce this distribution (Eq. 28), and the twin is transfer-learned to the large scale with a few runs. Large-scale data then flow back to *Design* for the next product.

Three cross-cutting layers make the loop trustworthy:

- a **data layer**: contextualised, FAIR batch records, with language models used to structure legacy text;
- a **trust layer**: uncertainty quantification, SHAP and mechanistic explanation, and model version control;
- a **governance layer**: risk-based validation aligned with QbD, ICH Q13 and FDA guidance.

![**Figure 5.** The Design–Sense–Decide–Scale (DSDS) loop proposed in this chapter. Solid arrows show the lifecycle sequence; dashed links show that every stage reads from and writes to the same hybrid digital twin. Three cross-cutting layers (data, trust, governance) make the loop suitable for regulated manufacturing.](../figures/Figure_5.png){width=16cm}

## 4.2 Formal definition of regime-matched scale-down

Let $\psi(t)=q_S\big(S(\mathbf{x}_c(t))\big)$ be the specific substrate-uptake rate experienced by a cell that follows the trajectory (lifeline) $\mathbf{x}_c(t)$ through the vessel. Uptake rate is used rather than concentration because it is what the cell's metabolism actually "feels". If the circulation is ergodic, the long-run distribution of $\psi$ over all cells and times equals the volume-weighted distribution over the tank:

$$
p_L(q)=\lim_{\mathcal{T}\to\infty}\frac{1}{\mathcal{T}}\int_0^{\mathcal{T}}\delta\big(q-\psi(t)\big)\,dt=\frac{1}{V}\int_V\delta\big(q-q_S(S(\mathbf{x}))\big)\,d\mathbf{x} \qquad (27)
$$

where $\mathcal{T}$ is the averaging time and $\delta$ is the Dirac delta. A scale-down design $d$ (for example an STR+PFR loop, or pulse feeding defined by a period and a duty cycle) produces its own distribution $p_d(q)$. We define the **regime-matched design** as

$$
d^{*}=\arg\min_{d\in\mathcal{D}}\ W_1\big(p_d,p_L\big)+\gamma\left[\ln\frac{\bar\tau_{d}}{\bar\tau_L}\right]^2
$$

$$
W_1(p,p')=\int_0^{q_{max}}\big|F_p(q)-F_{p'}(q)\big|\,dq \qquad (28)
$$

Here $W_1$ is the first Wasserstein (earth-mover's) distance between the two distributions, $F$ denotes a cumulative distribution function, and $\bar\tau$ is the mean duration of overflow excursions. The second term ensures that cells not only spend the right *fraction* of time in each regime but also stay there for a similar *duration*. This matters because cellular responses such as acetate formation or stress-gene induction have characteristic time constants. $\mathcal{D}$ is the set of practically feasible designs, and $\gamma$ weights the duration term.

To aid interpretation, the continuous distribution is also summarised over $R$ discrete regimes: starvation, limited, near-critical, overflow and strong overflow. The two discrete distributions are compared with the Jensen–Shannon divergence:

$$
JSD(p\|p')=\tfrac12 KL(p\|m)+\tfrac12 KL(p'\|m),\qquad m=\tfrac12(p+p')
$$

$$
KL(p\|m)=\sum_{r}p_r\log_2\frac{p_r}{m_r} \qquad (29)
$$

where $KL$ is the Kullback–Leibler divergence. The JSD lies between 0 (identical distributions) and 1. In a full implementation, $p_L$ is computed from a neural-operator surrogate of CFD for each candidate production vessel, making Eq. (28) cheap enough to embed in vessel-design optimisation. The novelty of this formulation is that it treats scale-up as a problem of *matching distributions*, which is the natural language of both cell biology and machine learning, instead of matching a single engineering number.

# 5. In-silico case studies: methodology

To move beyond narrative review, four reproducible case studies test the main claims of the DSDS framework. All data are **simulated** with a "virtual plant" whose true equations are hidden from the learning algorithms. This is a standard approach for benchmarking bioprocess algorithms when industrial data cannot be shared (Petsagkourakis et al., 2020; Rogers et al., 2023). Simulation allows the ground truth to be known exactly and experiments to be replicated many times. Numerical results should be read as *illustrative of mechanisms and relative performance*, not as predictions for a particular strain or plant. All code (Python 3.13; NumPy, SciPy, scikit-learn 1.9) is provided in the chapter's supplementary repository (folder `case_study/`).

## 5.1 Virtual plant

The virtual plant is a fed-batch *E. coli*-type cultivation governed by Eq. (2). Its true kinetics are:

$$
\mu=\mu_{ref}\,e^{-\left(\frac{T-T_{opt}}{T_w}\right)^2}\frac{S}{K_S+S}\,\frac{K_I}{K_I+S},\qquad q_S=\frac{\mu}{Y_{X/S}}+m_S\frac{S}{K_S+S}
$$

$$
q_P=(\alpha\mu+\beta)\,\frac{K_R}{K_R+S}\,e^{-0.15\,[\max(T-33,\,0)]^2} \qquad (30)
$$

The parameter values are $\mu_{ref}=0.60$ h⁻¹, $T_{opt}=34$ °C, $T_w=3$ °C, $K_S=0.15$ g/L, $K_I=15$ g/L, $K_R=1$ g/L, $Y_{X/S}=0.48$, $m_S=0.03$ g g⁻¹ h⁻¹, $\alpha=0.08$, $\beta=0.006$ g g⁻¹ h⁻¹ and $S_f=400$ g/L. The equations encode four realistic features: a temperature optimum for growth, substrate inhibition, catabolite repression of product formation, and a heat-labile product pathway.

The feed is exponential, $F(t)=F_0e^{kt}$, with $F_0=1.5$ mL/h. Runs last 16 h and are sampled hourly. Measurements carry heteroscedastic noise (5% relative plus an absolute floor). The two decision variables are the temperature $T$ and the feed exponent $k$.

## 5.2 Case A: hybrid versus black-box versus textbook models

Three model classes are trained on identical data:

- a **textbook mechanistic model**: Monod kinetics with constant yields, four parameters fitted by non-linear least squares;
- a **black-box network**: an MLP with two hidden layers of 32 units, mapping $(t,T,k)$ to $(X,S,P)$;
- a **hybrid model**: Eq. (16) with a 2–10–3 tanh network and a softplus output layer to guarantee non-negative rates. It is trained first incrementally on spline-derived rates and then end-to-end through a Runge–Kutta solver using L-BFGS (Eq. 17).

Training uses 12 runs from a Latin-hypercube design inside a narrow window ($T\in[32,36]$ °C, $k\in[0.12,0.26]$ h⁻¹). Three test sets of 8 runs each probe different kinds of generalisation:

- **interpolation**: runs inside the training window;
- **feed extrapolation**: $k\in[0.08,0.12)\cup(0.26,0.30]$ h⁻¹, so extrapolation is carried by the known mass balances;
- **temperature extrapolation**: $T\in[30.5,32)\cup(36,37.5]$ °C, so extrapolation must be carried by the learned kinetics.

The error metric is the normalised root-mean-square error, averaged over the three states:

$$
NRMSE=\frac{1}{3}\sum_{j\in\{X,S,P\}}\frac{\sqrt{\frac{1}{N}\sum(\hat\xi_j-\xi_j)^2}}{\max\xi_j-\min\xi_j}\times100\% \qquad (31)
$$

The whole experiment is repeated with 10 independent random replicates, each with new noise, a new design and a new network initialisation. Models are compared with the two-sided Wilcoxon signed-rank test, which pairs results within each replicate.

## 5.3 Case B: Raman soft sensor with uncertainty and drift detection

Raman spectra (400–1800 cm⁻¹, 701 points) are synthesised as linear combinations of Gaussian band profiles for glucose (bands at 520, 911, 1065, 1125 and 1365 cm⁻¹), lactate, cells/protein and medium. Realism is added through five effects:

- a random fluorescence baseline;
- detector noise;
- a wavenumber-calibration jitter of ±1 cm⁻¹;
- a laser-power variation of 3%;
- reference-analyser error of 3% CV.

There are 120 calibration spectra and 60 independent test spectra. Three models are compared:

1. **PLS** on Savitzky–Golay first-derivative spectra, with the number of LVs chosen by 10-fold cross-validation using a 2% parsimony rule;
2. **GP regression** on the PLS scores, which provides predictive intervals;
3. an **MLP** (64–16 units, early stopping) on the full pre-processed spectrum.

A separate 80-sample monitoring run then introduces a new, sugar-rich raw-material lot from sample 40 onwards. The interfering bands at 1060, 1110 and 1290 cm⁻¹ overlap the glucose bands. MSPC limits are set at 99% (Eqs. 10–11). Finally, the model is recalibrated by adding 12 spectra from the new lot.

## 5.4 Case C: regime-matched scale-down for a 50 m³ fermenter

**Classical criteria.** Eq. (7) is evaluated for scale-up from a 2 L to a 50 m³ geometrically similar vessel (H = T, Rushton turbine, $N_P=5$, $D_i/T_v=1/3$, 800 rpm at the laboratory scale).

**Large tank.** The production vessel is represented by 10 axial compartments of 5 m³ each, with glucose fed at the top. The inter-compartment exchange flow (2,080 L/s) is tuned so that the 95% mixing time equals 90 s, a typical value for multi-impeller tanks of this size. Uptake follows Michaelis–Menten kinetics with $q_{S,max}=1.2$ g g⁻¹ h⁻¹ and $K_S=0.05$ g/L, at $X=30$ g/L. The mean demand is $q_S=0.45$ g g⁻¹ h⁻¹, corresponding to $\mu\approx0.2$ h⁻¹.

**Regimes.** The uptake-rate axis is divided into five regimes, with boundaries at 0.12, 0.40, 0.70 and 0.95 g g⁻¹ h⁻¹: starvation, limited, near-critical, overflow and strong overflow. The overflow threshold is 0.70 g g⁻¹ h⁻¹. Above it, acetate is formed at a rate of $0.5\,(q_S-0.70)$.

**Cell lifelines.** These are simulated as a continuous-time Markov chain between compartments: 3,000 cells over 30 min, with a time step of 0.1 s.

**Scale-down candidates.** All candidates deliver the same mean feed:

- the homogeneous laboratory vessel;
- an STR+PFR loop sized by the common heuristic (10% of the volume in the PFR, PFR residence time equal to the large-scale mixing time);
- pulse feeding with a period equal to the mixing time and a 50% duty cycle;
- the regime-matched designs found by solving Eq. (28) by grid search with $\gamma=0.02$. The STR+PFR design is limited to at most 30% of the volume in the PFR; the pulse-feed design varies the period from 10 to 240 s and the duty cycle from 2% to 90%.

## 5.5 Case D: Bayesian optimisation of temperature and feeding

The goal is to maximise the final product mass $P\cdot V$ after 16 h over $T\in[30,38]$ °C and $k\in[0.08,0.32]$ h⁻¹. Each evaluation is one "bioreactor run" with 3% measurement noise, and each campaign has a budget of 15 runs. Three strategies are compared:

1. **BO**: a GP with a Matérn-5/2 kernel and expected improvement (Eq. 24), starting from 5 random runs;
2. **random search** with 15 runs;
3. **classical DoE + RSM**: a face-centred central composite design with 10 runs, a quadratic response-surface fit, and 5 confirmation runs around the predicted optimum.

Each strategy is repeated in 20 independent campaigns. The performance measure is the *true* (noise-free) value of each campaign's recommended condition, expressed as a percentage of the global optimum found by dense grid search. Strategies are compared with the one-sided Mann–Whitney U test.

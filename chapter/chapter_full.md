---
title: "Artificial Intelligence in Bioreactor Design, Monitoring, and Scale-Up: Mathematical Foundations, Critical Review, and an Integrated Hybrid Digital-Twin Framework"
subtitle: "Book chapter (Elsevier) — extended version with mathematical formulation and in-silico case studies"
---

# Abstract

Bioreactors sit at the heart of biomanufacturing. Yet their design, monitoring and scale-up still rely on empirical rules, sparse offline sampling and costly trial-and-error. A process that performs well in a 2 L vessel often loses titre, quality or robustness in a 50 m³ tank, because cells there pass through gradients of substrate, oxygen and pH that never existed at small scale. This chapter gives a mathematically grounded account of how artificial intelligence (AI) can close this gap. It first builds a common formal language: the balance equations of bioreactor engineering, the scale-up criteria that follow from them, and the statistical, machine-learning (ML), deep-learning (DL), physics-informed, reinforcement-learning and language-model tools that act on them. Recent literature (2020–2025) on design, monitoring, control and scale-up is then reviewed critically, and the evidence is synthesised into the *Design–Sense–Decide–Scale (DSDS)* framework. In this framework a single hybrid digital twin is carried across the whole bioreactor lifecycle. A central element is *regime-matched scale-up*: the scale-down design is chosen to minimise a distributional distance (the Wasserstein distance) between the conditions that cells experience at production and at laboratory scale. Four reproducible in-silico case studies, all based on simulated data, test the framework.

- **Case A (hybrid model).** A hybrid neural–mechanistic model reduced median prediction error from 8.9% to 4.4% compared with a black-box network (Wilcoxon p = 0.002).
- **Case B (Raman soft sensor).** A Raman partial least squares (PLS) soft sensor reached a prediction error of 0.26 g/L for glucose. Its multivariate control charts detected a raw-material drift within two samples of its onset.
- **Case C (scale-down design).** A regime-matched scale-down design reduced the environmental mismatch by 85% relative to a homogeneous laboratory vessel.
- **Case D (Bayesian optimisation).** Bayesian optimisation reached a median 97% of the true optimum in 15 runs, against 89% for a classical response-surface design.

The chapter closes with practical recommendations for industry, academia and regulators, and with a research agenda for trustworthy, data-efficient AI in bioprocessing.

**Keywords:** artificial intelligence; bioreactor; hybrid modelling; soft sensor; multivariate statistical process control; digital twin; scale-up; scale-down; Wasserstein distance; Bayesian optimisation; physics-informed machine learning; reinforcement learning

# 1. Introduction

A bioreactor is a vessel in which living cells or enzymes convert raw materials into valuable products under controlled conditions. Monoclonal antibodies, insulin, vaccines, industrial enzymes, organic acids, biofuels and cultivated meat all depend on it. The basic hardware has changed little in decades: a stirred tank with an impeller, a sparger, a cooling jacket and a handful of probes. The demands placed on it, however, have grown sharply. Regulators expect manufacturers to understand their processes scientifically under Quality by Design (QbD). Continuous manufacturing is now formally recognised in ICH Q13 (ICH, 2022). And companies must move new molecules from the bench to commercial production faster and at lower cost (Rathore et al., 2021).

Three long-standing problems make this difficult.

1. **Design.** Choosing vessel geometry, impeller type, gassing strategy and feeding profile requires many experiments, because cell metabolism is highly non-linear and only partly understood.
2. **Monitoring.** The variables that matter most are usually measured offline once or twice a day: viable cell density, glucose, lactate, titre and glycosylation. A process can therefore drift for hours before anyone notices (Gargalo et al., 2020).
3. **Scale-up.** A large tank is not simply a big flask. In a 50 m³ fermenter, mixing can take more than a minute, while a cell consumes the local glucose in a few seconds. As a result, each cell repeatedly passes between zones of excess and starvation (Nadal-Rey et al., 2021a). These fluctuations can reduce yield by 10–30% or trigger stress responses that change product quality. Such effects are difficult to predict from laboratory data (Wang et al., 2020).

Artificial intelligence (AI) here means computational methods that learn patterns from data and use them to predict, classify or decide. It has matured in exactly the areas these problems need.

- **Soft sensing.** ML models turn a Raman spectrum into a glucose concentration every few minutes (Esmonde-White et al., 2022).
- **Hybrid modelling.** Hybrid models embed neural networks inside mass balances and need far fewer experiments than purely empirical models (Narayanan et al., 2020; Sokolov et al., 2021).
- **Control.** Reinforcement learning (RL) can learn feeding policies in simulation (Petsagkourakis et al., 2020).
- **Fast simulation.** Physics-informed networks and neural operators can reproduce expensive computational fluid dynamics (CFD) in seconds (Karniadakis et al., 2021).
- **Language models.** Large language models open new ways to mine batch records and deviation reports.

Together with cheaper sensors and cloud data platforms, these tools underpin the vision of *Bioprocessing 4.0* (Isoko et al., 2024).

Yet the literature remains fragmented. Most reviews focus on one task or one model family (Mowbray et al., 2021; Helleckes et al., 2023; Rathore et al., 2023). Few provide the mathematics that connects bioreactor physics to the learning algorithms. Few show quantitatively how much each AI component actually contributes. Fewer still explain how a model built during development can be carried through monitoring to the production scale. Industrial uptake is further slowed by limited data, poor transferability between scales and cell lines, and regulatory uncertainty about "black-box" models (FDA, 2023).

This chapter addresses these gaps in five ways:

1. It derives the core mathematics of bioreactor transport, kinetics and scale-up. It then sets out the statistical, ML, DL, physics-informed, control and language-model formulations in a single notation (Section 2).
2. It critically reviews recent work on AI in design, monitoring, control and scale-up (Section 3).
3. It proposes the Design–Sense–Decide–Scale (DSDS) framework. This includes a formal definition of *regime-matched* scale-down as an optimal-transport problem (Section 4).
4. It tests the main claims in four reproducible in-silico case studies, using statistical hypothesis tests (Sections 5 and 6).
5. It distils findings, recommendations and a research agenda (Sections 7–9).

# 2. Theoretical overview and mathematical foundations

This section introduces the minimum mathematics needed to understand both the engineering problem and the AI methods. Symbols are defined at first use and summarised in Table 1.

## 2.1 Balance equations of a stirred-tank bioreactor

Any bioreactor model, whether written by an engineer or learned by a computer, must respect conservation of mass. Let $\boldsymbol{\xi}=[X,S,P,\dots]^\top$ be the vector of concentrations: biomass $X$, substrate $S$, product $P$ and so on. The reactor has volume $V$, a feed flow $F$ with feed composition $\boldsymbol{\xi}_{in}$, and a dilution rate $D=F/V$. The general fed-batch balance in the reaction-network form of Bastin and Dochain (1990) is

$$
\frac{d\boldsymbol{\xi}}{dt}= \mathbf{K}\,\boldsymbol{\varphi}(\boldsymbol{\xi},\mathbf{u}) + D\,(\boldsymbol{\xi}_{in}-\boldsymbol{\xi}) + \mathbf{Q}(\boldsymbol{\xi}), \qquad \frac{dV}{dt}=F \qquad (1)
$$

Here $\mathbf{K}$ is the stoichiometric (yield) matrix, $\boldsymbol{\varphi}$ is the vector of reaction rates, $\mathbf{u}$ is the vector of manipulated inputs (temperature $T$, pH, feed rate) and $\mathbf{Q}$ collects gas–liquid transfer terms. For the common three-state description, Eq. (1) becomes

$$
\frac{dX}{dt}=\mu X - DX,\qquad \frac{dS}{dt}=-q_S X + D\,(S_{f}-S),\qquad \frac{dP}{dt}=q_P X - DP \qquad (2)
$$

where $\mu$, $q_S$ and $q_P$ are the *specific* rates of growth, substrate uptake and product formation (per gram of cells per hour). The equations themselves are well established. The difficulty lies entirely in the kinetic functions $\mu(\cdot)$, $q_S(\cdot)$ and $q_P(\cdot)$. Classical choices include:

$$
\mu = \mu_{max}(T)\,\frac{S}{K_S+S}\,\frac{K_I}{K_I+S},\qquad q_S=\frac{\mu}{Y_{X/S}}+m_S,\qquad q_P=\alpha\mu+\beta \qquad (3)
$$

These combine the Monod/Haldane growth law, the Pirt maintenance concept and the Luedeking–Piret product equation. Real cells, however, respond to temperature, pH, osmolality, by-products and their own history in ways no closed-form expression captures. This mismatch is the main reason purely mechanistic models struggle, and the main entry point for machine learning (Narayanan et al., 2020).

**Oxygen transfer.** The dissolved-oxygen balance is

$$
\frac{dC_{O_2}}{dt}=k_La\,\big(C^{*}_{O_2}-C_{O_2}\big)-q_{O_2}X \qquad (4)
$$

The volumetric mass-transfer coefficient is usually correlated with power input and superficial gas velocity $v_s$ (van't Riet, 1979):

$$
k_La = C\left(\frac{P}{V}\right)^{\alpha} v_s^{\beta},\qquad \alpha\approx 0.4,\ \beta\approx 0.5 \qquad (5)
$$

## 2.2 Hydrodynamics, mixing and the mathematics of scale-up

The impeller power draw is $P=N_P\,\rho N^3 D_i^5$. Here $N_P$ is the power number (about 5 for a Rushton turbine), $\rho$ is the liquid density, $N$ is the stirrer speed and $D_i$ is the impeller diameter. The Reynolds number is $Re=\rho N D_i^2/\eta$, where $\eta$ is the liquid viscosity, and the tip speed is $u_{tip}=\pi N D_i$. For turbulent, single-impeller tanks, Grenville and Nienow (2004) give the 95% mixing time as

$$
\theta_{95}=\frac{5.20}{N_P^{1/3}\,N}\left(\frac{T_v}{D_i}\right)^2 \qquad (6)
$$

where $T_v$ is the vessel diameter. Under geometric similarity with linear scale factor $\lambda=T_{v,L}/T_{v,S}$ (large over small), keeping one quantity constant fixes the stirrer speed, and with it all the others:

$$
\frac{P}{V}=\text{const}\ \Rightarrow\ N_L=N_S\lambda^{-2/3};\qquad u_{tip}=\text{const}\ \Rightarrow\ N_L=N_S\lambda^{-1}
$$

$$
Re=\text{const}\ \Rightarrow\ N_L=N_S\lambda^{-2};\qquad \theta_{95}=\text{const}\ \Rightarrow\ N_L=N_S \qquad (7)
$$

Table 5 (Section 6) evaluates Eq. (7) for a 2 L to 50 m³ scale-up. It shows that no single criterion preserves all relevant quantities.

**Why gradients appear: the Damköhler number.** The decisive dimensionless group compares the time needed to mix the tank with the time cells need to consume the substrate:

$$
Da=\frac{\theta_{mix}}{\tau_{S}},\qquad \tau_{S}=\frac{\bar S}{q_S X} \qquad (8)
$$

As a worked example, take a glucose-limited *E. coli* fed-batch with $\bar S=0.03$ g/L, $q_S=0.45$ g g⁻¹ h⁻¹ and $X=30$ g/L. Then $\tau_S = 0.03/(0.45\times30/3600)\approx 8$ s. In a 2 L vessel ($\theta_{mix}\approx 2$ s), $Da\approx0.25$ and the liquid is effectively homogeneous. In a 50 m³ multi-impeller tank ($\theta_{mix}\approx 90$ s), $Da\approx 11$. Strong gradients are therefore unavoidable. This single number explains why scale-up so often fails, and it motivates the regime-based approach developed in Section 4.

## 2.3 Statistical and chemometric foundations

**Partial least squares (PLS).** Spectra $\mathbf{X}\in\mathbb{R}^{n\times p}$ (for Raman, $p\approx700$ wavenumbers) are highly collinear, so ordinary least squares is ill-posed. PLS finds weight vectors that maximise the covariance between spectral scores and the response (Wold et al., 2001):

$$
\mathbf{w}_a=\arg\max_{\|\mathbf{w}\|=1}\ \operatorname{cov}^2(\mathbf{X}_{a}\mathbf{w},\mathbf{y}),\qquad \mathbf{X}=\mathbf{T}\mathbf{P}^\top+\mathbf{E},\qquad \hat{\mathbf{y}}=\mathbf{T}\mathbf{q} \qquad (9)
$$

Here $\mathbf{T}$ are the scores, $\mathbf{P}$ the loadings and $\mathbf{E}$ the residuals. The number of latent variables (LVs) $A$ is chosen by cross-validation. Performance is reported as the root-mean-square error of prediction, $RMSEP=\sqrt{\tfrac{1}{n}\sum(\hat y_i-y_i)^2}$, together with $R^2$ and the ratio of performance to deviation, $RPD=s_y/RMSEP$. An RPD above 5 is usually considered adequate for process control.

**Multivariate statistical process control (MSPC).** A soft sensor must also recognise when *not* to trust itself. Two statistics computed from the PLS model do this (MacGregor and Kourti, 1995):

$$
T^2=\sum_{a=1}^{A}\frac{t_a^2}{s_{t_a}^2},\qquad T^2_{lim}=\frac{A(n-1)(n+1)}{n(n-A)}F_{0.99}(A,n-A) \qquad (10)
$$

$$
Q=\|\mathbf{x}-\hat{\mathbf{x}}\|^2=\|\mathbf{e}\|^2,\qquad Q_{lim}=g\,\chi^2_{0.99}(h),\quad g=\frac{\operatorname{var}(Q)}{2\,\overline{Q}},\ h=\frac{2\overline{Q}^{\,2}}{\operatorname{var}(Q)} \qquad (11)
$$

Hotelling's $T^2$ flags unusual *combinations* of known sources of variation. The squared prediction error $Q$ flags *new* sources of variation that the model has never seen, such as a new impurity or a fouled probe window.

## 2.4 Machine learning and deep learning

**Empirical risk minimisation.** Almost every supervised learner solves

$$
\boldsymbol{\theta}^*=\arg\min_{\boldsymbol{\theta}}\ \frac{1}{N}\sum_{i=1}^{N}\ell\big(f_{\boldsymbol{\theta}}(\mathbf{x}_i),y_i\big)+\lambda\,\Omega(\boldsymbol{\theta}) \qquad (12)
$$

where $\ell$ is a loss function (for example squared error) and $\Omega$ is a regulariser that controls the bias–variance trade-off. Bioprocess datasets are small: typically 5–50 runs. Regularisation and the choice of $f_{\boldsymbol\theta}$ therefore matter more than in data-rich domains.

**Feed-forward and recurrent networks.** A multilayer perceptron (MLP) composes affine maps and non-linear activations, $\mathbf{h}_l=\sigma(\mathbf{W}_l\mathbf{h}_{l-1}+\mathbf{b}_l)$. For time series, the long short-term memory (LSTM) cell (Hochreiter and Schmidhuber, 1997) keeps a memory $\mathbf{c}_t$ that is updated through gates:

$$
\mathbf{f}_t=\sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]),\qquad \mathbf{i}_t=\sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]),\qquad \mathbf{o}_t=\sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t])
$$

$$
\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]),\qquad \mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t) \qquad (13)
$$

where $\mathbf{f}_t$, $\mathbf{i}_t$ and $\mathbf{o}_t$ are the forget, input and output gates and $\odot$ is element-wise multiplication.

**Attention and transformers.** Transformers replace recurrence with *attention*, which lets the model weigh every past time point when predicting the next one (Vaswani et al., 2017):

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V} \qquad (14)
$$

For example, a temporal fusion transformer forecasting titre can learn to attend to the day of a temperature shift. Its attention weights then reveal which phase of the run drives the final outcome (Lim et al., 2021).

**Gaussian processes (GPs).** A GP places a prior directly on functions, $f\sim\mathcal{GP}(m,k)$, here with a Matérn or squared-exponential covariance kernel $k$. Given training data $(\mathbf{X},\mathbf{y})$ with noise variance $\sigma_n^2$, the prediction at a new point $\mathbf{x}_*$ is Gaussian with

$$
\mu(\mathbf{x}_*)=\mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{y},\qquad \sigma^2(\mathbf{x}_*)=k(\mathbf{x}_*,\mathbf{x}_*)-\mathbf{k}_*^\top(\mathbf{K}+\sigma_n^2\mathbf{I})^{-1}\mathbf{k}_* \qquad (15)
$$

(Rasmussen and Williams, 2006). The predictive variance $\sigma^2$ is what makes GPs attractive for small bioprocess datasets and for experiment design.

## 2.5 Hybrid (grey-box) models

A serial hybrid model keeps Eq. (1) and replaces only the uncertain rate vector with a neural network:

$$
\frac{d\boldsymbol{\xi}}{dt}=\mathbf{K}\,\boldsymbol{\varphi}_{NN}(\boldsymbol{\xi},\mathbf{u};\boldsymbol{\theta}) + D(\boldsymbol{\xi}_{in}-\boldsymbol{\xi}),\qquad \boldsymbol{\varphi}_{NN}\ge 0 \qquad (16)
$$

It is trained by minimising the trajectory error

$$
\min_{\boldsymbol\theta}\ \sum_{r=1}^{R}\sum_{k=1}^{K}\big\|\mathbf{W}^{-1}\big(\hat{\boldsymbol{\xi}}_r(t_k;\boldsymbol\theta)-\mathbf{y}_r(t_k)\big)\big\|^2+\lambda\|\boldsymbol\theta\|^2 \qquad (17)
$$

where $r$ indexes runs, $k$ indexes sampling times and $\mathbf{W}$ scales the states. Gradients of Eq. (17) are obtained by sensitivity equations, adjoint methods or automatic differentiation through the ODE solver. The last option is the idea behind *neural ordinary differential equations* (Chen et al., 2018). A cheaper alternative is *incremental* training: specific rates are first back-calculated from smoothed data, and the network is fitted to them directly. The two can be combined, as in Case A.

Positivity, mass conservation and known stoichiometry are then guaranteed by construction. This is why hybrid models extrapolate better along directions that are governed by the known physics, such as feed rate. Where the unknown kinetics dominate, for example at an unseen temperature, they extrapolate less well (Rogers et al., 2023). Case A quantifies exactly this distinction.

## 2.6 Physics-informed machine learning and neural operators

Physics-informed neural networks (PINNs) approximate the solution $u_{\boldsymbol\theta}(\mathbf{x},t)$ of a partial differential equation $\mathcal{N}[u]=0$, such as the Navier–Stokes or species-transport equations, by penalising the equation residual (Raissi et al., 2019):

$$
\mathcal{L}(\boldsymbol\theta)=\frac{1}{N_d}\sum\|u_{\boldsymbol\theta}-u_{obs}\|^2+\lambda_r\frac{1}{N_r}\sum\|\mathcal{N}[u_{\boldsymbol\theta}]\|^2+\lambda_b\frac{1}{N_b}\sum\|\mathcal{B}[u_{\boldsymbol\theta}]\|^2 \qquad (18)
$$

The three terms penalise misfit to observations, violation of the governing equation, and violation of the boundary conditions $\mathcal{B}$. *Neural operators* go one step further. Instead of learning a single solution, they learn the solution operator $\mathcal{G}:a\mapsto u$, which maps an input field (for example impeller speed and geometry) to an output field (for example the velocity or glucose field). One Fourier-neural-operator layer reads

$$
v_{l+1}(\mathbf{x})=\sigma\Big(\mathbf{W}v_l(\mathbf{x})+\mathcal{F}^{-1}\big(\mathbf{R}_l\cdot\mathcal{F}(v_l)\big)(\mathbf{x})\Big) \qquad (19)
$$

where $\mathcal{F}$ is the Fourier transform and $\mathbf{R}_l$ a learned spectral filter (Li et al., 2021). DeepONet writes $\mathcal{G}(a)(\mathbf{y})\approx\sum_k b_k(a)\,t_k(\mathbf{y})$, combining a "branch" network $b_k$ with a "trunk" network $t_k$ (Lu et al., 2021). Graph networks apply the same idea on unstructured CFD meshes (Pfaff et al., 2021). Once trained on a few dozen CFD runs, such surrogates evaluate a new tank design in milliseconds instead of days.

## 2.7 State estimation, model predictive control and reinforcement learning

**Extended Kalman filter (EKF).** An EKF fuses a model with noisy, multi-rate measurements $\mathbf{y}_k=h(\boldsymbol\xi_k)+\mathbf{v}_k$. Here $\mathbf{P}$ is the state-error covariance, $\mathbf{F}_k$ and $\mathbf{H}_k$ are the Jacobians of the model and of the measurement function $h$, and $\mathbf{Q}_w$, $\mathbf{R}_v$ are the process- and measurement-noise covariances. The prediction and update steps are

$$
\hat{\boldsymbol\xi}_{k|k-1}=f(\hat{\boldsymbol\xi}_{k-1}),\qquad \mathbf{P}_{k|k-1}=\mathbf{F}_k\mathbf{P}_{k-1}\mathbf{F}_k^\top+\mathbf{Q}_w
$$

$$
\mathbf{G}_k=\mathbf{P}_{k|k-1}\mathbf{H}_k^\top\big(\mathbf{H}_k\mathbf{P}_{k|k-1}\mathbf{H}_k^\top+\mathbf{R}_v\big)^{-1} \qquad (20)
$$

$$
\hat{\boldsymbol\xi}_{k}=\hat{\boldsymbol\xi}_{k|k-1}+\mathbf{G}_k\big(\mathbf{y}_k-h(\hat{\boldsymbol\xi}_{k|k-1})\big),\qquad \mathbf{P}_k=(\mathbf{I}-\mathbf{G}_k\mathbf{H}_k)\mathbf{P}_{k|k-1} \qquad (21)
$$

where $\mathbf{G}_k$ is the Kalman gain. When a hybrid model of Eq. (16) is used as $f$, the estimator inherits its physical consistency.

**Model predictive control (MPC).** At each sampling time, MPC solves a finite-horizon optimal-control problem and applies only the first move:

$$
\min_{\mathbf{u}_{k},\dots,\mathbf{u}_{k+H-1}}\ \sum_{j=1}^{H}\Big(\|\hat{\mathbf{y}}_{k+j}-\mathbf{r}_{k+j}\|^2_{\mathbf{Q}}+\|\Delta\mathbf{u}_{k+j-1}\|^2_{\mathbf{R}}\Big)
$$

$$
\text{subject to Eq. (16)},\qquad \mathbf{u}_{min}\le\mathbf{u}\le\mathbf{u}_{max},\qquad g(\hat{\boldsymbol\xi})\le 0 \qquad (22)
$$

Here $H$ is the prediction horizon, $\mathbf{r}$ is the reference trajectory, $\mathbf{Q}$ and $\mathbf{R}$ are weighting matrices, and $g$ collects state constraints such as a minimum dissolved oxygen.

**Reinforcement learning.** RL models the process as a Markov decision process, with states $s$, actions $a$, reward $r$ and discount factor $\gamma$. It seeks a policy $\pi_{\boldsymbol\theta}(a|s)$ that maximises the expected return $J(\boldsymbol\theta)=\mathbb{E}_{\pi}\big[\sum_t\gamma^t r_t\big]$. The policy-gradient theorem gives

$$
\nabla_{\boldsymbol\theta}J=\mathbb{E}_{\pi}\Big[\sum_t\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t|s_t)\,\big(G_t-b(s_t)\big)\Big] \qquad (23)
$$

where $G_t$ is the return from time $t$ and $b$ is a baseline (Sutton and Barto, 2018). Safety can be imposed as a chance constraint, $\Pr\big[g(\boldsymbol\xi_t)\le0\ \forall t\big]\ge1-\epsilon$ (Mowbray et al., 2022).

## 2.8 Bayesian optimisation, uncertainty quantification and explainability

**Bayesian optimisation (BO).** BO fits a GP, Eq. (15), to the experiments performed so far. It then chooses the next experiment by maximising an *acquisition function*, most often the expected improvement (Jones et al., 1998):

$$
EI(\mathbf{x})=\big(\mu(\mathbf{x})-f^{+}-\xi\big)\Phi(z)+\sigma(\mathbf{x})\phi(z),\qquad z=\frac{\mu(\mathbf{x})-f^{+}-\xi}{\sigma(\mathbf{x})} \qquad (24)
$$

Here $f^+$ is the best value so far, $\xi$ is a small exploration margin, and $\Phi$, $\phi$ are the standard normal cumulative and density functions. The two terms formalise *exploitation* (high predicted mean) and *exploration* (high uncertainty).

**Uncertainty quantification.** Deep ensembles estimate uncertainty from the spread of $M$ independently trained networks, $\sigma^2(\mathbf{x})\approx\frac{1}{M}\sum_m(f_m(\mathbf{x})-\bar f(\mathbf{x}))^2$ (Lakshminarayanan et al., 2017). *Split conformal prediction* turns any point predictor into an interval with guaranteed coverage (Angelopoulos and Bates, 2023):

$$
C(\mathbf{x})=\big[\hat f(\mathbf{x})-\hat q,\ \hat f(\mathbf{x})+\hat q\big],\qquad \hat q=\operatorname{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}\big(|y_i-\hat f(\mathbf{x}_i)|\big) \qquad (25)
$$

where the quantile is taken over the absolute residuals of $n$ held-out calibration samples, so that $\Pr[y\in C(\mathbf{x})]\ge1-\alpha$, provided the data are exchangeable.

**Explainability.** SHAP attributes a prediction $f(\mathbf{x})$ to input $j$ through the Shapley value (Lundberg and Lee, 2017):

$$
\phi_j=\sum_{S\subseteq\mathcal{M}\setminus\{j\}}\frac{|S|!\,(M-|S|-1)!}{M!}\Big[f_{S\cup\{j\}}(\mathbf{x})-f_S(\mathbf{x})\Big] \qquad (26)
$$

where $\mathcal{M}$ is the set of $M$ inputs and $f_S$ is the model evaluated using only the inputs in subset $S$. This makes it possible to rank critical process parameters from any ML model.

## 2.9 Natural language processing and large language models

A large share of bioprocess knowledge is not numeric. Batch records, deviation and CAPA (corrective and preventive action) reports, standard operating procedures, regulatory submissions and the scientific literature are all text. Transformer language models, Eq. (14), map text to vectors (embeddings) $\mathbf{e}\in\mathbb{R}^d$. Semantic similarity between two documents is then $\cos(\mathbf{e}_1,\mathbf{e}_2)=\mathbf{e}_1^\top\mathbf{e}_2/(\|\mathbf{e}_1\|\|\mathbf{e}_2\|)$.

*Retrieval-augmented generation (RAG)* combines such retrieval with a generative model, so that answers are grounded in a company's own validated documents (Lewis et al., 2020). Typical uses include:

- finding past deviations similar to a current foaming event;
- extracting process parameters from decades of PDF batch records into structured tables for ML;
- drafting first versions of investigation reports for human review.

Foundation models pre-trained on broad data (Bommasani et al., 2021) also promise transfer to spectra and time series. Their outputs, however, must be verified: hallucinated content is unacceptable in a GMP context. In the DSDS framework, language models therefore sit in the data and trust layers, as assistants that *structure* knowledge. They are not autonomous decision-makers.

Table: **Table 1.** Main symbols used in this chapter.

| Symbol | Meaning (units) | Symbol | Meaning (units) |
|---|---|---|---|
| $X, S, P$ | Biomass, substrate, product concentration (g/L) | $k_La$ | Volumetric O₂ transfer coefficient (h⁻¹) |
| $V, F, D$ | Volume (L), feed rate (L/h), dilution rate $F/V$ (h⁻¹) | $N, D_i, T_v$ | Stirrer speed (s⁻¹), impeller and vessel diameter (m) |
| $\mu, q_S, q_P$ | Specific growth, uptake, production rates (h⁻¹, g g⁻¹ h⁻¹) | $\theta_{95}$, $Da$ | 95% mixing time (s), Damköhler number (–) |
| $\mathbf{K}, \boldsymbol\varphi$ | Stoichiometric matrix, reaction-rate vector | $T^2, Q$ | Hotelling and squared-prediction-error statistics |
| $\boldsymbol\theta$ | Trainable model parameters | $W_1$, JSD | Wasserstein distance, Jensen–Shannon divergence |
| $\mathbf{u}$ | Manipulated inputs ($T$, pH, feed) | $EI$ | Expected improvement (acquisition function) |
# 3. Literature review

## 3.1 Scope and a map of the field

This review focuses on peer-reviewed studies published between 2020 and 2025. They were identified in Scopus, Web of Science and Google Scholar using combinations of the terms *bioreactor*, *fermentation*, *cell culture*, *machine learning*, *hybrid model*, *soft sensor*, *digital twin*, *reinforcement learning* and *scale-up*. A small number of earlier works are included where they introduced a method that later studies build on, such as physics-informed neural networks (Raissi et al., 2019) or coupled CFD–metabolic simulation (Haringa et al., 2018). Studies were retained if they reported an application to a real or simulated bioreactor. They were grouped by the lifecycle task they serve: design, monitoring, control or scale-up.

Two trends stand out. First, the field has moved from *data-only* models, such as artificial neural networks trained on a few batches, towards *knowledge-plus-data* models that respect conservation laws (Mowbray et al., 2021; Helleckes et al., 2023). Second, the central question has shifted. It is no longer "can ML predict this variable?" but "can the model be trusted, transferred and maintained in a GMP plant?" (Rathore et al., 2023; Isoko et al., 2024).

Figure 1 maps the model families by how much physical knowledge they embed and how much data they need, and links each family to the lifecycle stage where it has been used most. Table 2 compares the families in more detail.

![**Figure 1.** Map of AI model families used for bioreactors. Moving up adds embedded physical knowledge; moving right adds data demand. Hybrid models occupy the most favourable region and are the only family used across all four lifecycle stages. Positions are qualitative, synthesised from Mowbray et al. (2021) and Helleckes et al. (2023).](../figures/Figure_1.png){width=16cm}

Table: **Table 2.** Main AI model families used for bioreactors, their data needs and typical roles.

| Model family | Core idea (plain language) | Data needed | Interpretability | Typical bioreactor role | Representative studies |
|---|---|---|---|---|---|
| PLS / chemometrics | Finds the few spectral directions most correlated with a concentration | Tens of calibrated samples | High | Raman/NIR soft sensors for glucose, lactate, titre | Rafferty et al. (2020); Tulsyan et al. (2020) |
| Random forest, gradient boosting | Averages many decision trees, each splitting the data on simple rules | Hundreds of records | Medium (SHAP) | Batch-outcome and titre prediction from historical data | Gangadharan et al. (2021); Walsh et al. (2022) |
| Gaussian process | Predicts a value plus a confidence band from similarity between runs | Few runs (10–50) | Medium | Bayesian optimisation; cell-line transfer | Hutter et al. (2021) |
| Deep networks (CNN, LSTM, transformer) | Learns layered features from spectra, images or long time series | Large (thousands of points) | Low | Spectral calibration, forecasting, image-based counting | Lim et al. (2021); Khanal et al. (2023) |
| Hybrid (grey-box) model | Mass balances kept; network learns only unknown kinetics | Few runs (5–20) | High | Process development, digital twins, MPC | Narayanan et al. (2019); Bayer et al. (2020); Sokolov et al. (2021) |
| PINN, neural operator, graph network | Equations enforced in training, or whole flow fields learned as a function | Simulation data | Medium | Fast CFD surrogates, gradient prediction, vessel design | Karniadakis et al. (2021); Lu et al. (2021); Pfaff et al. (2021) |
| Reinforcement learning | Agent learns a policy by maximising a reward through trial | Simulator or many episodes | Low–medium | Feeding and temperature policies, constrained control | Petsagkourakis et al. (2020); Mowbray et al. (2022) |
| Language models (LLM, RAG) | Learns structure of text; retrieves and summarises documents | Large text corpora + company documents | Low (retrieval adds traceability) | Mining batch records, deviations, literature | Bommasani et al. (2021); Lewis et al. (2020) |

## 3.2 AI in bioreactor design and process development

Bioreactor *design* covers two kinds of decision. The first is the physical vessel: geometry, impeller, sparger and working volume. The second is the operating recipe: temperature, pH, dissolved-oxygen set point, feeding profile and medium. Traditionally both are explored by design of experiments (DoE), in which factors are varied according to a statistical plan and a quadratic response surface is fitted. For a CHO process with eight factors, a conventional DoE may need 30–50 bioreactor runs of two weeks each.

**Hybrid models and intensified DoE.** The most consistent finding in recent literature is that hybrid models, Eq. (16), substantially reduce this burden. Figure 2 shows the structure. Because the network learns only the "unknown part", it needs much less data and cannot predict physically impossible results such as negative concentrations (Narayanan et al., 2019).

Bayer et al. (2020) combined hybrid modelling with *intensified DoE*. In this approach, factors such as temperature are deliberately shifted within a single run, so that one bioreactor explores several conditions. In an *E. coli* case they characterised the process with far fewer runs than a classical DoE. In a follow-up CHO study, Bayer et al. (2021) showed that a hybrid model developed at small scale could be transferred to a larger bioreactor scale with few additional runs. Rogers et al. (2023) examined the *greyness* of hybrid models, meaning how much of the model is mechanistic and how much data-driven. They found that the best balance depends on data availability and on how far the model must extrapolate, a result that Case A (Section 6.1) reproduces quantitatively.

![**Figure 2.** Serial hybrid model for a fed-batch bioreactor (Eq. 16). The neural network predicts specific rates (μ, q_S, q_P) from process conditions; mass balances integrate them into concentrations. Training adjusts only the network, so conservation laws always hold. Adapted from the structures described by Narayanan et al. (2019) and Sokolov et al. (2021).](../figures/Figure_2.png){width=16cm}

**Bayesian optimisation and high-throughput systems.** When the goal is to find the best recipe rather than to understand it, Bayesian optimisation (BO, Eq. 24) is increasingly used. Combined with robotic, parallel mini-bioreactors of 10–250 mL, BO and model-based design allow dozens of conditions to be tested per week (Helleckes et al., 2023). Kim et al. (2023) ran model predictive control with moving-horizon estimation on a robotic *E. coli* platform. Their system adapted bolus feeds to each mini-bioreactor in real time, merging development and control in one loop.

**Knowledge transfer between products.** Each new cell line or molecule usually restarts development from zero. Hutter et al. (2021) proposed hybrid Gaussian-process models with *entity embeddings*: learned numerical "fingerprints" that capture how one cell line differs from others. With these embeddings, data from earlier products improved predictions for a new cell line from only a few runs. This is the bioprocess equivalent of a physician using experience with previous patients to treat a new one.

**AI for vessel and hydrodynamic design.** Physical design questions are usually answered with CFD, for example which impeller gives adequate oxygen transfer without damaging shear. A single CFD simulation of a large gassed tank can take days. del Rio-Chanona et al. (2019) used deep-learning surrogates to optimise microalgal photobioreactor design and operation, a task that would otherwise have needed thousands of simulations. Neural operators (Eq. 19) and graph networks are designed precisely for this purpose (Li et al., 2021; Lu et al., 2021; Pfaff et al., 2021). Their use for multiphase bioreactors is still at an early stage.

**Critical assessment.** Design studies remain mostly single-site and small-scale, often using simulated data or fewer than 20 real batches. Hybrid models are well validated for growth and titre but much less so for product-quality attributes. Neural-operator surrogates for gassed, non-Newtonian bioreactor flows have rarely been tested against plant measurements.

## 3.3 AI in bioreactor monitoring: soft sensors and process analytical technology

In a typical fed-batch CHO culture, pH, temperature and dissolved oxygen are measured continuously. The variables that define performance, by contrast, are sampled offline about once a day: viable cell density, glucose, lactate, ammonium, titre and glycosylation. Process analytical technology (PAT) aims to measure these in real time. A **soft sensor** is the AI component of PAT: a model that infers a hard-to-measure variable from easy-to-measure signals (Gargalo et al., 2020). A familiar analogy is a car's fuel gauge, which estimates the remaining fuel rather than weighing the tank.

**Spectroscopy-based soft sensors.** Raman spectroscopy has become the leading PAT tool for cell culture. Water interferes only weakly with the signal, and a single probe can track several compounds at once (Esmonde-White et al., 2022). The raw spectrum, however, contains hundreds of collinear wavenumbers, so a chemometric model, usually PLS (Eq. 9), is needed to convert it into concentrations. Rafferty et al. (2020) showed that spectral pre-processing and variable selection matter as much as the choice of algorithm.

A major practical problem is that a model calibrated on one product often fails on the next. Tulsyan et al. (2020) addressed this with an automated framework that calibrates *generic* Raman models across products. The framework also checks continuously whether each new spectrum lies inside the model's validity region, in essence the $T^2$/$Q$ logic of Eqs. (10–11). This turned Raman monitoring from a one-off project into a maintainable plant capability. Once glucose can be read every few minutes, it can also be controlled: feeding is triggered automatically to hold glucose at a set point.

**From single variables to whole states.** A soft sensor estimates one variable. A **state estimator** (Eqs. 20–21) reconstructs the whole internal state by combining a model with all available measurements, including slow offline ones. Kager et al. (2020) used model-based estimation in a *Penicillium chrysogenum* fed-batch. Kim et al. (2023) used moving-horizon estimation to cope with sparse, delayed measurements in parallel mini-bioreactors.

![**Figure 3.** AI soft-sensor pipeline for bioreactor monitoring. Spectra are pre-processed, converted into concentrations with an uncertainty band, and fused into a full state estimate that drives feeding control. A drift monitor (T²/Q statistics, Eqs. 10–11) checks each spectrum and triggers recalibration when the model leaves its validated region. Synthesised from Tulsyan et al. (2020), Esmonde-White et al. (2022) and Kim et al. (2023).](../figures/Figure_3.png){width=16cm}

**Predicting product quality.** Glycosylation, the pattern of sugars attached to an antibody, affects efficacy and safety. It is measured only at harvest, with slow analytical methods. Kotidis and Kontoravdi (2020) showed that artificial neural networks can predict glycan distributions from culture conditions and metabolite data. Walsh et al. (2022) reviewed how ML can support QbD by linking critical process parameters to critical quality attributes. They identified the shortage of large, well-annotated datasets as the main barrier.

**Learning from historical plant data.** Large manufacturing sites hold years of batch records. Gangadharan et al. (2021) used such data to predict biologics process performance early in a run, so that poorly performing batches could be flagged before resources were wasted. Udugama et al. (2020) argued that the value of such "big data" depends less on algorithm novelty than on data infrastructure: consistent tags, time alignment and context such as raw-material lots. Anomaly detection using MSPC, autoencoders or isolation forests can detect contamination, probe faults or foaming before an operator would (Mondal et al., 2023).

Table: **Table 3.** Representative AI applications in bioreactor monitoring.

| Monitored variable | Measurement basis | AI method | Organism / scale | Main benefit | Reference |
|---|---|---|---|---|---|
| Glucose, lactate, ammonium | In-line Raman | PLS with automated calibration and validity checks | CHO, development to manufacturing | Generic models reused across products | Tulsyan et al. (2020) |
| Multiple metabolites | In-line Raman | Comparison of chemometric models | CHO cell culture | Guidance on pre-processing and variable selection | Rafferty et al. (2020) |
| Biomass and substrate states | Off-gas, base addition, delayed offline samples | Model-based state estimation | *P. chrysogenum* fed-batch | Real-time states for advanced control | Kager et al. (2020) |
| Biomass and glucose in parallel runs | Sparse at-line samples | Moving-horizon estimation | *E. coli*, robotic mini-bioreactors | Run-specific adaptive feeding | Kim et al. (2023) |
| Glycan distribution | Culture conditions, metabolites | Artificial neural network | CHO, antibody production | Quality predicted before harvest | Kotidis and Kontoravdi (2020) |
| Final batch performance | Historical batch records | ML on early-process data | Biologics manufacturing | Early warning for under-performing batches | Gangadharan et al. (2021) |

**Critical assessment.** Soft sensors are the most industrially mature AI application for bioreactors. The open problems concern maintenance and trust rather than accuracy. Models drift as raw materials, probes and cell lines change, and few studies report long-term behaviour. Calibrated uncertainty estimates, which tell the operator when *not* to believe a prediction, are still the exception. Case B (Section 6.2) shows how much accuracy a single raw-material change can cost, and how MSPC recovers it.

## 3.4 AI in bioreactor control and optimisation

Monitoring tells us what is happening; control decides what to do about it. Most industrial bioreactors still use PID loops for temperature, pH and dissolved oxygen, and fixed feeding profiles for nutrients. A PID loop reacts only to the present error. It cannot anticipate, for example, that the culture will run out of glucose in three hours.

**Model predictive control (MPC).** MPC (Eq. 22) uses a process model to predict the next few hours. It chooses the control moves that best meet the targets while respecting constraints, applies only the first move, and repeats the calculation when new data arrive. The approach resembles a driver who watches the road ahead rather than the bonnet. Kager et al. (2020) compared MPC, PID and model-inversion control in a *P. chrysogenum* fed-batch and reported that the model-based strategies tracked the specific-growth-rate target more closely. Rathore et al. (2021) concluded that the limiting factor for MPC is a model that is both accurate and cheap to run. This is precisely where hybrid models and AI surrogates help.

**Reinforcement learning.** RL (Eq. 23) learns a *policy*, a rule mapping the current state to an action, by trial and error guided by a reward. Petsagkourakis et al. (2020) showed that policy-gradient RL could optimise batch bioprocesses despite plant–model mismatch. The agent was pre-trained on an approximate model and then refined on the "real" plant. Mowbray et al. (2022) addressed the risk of unsafe exploration with chance-constrained RL. Nikita et al. (2021) applied RL to chromatography loading coupled to continuous upstream processing.

**Optimising whole campaigns.** Beyond real-time control, AI is used to optimise run-to-run decisions: the next feed recipe, the harvest time, or perfusion rates in continuous processes. Puranik et al. (2022) and Cheng et al. (2023) noted that most RL studies are still simulation-based, because every real experiment is expensive.

**Critical assessment.** The literature suggests a practical hierarchy:

- PID remains the right tool for fast physical loops.
- Hybrid-model MPC is ready for nutrient feeding and growth-rate control wherever a validated model exists.
- RL is best viewed today as a way to generate candidate policies *in silico*, to be checked by MPC or by human experts before use.
- BO is the most data-efficient tool for run-to-run optimisation. Case D (Section 6.4) quantifies its advantage over classical DoE.

## 3.5 AI in bioreactor scale-up and scale-down

**Why scale-up is hard.** As shown by Eq. (8), the Damköhler number in a large fed-batch tank can exceed 10. Substrate is then high near the feed point and depleted elsewhere, while dissolved oxygen is high near the sparger and low at the top (Nadal-Rey et al., 2021a). A single cell travels through these zones many times per hour, like a swimmer passing repeatedly between warm and cold currents. *E. coli* and yeast respond by switching metabolic pathways, forming by-products such as acetate or ethanol, and losing yield.

**Classical criteria and their limits.** Engineers traditionally scale up by holding one quantity constant: P/V, $k_La$, tip speed or mixing time (Eq. 7). These cannot all be held constant at once. Each criterion protects one aspect of the environment while ignoring the others, which is why scale-up remains partly empirical (Wang et al., 2020).

**Compartment models and cell lifelines.** Two modelling advances underpin the AI approaches.

- *Compartment models* divide the tank into well-mixed zones connected by flows derived from CFD (Nadal-Rey et al., 2021b).
- *Lifeline* analysis tracks thousands of virtual cells through a flow field and records the sequence of concentrations each one experiences. Haringa et al. (2018) coupled lifelines to a nine-pool metabolic model of *P. chrysogenum* to design rational scale-down experiments. Wang et al. (2020) proposed a general computational framework built on CFD, lifelines and cell models.

**Where AI enters.** AI contributes in three ways.

1. *Surrogates*: neural operators and graph networks predict concentration fields for new designs in seconds.
2. *Learning the cell response*: hybrid models learn how metabolism responds to fluctuating rather than constant conditions.
3. *Transfer learning*: a small-scale model is adapted to large scale using a handful of plant runs (Bayer et al., 2021; Rogers et al., 2023; Hutter et al., 2021).

**From single criterion to regime matching.** Combining these strands leads to an idea that is implicit in the literature but rarely formalised. Scale-down should reproduce the full *distribution* of conditions that cells experience at large scale, not a single number such as P/V or mixing time. Section 4 formalises this as an optimal-transport problem, and Case C tests it. Figure 4 outlines the workflow and Table 4 contrasts it with classical practice.

![**Figure 4.** AI-assisted, regime-matched scale-up workflow. A few expensive CFD runs train a fast surrogate; virtual cell lifelines yield the distribution of regimes in the large tank; a laboratory scale-down system is designed to reproduce that distribution; and the hybrid model is transfer-learned and verified at scale before a go/no-go decision.](../figures/Figure_4.png){width=16cm}

Table: **Table 4.** Classical versus AI-enhanced scale-up strategies.

| Aspect | Classical approach | AI-enhanced approach | Key enabling studies |
|---|---|---|---|
| Scale-up criterion | Keep one parameter constant (P/V, $k_La$, tip speed, mixing time) | Match the distribution of regimes and excursion times cells experience (Eq. 28) | Haringa et al. (2018); Wang et al. (2020) |
| Description of the large tank | Correlations for $k_La$ and mixing time | CFD and compartment models replaced by neural-operator or graph-network surrogates | Nadal-Rey et al. (2021b); Li et al. (2021); Pfaff et al. (2021) |
| Cell response | Assumed equal to response at constant lab conditions | Hybrid or deep models trained on fluctuating, lifeline-like conditions | Haringa et al. (2018); Sokolov et al. (2021) |
| Scale-down experiments | Ad hoc pulses or STR+PFR set-ups sized by mixing time | Design optimised to minimise a distributional distance to production | This chapter (Case C) |
| Model transfer to plant | Re-fit empirical model from scratch | Transfer learning of hybrid model with uncertainty bounds | Bayer et al. (2021); Hutter et al. (2021); Rogers et al. (2023) |
| Decision support | Expert judgement and pilot runs | Digital twin ranks designs and flags risk before construction | Chen et al. (2020); Isoko et al. (2024) |

**Critical assessment.** AI for scale-up is the least mature of the four areas. Most studies rely on simulation or pilot-scale data. Metabolic response models exist for only a few organisms, and validation at 10–100 m³ is rare because such data are commercially sensitive. Yet this is where AI could deliver the largest economic benefit, since a failed scale-up can delay a product launch by months.

## 3.6 Digital twins, explainability and regulation

**Digital twins.** A digital twin is a model continuously synchronised with a specific physical bioreactor. It is used to predict that bioreactor's future, to test decisions virtually and to feed recommendations back (Chen et al., 2020). Sokolov et al. (2021) argued that hybrid models are the natural core of bioprocess twins. Isoko et al. (2024) observed that many self-described "digital twins" are in fact offline models without a live data connection. Data infrastructure is repeatedly identified as the bottleneck (Udugama et al., 2020).

**Explainability.** Process engineers and quality units must understand why a model makes a recommendation. SHAP values (Eq. 26) are now used to rank critical process parameters (Walsh et al., 2022). Hybrid models offer a stronger form of explainability: their mechanistic part can be read directly, and the learned kinetic terms can be plotted and compared with biological knowledge.

**Regulation.** The FDA discussion paper on AI in drug manufacturing (FDA, 2023) identified model credibility, data management, continuously learning models and cybersecurity as key concerns. It also signalled openness to AI within existing quality frameworks. Together with ICH Q13 (ICH, 2022), this creates a pathway in which an AI model is treated like any other process model. Its risk is assessed by its impact on product quality, and its validation effort is scaled accordingly.

**Emerging directions.** Foundation and language models (Section 2.9) could be fine-tuned for a new product with very little data. Reviews by Khanal et al. (2023), Mondal et al. (2023) and Cheng et al. (2023) also highlight AI in non-pharmaceutical bioprocesses, such as biofuels, wastewater and food fermentation, where cost constraints are tighter and sensors cheaper.

## 3.7 Synthesis of research gaps

Across the four areas, the review identifies six recurring gaps:

1. **Fragmentation.** Separate models are built for design, monitoring, control and scale-up, so knowledge is lost at each hand-over.
2. **Data scarcity at large scale.** Models are rarely validated above pilot scale.
3. **Single-criterion scale-up.** Few works capture the full environment that cells experience.
4. **Weak uncertainty quantification.** Predictions are often reported without calibrated intervals.
5. **Model lifecycle management.** Drift detection, retraining and version control are seldom addressed.
6. **Quantitative evidence.** Few studies compare model classes head-to-head under controlled interpolation and extrapolation conditions with statistical tests.

These gaps motivate the framework of Section 4 and the case studies of Sections 5–6.
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
# References

Angelopoulos, A.N., Bates, S., 2023. Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning 16, 494–591.

Bastin, G., Dochain, D., 1990. On-line Estimation and Adaptive Control of Bioreactors. Elsevier, Amsterdam.

Bayer, B., Striedner, G., Duerkop, M., 2020. Hybrid modeling and intensified DoE: an approach to accelerate upstream process characterization. Biotechnology Journal 15, 2000121.

Bayer, B., Duerkop, M., Striedner, G., Sissolak, B., 2021. Model transferability and reduced experimental burden in cell culture process development facilitated by hybrid modeling and intensified design of experiments. Frontiers in Bioengineering and Biotechnology 9, 740215.

Bommasani, R., Hudson, D.A., Adeli, E., et al., 2021. On the opportunities and risks of foundation models. arXiv:2108.07258.

Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D., 2018. Neural ordinary differential equations. In: Advances in Neural Information Processing Systems 31 (NeurIPS 2018).

Chen, Y., Yang, O., Sampat, C., Bhalode, P., Ramachandran, R., Ierapetritou, M., 2020. Digital twins in pharmaceutical and biopharmaceutical manufacturing: a literature review. Processes 8, 1088.

Cheng, Y., Bi, X., Xu, Y., Liu, Y., Li, J., Du, G., Lv, X., Liu, L., 2023. Artificial intelligence technologies in bioprocess: opportunities and challenges. Bioresource Technology 369, 128451.

del Rio-Chanona, E.A., Wagner, J.L., Ali, H., Fiorelli, F., Zhang, D., Hellgardt, K., 2019. Deep learning-based surrogate modeling and optimization for microalgal biofuel production and photobioreactor design. AIChE Journal 65, 915–923.

Esmonde-White, K.A., Cuellar, M., Lewis, I.R., 2022. The role of Raman spectroscopy in biopharmaceuticals from development to manufacturing. Analytical and Bioanalytical Chemistry 414, 969–991.

FDA (U.S. Food and Drug Administration), 2023. Artificial Intelligence in Drug Manufacturing. Discussion paper, Center for Drug Evaluation and Research, Silver Spring, MD.

Gangadharan, N., Sewell, D., Turner, R., Field, R., Cheeks, M., Oliver, S.G., Slater, N.K.H., Dikicioglu, D., 2021. Data intelligence for process performance prediction in biologics manufacturing. Computers & Chemical Engineering 146, 107226.

Gargalo, C.L., Udugama, I., Pontius, K., Lopez, P.C., Nielsen, R.F., Hasanzadeh, A., Mansouri, S.S., Bayer, C., Junicke, H., Gernaey, K.V., 2020. Towards smart biomanufacturing: a perspective on recent developments in industrial measurement and monitoring technologies for bio-based production processes. Journal of Industrial Microbiology & Biotechnology 47, 947–964.

Grenville, R.K., Nienow, A.W., 2004. Blending of miscible liquids. In: Paul, E.L., Atiemo-Obeng, V.A., Kresta, S.M. (Eds.), Handbook of Industrial Mixing: Science and Practice. Wiley, Hoboken, NJ, pp. 507–542.

Haringa, C., Tang, W., Wang, G., Deshmukh, A.T., van Winden, W.A., Chu, J., van Gulik, W.M., Heijnen, J.J., Mudde, R.F., Noorman, H.J., 2018. Computational fluid dynamics simulation of an industrial *P. chrysogenum* fermentation with a coupled 9-pool metabolic model: towards rational scale-down and design optimization. Chemical Engineering Science 175, 12–24.

Helleckes, L.M., Hemmerich, J., Wiechert, W., von Lieres, E., Grünberger, A., 2023. Machine learning in bioprocess development: from promise to practice. Trends in Biotechnology 41, 817–835.

Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural Computation 9, 1735–1780.

Hutter, C., von Stosch, M., Cruz Bournazou, M.N., Butté, A., 2021. Knowledge transfer across cell lines using hybrid Gaussian process models with entity embedding vectors. Biotechnology and Bioengineering 118, 4389–4401.

ICH (International Council for Harmonisation), 2022. Q13: Continuous Manufacturing of Drug Substances and Drug Products. ICH Harmonised Guideline, Geneva.

Isoko, K., Cordiner, J.L., Kis, Z., Moghadam, P.Z., 2024. Bioprocessing 4.0: a pragmatic review and future perspectives. Digital Discovery 3, 1662–1681.

Jones, D.R., Schonlau, M., Welch, W.J., 1998. Efficient global optimization of expensive black-box functions. Journal of Global Optimization 13, 455–492.

Kager, J., Tuveri, A., Ulonska, S., Kroll, P., Herwig, C., 2020. Experimental verification and comparison of model predictive, PID and model inversion control in a *Penicillium chrysogenum* fed-batch process. Process Biochemistry 90, 1–11.

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L., 2021. Physics-informed machine learning. Nature Reviews Physics 3, 422–440.

Khanal, S.K., Tarafdar, A., You, S., 2023. Artificial intelligence and machine learning for smart bioprocesses. Bioresource Technology 375, 128826.

Kim, J.W., Krausch, N., Aizpuru, J., Barz, T., Lucia, S., Neubauer, P., Cruz Bournazou, M.N., 2023. Model predictive control and moving horizon estimation for adaptive optimal bolus feeding in high-throughput cultivation of *E. coli*. Computers & Chemical Engineering 172, 108158.

Kotidis, P., Kontoravdi, C., 2020. Harnessing the potential of artificial neural networks for predicting protein glycosylation. Metabolic Engineering Communications 10, e00131.

Lakshminarayanan, B., Pritzel, A., Blundell, C., 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017).

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial differential equations. In: International Conference on Learning Representations (ICLR 2021).

Lim, B., Arık, S.Ö., Loeff, N., Pfister, T., 2021. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting 37, 1748–1764.

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E., 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218–229.

Lundberg, S.M., Lee, S.-I., 2017. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 4765–4774.

MacGregor, J.F., Kourti, T., 1995. Statistical process control of multivariate processes. Control Engineering Practice 3, 403–414.

Mondal, P.P., Galodha, A., Verma, V.K., Singh, V., Show, P.L., Awasthi, M.K., Lall, B., Anees, S., Pollmann, K., Jain, R., 2023. Review on machine learning-based bioprocess optimization, monitoring, and control systems. Bioresource Technology 370, 128523.

Mowbray, M., Savage, T., Wu, C., Song, Z., Cho, B.A., Del Rio-Chanona, E.A., Zhang, D., 2021. Machine learning for biochemical engineering: a review. Biochemical Engineering Journal 172, 108054.

Mowbray, M., Petsagkourakis, P., del Rio-Chanona, E.A., Zhang, D., 2022. Safe chance constrained reinforcement learning for batch process control. Computers & Chemical Engineering 157, 107630.

Nadal-Rey, G., McClure, D.D., Kavanagh, J.M., Cornelissen, S., Fletcher, D.F., Gernaey, K.V., 2021a. Understanding gradients in industrial bioreactors. Biotechnology Advances 46, 107660.

Nadal-Rey, G., McClure, D.D., Kavanagh, J.M., Cassells, B., Cornelissen, S., Fletcher, D.F., Gernaey, K.V., 2021b. Development of dynamic compartment models for industrial aerobic fed-batch fermentation processes. Chemical Engineering Journal 420, 130402.

Narayanan, H., Sokolov, M., Morbidelli, M., Butté, A., 2019. A new generation of predictive models: the added value of hybrid models for manufacturing processes of therapeutic proteins. Biotechnology and Bioengineering 116, 2540–2549.

Narayanan, H., Luna, M.F., von Stosch, M., Cruz Bournazou, M.N., Polotti, G., Morbidelli, M., Butté, A., Sokolov, M., 2020. Bioprocessing in the digital age: the role of process models. Biotechnology Journal 15, 1900172.

Nikita, S., Tiwari, A., Sonawat, D., Kodamana, H., Rathore, A.S., 2021. Reinforcement learning based optimization of process chromatography for continuous processing of biopharmaceuticals. Chemical Engineering Science 230, 116171.

Petsagkourakis, P., Sandoval, I.O., Bradford, E., Zhang, D., del Rio-Chanona, E.A., 2020. Reinforcement learning for batch bioprocess optimization. Computers & Chemical Engineering 133, 106649.

Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P.W., 2021. Learning mesh-based simulation with graph networks. In: International Conference on Learning Representations (ICLR 2021).

Puranik, A., Dandekar, P., Jain, R., 2022. Exploring the potential of machine learning for more efficient development and production of biopharmaceuticals. Biotechnology Progress 38, e3291.

Rafferty, C., Johnson, K., O'Mahony, J., Burgoyne, B., Rea, R., Balss, K.M., 2020. Analysis of chemometric models applied to Raman spectroscopy for monitoring key metabolites of cell culture. Biotechnology Progress 36, e2977.

Raissi, M., Perdikaris, P., Karniadakis, G.E., 2019. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics 378, 686–707.

Rasmussen, C.E., Williams, C.K.I., 2006. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA.

Rathore, A.S., Mishra, S., Nikita, S., Priyanka, P., 2021. Bioprocess control: current progress and future perspectives. Life 11, 557.

Rathore, A.S., Nikita, S., Thakur, G., Mishra, S., 2023. Artificial intelligence and machine learning applications in biopharmaceutical manufacturing. Trends in Biotechnology 41, 497–510.

Rogers, A.W., Song, Z., Vega-Ramon, F., Jing, K., Zhang, D., 2023. Investigating 'greyness' of hybrid model for bioprocess predictive modelling. Biochemical Engineering Journal 190, 108761.

Sokolov, M., von Stosch, M., Narayanan, H., Feidl, F., Butté, A., 2021. Hybrid modeling — a key enabler towards realizing digital twins in biopharma? Current Opinion in Chemical Engineering 34, 100715.

Sutton, R.S., Barto, A.G., 2018. Reinforcement Learning: An Introduction, 2nd ed. MIT Press, Cambridge, MA.

Tulsyan, A., Wang, T., Schorner, G., Khodabandehlou, H., Coufal, M., Undey, C., 2020. Automatic real-time calibration, assessment, and maintenance of generic Raman models for online monitoring of cell culture processes. Biotechnology and Bioengineering 117, 406–416.

Udugama, I.A., Gargalo, C.L., Yamashita, Y., Taube, M.A., Palazoglu, A., Young, B.R., Gernaey, K.V., Kulahci, M., Bayer, C., 2020. The role of big data in industrial (bio)chemical process operations. Industrial & Engineering Chemistry Research 59, 15283–15297.

van't Riet, K., 1979. Review of measuring methods and results in nonviscous gas–liquid mass transfer in stirred vessels. Industrial & Engineering Chemistry Process Design and Development 18, 357–364.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017).

Walsh, I., Myint, M., Nguyen-Khuong, T., Ho, Y.S., Ng, S.K., Lakshmanan, M., 2022. Harnessing the potential of machine learning for advancing "Quality by Design" in biomanufacturing. mAbs 14, 2013593.

Wang, G., Haringa, C., Noorman, H., Chu, J., Zhuang, Y., 2020. Developing a computational framework to advance bioprocess scale-up. Trends in Biotechnology 38, 846–856.

Wold, S., Sjöström, M., Eriksson, L., 2001. PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems 58, 109–130.

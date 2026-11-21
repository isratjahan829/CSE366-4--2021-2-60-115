---
title: "Artificial Intelligence in Bioreactor Design, Monitoring, and Scale-Up: Mathematical Foundations, Critical Review, and an Integrated Hybrid Digital-Twin Framework"
subtitle: "Book chapter (Elsevier) — extended version with mathematical formulation and in-silico case studies"
---

# Abstract

Bioreactors are the workhorses of biomanufacturing, yet their design, monitoring and scale-up still rest on empirical correlations, sparse sampling and trial-and-error. The cost of this empiricism peaks at scale-up. In a 50 m³ fermenter, the liquid takes longer to mix than cells take to exhaust the local substrate, so each cell cycles between feast and famine, and processes that excel at 2 L lose yield and quality in production. Here we examine how artificial intelligence (AI) can replace this empiricism with predictive, data-efficient and verifiable engineering. We unify bioreactor transport and kinetics with statistical, machine-learning, deep-learning, physics-informed, reinforcement-learning and language-model methods in one mathematical framework, and critically review the literature from 2020–2025. From this synthesis we propose the Design–Sense–Decide–Scale (DSDS) framework, in which one hybrid digital twin spans the entire bioreactor lifecycle. Within it, we formalise regime-matched scale-down as the minimisation of a Wasserstein distance between the environments that cells experience at laboratory and at production scale. Four reproducible in-silico case studies quantify the gains. Hybrid models halved prediction error relative to black-box networks (4.4% versus 8.9%; p = 0.002). A Raman soft sensor predicted glucose to within 0.26 g/L and detected a raw-material drift within two samples. Regime matching reduced environmental mismatch by 85%, and Bayesian optimisation reached a median 97% of the optimum within 15 runs. The case studies also delineate a limit: no model class extrapolated reliably beyond the temperature range covered by its training data. These results offer a quantitative roadmap towards trustworthy, AI-enabled bioprocessing.

**Keywords:** artificial intelligence; bioreactor; hybrid modelling; soft sensor; multivariate statistical process control; digital twin; scale-up; scale-down; Wasserstein distance; Bayesian optimisation; physics-informed machine learning; reinforcement learning

# 1. Introduction

Biomanufacturing now underpins a substantial share of modern medicine and an expanding part of the industrial economy. Monoclonal antibodies, insulin, vaccines, industrial enzymes, organic acids, biofuels and cultivated meat are all made by living cells growing in a bioreactor. The core hardware has barely changed in decades: a stirred tank, an impeller, a sparger, a cooling jacket and a handful of probes. What has changed is the pressure on it. Regulators expect processes to be understood scientifically rather than merely reproduced, under the Quality by Design (QbD) paradigm. Continuous manufacturing has been formally codified in ICH Q13 (ICH, 2022). And every month a new therapy spends in development delays patient access and adds cost (Rathore et al., 2021). The bioreactor has thus become both the most valuable and the least predictable step in the production chain.

Three long-standing problems explain why. The first is *design*. Choosing vessel geometry, impeller type, gassing strategy and feeding profile demands many costly experiments, because cell metabolism is non-linear and only partly understood. The second is *monitoring*. The variables that define performance, including viable cell density, glucose, lactate, titre and glycosylation, are typically measured offline once or twice a day, so a process can drift for hours before anyone notices (Gargalo et al., 2020). The third, and most consequential, is *scale-up*. A production tank is not simply a large flask. In a 50 m³ fermenter, mixing can take more than a minute, while a cell consumes the glucose in its immediate surroundings within seconds. Cells therefore circulate repeatedly between zones of excess and starvation (Nadal-Rey et al., 2021a). These fluctuations shift substrate yield, by-product formation and productivity in ways that laboratory experiments do not reveal, which is why a process optimised at 2 L so often under-performs at production scale (Wang et al., 2020).

Artificial intelligence (AI), meaning computational methods that learn patterns from data and use them to predict, classify or decide, has matured in precisely the areas these problems require. Machine-learning soft sensors convert an in-line Raman spectrum into a glucose concentration every few minutes (Esmonde-White et al., 2022). Hybrid models embed neural networks within mass balances and need far fewer experiments than purely empirical models (Narayanan et al., 2020; Sokolov et al., 2021). Reinforcement learning can discover feeding policies in simulation (Petsagkourakis et al., 2020). Physics-informed networks and neural operators can emulate hours of computational fluid dynamics (CFD) in seconds (Karniadakis et al., 2021). Large language models open new ways to mine decades of batch records and deviation reports. Combined with cheaper sensors and cloud data platforms, these tools underpin the vision of *Bioprocessing 4.0* (Isoko et al., 2024).

Yet three gaps prevent this promise from translating into routine practice. First, the literature is fragmented. Most reviews address a single task or a single model family (Mowbray et al., 2021; Helleckes et al., 2023; Rathore et al., 2023), so a model built during development is rarely carried forward to monitoring, control and scale-up, and knowledge is lost at every hand-over. Second, the mathematics that connects bioreactor physics to learning algorithms is seldom made explicit, which leaves engineers and data scientists without a shared language. Third, quantitative evidence is scarce. Few studies compare model classes head-to-head under controlled conditions, separate interpolation from extrapolation, or apply statistical tests, and almost none address scale-up as a problem of reproducing the environment that cells actually experience. Limited data, poor transferability between scales and cell lines, and regulatory uncertainty about "black-box" models compound these gaps (FDA, 2023).

This chapter responds by combining mathematical synthesis, critical review, conceptual design and quantitative testing in a single account. It treats the bioreactor lifecycle as one connected problem rather than four separate ones, and it argues that the persistent asset of a bioprocess should be a continuously learning model, not a series of disconnected experiments. Its central conceptual proposal is that scale-down should be designed to reproduce the full *distribution* of conditions experienced by cells at production scale, rather than a single engineering parameter such as power input per volume or mixing time.

Accordingly, the chapter pursues four objectives. The first is to establish a unified mathematical foundation that links bioreactor transport, kinetics and scale-up criteria to the statistical, machine-learning, deep-learning, physics-informed, control and language-model methods that act on them. The second is to critically evaluate recent evidence (2020–2025) on AI for bioreactor design, monitoring, control and scale-up, and to identify the gaps that limit industrial adoption. The third is to develop an integrated Design–Sense–Decide–Scale framework centred on a hybrid digital twin, including a formal, optimal-transport definition of regime-matched scale-down. The fourth is to quantify, through four reproducible in-silico case studies with statistical testing, how much hybrid modelling, soft sensing with drift detection, regime matching and Bayesian optimisation actually improve on conventional practice, and where their limits lie.

The remainder of the chapter follows these objectives. Section 2 develops the theoretical and mathematical foundations, and Section 3 reviews the literature. Section 4 presents the DSDS framework, Section 5 describes the case-study methodology and Section 6 reports the results. Sections 7–9 distil the findings, discuss their implications with recommendations for industry, academia and regulators, and set out directions for future research.

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

The impeller power draw is $P=N_P\,\rho N^3 D_i^5$. Here $N_P$ is the power number (about 5 for a Rushton turbine), $\rho$ is the liquid density, $N$ is the stirrer speed and $D_i$ is the impeller diameter. The Reynolds number is $Re=\rho N D_i^2/\eta$, where $\eta$ is the liquid viscosity, and the tip speed is $u_{tip}=\pi N D_i$. For turbulent, single-impeller tanks, Grenville and Nienow (2003) give the 95% mixing time as

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

*Retrieval-augmented generation (RAG)* combines such retrieval with a generative model, so that answers are grounded in a company's own validated documents (Lewis et al., 2020). In bioprocessing, such systems can retrieve past deviations that resemble a current foaming event, extract process parameters from decades of scanned batch records into structured tables for machine learning, and draft first versions of investigation reports for expert review.

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

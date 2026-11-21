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

Bayer et al. (2020) combined hybrid modelling with *intensified DoE*. In this approach, factors such as temperature are deliberately shifted within a single run, so that one bioreactor explores several conditions. In 20 L *E. coli* fed-batch cultivations, the intensified hybrid model needed only one-third of the training data of a full-factorial description, an experimental saving of more than 66%. In a follow-up CHO study, Bayer et al. (2021) trained a hybrid model on shaker-scale data and used it to predict 15 L bioreactor runs, a 1:50 scale-up. Rogers et al. (2023) examined the *greyness* of hybrid models, meaning how much of the model is mechanistic and how much data-driven. They found that the best balance depends on data availability and on how far the model must extrapolate, a result that Case A (Section 6.1) reproduces quantitatively.

![**Figure 2.** Serial hybrid model for a fed-batch bioreactor (Eq. 16). The neural network predicts specific rates (μ, q_S, q_P) from process conditions; mass balances integrate them into concentrations. Training adjusts only the network, so conservation laws always hold. Adapted from the structures described by Narayanan et al. (2019) and Sokolov et al. (2021).](../figures/Figure_2.png){width=16cm}

**Bayesian optimisation and high-throughput systems.** When the goal is to find the best recipe rather than to understand it, Bayesian optimisation (BO, Eq. 24) is increasingly used. Combined with robotic, parallel mini-bioreactors of 10–250 mL, BO and model-based design allow dozens of conditions to be tested per week (Helleckes et al., 2023). Kim et al. (2023) combined model predictive control with multi-rate moving-horizon estimation for bolus-fed *E. coli* cultivations. In a simulation study of 24 parallel mini-bioreactors, their framework adapted the feeding of each vessel individually, merging development and control in one loop.

**Knowledge transfer between products.** Each new cell line or molecule usually restarts development from zero. Hutter et al. (2021) proposed hybrid Gaussian-process models with *entity embeddings*: learned numerical "fingerprints" that capture how one cell line differs from others. With these embeddings, data from earlier products improved predictions for a new cell line from only a few runs. This is the bioprocess equivalent of a physician using experience with previous patients to treat a new one.

**AI for vessel and hydrodynamic design.** Physical design questions are usually answered with CFD, for example which impeller gives adequate oxygen transfer without damaging shear. A single CFD simulation of a large gassed tank can take days. del Rio-Chanona et al. (2019) used deep-learning surrogates to optimise microalgal photobioreactor design and operation, a task that would otherwise have needed thousands of simulations. Neural operators (Eq. 19) and graph networks are designed precisely for this purpose (Li et al., 2021; Lu et al., 2021; Pfaff et al., 2021). Their use for multiphase bioreactors is still at an early stage.

**Critical assessment.** Design studies remain mostly single-site and small-scale, often using simulated data or fewer than 20 real batches. Hybrid models are well validated for growth and titre but much less so for product-quality attributes. Neural-operator surrogates for gassed, non-Newtonian bioreactor flows have rarely been tested against plant measurements.

## 3.3 AI in bioreactor monitoring: soft sensors and process analytical technology

In a typical fed-batch CHO culture, pH, temperature and dissolved oxygen are measured continuously. The variables that define performance, by contrast, are sampled offline about once a day: viable cell density, glucose, lactate, ammonium, titre and glycosylation. Process analytical technology (PAT) aims to measure these in real time. A **soft sensor** is the AI component of PAT: a model that infers a hard-to-measure variable from easy-to-measure signals (Gargalo et al., 2020). A familiar analogy is a car's fuel gauge, which estimates the remaining fuel rather than weighing the tank.

**Spectroscopy-based soft sensors.** Raman spectroscopy has become the leading PAT tool for cell culture. Water interferes only weakly with the signal, and a single probe can track several compounds at once (Esmonde-White et al., 2022). The raw spectrum, however, contains hundreds of collinear wavenumbers, so a chemometric model, usually PLS (Eq. 9), is needed to convert it into concentrations. Rafferty et al. (2020) compared PLS with support-vector machines, random forests and Cubist models for glucose, lactate and ammonia. Cubist gave the best average performance, showing that the linear PLS default is worth challenging.

A major practical problem is that a model calibrated on one product often fails on the next. Tulsyan et al. (2020) addressed this with an automated framework that calibrates *generic* Raman models across products. The framework also checks continuously whether each new spectrum lies inside the model's validity region, in essence the $T^2$/$Q$ logic of Eqs. (10–11). This turned Raman monitoring from a one-off project into a maintainable plant capability. Once glucose can be read every few minutes, it can also be controlled: feeding is triggered automatically to hold glucose at a set point.

**From single variables to whole states.** A soft sensor estimates one variable. A **state estimator** (Eqs. 20–21) reconstructs the whole internal state by combining a model with all available measurements, including slow offline ones. Kager et al. (2020) used model-based estimation in a *Penicillium chrysogenum* fed-batch. Kim et al. (2023) used moving-horizon estimation to cope with sparse, delayed measurements in parallel mini-bioreactors.

![**Figure 3.** AI soft-sensor pipeline for bioreactor monitoring. Spectra are pre-processed, converted into concentrations with an uncertainty band, and fused into a full state estimate that drives feeding control. A drift monitor (T²/Q statistics, Eqs. 10–11) checks each spectrum and triggers recalibration when the model leaves its validated region. Synthesised from Tulsyan et al. (2020), Esmonde-White et al. (2022) and Kim et al. (2023).](../figures/Figure_3.png){width=16cm}

**Predicting product quality.** Glycosylation, the pattern of sugars attached to an antibody, affects efficacy and safety. It is measured only at harvest, with slow analytical methods. Kotidis and Kontoravdi (2020) showed that artificial neural networks can predict glycan distributions from culture conditions and metabolite data. Walsh et al. (2022) reviewed how ML can support QbD by linking critical process parameters to critical quality attributes. They identified the shortage of large, well-annotated datasets as the main barrier.

**Learning from historical plant data.** Large manufacturing sites hold years of batch records. Gangadharan et al. (2021) used such data to predict biologics process performance early in a run, so that poorly performing batches could be flagged before resources were wasted. Udugama et al. (2020) argued that the value of such "big data" depends less on algorithm novelty than on data infrastructure: consistent tags, time alignment and context such as raw-material lots. Anomaly detection using MSPC, autoencoders or isolation forests can detect contamination, probe faults or foaming before an operator would (Mondal et al., 2023).

Table: **Table 3.** Representative AI applications in bioreactor monitoring.

| Monitored variable | Measurement basis | AI method | Organism / scale | Main benefit | Reference |
|---|---|---|---|---|---|
| Glucose, lactate, ammonium | In-line Raman | PLS with automated calibration and validity checks | CHO, development to manufacturing | Generic models reused across products | Tulsyan et al. (2020) |
| Glucose, lactate, ammonia | In-line Raman | PLS vs SVM, random forest and Cubist | CHO cell culture | Non-linear Cubist models outperformed PLS on average | Rafferty et al. (2020) |
| Biomass and substrate states | Off-gas, base addition, delayed offline samples | Model-based state estimation | *P. chrysogenum* fed-batch | Real-time states for advanced control | Kager et al. (2020) |
| Biomass and glucose in parallel runs | Sparse at-line samples | Moving-horizon estimation | *E. coli*, robotic mini-bioreactors | Run-specific adaptive feeding | Kim et al. (2023) |
| Glycan distribution | Culture conditions, metabolites | Artificial neural network | CHO, antibody production | Quality predicted before harvest | Kotidis and Kontoravdi (2020) |
| Final batch performance | Historical batch records | ML on early-process data | Biologics manufacturing | Early warning for under-performing batches | Gangadharan et al. (2021) |

**Critical assessment.** Soft sensors are the most industrially mature AI application for bioreactors. The open problems concern maintenance and trust rather than accuracy. Models drift as raw materials, probes and cell lines change, and few studies report long-term behaviour. Calibrated uncertainty estimates, which tell the operator when *not* to believe a prediction, are still the exception. Case B (Section 6.2) shows how much accuracy a single raw-material change can cost, and how MSPC recovers it.

## 3.4 AI in bioreactor control and optimisation

Monitoring tells us what is happening; control decides what to do about it. Most industrial bioreactors still use PID loops for temperature, pH and dissolved oxygen, and fixed feeding profiles for nutrients. A PID loop reacts only to the present error. It cannot anticipate, for example, that the culture will run out of glucose in three hours.

**Model predictive control (MPC).** MPC (Eq. 22) uses a process model to predict the next few hours. It chooses the control moves that best meet the targets while respecting constraints, applies only the first move, and repeats the calculation when new data arrive. The approach resembles a driver who watches the road ahead rather than the bonnet. Kager et al. (2020) implemented a non-linear MPC in a *P. chrysogenum* fed-batch process and compared it experimentally with PID control and a model-inversion (open-loop, model-based) feeding strategy. Rathore et al. (2021) concluded that the limiting factor for MPC is a model that is both accurate and cheap to run. This is precisely where hybrid models and AI surrogates help.

**Reinforcement learning.** RL (Eq. 23) learns a *policy*, a rule mapping the current state to an action, by trial and error guided by a reward. Petsagkourakis et al. (2020) showed that policy-gradient RL could optimise batch bioprocesses despite plant–model mismatch. The agent was pre-trained on an approximate model and then refined on the "real" plant. Mowbray et al. (2022) addressed the risk of unsafe exploration with chance-constrained RL. Nikita et al. (2021) applied RL to chromatography loading coupled to continuous upstream processing.

**Optimising whole campaigns.** Beyond real-time control, AI is used to optimise run-to-run decisions: the next feed recipe, the harvest time, or perfusion rates in continuous processes. Puranik et al. (2022) and Cheng et al. (2023) noted that most RL studies are still simulation-based, because every real experiment is expensive.

**Critical assessment.** Taken together, the evidence points to a pragmatic division of labour rather than a single winning method. PID control remains the right tool for fast physical loops such as temperature and pH. Hybrid-model MPC is ready for nutrient feeding and growth-rate control wherever a validated model exists. Reinforcement learning is best regarded, for now, as a generator of candidate policies *in silico* that MPC or human experts must check before use. For run-to-run optimisation, Bayesian optimisation is the most data-efficient option, and Case D (Section 6.5) quantifies its advantage over classical DoE.

## 3.5 AI in bioreactor scale-up and scale-down

**Why scale-up is hard.** As shown by Eq. (8), the Damköhler number in a large fed-batch tank can exceed 10. Substrate is then high near the feed point and depleted elsewhere, while dissolved oxygen is high near the sparger and low at the top (Nadal-Rey et al., 2021a). A single cell travels through these zones many times per hour, like a swimmer passing repeatedly between warm and cold currents. *E. coli* and yeast respond by switching metabolic pathways, forming by-products such as acetate or ethanol, and losing yield.

**Classical criteria and their limits.** Engineers traditionally scale up by holding one quantity constant: P/V, $k_La$, tip speed or mixing time (Eq. 7). These cannot all be held constant at once. Each criterion protects one aspect of the environment while ignoring the others, which is why scale-up remains partly empirical (Wang et al., 2020).

**Compartment models and cell lifelines.** Two modelling advances underpin the AI approaches. *Compartment models* divide the tank into well-mixed zones connected by flows derived from CFD, giving a spatially resolved yet fast description of the vessel (Nadal-Rey et al., 2021b). *Lifeline* analysis goes a step further: it tracks thousands of virtual cells through a flow field and records the sequence of concentrations that each one experiences. Haringa et al. (2018) coupled such lifelines to a nine-pool metabolic model of *P. chrysogenum* to design rational scale-down experiments, and Wang et al. (2020) generalised the idea into a computational framework built on CFD, lifelines and cell models.

**Where AI enters.** These simulations are powerful but slow, which confines them to a handful of what-if cases. AI removes this bottleneck in three complementary ways. Neural-operator and graph-network *surrogates* predict concentration fields for new designs in seconds. Hybrid models can *learn the cell response* to fluctuating rather than constant conditions. And *transfer learning* adapts a small-scale model to production scale using only a handful of plant runs (Bayer et al., 2021; Hutter et al., 2021; Rogers et al., 2023).

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

**Emerging directions.** Foundation and language models (Section 2.9) could be fine-tuned for a new product with very little data. Reviews by Mondal et al. (2023) and Cheng et al. (2023), and a special-issue editorial by Khanal et al. (2023), also highlight AI in non-pharmaceutical bioprocesses, such as biofuels, wastewater and food fermentation, where cost constraints are tighter and sensors cheaper.

## 3.7 Synthesis of research gaps

Across the four areas, the review identifies six recurring gaps:

1. **Fragmentation.** Separate models are built for design, monitoring, control and scale-up, so knowledge is lost at each hand-over.
2. **Data scarcity at large scale.** Models are rarely validated above pilot scale.
3. **Single-criterion scale-up.** Few works capture the full environment that cells experience.
4. **Weak uncertainty quantification.** Predictions are often reported without calibrated intervals.
5. **Model lifecycle management.** Drift detection, retraining and version control are seldom addressed.
6. **Quantitative evidence.** Few studies compare model classes head-to-head under controlled interpolation and extrapolation conditions with statistical tests.

These gaps motivate the framework of Section 4 and the case studies of Sections 5–6.

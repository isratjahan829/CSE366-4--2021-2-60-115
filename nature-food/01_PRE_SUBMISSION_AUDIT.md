# Pre-submission audit — FAO/FPMU Consumer Awareness Survey (ToR-15) → Nature Food

**Source documents (read in full):**
1. *Survey on Consumer Awareness of Nutrition, Food Safety and Hygiene — Final Report* (LoA-1507316), BRAC JPGSPH and BAU, 28 Feb 2021 (71 pp.)
2. *Questionnaire_FAO_ToR_15_Draft_final* (English, 48 pp.)
3. *Questionnaire_FAO_project_DRAFT_Bangla_27.9.2020_Final2_2* (Bangla field version, 58 pp.)

**Bottom line.** The dataset is strong: nationally representative, 5,000 households, four respondent types, objective anthropometry and observed WASH. The **final report, however, cannot go to Nature Food as written.** It is a descriptive knowledge, attitudes and practices (KAP) baseline, and it has several method problems that reviewers will find. A Nature Food paper needs (i) one sharp food-systems question, (ii) re-analysis of the **raw microdata**, and (iii) the fixes listed below. Sections A–D list the blockers. Section E gives the recommended paper angle and Section F the analysis plan.

---

## A. Blockers to resolve BEFORE writing further (non-negotiable)

| # | Issue | Why it matters | Action |
|---|---|---|---|
| A1 | **Data ownership and permission.** The survey was commissioned by FAO (MUCH project) and FPMU, Ministry of Food, and funded by the EU and USAID under an LoA. | Publishing without written clearance can cause retraction or contract breach. | Get written permission from FAO-MUCH and FPMU, plus the funder acknowledgement wording. |
| A2 | **Authorship.** The PI is Dr Nepal C. Dey (JPGSPH); the co-investigators are Awal, Shamim, Mridha and Afsana. | Nature Portfolio requires all contributors to agree, and the Inclusion & Ethics statement needs local investigators. | Agree an author list and CRediT roles with the PI and team first. |
| A3 | **Ethics approval number.** The report gives no IRB number or date. | Mandatory for Nature Food. | Get the JPGSPH IRB approval (reference no. and date). |
| A4 | **Adolescent consent and assent.** 2,473 respondents aged 10–19 were interviewed. The consent form is adult-only and has no assent or parental consent text. | Reviewers and editors will ask for it, and it is an ethics red flag. | Document the parental consent and adolescent assent procedure that was actually used. If none was used, disclose that and consult the IRB. |
| A5 | **Raw microdata (KoboCollect export) and sampling weights.** | Every Nature Food-level analysis below needs record-level data; the report tables are not enough. | Get the full dataset, codebook, weights and PSU/strata IDs. |

---

## B. Methodological flaws in the report (reviewers WILL catch these)

1. **The "one-third have low knowledge" headline is circular.** Knowledge was cut into *sample tertiles* (Methods 3.6.2), so by construction about 33% of people are "low", 33% "medium" and 33% "high". Headlines such as "one-third had low basic nutrition knowledge" are therefore an artefact of the cut-off, not a finding.
   - **Fix:** report the **continuous score (0–100% correct)**. Add an **absolute, criterion-based cut-off** (for example ≥60% of the maximum score, or the FAO KAP guideline approach), and treat tertiles only as a sensitivity analysis.
2. **The wrong model for an ordered outcome.** Multinomial logit was used for low < medium < high.
   - **Fix:** use design-based **linear regression on the continuous score** (primary model) and an **ordinal logit** (proportional odds, with a Brant test) as a secondary model.
3. **Quasi-complete separation.** The tables contain impossible odds ratios (ORs): 2.05e−06 and 3.77e−08 (Table 20 and "Table 19"), 5.04e+06 (A21), 1.64e+07 (A23), 0.0052 for obese adolescents (A22) and 585,939 (cooks, "very bad health").
   - **Fix:** collapse sparse categories (merge "bad" with "very bad", and HSC+ with secondary in the slum sample), or use **Firth penalised regression**. Report 95% confidence intervals (CIs), not SEs.
4. **Significance at p<0.1 (`*`).** This is not acceptable for Nature Food.
   - **Fix:** use two-sided α = 0.05, report exact P values and CIs, and correct for multiple comparisons (Benjamini–Hochberg) across the many item-level tests.
5. **Variable selection by bivariate significance** (Methods 3.5.1).
   - **Fix:** pre-specify covariates using a directed acyclic graph (DAG), and avoid data-driven selection.
6. **Adolescent BMI uses adult cut-offs.** For example, 12.6% of adolescents are classed "obese", and 96.1% of the obese have low knowledge.
   - **Fix:** use **WHO 2007 BMI-for-age z-scores** (thinness < −2 SD, overweight > +1 SD, obesity > +2 SD).
7. **BMI distribution in slum adults looks implausible (Table A8).** It shows 59.5% underweight and 11.3% normal among purchasers, and 55.4% underweight and 9.5% normal among cooks. The rural-urban pattern is the reverse (62.7% normal). This is probably a column swap or a coding error.
   - **Fix:** recompute BMI from the raw height and weight, and flag biologically implausible values.
8. **Same person counted as two respondent types.** N = 5,000 purchasers *and* 5,000 cooks from 5,000 households, so one person can be both.
   - **Fix:** report the overlap. Analyse at person level with household clustering, or in separate role-specific samples with this stated clearly.
9. **Sex comparisons among cooks are meaningless.** Only 0.7% of cooks are male (n ≈ 35), so findings like "low knowledge more prevalent among male cooks" should be dropped.
10. **Self-reported handwashing is near-ceiling.** For example, 96.5% say they wash with soap before eating, which reflects social desirability during COVID-19 messaging. The questionnaire also has **observed** data that the report never used: the handwashing place, soap or ash present within one minute, water present, the device, toilet cleanliness and kitchen pollution sources. **Use these for triangulation.**
11. **The "bias-correction matrix" is self-reported recall plus intention.** It does not correct bias. Rename it "self-reported COVID-19-attributed adoption" and discuss recall and intention-behaviour bias.
12. **The "unfavourable nutritional behaviour" definition is too broad.** It counts any savoury snack, sweet or sugar-sweetened beverage (SSB) in the last 24 hours *or any "other beverage" in the last 7 days*; tea with sugar alone could qualify. The "favourable" definition (any staple *or* any vegetable) is nearly universal.
   - **Fix:** use validated indicators: **MDD-W** (≥5 of 10 groups, women 15–49), **DQQ/GDR-style** NCD-protect and NCD-risk scores, and separate SSB, sweets and salty-fried snack indicators.
13. **Design-based inference.** The report says weights were used, but there is no evidence that standard errors account for clustering and stratification.
   - **Fix:** use `svyset psu [pw=wt], strata(division×urban/rural/slum)` in Stata, or `survey::svydesign` in R.

---

## C. Internal inconsistencies to correct (typos, numbers)

| Location | Inconsistency |
|---|---|
| Exec. summary vs Results vs Conclusion | Low food-safety knowledge is given as "one-fifth" (summary), 35%/32% (Results, Fig. 1) and "one-third" (Conclusion). |
| Exec. summary | Says adolescents "living in urban areas" had the lowest knowledge. Results say **slum** adolescents (52.0% low). |
| Table 6 | The Overall FC/PC/IC rows do not sum to 100. For example, Q5 reads 0.0 / 95.9 / 95.9 (the IC column copies PC), and the same happens in Q6–Q12. |
| Table 7 | The layout is broken, so values cannot be traced to cells. |
| Table 1 | Dhaka has 30 rural PSUs × 20 = 600, but 589 are reported. The total of 3,989 vs 4,000 is footnoted; keep that footnote. |
| §3.1.3 | "73% rural / 27% urban (slum and non-slum)" conflicts with Table 5 (58.4% rural, 21.4% urban, 20.2% slum). |
| §3.1.2 | The MICS sample-size formula is referenced, but the equation is missing. |
| Table numbering | There are two "Table 19"s; Table 17 is cited as A17; Tables A20–A21 are cited but A20 is missing. |
| Sections | 3.6.3 is missing; there are "Chapter 4" and then "Chapter 9". |
| References | Placeholder "(Add Ref)" in the Introduction. The report is dated 28 Feb 2021 but cites a source "accessed 7 March 2021". |
| Table 13 text | "female cooks compared to boys" |
| Table 14 | Rajshahi 19.2 / 71.8 / 9.0 is an outlier; verify it. |
| Slum Table A13 text | "Comparatively a higher proportion of cooks from lowest wealth quintile (65.1%)" — the table shows 60.47%. |

---

## D. Data the report collected but never analysed (this is where the novelty is)

| Module | Variable | Use in paper |
|---|---|---|
| A3 | **Household Food Insecurity Access Scale (9 HFIAS items, 4-week recall) + 6 coping strategies** | Key moderator: does knowledge translate into diets *when households are food-insecure*? |
| A2.3 | Monthly income, **food expenditure**, health expenditure, remittances, safety-net programmes | Food budget share; affordability |
| A2.3 | **Internet access and device** | Digital information channel |
| A3 | Drinking-water source, **water treatment**, time to collect water, toilet type and sharing | WASH and food-safety environment |
| A3 (observed) | **Handwashing station, soap/ash present, water present, device** | Objective hygiene practice (vs self-report) |
| B1 (observed) | **Food stored near chemicals**, kitchen pollution sources, overall cleanliness | Observed food-safety practice |
| B1 | Iodised salt, packaged vs open oil, **vitamin A/D fortified oil (observed packet)**, salt g/week | Fortification reach in the market |
| B2/D3/E2 | **Media exposure, trusted sources, food advertising noticed, labelling use** | Food environment and marketing |
| D2 | **Label reading, purchase drivers (price, taste, safety, brand)** | Consumer food choices |
| E1/E3 | Adolescent self-purchase, **home food environment (rules, availability of snacks/SSB)** | Adolescent food environment |
| B4/D5/E5 | **24-h and 7-day food frequency (24 groups)** | MDD-W, NCD-risk score, ultra-processed or discretionary foods |
| B5/D6/E6 | Measured height, weight, **MUAC**; EQ-VAS (0–100); MEHM (3 items); *pre-COVID weight change* | Nutritional status, health |
| C1–C4 | IYCF practices, **child faeces disposal**, child MUAC, MNP use | Child diet and hygiene |

---

## E. Recommended paper angle for Nature Food

### Primary (recommended): "Knowing is not eating"
**Question:** In a nationally representative LMIC population, is household food and nutrition knowledge linked to what people actually eat? And does food insecurity or poverty cut that link?

Why it fits Nature Food: it is about food environments and the limits of information-only nutrition policy in a country going through the nutrition transition (double burden). Behaviour change communication (BCC) is the main nutrition-sensitive strategy in Bangladesh. The report's own tables already hint at a **weak knowledge–diet gradient**:
- Food purchasers: high-knowledge share is 35.9% (adequate diet diversity) vs 30.8% (inadequate diet).
- Cooks: high knowledge is *more* common among those with **unfavourable** snack/SSB behaviour (38.7%) than favourable (27.5%).
- Cooks' food-safety knowledge is not associated with diet diversity (P = 0.549).
- 78–86% of adolescents consumed discretionary foods in the last 24 hours, while knowledge showed a 41-fold education gradient.

**Working title (≤90 characters):**
*"Nutrition knowledge is socially patterned but weakly linked to diets across Bangladesh"*

### Secondary (Extended Data / companion paper): the household food-safety "last mile"
Food-safety knowledge–practice gaps: reheating, storage near chemicals, vegetable washing, observed vs reported handwashing, and COVID-19-attributed adoption.

### Fallback journals if Nature Food desk-rejects
*Global Food Security* · *Food Policy* · *BMJ Global Health* · *Public Health Nutrition* · *Food Security* · *Maternal & Child Nutrition* (IYCF module) · *npj Science of Food*.

---

## F. Analysis plan (pre-specified; run on raw data)

**Design:** `svyset psu [pweight=wt], strata(stratum)`, with strata = division × {rural, urban, slum}. Use person-level analysis by respondent role.

**Exposures**
- *Knowledge scores* (continuous, % of maximum; FAO KAP scoring with fractional credit), by domain:
  - Basic nutrition: B1 items 1–25, D1, E1
  - Cooking practice: B1 items 26–39
  - Food safety and hygiene: B1 items 51–70 plus A3 items 13–18
  - IYCF: C1
- Internal consistency (Cronbach's α / KR-20) and a confirmatory factor analysis (CFA) or item response theory (IRT) check, reported in the SI.

**Outcomes**
1. **MDD-W** (women 15–49, cooks) and a 10-group dietary diversity score (DDS) for all adults, from the 24-h recall (B4/D5).
2. **Discretionary-food score**: salty/fried snacks + sweets + SSB in the last 24 h (count 0–3), plus the 7-day frequency.
3. Adolescents: DDS, discretionary score, BMI-for-age z-score.
4. Child (6–23 mo): MDD (≥5/8), minimum meal frequency (MMF), minimum acceptable diet (MAD).
5. Food safety: **observed** practice index (soap+water at station, food not stored near chemicals, clean kitchen surroundings).

**Moderators**
- HFIAS category (secure / mild / moderate / severe; Coates et al. 2007)
- Wealth quintile (PCA, as in the report)
- Residence (rural / urban / slum)

**Models**
- **M1 (social patterning):** knowledge ~ education + wealth + age + sex (where it varies) + residence + division + internet + media exposure (design-based linear). Present average marginal effects (AMEs).
- **M2 (knowledge → diet):** diet outcome ~ knowledge (per 10 pp) + M1 covariates + HFIAS + food expenditure per adult equivalent (Poisson/negative binomial for counts; logistic for MDD-W).
- **M3 (moderation):** M2 + knowledge × HFIAS and knowledge × wealth. Plot predicted outcomes.
- **M4 (knowledge–practice gap, food safety):** observed practice ~ food-safety knowledge + WASH infrastructure (water source, time to water) + wealth.
- **Decomposition:** Oaxaca–Blinder or Shapley decomposition of the rich–poor gap in diet diversity into knowledge, affordability (HFIAS, food expenditure) and access (residence, market) components. This gives the "headline number": *the share of the diet gap explained by knowledge vs. affordability*.

**Robustness**
- Tertile and absolute cut-offs; complete-case vs multiple imputation (MICE, m = 20)
- Excluding slum, and analysing the slum sample separately
- Excluding cooks who are also purchasers
- Firth logit for sparse cells
- E-values for unmeasured confounding
- Leave-one-division-out analysis

**Software:** R 4.x (`survey`, `srvyr`, `marginaleffects`, `logistf`, `oaxaca`, `mice`) or Stata 18 (`svy`, `margins`, `firthlogit`, `oaxaca`).

### R skeleton
```r
library(survey); library(srvyr); library(marginaleffects)
d <- readRDS("hh_person_level.rds")
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wt, data = d, nest = TRUE)
# M2/M3: knowledge -> MDD-W, moderated by HFIAS
m3 <- svyglm(mddw ~ know_nut10 * hfias_cat + edu + wealth_q + age + residence +
               division + internet + food_exp_ae,
             design = subset(des, role == "cook" & age %in% 15:49),
             family = quasibinomial())
avg_slopes(m3, variables = "know_nut10", by = "hfias_cat")
```

---

## G. Items to request from the PI team
1. Raw KoboCollect data (all modules), codebook, weights and PSU/strata IDs
2. The scoring key used for each knowledge item, and the "fully/partially correct" definitions
3. IRB approval letter; consent and assent procedure for adolescents
4. GPS or cluster list (Annexures 1–2; not included in the PDF) for the Fig. 1 map
5. FAO/FPMU written permission and the funder acknowledgement text
6. Any pilot or pre-test report (questionnaire validation)

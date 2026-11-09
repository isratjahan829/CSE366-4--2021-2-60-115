# Nature Food — Manuscript Format & High-Impact Writing Guide

Reference guide for preparing a field-data research paper for **Nature Food**
(https://www.nature.com/natfood/). Use it as the checklist when the draft arrives.

> **Verification note (Nov 2026):** nature.com could not be reached from the
> working environment, so these numbers come from the Nature Portfolio author
> guidelines as known, plus a web search. Before submission, check every limit
> marked ⚠ against the live pages:
> - Content types: https://www.nature.com/natfood/content
> - Submission guidelines: https://www.nature.com/natfood/submission-guidelines
> - Reporting standards: https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards

---

## 1. Scope: is this paper a fit?

Nature Food publishes research on **food systems as a whole**: production,
processing, distribution, consumption, nutrition, food security, food safety,
waste, environment and climate links, livelihoods and policy. It covers natural,
social and health sciences.

The editors look for:
- **A finding that matters beyond one place.** Local field data has to answer a
  question that people in other regions or at global level care about.
- **A food-systems angle.** It should touch more than one link in the chain, or
  have trade-offs such as yield vs. environment or income vs. nutrition.
- **Policy relevance.** The paper should give clear implications for
  governments, donors or industry, and ideally link to the SDGs (2, 3, 12, 13).
- **Interdisciplinary, rigorous methods.** Agronomy plus economics, nutrition
  plus remote sensing, survey plus modelling, and so on.

Papers that are *only* descriptive surveys, or only technical (for example,
one food-chemistry assay), usually get desk-rejected and sent to sister
journals (Nature Communications, npj Science of Food, Communications Earth &
Environment).

---

## 2. Content types (main ones) ⚠

| Type | Main text | Abstract | Display items | References |
|---|---|---|---|---|
| **Article** (primary research) | ~3,000–4,000 words (intro + results + discussion) | ≤150 words, no references | ≤6 (figures + tables) | ~50 in main text; Methods refs extra |
| **Analysis** (new analysis of existing/compiled data) | as Article | ≤150 words | ≤6 | ~50 |
| **Brief Communication** | ~1,500 words | ~70 words | ≤3 | ~20 |

The following do **not** count toward the main-text words: Methods (aim for
~3,000 words or fewer), references, figure legends, Data/Code availability and
the end statements.

Main-text **figure legends**: ≤350 words each.
**Extended Data**: up to 10 items (figures/tables) with legends; these are
peer-reviewed and published with the paper.
**Supplementary Information**: one combined PDF (methods detail, extra tables,
questionnaire, robustness checks).

**Recommendation for field data:** use an **Article**. Use an **Analysis** if the
core is secondary or compiled datasets.

---

## 3. Manuscript structure (in this order)

1. **Title.** ≤~90 characters ⚠. Make it a statement of the finding, not a
   description of the study. Avoid acronyms, place names in the lead (if
   possible), and question titles.
   - Weak: "A survey of rice farmers' adaptation practices in coastal Bangladesh"
   - Strong: "Salinity-tolerant rice adoption halves climate-driven yield losses
     for coastal smallholders"
2. **Author list and affiliations.** Include the corresponding author email and
   ORCID. Under the Inclusion & Ethics policy, local researchers should be
   credited as authors.
3. **Abstract (≤150 words, unreferenced).** Use this sequence:
   - 1–2 sentences: the global problem and why it matters
   - 1 sentence: the gap ("However, … remains unknown")
   - 1 sentence: what you did, with the scale (n households, regions, years)
   - 2–3 sentences: key results **with numbers** (effect size, %)
   - 1 sentence: implication for policy or the food system
4. **Main text.**
   - **Introduction** (no heading, ~500–700 words). Go from the global context
     to the specific gap, then the study design, then a one-paragraph preview of
     the main findings. End by stating what is new.
   - **Results** with informative subheadings. Each subheading states a finding
     (≤~60 characters ⚠), for example "Diversified farms show higher dietary
     diversity". Refer to Fig. 1–6 and Extended Data in sequence. Put the
     statistics inline: estimate, 95% CI, n, test, *P*.
   - **Discussion** (no subheadings or few, ~800–1,000 words). Cover the main
     message, comparison with the literature, mechanisms, generalisability,
     **limitations** (be candid), and policy implications. Close with a strong
     final sentence.
5. **Methods** (after the main text, with subheadings). Cover the study area,
   sampling frame and design (power and sample size), questionnaire and
   instruments, measurements (yield, dietary diversity such as HDDS/FIES/MDD-W,
   income), data cleaning, statistical models (equations), causal
   identification, robustness checks, software, ethics approval and informed
   consent.
6. **Data availability** (mandatory). Deposit anonymised data in a public
   repository (Zenodo, Figshare, Dryad, Harvard Dataverse) and give a DOI. If
   access must be restricted, say why and how to get access.
7. **Code availability.** Give a GitHub or Zenodo DOI for the analysis scripts.
8. **References.** Nature style, numbered in order of appearance, in
   superscript. Main-text refs come first, then Methods refs continue the
   numbering. List up to 5 authors, then "et al.".
   Format: `Author, A. B., Author, C. D. & Author, E. F. Title of article. *Nat. Food* **7**, 66–73 (2026).`
9. **Acknowledgements and funding.** Thank enumerators, farmers and
   communities.
10. **Author contributions** (CRediT style initials).
11. **Competing interests** (mandatory statement, even if none).
12. **Inclusion & Ethics statement.** Required for research done in a country
    other than the authors' own, or in low- and middle-income countries
    (LMICs). Say how local researchers were involved, local ethics approval,
    and how benefit is shared.
13. **Figure legends.** Start each with a bold title sentence, then describe the
    panels, n, error-bar definition and statistical test.
14. **Extended Data figure and table legends.**
15. **Tables**, if any, in editable format.

---

## 4. Figures: what high-impact Nature Food figures look like

- **Fig. 1 = the whole story at a glance.** Use a study-area map plus a
  conceptual framework, or a headline result. Editors and reviewers judge the
  paper on Fig. 1.
- Use multi-panel figures (a, b, c…, bold lowercase) that are dense but clean.
- Show distributions, not only bars: violin, box with points, raincloud,
  coefficient (forest) plots for regression effects, maps with spatial
  heterogeneity, and Sankey diagrams for food flows.
- Use colour-blind-safe palettes. Do not use red and green together.
- Sizes: single column 89 mm, double column 183 mm. Use Arial or Helvetica at
  5–7 pt in the final size. Supply vector files (PDF/EPS/AI) or ≥300 dpi
  images.
- Error bars must be defined (s.d., s.e.m. or 95% CI) and n must be stated.
- No image manipulation. Keep the raw data for every plot (source data files
  are encouraged).

---

## 5. Statistics and reporting (reviewers check these closely)

- Submit the **Nature Portfolio Reporting Summary** with the paper. For survey
  or field data, use the *Behavioural & social sciences* or *Ecological,
  evolutionary & environmental sciences* sections as relevant.
- For every test, report the exact n (unit: household, plot, person), the test
  name, whether it is one- or two-sided, the exact *P*, the effect size and the
  95% CI.
- Correct for multiple comparisons where relevant, and check model assumptions.
- For **survey data**, describe the sampling strategy (multistage, stratified,
  cluster), use survey weights, cluster standard errors at the village or
  union level, and report non-response and attrition.
- **Causal claims** need identification: panel FE, DiD, IV, PSM or IPW, RDD or
  RCT. Without it, write "associated with", not "caused". Nature Food
  reviewers often ask for this.
- Robustness: alternative specifications, placebo tests, sensitivity to
  outliers, heterogeneity (gender, farm size, region).
- Report sex and gender (SAGER guidelines): give sex-disaggregated results if
  you have them.
- Ethics: give the IRB or ethics committee name and approval number, informed
  consent (verbal or written), compensation and anonymisation.

---

## 6. What makes Nature Food papers high-impact (patterns)

1. **A headline number.** For example, "small-scale farms supply ~1/3 of food
   consumed in high-income nations" (Taherzadeh et al., *Nat. Food* 7, 66–73,
   2026). Find the one quantified message of your paper.
2. **Scale-up or extrapolation.** Link local field results to national or
   global stakes. Use scenarios, upscaling with census or remote-sensing data,
   or "if adopted by X% of farmers…" estimates.
3. **Trade-offs and synergies.** Look at yield, nutrition, income, emissions,
   water and gender equity together.
4. **Climate-food nexus.** Climate shocks, adaptation, resilience and
   salinity or flooding are highly topical, and Bangladesh field data is very
   relevant here.
5. **Nutrition outcomes, not only production.** Dietary diversity, the
   affordability of healthy diets, micronutrients.
6. **Equity lens.** Women, the landless, tenants, the poorest quintile.
7. **Mixed methods.** Quantitative survey plus qualitative (FGD/KII) gives a
   mechanism story. Put the qualitative quotes in the SI.
8. **Policy translation.** Close the Discussion with concrete, targeted
   recommendations, and avoid generic ones.

---

## 7. Writing style (Nature house style)

- Write for a broad, non-specialist audience. Explain jargon or avoid it.
- Use active voice ("We surveyed 1,200 households"), short sentences and one
  idea per paragraph.
- Use British or American spelling, but use it consistently. Nature uses
  "per cent" as "%", SI units, and a space between number and unit (5 kg).
- Write numbers from one to nine in words, except with units.
- Define abbreviations at first use and keep them to a minimum.
- No "novel" or "for the first time" claims without evidence. Show novelty
  through the gap statement instead.
- Avoid overclaiming. Scope the conclusions to the sample and the design.

---

## 8. Cover letter (critical for passing the desk)

One page, addressed to the Chief Editor or Handling Editor:
1. Title and type (Article).
2. **The question and why it matters to Nature Food's broad readership.**
3. Key findings: 2–3 bullet points with numbers.
4. What is new compared with the existing literature (cite 2–3 key papers,
   including recent Nature Food ones).
5. Policy or real-world significance.
6. Confirm the paper is not under consideration elsewhere, and declare any
   preprint (preprints are allowed).
7. Suggested reviewers (3–5, with no conflicts) and any excluded reviewers.

Option: send a **presubmission enquiry** (abstract plus cover letter) first to
check editor interest.

---

## 9. Submission checklist

- [ ] Article type chosen, and word count within limits ⚠
- [ ] Abstract ≤150 words, unreferenced, with numbers
- [ ] Title ≤~90 characters and states the finding
- [ ] ≤6 main display items, ≤10 Extended Data items
- [ ] Figure legends ≤350 words, each with n, error bars and test
- [ ] Methods complete: sampling, instruments, models, ethics
- [ ] Data availability with repository DOI (anonymised)
- [ ] Code availability
- [ ] Reporting Summary completed
- [ ] Inclusion & Ethics in Global Research statement
- [ ] Ethics approval number and informed consent stated
- [ ] Competing interests and author contributions
- [ ] References in Nature style; main text ~≤50
- [ ] Supplementary Information as one PDF (questionnaire included)
- [ ] Cover letter and suggested reviewers
- [ ] Editable source files (Word or LaTeX) ready for revision. The first
      submission can be a single PDF ("format-free" initial submission is
      generally accepted in Nature Portfolio journals ⚠)

---

## Sources
- Nature Food homepage: https://www.nature.com/natfood/
- Taherzadeh, O., Cai, Z. & Mogollón, J. M. *Nat. Food* **7**, 66–73 (2026),
  doi:10.1038/s43016-025-01276-y (example of headline-number framing)
- Nature Portfolio formatting guide (general): https://www.nature.com/nature/authors/gta
- Manusights Nature Food submission guide: https://manusights.com/blog/nature-food-submission-guide

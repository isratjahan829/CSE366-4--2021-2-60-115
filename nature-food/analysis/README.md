# Analysis (microdata)

Run with Python 3 (pandas, pyreadstat, statsmodels, matplotlib). The raw survey data are **not** in this
repository (FAO/FPMU-owned; download the Stata files separately). Identifiers (names, phone numbers,
addresses) are dropped on load.

1. `01_construct.py <data_dir>` — builds knowledge scores, diets, HFIAS, wealth index, WASH and anthropometry
   (expects `clean.pkl` = raw .dta without identifiers, with `S_weight` merged from the original file).
2. `02_validate.py <data_dir>` — compares weighted estimates with the published survey report.
3. `03_models.py <data_dir> microdata_results.json` — survey-weighted models (PSU-clustered SEs).
4. `05_figures_v2.py <data_dir>` — manuscript figures (editorial style, Inter font; bootstrap bands cached in <data_dir>).
5. `06_ml_feasibility.py <data_dir> ml_feasibility_results.json` — feasibility of a short household risk-screening model (PSU-grouped cross-validation).

`microdata_results.json` holds aggregate results only. `legacy_report_tables/` contains the earlier
analysis based on the report's published tables (superseded), plus the first figure script.

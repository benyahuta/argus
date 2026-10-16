# Forecasting Medicare Fraud Exclusions from Public Billing Data

[![ci](https://github.com/benyahuta/medicare-fraud-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/benyahuta/medicare-fraud-forecasting/actions/workflows/ci.yml)
**[Interactive results →](https://benyahuta.github.io/medicare-fraud-forecasting/)** · [Model card](MODEL_CARD.md) · [Full results tables](reports/RESULTS.md)

Improper payments in Medicare fee-for-service run to tens of billions of dollars a year, while the HHS Office
of Inspector General (OIG) can investigate only a sliver of the ~1.2 million clinicians who bill it. This project
asks a concrete question: **using only data CMS already publishes, can we tell investigators which providers to
look at first, and who would those leads unfairly land on?**

The model reads one year of a provider's public Medicare billing (Part B services and Part D prescribing) and ranks
how likely they are to be **excluded by OIG for fraud, abuse or misconduct within the next three years**. Everything is
evaluated strictly forward in time: the model is trained on exclusions through 2021 and tested on what OIG actually did in
2022–2024.

<p align="center"><img src="reports/figures/recall_at_budget.png" width="720"></p>

## Headline results (2021 billing → exclusions in 2022–24)

{{HEADLINE}}

## What's in here

| | |
|---|---|
| **Data engineering** | ~25 GB of raw CMS files → 4.5M labelled provider-years, ~70M billing lines (Polars, Parquet cache) |
| **Feature engineering** | ~110 features in 8 groups, all computed *within year* against specialty peers so they survive coding changes such as the 2021 E/M overhaul |
| **Models** | peer-payment rule, Isolation Forest, logistic regression, tuned LightGBM (5-seed bag), a **ClaimSet Transformer** over raw billing lines (PyTorch / Apple MPS), and a hybrid using its self-supervised embeddings |
| **Evaluation** | forward-in-time split, stratified-bootstrap CIs, paired significance tests, audit-budget metrics, dollar impact, lead time, rolling-origin test years 2019–2022 |
| **Responsible ML** | fairness audit of false positives by patient population and rurality, a stratified-allocation mitigation with its measured cost, a model card, no per-provider scores published |
| **Engineering** | typed package (`src/fraudml`), unit tests on synthetic data, ruff, GitHub Actions CI, one-command reproduction |

## Data

| Source | What it gives | Rows |
|---|---|---|
| [CMS Medicare Physician & Other Practitioners — by Provider](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider) | volume, payments, patient demographics & chronic conditions | ~1.1M / year |
| [… — by Provider and Service](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service) | every (provider, HCPCS code, place of service) line | ~10M / year |
| [Medicare Part D Prescribers — by Provider](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider) | prescribing incl. opioids, brand share, cost | ~1.1M / year |
| [OIG List of Excluded Individuals/Entities (LEIE)](https://oig.hhs.gov/exclusions/exclusions_list.asp) | who was excluded, when, under which authority | 84k |

**Label.** A provider-year is positive if the NPI appears in the LEIE under a fraud, abuse or misconduct authority
(§1128(a)(1)–(4), §1128(b)(1)–(8), §1128A) with an exclusion date in the three calendar years after the
feature year. Providers already excluded are dropped. See the [model card](MODEL_CARD.md) for why
*exclusion ≠ fraud*.

**Split.**

| Cohort | Feature year | Label window | Providers | Positives |
|---|---|---|---|---|
{{SPLITS}}

All training labels close before the first test label (2022), so the final model can be refit on train + valid
without seeing the future.

## Features

All peer comparisons are within *(year, specialty)*, using robust z-scores (median / MAD) or percentiles.

| Group | Examples | Why |
|---|---|---|
| volume | payment per patient, services per patient, charge ÷ allowed markup | raw billing intensity |
| bene | % dual-eligible, % under 65, risk score, chronic-condition prevalence | case-mix context |
| peer | each volume metric as a z-score vs. same-specialty peers | "unusual for a cardiologist", not "large" |
| service | per-code services-per-patient vs. specialty median (payment-weighted), rare-code share, payment HHI | billing many more units of a code than peers |
| em | share of office visits billed at level 4–5, mean E/M level, both vs. peers | classic **upcoding** signal |
| mismatch | out-of-fold surprise −log p(declared specialty ∣ code mix) from a TF-IDF→SVD→multinomial model | a "dermatologist" billing like a lab |
| rx | opioid share, long-acting opioids, opioid days per patient, brand share, cost per claim (Part D) | pill-mill and kickback patterns |
| meta | specialty, organization vs. individual, rurality | |

<p align="center"><img src="reports/figures/em_upcoding.png" width="620"></p>

## Results

<p align="center"><img src="reports/figures/pr_curves.png" width="560"></p>

{{RESULTS_BODY}}

### What drives the score

<p align="center"><img src="reports/figures/ablation.png" width="620"><br><img src="reports/figures/shap_summary.png" width="640"></p>

{{ABLATION_TEXT}}

An anonymized example (values rounded so the public files can't be searched for an exact match):

<p align="center"><img src="reports/figures/case_study.png" width="700"></p>

### Early warning

<p align="center"><img src="reports/figures/lead_time.png" width="600"></p>

{{LEAD_TEXT}}

### Does it hold up across years?

<p align="center"><img src="reports/figures/rolling.png" width="600"></p>

{{ROLLING_TEXT}}

### Can a neural network learn this from raw billing lines?

The **ClaimSet Transformer** (`src/fraudml/deep.py`) treats each provider-year as an unordered set of up to 32 billing
lines: HCPCS embedding, place of service, and 9 numeric features such as volume, price, and intensity vs. peers. A
transformer encoder with attention pooling feeds two heads. One predicts exclusion; an auxiliary head predicts the
provider's specialty from the code mix alone, which gives the encoder ~1M labels per year instead of a few hundred
positives.

{{DEEP_TEXT}}

## Who gets wrongly flagged? (fairness audit)

A model that steers audits toward providers *because of who their patients are* would harm exactly the clinics that
low-income and disabled Medicare beneficiaries depend on. So we measure, at a 1% audit budget, each group's
false-positive rate relative to the overall rate.

<p align="center"><img src="reports/figures/fairness.png" width="680"></p>

{{FAIRNESS_TEXT}}

<p align="center"><img src="reports/figures/fairness_tradeoff.png" width="560"></p>

{{TRADEOFF_TEXT}}

## Limitations

* **Labels are enforcement outcomes, not ground truth.** Undetected fraud counts as a negative, and any bias in whom OIG
  pursues is learned. Exclusions under §1128(b)(4), license actions, include misconduct unrelated to billing.
* **Small positive counts** (~100–320 per cohort) make metrics noisy; that is why every number has a bootstrap CI and
  hyper-parameter choice on the 105-positive validation cohort is itself noisy.
* **Aggregate public data only**: no claims, diagnoses or patient-level detail, and CMS suppresses small cells.
* **Not a decision system.** See the [model card](MODEL_CARD.md). This repository intentionally publishes no
  per-provider scores.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/), ~30 GB disk and ~16 GB RAM. On macOS, LightGBM needs `brew install libomp`.

```bash
uv sync --extra deep   # include PyTorch for the neural models
make data          # download CMS + OIG files (~25 GB)
make features      # build data/interim/dataset.parquet (~10 min)
make experiments   # baselines, tuned LightGBM, ablations, fairness (~3 h on an M-series laptop)
uv run python scripts/run_rolling.py   # rolling-origin evaluation
uv run python scripts/run_deep.py      # ClaimSet Transformer, end to end (uses MPS/CUDA if available)
uv run python scripts/run_hybrid.py    # self-supervised embeddings → LightGBM
make figures && uv run python scripts/render_tables.py && uv run python scripts/export_site.py
make test
```

## Repository layout

```
src/fraudml/
  config.py      cohorts, label windows, exclusion authorities
  data.py        CMS / OIG loaders → typed Parquet
  features.py    feature groups (volume, peer, service, E/M, mismatch, Part D, …)
  labels.py      forward-looking LEIE labels
  models.py      baselines, LightGBM tuning and seed bagging
  deep.py        ClaimSet Transformer (PyTorch)
  evaluate.py    audit-budget metrics, stratified & paired bootstrap
scripts/         build_dataset · run_experiments · run_rolling · run_deep · run_hybrid · make_figures · export_site
reports/         RESULTS.md, results JSON, figures
docs/            interactive results page (GitHub Pages)
tests/           unit tests on synthetic data
```

## License & data use

Code: MIT. Data: public-use files from CMS and HHS-OIG, which are not redistributed here (download with `make data`).
This project is an independent research exercise and is not affiliated with CMS or OIG.

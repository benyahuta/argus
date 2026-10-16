# Forecasting Medicare Fraud Exclusions from Public Billing Data

[![ci](https://github.com/benyahuta/medicare-fraud-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/benyahuta/medicare-fraud-forecasting/actions/workflows/ci.yml)
**[Interactive results →](https://benyahuta.github.io/medicare-fraud-forecasting/)** · [Model card](MODEL_CARD.md) · [Full results tables](reports/RESULTS.md)

## The short version

Medicare is the U.S. government health insurance program for people over 65 and people with disabilities. Doctors,
nurses, labs and clinics bill it for the care they provide, about $100 billion a year for the services studied here.
A small number of them commit fraud: billing for services that never happened, billing for more expensive services than
were provided ("upcoding"), prescribing opioids without medical need, or taking kickbacks.

When a provider is caught, the government watchdog, the **HHS Office of Inspector General (OIG)**, can ban them from
billing Medicare. That ban is called an **exclusion**, and OIG publishes the full list of excluded providers. But OIG
can only investigate a tiny fraction of the ~1.2 million providers who bill Medicare, so **deciding whom to look at
first matters a lot**.

This project asks two questions:

1. **Using only data the government already publishes, can a model point investigators toward providers who will be
   excluded in the next few years?**
2. **If so, does the model unfairly point at doctors who serve poor or disabled patients?**

**How it works, in one paragraph.** For every provider, the government publishes what they billed Medicare each year:
which services, how many, to how many patients, and what they prescribed. I turned that into 86 measurements of
billing behavior, mostly "how unusual is this provider compared with others in the same specialty?" I then trained a
model on 2016–2019 data to recognize the billing patterns of providers who were later excluded. Finally I tested it
the honest way: give it providers' **2021** billing, ask it to rank them, and check the ranking against who OIG
actually excluded in **2022–2024**. The model never saw any of those later outcomes during training.

<p align="center"><img src="reports/figures/recall_at_budget.png" width="720"></p>

*How to read this chart: move right along the bottom and investigators review more providers, starting with the
model's highest-risk picks. The height of each line shows the share of future-excluded providers that review would
catch. Higher is better. The blue line is this project's model; the gray lines are simpler approaches.*

## Results in plain English

Out of **1,198,748** providers billing Medicare in 2021, **319** were excluded for fraud or abuse in 2022–24, or about
**1 in 3,800**. Finding them is a needle-in-a-haystack problem.

* **If investigators review the 1% of providers the model ranks riskiest (about 12,000 people), they find 17% of the
  319 future exclusions.** Picking providers at random would find 1%. That makes the model **17× better than random**,
  and **3.4× better than the common-sense approach** of reviewing whoever bills the most per patient compared with peers.
* **It gives early warning.** The providers it flags were excluded a median of **26 months later**: about two years in
  which payments could have been reviewed.
* **It can point toward the money.** Ranking by "risk × how much Medicare paid them" puts providers holding **59%** of
  the dollars paid to future-excluded providers into a 1% review list. (That figure leaves out two very large labs; with
  them it's 88%.)
* **It works year after year.** Retrained and retested on four different years (2019–2022), performance barely moves.
* **Fairness: a problem found, and a fix.** Out of the box, the model wrongly flags honest doctors who serve many
  low-income or disabled patients **3.4× as often** as other doctors. Removing patient-demographic information and
  splitting the review list fairly across patient groups brings that down to **1.4×**, while still catching the same
  number of cases.
* **A negative result.** A deep-learning model (a transformer) that reads each provider's raw billing did *worse* than the
  carefully designed measurements. With only a few hundred examples of fraud to learn from, domain knowledge beat raw
  model power.

> **Important caveat.** Being excluded is not the same as committing fraud, and not being excluded is not proof of
> innocence. The model learns from whom OIG *caught*. Its scores are meant to help a human decide where to look
> first, never to accuse anyone. This repository deliberately publishes no score for any individual provider. See the
> [model card](MODEL_CARD.md).

### The same results, in the standard metrics

For readers who want the technical numbers, with 95% confidence intervals. All terms are defined in the
[glossary](#glossary).

| | LightGBM (this project) | Logistic regression | Peer-payment rule¹ | Isolation Forest |
|---|---|---|---|---|
| **Recall when reviewing the top 1%** | **17.2%** (13.3–21.3) | 14.1% | 5.0% | 3.4% |
| Lift over random at 1% | **17×** | 14× | 5× | 3× |
| ROC-AUC | **0.823** (0.800–0.850) | 0.776 | 0.582 | 0.644 |
| Average precision (random = 0.0003) | **0.0028** | 0.0018 | 0.0005 | 0.0006 |

¹ *Rank providers by how much more they're paid per patient than others in their specialty: roughly what a spreadsheet audit would do.*

In a paired bootstrap the main model beats both simple baselines in all 300 replicates, and logistic regression in 95%
of them.

## What's in here

*Everything below this point goes into more technical detail. The [glossary](#glossary) at the bottom defines the terms.*

| | |
|---|---|
| **Data engineering** | ~25 GB of raw CMS files → 4.5M labelled provider-years, ~70M billing lines (Polars, Parquet cache) |
| **Feature engineering** | 86 features in 8 groups, all computed *within year* against specialty peers so they survive coding changes such as the 2021 E/M overhaul |
| **Models** | peer-payment rule, Isolation Forest, logistic regression, tuned LightGBM (5-seed bag), a **ClaimSet Transformer** over raw billing lines (PyTorch / Apple MPS), and a hybrid using its self-supervised embeddings |
| **Evaluation** | forward-in-time split, stratified-bootstrap CIs, paired significance tests, audit-budget metrics, dollar impact, lead time, rolling-origin test years 2019–2022 |
| **Responsible ML** | fairness audit of false positives by patient population and rurality, a stratified-allocation mitigation with its measured cost, a model card, no per-provider scores published |
| **Engineering** | installable package (`src/fraudml`), unit tests on synthetic data, ruff, GitHub Actions CI, Makefile-driven reproduction |

## Data

| Source | What it gives | Rows |
|---|---|---|
| [CMS Medicare Physician & Other Practitioners — by Provider](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider) | volume, payments, patient demographics & chronic conditions | ~1.1M / year |
| [… — by Provider and Service](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service) | every (provider, HCPCS code, place of service) line | ~10M / year |
| [Medicare Part D Prescribers — by Provider](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider) | prescribing incl. opioids, brand share, cost | ~1.1M / year |
| [OIG List of Excluded Individuals/Entities (LEIE)](https://oig.hhs.gov/exclusions/exclusions_list.asp) | who was excluded, when, under which authority | 84k |

**What counts as a "positive".** In plain terms: a provider counts as a hit if they were banned for fraud, abuse or
misconduct within three years of the billing year the model looks at. Precisely: a provider-year is positive if the NPI appears in the LEIE under a fraud, abuse or misconduct authority
(§1128(a)(1)–(4), §1128(b)(1)–(8), §1128A) with an exclusion date in the three calendar years after the
feature year. Providers already excluded are dropped. See the [model card](MODEL_CARD.md) for why
*exclusion ≠ fraud*.

**Split.** The model learns from older years and is graded on a newer year it never saw, the way it would actually be used.

| Cohort | Feature year | Label window | Providers | Positives |
|---|---|---|---|---|
| train | 2016 | 2017–2019 | 1,053,934 | 309 |
| train | 2017 | 2018–2020 | 1,088,666 | 261 |
| valid | 2019 | 2020–2021 (2 yr) | 1,155,862 | 105 |
| **test** | **2021** | **2022–2024** | **1,198,748** | **319** |

All training labels close before the first test label (2022), so the final model can be refit on train + valid
without seeing the future.

## Features

The key idea: a large bill isn't suspicious on its own (a busy cardiologist bills a lot), but billing very differently
from others *in the same specialty* can be. So most measurements compare each provider with their specialty peers in
the same year. All peer comparisons are within *(year, specialty)*, using robust z-scores (median / MAD) or percentiles.

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

Full tables, including per-specialty results and significance tests, are in [`reports/RESULTS.md`](reports/RESULTS.md).

**Lesson learned: how you choose settings mattered more than which model you choose.** In plain terms: my first
attempt tuned the model's settings against a small check set and picked settings that only looked good by luck. Testing
settings across three larger, separate groups of providers fixed it. The details: My first pass tuned LightGBM on the single 2019 validation
cohort, with 105 positives. Validation AP swung from 0.002 to 0.033 between configurations, the "best" one was noise, and on
test it did *worse* than logistic regression (recall@1% 12.2% vs 14.1%). Switching to **provider-grouped 3-fold CV on
the training cohorts** (570 positives, every configuration scored on three disjoint sets of providers) plus **balanced bagging** (each of five
members sees all positives and a different 25% of negatives) picked a heavily regularized model: extremely randomized
trees, 7 leaves, at least 2,000 samples per leaf. It wins on every metric. The change is in the
git history (`Grouped-CV model selection, balanced bagging …`).

| Budget (share of providers reviewed) | 0.1% | 0.5% | 1% | 5% |
|---|---|---|---|---|
| Providers reviewed | 1,199 | 5,994 | 11,987 | 59,937 |
| Later-excluded providers found | 11 | 30 | 55 | 128 |
| Precision | 0.9% | 0.5% | 0.46% | 0.21% |

In plain terms: reviewing 12,000 providers turns up 55 who were later banned. A 0.5–1% hit rate sounds low, but the base rate is 0.027%. And "not excluded within 3 years" is not the same as
"not committing fraud", so some of the remaining flags are probably undetected cases rather than false alarms.

### What drives the score

Which information helps the model most? The first chart removes one group of measurements at a time and re-scores the
model. The second (SHAP) shows, for individual providers, which measurements pushed their risk up (right) or down (left).

<p align="center"><img src="reports/figures/ablation.png" width="620"><br><img src="reports/figures/shap_summary.png" width="640"></p>

* The engineered features **roughly double** average precision over the CMS summary file alone (0.0028 vs 0.0013).
* No single group is indispensable. Billing behaviour shows up redundantly across per-code intensity, Part D prescribing
  and E/M mix, and removing any one costs at most 7%.
* **Removing the patient-mix group did not hurt, and test AP even rose (0.0034 vs 0.0028).** With 319 test positives
  that gap is within bootstrap noise, but it points the same way as the fairness results below: patient-mix features add
  little forward-looking signal and most of the disparity.
* SHAP shows what the model keys on: specialty, office-only practices (low facility share), a high share of prescriptions
  to low-income-subsidy patients, payments concentrated in codes that peers rarely bill, *below*-peer allowed prices per
  code, and opioid prescribing above peers.
* The E/M chart above is a useful negative result. The textbook upcoding signal, "bills level 4–5 visits more than peers",
  is weak on its own: later-excluded internists and family physicians cluster at *both* extremes of the level mix.

An anonymized example of why one provider who was later banned got a high score: they were a pain-management practice
with very high opioid prescribing and a billing mix unlike their peers. Values are rounded so the public files can't be searched for an exact match.

<p align="center"><img src="reports/figures/case_study.png" width="700"></p>

### Early warning

<p align="center"><img src="reports/figures/lead_time.png" width="600"></p>

Most exclusions come 2–3 years after the billing year scored. The flags fire well before OIG acts: median 26 months,
IQR 22–32 months. That is long enough for a payment review or a targeted audit to matter.

### Does it hold up across years?

<p align="center"><img src="reports/figures/rolling.png" width="600"></p>

Each test year *T* gets its own model, trained only on cohorts whose three-year label windows close by the end of *T*,
before any exclusion it is scored on. This mimics an analyst retraining every January. Hyper-parameters are frozen
from the main experiment, which selected them on the 2016–17 cohorts. For test years 2019–20 that choice saw
exclusions from overlapping years, so those two rows are slightly optimistic.

| Test year | Trained on | Positives | ROC-AUC | Recall @1% | Peer-rule recall @1% |
|---|---|---|---|---|---|
| 2019 | 2016 | 258 | 0.818 | 20.2% | 4.3% |
| 2020 | 2016–17 | 260 | 0.825 | 20.4% | 2.7% |
| 2021 | 2016–18 | 319 | 0.825 | 16.9% | 5.0% |
| 2022 | 2016–19 | 288 | 0.821 | 14.6% | 4.9% |

Ranking quality (AUC) is essentially constant across four years, through COVID-19 and the 2021 E/M coding overhaul.
Top-of-list recall drifts down slightly in later years, which is worth monitoring if this were deployed.

### Can a neural network learn this from raw billing lines?

In plain terms: instead of hand-designing 86 measurements, can a modern deep-learning model work out what matters by
reading each provider's raw list of billed services? Here, no. It needs far more examples of fraud than exist.

The **ClaimSet Transformer** (`src/fraudml/deep.py`) treats each provider-year as an unordered set of up to 32 billing
lines: HCPCS embedding, place of service, and 9 numeric features such as volume, price, and intensity vs. peers. A
transformer encoder with attention pooling feeds two heads. One predicts exclusion; an auxiliary head predicts the
provider's specialty from the code mix alone, which gives the encoder ~1M labels per year instead of a few hundred
positives.

| Model (2021 test cohort) | AP | ROC-AUC | Recall @1% |
|---|---|---|---|
| ClaimSet Transformer, end-to-end | 0.0010 | 0.718 | – |
| LightGBM, engineered features (3 seeds) | 0.0027 | 0.824 | 17.9% |
| LightGBM + 16 self-supervised ClaimSet embedding dims + specialty surprise | 0.0028 | 0.822 | 16.3% |

* **End-to-end training loses badly.** With ~700 positive examples the network memorizes the positives within a few
  epochs: training loss keeps falling while validation AP stays flat. Heavy regularization (2 layers, d = 64, dropout
  0.3, three positive repeats per epoch) helped but did not close the gap.
* **Self-supervised pretraining works as representation learning, but adds nothing here.** Trained only on the specialty
  task over 3.3M provider-years, the encoder recovers the declared specialty with p > 0.5 for 54% of 2021 providers across
  80 classes. Adding its embeddings to LightGBM changes AP by **+0.0002 (95% CI −0.0001 to +0.0010)**, which is not
  significant.
* **Takeaway:** with labels this scarce, domain-specific peer-relative features carry the signal that a sequence model
  would otherwise have to discover from a few hundred examples. I'd expect the neural approach to pay off with
  claim-level data or far more positives (e.g. state Medicaid fraud-unit referrals).

## Who gets wrongly flagged? (fairness audit)

A model that steers audits toward providers *because of who their patients are* would harm exactly the clinics that
low-income and disabled Medicare beneficiaries depend on. So we measure, at a 1% audit budget, each group's
false-positive rate relative to the overall rate.

<p align="center"><img src="reports/figures/fairness.png" width="680"></p>

* The raw model's false positives fall **3.4×** more often on providers in the top quartile of dual-eligible (low-income)
  patients, and **3.4×** on those in the top quartile of disabled (<65) patients. Rural providers are near parity (1.1×).
* Part of this reflects real differences in base rates: exclusions are 1.6× more common in those groups. But
  false-positive rate is measured *only on providers who were never excluded*, so a 3.4× ratio means extra scrutiny of
  honest clinics serving vulnerable patients.
* Deleting the five patient-demographic features only reduces it to 2.9×. Billing patterns such as low-income-subsidy
  prescription share and chronic-condition mix act as proxies.

<p align="center"><img src="reports/figures/fairness_tradeoff.png" width="560"></p>

**The fix: a fairer review list.** In plain terms: instead of one ranking across everyone, providers are also ranked
*within groups of similar patient populations*, and the review list draws from every group. A dial, α, sets how much
weight goes to the within-group ranking (0 = none, 1 = every group reviewed at the same rate). Precisely: each provider's score is blended with its percentile *within its
patient-mix stratum* (dual-eligible quartile × disabled quartile, 25 strata). At α = 1 every stratum is reviewed at the same
rate.

| Configuration | Recall @1% | AP | Worst-group FPR ratio |
|---|---|---|---|
| LightGBM (all features) | 17.2% | 0.0028 | 3.42 |
| No patient demographics | 16.9% | 0.0030 | 2.92 |
| No demographics + stratified α = 0.5 | 16.9% | 0.0029 | 1.81 |
| **No demographics + stratified α = 0.75** | **16.9%** | 0.0027 | **1.39** |
| No demographics + stratified α = 1 | 14.7% | 0.0024 | 1.13 |

Without demographics, recall on the test cohort is unchanged all the way to α = 0.75, while the worst-group
false-positive ratio falls from 2.9× to 1.4×. Full parity costs about 2.5 points of recall, which is still above logistic
regression with no mitigation. These α values were *read off* the test cohort rather than chosen in advance, so an agency
should set α on its own validation data. **The trade-off belongs to the people deploying the tool, so the code exposes α
rather than hiding it.**

## Limitations

* **Labels are enforcement outcomes, not ground truth.** Undetected fraud counts as a negative, and any bias in whom OIG
  pursues is learned. Exclusions under §1128(b)(4), license actions, include misconduct unrelated to billing.
* **Small positive counts** (~100–320 per cohort) make metrics noisy. That is why every headline number has a bootstrap
  CI, and why model selection uses grouped CV rather than the 105-positive validation cohort.
* **Probabilities are not calibrated forward in time.** The model expects 165 exclusions in the 2021 cohort and 319
  happened, so use scores for ranking only.
* **Aggregate public data only**: no claims, diagnoses or patient-level detail, and CMS suppresses small cells.
* **Not a decision system.** See the [model card](MODEL_CARD.md). This repository intentionally publishes no
  per-provider scores.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/), ~30 GB disk and ~16 GB RAM. On macOS, LightGBM needs `brew install libomp`.

```bash
uv sync --extra deep   # include PyTorch for the neural models
make data          # download CMS + OIG files (~25 GB)
make features      # build data/interim/dataset.parquet (~10 min)
make experiments   # baselines, tuned LightGBM, ablations, fairness, dollar queue (~1 h on an M-series laptop)
make rolling       # rolling-origin evaluation, test years 2019–2022
make deep          # ClaimSet Transformer end to end, then the self-supervised hybrid (uses MPS/CUDA if available)
make report        # figures, reports/RESULTS.md, docs/data.js
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

## Glossary

**Medicare / healthcare terms**

| Term | Meaning |
|---|---|
| **CMS** | Centers for Medicare & Medicaid Services, the agency that runs Medicare and publishes the billing data used here. |
| **OIG** | HHS Office of Inspector General, the watchdog that investigates fraud and decides exclusions. |
| **Exclusion / LEIE** | A ban on billing federal health programs. The LEIE (List of Excluded Individuals/Entities) is OIG's public list of them. |
| **NPI** | National Provider Identifier, a unique ID number for every provider. Used to link billing data to the exclusion list. |
| **Part B / Part D** | Part B pays for doctor visits and outpatient services; Part D pays for prescription drugs. |
| **HCPCS code** | The standardized code for each billable service (e.g. 99213 = a routine office visit). |
| **E/M visit levels, upcoding** | Office visits are billed at levels 1–5; higher levels pay more. *Upcoding* means billing a higher level than the visit justified. |
| **Dual-eligible** | Patients who qualify for both Medicare and Medicaid, a standard marker of low income. |
| **Under-65 patients** | People on Medicare before 65 are almost always there because of a disability. |
| **Specialty / peers** | Providers are compared with others who declared the same specialty, because a normal bill for a cardiologist looks nothing like one for a chiropractor. |

**Machine-learning terms**

| Term | Meaning |
|---|---|
| **Training / test data** | The model learns from training data (2016–2019) and is graded on test data it never saw (2021 billing → 2022–24 exclusions). |
| **Feature** | One measurement the model sees per provider, e.g. "payment per patient compared with peers". |
| **Recall at 1%** | Of all providers who were later excluded, the share that appear in the model's top 1% of risk. The most practical number here. |
| **Precision** | Of the providers flagged, the share who were later excluded. |
| **Lift** | How many times better than random picking. 17× means 17 times as many hits as choosing providers at random. |
| **ROC-AUC** | Chance the model ranks a randomly chosen later-excluded provider above a randomly chosen other provider. 0.5 = coin flip, 1.0 = perfect. |
| **Average precision (AP)** | A summary of precision across all cut-offs. Looks tiny here because only 0.027% of providers are positive; compare it with the random baseline of 0.0003. |
| **False-positive rate (FPR)** | Among providers who were *not* later excluded, the share the model flagged anyway, i.e. honest providers sent for review. |
| **Confidence interval (CI)** | The range the true value plausibly falls in, given the limited number of cases. Computed by bootstrap resampling. |
| **LightGBM** | A widely used gradient-boosted decision-tree model, strong on tabular data. |
| **Logistic regression, Isolation Forest** | Simpler baselines: a linear model, and an anomaly detector that uses no fraud labels at all. |
| **Cross-validation (CV)** | Repeatedly training on part of the data and checking on the rest, to choose settings without touching the test set. |
| **SHAP** | A method that explains which features pushed a given prediction up or down. |
| **Ablation** | Removing a group of features and re-measuring, to see how much they matter. |
| **Transformer** | The neural-network architecture behind modern AI models; here applied to a provider's list of billed services. |

## License & data use

Code: MIT. Data: public-use files from CMS and HHS-OIG, which are not redistributed here (download with `make data`).
This project is an independent research exercise and is not affiliated with CMS or OIG.

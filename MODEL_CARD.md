# Model card: Medicare exclusion-risk ranker

## What it does
Given one calendar year of a provider's **public** Medicare billing (Part B services and
Part D prescribing, as published by CMS), the model outputs a risk score. The score ranks
how likely the provider is to be **excluded from federal health programs by the HHS Office
of Inspector General (OIG)** within the next three years, under a fraud, abuse or
misconduct authority (Social Security Act §1128(a)(1)-(4), §1128(b)(1)-(8), §1128A).

## Intended use
* **Prioritizing human review** under a fixed audit budget, for example "which 1% of
  providers should an analyst look at first". The output is a queue for a human, never
  a decision.
* Research on fraud-detection methods and on the fairness of audit targeting.

## Out of scope / must not be used for
* Deciding that any individual committed fraud, or taking any adverse action (payment
  suspension, denial, publication) on the basis of the score.
* Publishing ranked lists of named providers. This repository deliberately does not
  publish per-provider scores.
* Non-Medicare populations, or years far outside 2016–2022, without re-validation.

## Training data
* CMS *Medicare Physician & Other Practitioners* (by Provider; by Provider and Service)
  and *Medicare Part D Prescribers – by Provider*, years 2016–2022. These are public,
  aggregate files; CMS suppresses cells with fewer than 11 beneficiaries.
* Labels: the OIG *List of Excluded Individuals/Entities* (LEIE), joined by NPI.

## Label caveats (read before trusting any number)
* **Exclusion ≠ fraud, and no exclusion ≠ innocence.** The LEIE only contains providers who
  were *caught and sanctioned*. Undetected fraud shows up as false negatives, and the
  model partly learns "what OIG has historically pursued", including any enforcement bias.
* Exclusions under §1128(b)(4) (license revocation) are included. Some of these involve
  misconduct unrelated to billing.
* The public LEIE "UPDATED" file drops providers who were later reinstated, so older
  cohorts slightly undercount positives.
* Only exclusions that carry an NPI can be linked (most do for individual clinicians).

## Evaluation design
Strictly forward in time. Features from year *Y* predict exclusions in *Y+1 … Y+3*. The
headline test cohort is 2021 (exclusions 2022–2024). Training labels all end before
2022, and the final model is refit on train+valid only. A rolling-origin evaluation
repeats this for test years 2019–2022. Metrics are reported as recall / precision /
lift at fixed audit budgets plus average precision, with stratified-bootstrap 95% CIs.

## Fairness
At a 1% audit budget, an unconstrained model wrongly flags providers who serve many
dual-eligible (low-income) or disabled patients at a higher rate than average. Dropping
patient-demographic features does **not** fix this, because billing patterns act as
proxies. The repository measures a *stratified audit allocation* that equalizes review
rates across patient-mix strata, and reports what that costs in recall (see the
README). Anyone deploying a tool like this should choose that trade-off explicitly
rather than inherit it by default.

## Known limitations
* Small number of positives per cohort (~100–320), so metric CIs are wide.
* Aggregate public data only: no claim-level, diagnosis or beneficiary-level detail.
* Organizations (e.g. clinics) and individuals are scored by the same model.
* Specialty labels are self-reported to CMS.

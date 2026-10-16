# Results

Test cohort: 1,198,748 providers billing Medicare in 2021, of whom 319 were excluded by OIG in the following three years (prevalence 0.0266%). 95% CIs from a stratified bootstrap.

## Headline comparison

| Model | AP | ROC-AUC | Recall @0.1% | Recall @1% | Lift @1% | Recall @5% |
|---|---|---|---|---|---|---|
| Peer payment rule | 0.001 (0.000–0.001) | 0.582 (0.551–0.619) | 0.003 | 0.050 (0.028–0.077) | 5.0× | 0.150 |
| Isolation Forest | 0.001 (0.000–0.001) | 0.644 (0.613–0.675) | 0.009 | 0.034 (0.016–0.055) | 3.4× | 0.157 |
| Logistic regression | 0.002 (0.001–0.003) | 0.776 (0.751–0.801) | 0.025 | 0.141 (0.103–0.182) | 14.1× | 0.304 |
| LightGBM (ours) | 0.003 (0.002–0.005) | 0.823 (0.800–0.850) | 0.034 | 0.172 (0.133–0.213) | 17.2× | 0.401 |
| LightGBM + LR blend | 0.002 (0.002–0.004) | 0.808 (0.786–0.833) | 0.034 | 0.157 (0.119–0.197) | 15.7× | 0.351 |
| LightGBM (no patient demographics) | 0.003 (0.002–0.005) | 0.822 (0.797–0.846) | 0.044 | 0.169 (0.132–0.216) | 16.9× | 0.398 |
| LightGBM + stratified audit | 0.002 (0.002–0.003) | 0.810 (0.781–0.835) | 0.034 | 0.138 (0.097–0.177) | 13.8× | 0.379 |
| ClaimSet Transformer (end-to-end) | 0.001 | 0.718 | | | | |
| LightGBM + ClaimSet embeddings (3 seeds) | 0.003 | 0.822 | 0.031 | 0.163 | 16.3× | 0.411 |

Hybrid vs. same-seed LightGBM baseline (AP 0.003): ΔAP = +0.0002 (95% CI -0.0001 to +0.0010).

## Paired bootstrap: LightGBM (ours) minus baseline, ΔAP

| Baseline | ΔAP | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| Peer payment rule | +0.0025 | +0.0014 to +0.0046 | 0.000 |
| Isolation Forest | +0.0024 | +0.0013 to +0.0044 | 0.000 |
| Logistic regression | +0.0011 | -0.0002 to +0.0028 | 0.047 |
| LightGBM + LR blend | +0.0005 | -0.0000 to +0.0017 | 0.033 |

## Feature-group ablation (test AP, 3-seed LightGBM)

| Variant | AP | Δ vs. all |
|---|---|---|
| summary-file only (volume+meta) | 0.0013 | -52% |
| - service | 0.0026 | -7% |
| - rx | 0.0027 | -3% |
| - em | 0.0027 | -1% |
| all features | 0.0028 | +0% |
| - meta | 0.0029 | +5% |
| - mismatch | 0.0029 | +6% |
| - volume | 0.0029 | +7% |
| - peer | 0.0029 | +7% |
| - bene | 0.0034 | +24% |

## Dollars

Standardized Medicare payments in the test year: $106.9B in total, $132.2M to providers later excluded.

| Budget | Providers reviewed | Later-excluded found | Their payments | Share of those $ |
|---|---|---|---|---|
| 0.1% | 1,199 | 11 | $0.5M | 0% |
| 0.5% | 5,994 | 30 | $5.9M | 4% |
| 1% | 11,987 | 55 | $11.3M | 9% |
| 5% | 59,937 | 128 | $55.0M | 42% |

Lead time for providers caught at 1%: median 26 months (IQR 22–32) between the end of the scored year and the exclusion.

## Fairness at a 1% audit budget

False-positive rate of each group relative to all providers (1.0 = parity).

| Group | LightGBM (ours) | LightGBM (no patient demographics) | Peer payment rule |
|---|---|---|---|
| all providers | 1.00 | 1.00 | 1.00 |
| high dual-eligible share (top quartile) | 3.42 | 2.92 | 1.34 |
| high disabled (<65) share (top quartile) | 3.35 | 2.76 | 1.19 |
| rural (RUCA >= 4) | 1.10 | 1.10 | 0.85 |
| urban (RUCA < 4) | 0.99 | 0.99 | 1.02 |

Stratified-allocation trade-off (α = weight on within-stratum ranking):

| α | AP | Recall @1% | Worst-group FPR ratio |
|---|---|---|---|
| 0.00 | 0.0028 | 0.172 | 3.42 |
| 0.25 | 0.0027 | 0.169 | 2.64 |
| 0.50 | 0.0027 | 0.157 | 2.00 |
| 0.75 | 0.0025 | 0.157 | 1.49 |
| 1.00 | 0.0021 | 0.138 | 1.11 |

## Recommended configuration (fairness without losing recall)

| Variant | AP | Recall @1% | Worst-group FPR ratio |
|---|---|---|---|
| LightGBM (ours) | 0.0028 | 0.172 | 3.42 |
| No demographics | 0.0030 | 0.169 | 2.92 |
| No demographics + stratified α=0.5 | 0.0029 | 0.169 | 1.81 |
| No demographics + stratified α=0.75 | 0.0027 | 0.169 | 1.39 |
| No demographics + stratified α=1 | 0.0024 | 0.147 | 1.13 |

## Case queue vs. dollar queue (1% budget)

Two clinical laboratories account for 72% of all 2021 payments to later-excluded providers, so dollar shares are also shown without them.

| Ranking | Later-excluded found | Recall | Their 2021 payments | $ share | $ share excl. 2 largest |
|---|---|---|---|---|---|
| risk | 55 | 17.2% | $11.3M | 9% | 31% |
| risk × dollars | 37 | 11.6% | $117.0M | 88% | 59% |
| dollars only | 8 | 2.5% | $108.2M | 82% | 35% |

Calibration: after undoing negative subsampling the model expects 165 exclusions in the test cohort; 319 occurred (rates rose relative to the training years), so scores should be used for ranking, not as probabilities.

## Rolling-origin evaluation (3-year label window)

| Test year | Train years | Positives | Model | AP | ROC-AUC | Recall @1% |
|---|---|---|---|---|---|---|
| 2019 | 2016–2016 | 258 | LightGBM (ours) | 0.003 (0.002–0.005) | 0.818 | 0.202 (0.159–0.252) |
| 2019 | 2016–2016 | 258 | Peer payment rule | 0.000 (0.000–0.001) | 0.595 | 0.043 (0.016–0.070) |
| 2020 | 2016–2017 | 260 | LightGBM (ours) | 0.003 (0.002–0.006) | 0.825 | 0.204 (0.158–0.258) |
| 2020 | 2016–2017 | 260 | Peer payment rule | 0.000 (0.000–0.001) | 0.613 | 0.027 (0.012–0.046) |
| 2021 | 2016–2018 | 319 | LightGBM (ours) | 0.003 (0.002–0.005) | 0.825 | 0.169 (0.129–0.207) |
| 2021 | 2016–2018 | 319 | Peer payment rule | 0.001 (0.000–0.001) | 0.582 | 0.050 (0.028–0.072) |
| 2022 | 2016–2019 | 288 | LightGBM (ours) | 0.003 (0.002–0.005) | 0.821 | 0.146 (0.111–0.188) |
| 2022 | 2016–2019 | 288 | Peer payment rule | 0.000 (0.000–0.001) | 0.554 | 0.049 (0.028–0.073) |

## Per-specialty (specialties with ≥ 8 test positives)

| Specialty | Providers | Positives | ROC-AUC | Recall @ global 1% |
|---|---|---|---|---|
| Family Practice | 83,929 | 55 | 0.768 | 0.20 |
| Internal Medicine | 91,072 | 45 | 0.843 | 0.24 |
| Nurse Practitioner | 162,568 | 24 | 0.733 | 0.04 |
| Chiropractic | 33,885 | 18 | 0.696 | 0.06 |
| Podiatry | 15,433 | 13 | 0.721 | 0.31 |
| Psychiatry | 20,760 | 12 | 0.799 | 0.17 |
| Emergency Medicine | 46,367 | 11 | 0.792 | 0.18 |
| Obstetrics & Gynecology | 24,892 | 11 | 0.691 | 0.00 |
| General Practice | 4,150 | 9 | 0.814 | 0.33 |
| General Surgery | 20,549 | 9 | 0.906 | 0.00 |
| Certified Registered Nurse Anesthetist (CRNA) | 48,405 | 8 | 0.745 | 0.00 |
| Physician Assistant | 91,922 | 8 | 0.808 | 0.00 |

## Top SHAP features

| Feature | mean abs SHAP |
|---|---|
| `specialty_cat` | 0.6019 |
| `svc_facility_share` | 0.2530 |
| `rx_lis_claim_share` | 0.1275 |
| `bene_under65_share` | 0.1169 |
| `bene_female_share` | 0.1130 |
| `svc_rare_code_pay_share` | 0.1105 |
| `cc_cancer6` | 0.1067 |
| `charge_markup` | 0.0957 |
| `rx_antipsychotic_65_share` | 0.0882 |
| `svc_pay_excess` | 0.0881 |
| `bene_dual_share` | 0.0833 |
| `drug_pay_share` | 0.0816 |
| `cc_alz` | 0.0780 |
| `peer_z_charge_markup` | 0.0714 |
| `peer_z_rx_opioid_share` | 0.0675 |

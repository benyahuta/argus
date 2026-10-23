# argus

Predicting which Medicare providers will get excluded by the HHS Office of Inspector General (OIG) for fraud or abuse, using only public CMS billing data.

Results page: https://benyahuta.github.io/argus/

## What it does

For every provider who bills Medicare, CMS publishes what they billed and prescribed each year. I turned that into features (mostly "how unusual is this provider compared to others in their specialty") and trained a model to rank providers by how likely they are to be excluded in the next 3 years.

Trained on 2016-2019, tested on 2021 billing against the exclusions OIG actually made in 2022-2024.

## Results

Test set: 1,198,748 providers, 319 later excluded.

| Model | Recall in top 1% | ROC-AUC |
|---|---|---|
| LightGBM | 17.2% | 0.823 |
| Logistic regression | 14.1% | 0.776 |
| Peer-payment rule | 5.0% | 0.582 |
| Isolation Forest | 3.4% | 0.644 |

I also checked who gets wrongly flagged. The model flags providers with lots of low-income or disabled patients about 3.4x more often than average. Dropping demographic features and ranking within patient-mix groups got that down to about 1.4x without losing recall.

More detail is in [reports/RESULTS.md](reports/RESULTS.md) and the [model card](MODEL_CARD.md).

## Caveats

Getting excluded isn't the same as committing fraud, so the model learns who OIG catches. There are only a few hundred positives per year, so the numbers are noisy. The repo doesn't publish scores for individual providers.

## Running it

Needs [uv](https://docs.astral.sh/uv/) and about 30 GB of disk.

```bash
uv sync
make data
make features
make experiments
make report
```

Data comes from CMS and HHS-OIG public files (not included, `make data` downloads them). Code is MIT.

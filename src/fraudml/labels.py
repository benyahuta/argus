"""Forward-looking exclusion labels from the OIG LEIE."""

from __future__ import annotations

import polars as pl

from .config import FRAUD_EXCLUSION_TYPES, Cohort


def attach_labels(features: pl.DataFrame, leie: pl.DataFrame, cohort: Cohort) -> pl.DataFrame:
    """Label each provider 1 if excluded (fraud/abuse authority) inside the cohort's window.

    Providers already excluded on or before the end of the feature year are dropped:
    they should not be billing Medicare at all, so they are not a forecasting target.
    """
    fraud = leie.filter(pl.col("excl_type").is_in(FRAUD_EXCLUSION_TYPES))
    # A provider can appear more than once; the earliest exclusion is the event.
    first = fraud.group_by("npi").agg(
        pl.col("excl_date").min(), pl.col("excl_type").sort_by("excl_date").first()
    )
    already = leie.filter(pl.col("excl_date") < cohort.label_start).select("npi").unique()

    out = (
        features.join(already, on="npi", how="anti")
        .join(first, on="npi", how="left")
        .with_columns(
            pl.col("excl_date").is_between(cohort.label_start, cohort.label_end)
            .fill_null(False).cast(pl.Int8).alias("label"),
            pl.lit(cohort.name).alias("split"),
        )
    )
    # Exclusion details are only meaningful for positives; never use them as features.
    return out.with_columns(
        pl.when(pl.col("label") == 1).then(pl.col("excl_date")).alias("excl_date"),
        pl.when(pl.col("label") == 1).then(pl.col("excl_type")).alias("excl_type"),
    )

"""Build the labelled provider-year dataset for every cohort in the experiment config."""

from __future__ import annotations

import time

import polars as pl

from fraudml.config import ALL_YEARS, EXP, INTERIM
from fraudml.data import load_leie, load_partd, load_provider, load_service
from fraudml.features import build_year
from fraudml.labels import attach_labels


def features(year: int) -> pl.DataFrame:
    path = INTERIM / f"features_{year}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    t = time.time()
    feats = build_year(load_provider(year), load_service(year), load_partd(year),
                       min_benes=EXP.min_benes, seed=EXP.seed)
    feats.write_parquet(path)
    print(f"features {year}: {feats.height:,} rows [{time.time() - t:.0f}s]", flush=True)
    return feats


def main() -> None:
    leie = load_leie()
    # Every year with data, so the rolling-origin evaluation can reuse the cache.
    for year in ALL_YEARS:
        features(year)
    parts = []
    for cohort in (*EXP.train, EXP.valid, EXP.test):
        t = time.time()
        feats = features(cohort.feature_year)
        labelled = attach_labels(feats, leie, cohort)
        n_pos = int(labelled["label"].sum())
        print(f"{cohort.name:5s} {cohort.feature_year}: {labelled.height:>9,} providers, "
              f"{n_pos:>4} excluded in {cohort.label_start // 10000}-{cohort.label_end // 10000}"
              f" ({n_pos / labelled.height:.4%})  [{time.time() - t:.0f}s]")
        parts.append(labelled)

    ds = pl.concat(parts, how="vertical_relaxed")
    # Training positives never reappear as validation/test positives (they are excluded).
    ds.write_parquet(INTERIM / "dataset.parquet")
    print(f"wrote {ds.height:,} rows -> {INTERIM / 'dataset.parquet'}")


if __name__ == "__main__":
    main()

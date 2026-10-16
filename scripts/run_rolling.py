"""Rolling-origin evaluation: does the model hold up across test years 2019-2022?

For each test year T the model only sees cohorts whose 3-year label windows close by
the end of T (feature years <= T-3), mimicking an investigator retraining every
January with whatever exclusions have been published so far. Hyper-parameters are
frozen to the values chosen in run_experiments.py.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import polars as pl

from fraudml.config import EXP, INTERIM, REPORTS, Cohort
from fraudml.data import load_leie
from fraudml.evaluate import bootstrap_ci, summarize
from fraudml.features import ALL_FEATURES
from fraudml.labels import attach_labels
from fraudml.models import fit_gbm_ensemble, predict_ensemble, rule_score, to_pandas

WINDOW = 3
TEST_YEARS = [2019, 2020, 2021, 2022]
FIRST_YEAR = 2016


def main() -> None:
    t0 = time.time()
    leie = load_leie()
    tuned = json.loads((REPORTS / "results.json").read_text())["tuning"]
    params = tuned["params"] | {"verbose": -1}
    n_iter = int(tuned["best_iter"] * 1.2)
    feats = {y: pl.read_parquet(INTERIM / f"features_{y}.parquet")
             for y in range(FIRST_YEAR, max(TEST_YEARS) + 1)}
    specialties = sorted(set().union(*[set(f["specialty"].unique()) for f in feats.values()]))

    rows = []
    for T in TEST_YEARS:
        train_years = list(range(FIRST_YEAR, T - WINDOW + 1))
        tr = pl.concat([attach_labels(feats[y], leie, Cohort("train", y, WINDOW))
                        for y in train_years], how="vertical_relaxed")
        te = attach_labels(feats[T], leie, Cohort("test", T, WINDOW))
        Xtr, Xte = (to_pandas(d, ALL_FEATURES, specialties) for d in (tr, te))
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
        models = fit_gbm_ensemble(params, n_iter, Xtr, ytr, seeds=(0, 1, 2))
        for name, s in (("LightGBM (ours)", predict_ensemble(models, Xte)),
                        ("Peer payment rule", rule_score(Xte))):
            r = summarize(yte, s, EXP.audit_budgets)
            r["ci"] = bootstrap_ci(yte, s, n_boot=200, seed=EXP.seed)
            rows.append({"test_year": T, "train_years": train_years, "model": name,
                         "n_train_pos": int(ytr.sum()), **r})
            print(f"[{time.time() - t0:5.0f}s] T={T} train={train_years} {name:18s} "
                  f"pos={int(yte.sum())} AP={r['ap']:.4f} AUC={r['roc_auc']:.3f} "
                  f"recall@1%={r['recall@1pct']:.3f}", flush=True)
    (REPORTS / "rolling.json").write_text(json.dumps(rows, indent=2, default=float))
    print(pd.DataFrame(rows)[["test_year", "model", "n_pos", "ap", "roc_auc", "recall@1pct"]])


if __name__ == "__main__":
    main()

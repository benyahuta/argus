"""Model zoo: unsupervised and rule baselines, logistic regression and LightGBM."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer


def to_pandas(df, features: list[str], specialties: list[str]) -> pd.DataFrame:
    """Polars -> pandas with a fixed specialty category order shared by all cohorts."""
    pdf = df.select([f for f in features if f != "specialty_cat"] + ["specialty"]).to_pandas()
    if "specialty_cat" in features:
        pdf["specialty_cat"] = pd.Categorical(pdf["specialty"], categories=specialties)
    return pdf.drop(columns="specialty")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def rule_score(pdf: pd.DataFrame) -> np.ndarray:
    """What a spreadsheet audit does: rank by payment-per-beneficiary vs. specialty peers."""
    return pdf["peer_z_log_pay_per_bene"].fillna(0).to_numpy()


def isolation_forest(X_tr: pd.DataFrame, X_te: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Label-free anomaly detection (what you'd deploy with no investigation history)."""
    num = [c for c in X_tr.columns if c != "specialty_cat"]
    imp = make_pipeline(SimpleImputer(strategy="median"),
                        QuantileTransformer(n_quantiles=200, subsample=200_000,
                                            random_state=seed))
    tr = imp.fit_transform(X_tr[num].sample(n=min(300_000, len(X_tr)), random_state=seed))
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=-1)
    iso.fit(tr)
    return -iso.score_samples(imp.transform(X_te[num]))


def logistic(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, C: float = 0.05,
             seed: int = 0) -> np.ndarray:
    num = [c for c in X_tr.columns if c != "specialty_cat"]

    def design(X: pd.DataFrame) -> pd.DataFrame:
        if "specialty_cat" not in X:
            return X[num]
        dummies = pd.get_dummies(X["specialty_cat"], prefix="spec", dtype=float)
        return pd.concat([X[num], dummies], axis=1)

    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000,
                            random_state=seed),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(design(X_tr), y_tr)
    return pipe.predict_proba(design(X_te))[:, 1]


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
    scale_pos_weight=20.0, max_cat_to_onehot=4, cat_smooth=50, min_data_per_group=500,
    verbose=-1, num_threads=0,
)


def search_space(rng: np.random.Generator) -> dict:
    return dict(
        num_leaves=int(rng.choice([7, 15, 31, 63])),
        min_child_samples=int(rng.choice([50, 200, 500, 1000])),
        feature_fraction=float(rng.choice([0.4, 0.6, 0.8])),
        lambda_l2=float(rng.choice([1.0, 10.0, 50.0])),
        scale_pos_weight=float(rng.choice([1.0, 10.0, 50.0])),
        learning_rate=float(rng.choice([0.02, 0.05])),
    )


TUNED = ["num_leaves", "min_child_samples", "feature_fraction", "lambda_l2",
         "scale_pos_weight", "learning_rate"]


@dataclass
class GBMResult:
    params: dict
    best_iter: int
    valid_ap: float
    trials: list[dict] = field(default_factory=list)


def _fit(params: dict, X_tr, y_tr, X_va=None, y_va=None, rounds: int = 3000,
         seed: int = 0) -> lgb.Booster:
    p = {**params, "seed": seed}
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    if X_va is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    return lgb.train(p | {"metric": "average_precision"}, dtr, num_boost_round=rounds,
                     valid_sets=[dva],
                     callbacks=[lgb.early_stopping(150, verbose=False)])


def tune_gbm(X_tr, y_tr, X_va, y_va, n_trials: int = 16, seed: int = 0,
             log=print) -> GBMResult:
    rng = np.random.default_rng(seed)
    trials = []
    best = None
    for t in range(n_trials):
        params = BASE_PARAMS | ({} if t == 0 else search_space(rng))
        booster = _fit(params, X_tr, y_tr, X_va, y_va, seed=seed)
        pred = booster.predict(X_va, num_iteration=booster.best_iteration)
        ap = average_precision_score(y_va, pred)
        trials.append({**{k: params[k] for k in TUNED}, "best_iter": booster.best_iteration,
                       "valid_ap": ap})
        log(f"  trial {t:2d}  valid AP={ap:.4f}  iters={booster.best_iteration}  "
            f"leaves={params['num_leaves']} mcs={params['min_child_samples']} "
            f"spw={params['scale_pos_weight']}")
        if best is None or ap > best.valid_ap:
            best = GBMResult(params, booster.best_iteration, ap)
    best.trials = trials
    return best


def fit_gbm_ensemble(params: dict, n_iter: int, X, y, seeds=(0, 1, 2, 3, 4)) -> list[lgb.Booster]:
    """Seed-bagged final models (reduces variance, which matters with ~1k positives)."""
    return [_fit(params, X, y, rounds=max(50, n_iter), seed=s) for s in seeds]


def predict_ensemble(models: list[lgb.Booster], X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)

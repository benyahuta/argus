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
    # Balanced sample (all positives, 25% of negatives) keeps lbfgs fast on 3M rows;
    # class_weight="balanced" already reweights the classes.
    idx = balanced_sample(np.asarray(y_tr), 0.25, np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(design(X_tr.iloc[idx]), np.asarray(y_tr)[idx])
    return pipe.predict_proba(design(X_te))[:, 1]


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------

BASE_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
    scale_pos_weight=20.0, max_cat_to_onehot=4, cat_smooth=50, min_data_per_group=500,
    extra_trees=False, verbose=-1, num_threads=0,
)


def search_space(rng: np.random.Generator) -> dict:
    return dict(
        num_leaves=int(rng.choice([7, 15, 31, 63])),
        min_child_samples=int(rng.choice([50, 200, 500, 2000])),
        feature_fraction=float(rng.choice([0.3, 0.5, 0.7])),
        lambda_l2=float(rng.choice([1.0, 10.0, 100.0])),
        scale_pos_weight=float(rng.choice([1.0, 5.0, 20.0])),
        learning_rate=float(rng.choice([0.02, 0.05])),
        extra_trees=bool(rng.choice([False, True])),
    )


TUNED = ["num_leaves", "min_child_samples", "feature_fraction", "lambda_l2",
         "scale_pos_weight", "learning_rate", "extra_trees"]


@dataclass
class GBMResult:
    params: dict
    best_iter: int
    cv_ap: float
    trials: list[dict] = field(default_factory=list)


def balanced_sample(y: np.ndarray, neg_frac: float, rng: np.random.Generator) -> np.ndarray:
    """All positives plus a random ``neg_frac`` of negatives (balanced bagging)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = rng.choice(neg, int(len(neg) * neg_frac), replace=False)
    return np.sort(np.concatenate([pos, keep]))


def _fit(params: dict, X_tr, y_tr, rounds: int, seed: int = 0, X_ev=None, y_ev=None,
         record: dict | None = None) -> lgb.Booster:
    p = {**params, "seed": seed}
    dtr = lgb.Dataset(X_tr, y_tr)
    if X_ev is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dev = lgb.Dataset(X_ev, y_ev, reference=dtr)
    return lgb.train(p | {"metric": "average_precision"}, dtr, num_boost_round=rounds,
                     valid_sets=[dev], valid_names=["ev"],
                     callbacks=[lgb.record_evaluation(record)])


def tune_gbm_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_trials: int = 16,
                n_folds: int = 3, neg_frac: float = 0.25, max_rounds: int = 2000,
                seed: int = 0, log=print) -> GBMResult:
    """Random search scored by provider-grouped K-fold CV on the training cohorts.

    A single later cohort with ~100 positives proved far too noisy to rank
    configurations, so every config is scored on the mean AP curve over K folds
    (each fold's ~190 positives are providers never seen in training). The number of
    boosting rounds is the argmax of that mean curve, not per-fold early stopping.
    Training folds use balanced bagging; held-out folds keep all positives and the same
    share of negatives, so the AP scale is comparable across configs.
    """
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    folds = list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    trials, best = [], None
    for t in range(n_trials):
        params = BASE_PARAMS | ({} if t == 0 else search_space(rng))
        curves = []
        for k, (tr, ev) in enumerate(folds):
            frng = np.random.default_rng(seed + k)
            tr = tr[balanced_sample(y[tr], neg_frac, frng)]
            ev = ev[balanced_sample(y[ev], neg_frac, frng)]
            rec: dict = {}
            _fit(params, X.iloc[tr], y[tr], max_rounds, seed, X.iloc[ev], y[ev], rec)
            curves.append(np.asarray(rec["ev"]["average_precision"]))
        mean = np.mean(curves, axis=0)
        it = int(np.argmax(mean)) + 1
        fold_aps = [float(c[it - 1]) for c in curves]
        trials.append({**{k: params[k] for k in TUNED}, "best_iter": it,
                       "cv_ap": float(mean[it - 1]), "fold_ap": fold_aps})
        folds_txt = ", ".join(f"{a:.3f}" for a in fold_aps)
        log(f"  trial {t:2d}  CV AP={mean[it - 1]:.4f} (folds {folds_txt})  iters={it}  "
            f"leaves={params['num_leaves']} mcs={params['min_child_samples']} "
            f"spw={params['scale_pos_weight']} et={params.get('extra_trees', False)}")
        if best is None or mean[it - 1] > best.cv_ap:
            best = GBMResult(params, it, float(mean[it - 1]))
    best.trials = trials
    return best


def fit_gbm_ensemble(params: dict, n_iter: int, X, y, seeds=(0, 1, 2, 3, 4),
                     neg_frac: float = 0.25) -> list[lgb.Booster]:
    """Balanced bagging: each member sees all positives and a different negative sample."""
    models = []
    for s in seeds:
        idx = balanced_sample(np.asarray(y), neg_frac, np.random.default_rng(1000 + s))
        Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        models.append(_fit(params, Xs, np.asarray(y)[idx], rounds=max(50, n_iter), seed=s))
    return models


def predict_ensemble(models: list[lgb.Booster], X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)

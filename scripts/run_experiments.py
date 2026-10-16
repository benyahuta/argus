"""Train every model, evaluate on the held-out future cohort, and write all results.

Outputs
  artifacts/test_scores.parquet   per-provider test scores for every model (for figures)
  artifacts/shap_*.npy            SHAP values on a test sample
  reports/results.json            every number quoted in the README
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudml.config import ARTIFACTS, EXP, INTERIM, REPORTS
from fraudml.evaluate import bootstrap_ci, paired_bootstrap_delta, summarize
from fraudml.features import ALL_FEATURES, FEATURE_GROUPS
from fraudml.models import (
    fit_gbm_ensemble,
    isolation_forest,
    logistic,
    predict_ensemble,
    rule_score,
    to_pandas,
    tune_gbm,
)

# Features describing *who the patients are* rather than *how the provider bills*.
# Used in the fairness experiment: a model should not learn "serves poor/disabled
# patients" as a proxy for fraud.
PATIENT_DEMOGRAPHIC = ["bene_under65_share", "bene_dual_share", "bene_female_share",
                       "bene_avg_age", "bene_over84_share"]

T0 = time.time()
# SMOKE=1 runs the whole pipeline on a 3% negative subsample with few trials (for testing).
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    ARTIFACTS, REPORTS = ARTIFACTS / "smoke", ARTIFACTS / "smoke"  # noqa: F811
N_BOOT = 20 if SMOKE else 300


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ds = pl.read_parquet(INTERIM / "dataset.parquet")
    if SMOKE:
        ds = ds.filter((pl.col("label") == 1) | (pl.col("npi").hash(0) % 33 == 0))
    specialties = sorted(ds["specialty"].unique().to_list())

    tr = ds.filter(pl.col("split") == "train")
    va = ds.filter(pl.col("split") == "valid")
    te = ds.filter(pl.col("split") == "test")
    y_tr, y_va, y_te = (d["label"].to_numpy() for d in (tr, va, te))
    log(f"train {tr.height:,} ({y_tr.sum()} pos)  valid {va.height:,} ({y_va.sum()} pos)  "
        f"test {te.height:,} ({y_te.sum()} pos)")

    X = {k: to_pandas(d, ALL_FEATURES, specialties) for k, d in
         (("tr", tr), ("va", va), ("te", te))}
    X_trva = pd.concat([X["tr"], X["va"]], ignore_index=True)
    y_trva = np.concatenate([y_tr, y_va])

    results: dict = {"data": {
        "train": {"n": tr.height, "pos": int(y_tr.sum()),
                  "years": [c.feature_year for c in EXP.train]},
        "valid": {"n": va.height, "pos": int(y_va.sum()), "years": [EXP.valid.feature_year]},
        "test": {"n": te.height, "pos": int(y_te.sum()), "years": [EXP.test.feature_year]},
        "n_features": len(ALL_FEATURES),
    }}
    scores: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ baselines
    log("rule baseline")
    scores["Peer payment rule"] = rule_score(X["te"])
    log("isolation forest")
    scores["Isolation Forest"] = isolation_forest(X_trva, X["te"], seed=EXP.seed)
    log("logistic regression")
    scores["Logistic regression"] = logistic(X_trva, y_trva, X["te"], seed=EXP.seed)

    # ------------------------------------------------------------------ LightGBM
    log("tuning LightGBM on the 2019 validation cohort")
    tuned = tune_gbm(X["tr"], y_tr, X["va"], y_va, n_trials=2 if SMOKE else 16, seed=EXP.seed,
                     log=log)
    log(f"best valid AP {tuned.valid_ap:.4f} at {tuned.best_iter} iters")
    results["tuning"] = {"best_valid_ap": tuned.valid_ap, "best_iter": tuned.best_iter,
                         "params": {k: v for k, v in tuned.params.items() if k != "verbose"},
                         "trials": tuned.trials}
    # Refit on train+valid (labels end 2021, before any test label) with ~20% more rounds
    # for the larger training set.
    n_iter = int(tuned.best_iter * 1.2)
    log("fitting 5-seed LightGBM ensemble on train+valid")
    gbm = fit_gbm_ensemble(tuned.params, n_iter, X_trva, y_trva)
    scores["LightGBM (ours)"] = predict_ensemble(gbm, X["te"])
    for i, m in enumerate(gbm):
        m.save_model(str(ARTIFACTS / f"lgbm_seed{i}.txt"))

    # ------------------------------------------------------------------ headline metrics
    log("metrics + bootstrap CIs")
    results["models"] = {}
    for name, s in scores.items():
        r = summarize(y_te, s, EXP.audit_budgets)
        r["ci"] = bootstrap_ci(y_te, s, n_boot=N_BOOT, seed=EXP.seed)
        results["models"][name] = r
        log(f"  {name:22s} AP={r['ap']:.4f} AUC={r['roc_auc']:.3f} "
            f"recall@1%={r['recall@1pct']:.3f} lift@1%={r['lift@1pct']:.1f}")
    ours = scores["LightGBM (ours)"]
    results["significance"] = {
        name: paired_bootstrap_delta(y_te, ours, s, n_boot=N_BOOT, seed=EXP.seed)
        for name, s in scores.items() if name != "LightGBM (ours)"
    }

    # ------------------------------------------------------------------ ablations
    log("feature-group ablations")
    abl_seeds = (0,) if SMOKE else (0, 1, 2)
    abl = {"all features": average_precision_score(y_te, ours)}
    variants = {f"- {g}": [c for c in X_trva.columns if c not in cols]
                for g, cols in FEATURE_GROUPS.items()}
    variants["summary-file only (volume+meta)"] = FEATURE_GROUPS["volume"] + FEATURE_GROUPS["meta"]
    for name, keep in variants.items():
        m = fit_gbm_ensemble(tuned.params, n_iter, X_trva[keep], y_trva, seeds=abl_seeds)
        abl[name] = average_precision_score(y_te, predict_ensemble(m, X["te"][keep]))
        log(f"  {name:32s}: AP={abl[name]:.4f}")
    results["ablation_ap"] = abl

    # ------------------------------------------------------------------ fairness
    log("fairness audit")
    keep = [c for c in X_trva.columns if c not in PATIENT_DEMOGRAPHIC]
    fair = fit_gbm_ensemble(tuned.params, n_iter, X_trva[keep], y_trva)
    scores["LightGBM (no patient demographics)"] = predict_ensemble(fair, X["te"][keep])
    results["fairness"] = fairness_audit(te, y_te, {
        "LightGBM (ours)": ours,
        "LightGBM (no patient demographics)": scores["LightGBM (no patient demographics)"],
        "Peer payment rule": scores["Peer payment rule"],
    })
    results["fairness_mitigation"] = fairness_tradeoff(te, y_te, ours)
    scores["LightGBM + stratified audit"] = stratified_score(te, ours, alpha=1.0)
    for name in ("LightGBM + stratified audit",):
        r = summarize(y_te, scores[name], EXP.audit_budgets)
        r["ci"] = bootstrap_ci(y_te, scores[name], n_boot=N_BOOT)
        results["models"][name] = r
    r = summarize(y_te, scores["LightGBM (no patient demographics)"], EXP.audit_budgets)
    r["ci"] = bootstrap_ci(y_te, scores["LightGBM (no patient demographics)"], n_boot=N_BOOT)
    results["models"]["LightGBM (no patient demographics)"] = r

    # ------------------------------------------------------------------ dollars + lead time
    log("dollar impact and lead time")
    results["impact"] = impact(te, y_te, scores)

    # ------------------------------------------------------------------ per specialty
    results["by_specialty"] = by_specialty(te, y_te, ours)

    # ------------------------------------------------------------------ SHAP
    log("SHAP")
    import shap
    rng = np.random.default_rng(EXP.seed)
    idx = np.concatenate([np.flatnonzero(y_te == 1),
                          rng.choice(np.flatnonzero(y_te == 0), 20_000, replace=False)])
    Xs = X["te"].iloc[idx]
    if SMOKE:
        idx = idx[:3000]
        Xs = X["te"].iloc[idx]
    sv = np.mean([shap.TreeExplainer(m).shap_values(Xs) for m in gbm], axis=0)
    np.save(ARTIFACTS / "shap_values.npy", sv)
    Xs.assign(label=y_te[idx], score=ours[idx]).to_parquet(ARTIFACTS / "shap_sample.parquet")
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    results["shap_importance"] = imp.head(25).round(5).to_dict()
    results["shap_group_importance"] = {
        g: float(imp[[c for c in cols if c in imp.index]].sum())
        for g, cols in FEATURE_GROUPS.items()
    }

    # ------------------------------------------------------------------ save
    out = te.select("npi", "specialty", "state", "label", "excl_date", "excl_type",
                    "log_payment", "bene_dual_share", "ruca_code", "bene_under65_share")
    out = out.with_columns([pl.Series(k, v) for k, v in scores.items()])
    out.write_parquet(ARTIFACTS / "test_scores.parquet")
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log("done")


def audit_groups(te: pl.DataFrame) -> dict[str, np.ndarray]:
    dual = te["bene_dual_share"].to_numpy()
    u65 = te["bene_under65_share"].to_numpy()
    ruca = te["ruca_code"].to_numpy()
    return {
        "all providers": np.ones(te.height, bool),
        "high dual-eligible share (top quartile)": dual >= np.nanquantile(dual, 0.75),
        "high disabled (<65) share (top quartile)": u65 >= np.nanquantile(u65, 0.75),
        "rural (RUCA >= 4)": ruca >= 4,
        "urban (RUCA < 4)": ruca < 4,
    }


def patient_mix_strata(te: pl.DataFrame) -> np.ndarray:
    """Dual-eligible quartile x disabled quartile (missing = its own bin): 25 strata."""
    def q(x):
        edges = np.nanquantile(x, [0.25, 0.5, 0.75])
        return np.where(np.isnan(x), 4, np.searchsorted(edges, x, side="right"))
    return q(te["bene_dual_share"].to_numpy()) * 5 + q(te["bene_under65_share"].to_numpy())


def stratified_score(te: pl.DataFrame, s: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the global percentile with the within-patient-mix-stratum percentile.

    alpha=0 is the raw model ranking; alpha=1 audits every stratum at the same rate, so no
    group of providers is reviewed more often just because of who their patients are.
    """
    strata = patient_mix_strata(te)
    g = pd.Series(s).rank(pct=True).to_numpy()
    w = pd.Series(s).groupby(strata).rank(pct=True).to_numpy()
    return (1 - alpha) * g + alpha * w + 1e-9 * g  # tiny global tie-break


def fairness_tradeoff(te: pl.DataFrame, y: np.ndarray, s: np.ndarray,
                      alphas=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    rows = []
    for a in alphas:
        sa = stratified_score(te, s, a)
        fa = fairness_audit(te, y, {"m": sa})["m"]
        r = summarize(y, sa, EXP.audit_budgets)
        rows.append({
            "alpha": a, "ap": r["ap"], "recall@1pct": r["recall@1pct"],
            "fpr_ratio": {g: v["fpr_ratio_vs_all"] for g, v in fa.items()},
            "max_fpr_ratio": max(v["fpr_ratio_vs_all"] for v in fa.values()),
        })
        log(f"  alpha={a:.2f}  recall@1%={r['recall@1pct']:.3f}  "
            f"max FPR ratio={rows[-1]['max_fpr_ratio']:.2f}")
    return rows


def fairness_audit(te: pl.DataFrame, y: np.ndarray, models: dict[str, np.ndarray],
                   budget: float = 0.01) -> dict:
    """Flag rates and false-positive rates at a 1% audit budget for vulnerable-patient groups.

    The concern: a fraud model could learn that serving poor, disabled or rural patients is
    'suspicious', steering audits at the providers those patients depend on.
    """
    groups = audit_groups(te)
    out = {}
    for name, s in models.items():
        k = int(round(budget * len(y)))
        flagged = np.zeros(len(y), bool)
        flagged[np.argpartition(-s, k - 1)[:k]] = True
        rows = {}
        for g, mask in groups.items():
            neg = mask & (y == 0)
            pos = mask & (y == 1)
            rows[g] = {
                "n": int(mask.sum()),
                "flag_rate": float(flagged[mask].mean()),
                "false_positive_rate": float(flagged[neg].mean()),
                "recall": float(flagged[pos].mean()) if pos.any() else None,
                "prevalence": float(y[mask].mean()),
            }
        base = rows["all providers"]["false_positive_rate"]
        for g in rows:
            rows[g]["fpr_ratio_vs_all"] = rows[g]["false_positive_rate"] / base
        out[name] = rows
    return out


def impact(te: pl.DataFrame, y: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    """Medicare dollars (standardized) paid in the test year to providers later excluded."""
    pay = np.exp(te["log_payment"].to_numpy()) - 1
    total_pos_pay = float(pay[y == 1].sum())
    excl = te["excl_date"].to_numpy()
    out = {"total_payments_test_year": float(pay.sum()),
           "payments_to_future_excluded": total_pos_pay, "models": {}}
    for name, s in scores.items():
        row = {}
        for b in EXP.audit_budgets:
            k = int(round(b * len(y)))
            top = np.argpartition(-s, k - 1)[:k]
            hit = top[y[top] == 1]
            row[f"{b * 100:g}pct"] = {
                "providers_reviewed": k,
                "future_excluded_found": int(len(hit)),
                "their_payments": float(pay[hit].sum()),
                "share_of_future_excluded_payments": float(pay[hit].sum() / total_pos_pay),
            }
            if b == 0.01 and len(hit):
                d = pd.to_datetime(pd.Series(excl[hit]).astype(int).astype(str))
                lead = (d - pd.Timestamp(f"{EXP.test.feature_year}-12-31")).dt.days / 30.4
                row["lead_time_months_at_1pct"] = {
                    "median": float(lead.median()), "q25": float(lead.quantile(0.25)),
                    "q75": float(lead.quantile(0.75)),
                }
        out["models"][name] = row
    return out


def by_specialty(te: pl.DataFrame, y: np.ndarray, s: np.ndarray, min_pos: int = 8) -> list:
    spec = te["specialty"].to_numpy()
    rows = []
    for sp in np.unique(spec):
        m = spec == sp
        if y[m].sum() >= min_pos and (y[m] == 0).sum() > 0:
            rows.append({"specialty": sp, "n": int(m.sum()), "pos": int(y[m].sum()),
                         "roc_auc": float(roc_auc_score(y[m], s[m])),
                         "ap": float(average_precision_score(y[m], s[m])),
                         "prevalence": float(y[m].mean()),
                         "recall@1pct_global": float(_global_recall(y, s, m))})
    return sorted(rows, key=lambda r: -r["pos"])


def _global_recall(y, s, mask, budget=0.01):
    k = int(round(budget * len(y)))
    thr = np.partition(s, len(s) - k)[len(s) - k]
    return ((s >= thr) & (y == 1) & mask).sum() / max(1, (y[mask] == 1).sum())


if __name__ == "__main__":
    main()

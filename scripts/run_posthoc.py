"""Post-hoc analyses on saved test scores (no retraining).

* Dollar-weighted queue: rank by risk × Medicare payments, i.e. expected dollars at
  stake, instead of risk alone. Probability ranking finds the most cases; dollar ranking
  finds the most money. An investigator would pick based on mission.
* Calibration of the bagged LightGBM probabilities (after undoing negative subsampling).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from fraudml.config import ARTIFACTS, EXP, REPORTS

NEG_FRAC = 0.25  # balanced-bagging negative sampling rate used in training


def main() -> None:
    df = pl.read_parquet(ARTIFACTS / "test_scores.parquet")
    y = df["label"].to_numpy()
    pay = np.exp(df["log_payment"].to_numpy()) - 1
    s = df["LightGBM (ours)"].to_numpy()
    # Undo the prior shift from training on 25% of negatives: odds_true = odds_model * frac.
    odds = s / (1 - s) * NEG_FRAC
    p = odds / (1 + odds)

    # Two very large labs dominate the dollar totals; report shares without them too.
    big = np.argsort(-(pay * y))[:2]
    robust = np.ones(len(y), bool)
    robust[big] = False
    out = {"queues": {}, "top2_positive_payments": [float(pay[i]) for i in big],
           "top2_share_of_excluded_dollars": float(pay[big].sum() / pay[y == 1].sum())}
    for name, score in (("risk", p), ("risk × dollars", p * pay), ("dollars only", pay)):
        rows = {}
        order = np.argsort(-score)
        for b in EXP.audit_budgets:
            k = int(round(b * len(y)))
            top = order[:k]
            hit = top[y[top] == 1]
            hit_r = hit[robust[hit]]
            rows[f"{b * 100:g}pct"] = {
                "dollar_share_excl_top2": float(pay[hit_r].sum()
                                                / pay[(y == 1) & robust].sum()),
                "found": int(len(hit)), "recall": float(len(hit) / y.sum()),
                "dollars_found": float(pay[hit].sum()),
                "dollar_share": float(pay[hit].sum() / pay[y == 1].sum()),
                "dollars_reviewed": float(pay[top].sum()),
            }
        out["queues"][name] = rows

    # Calibration: deciles of predicted probability vs. observed rate.
    q = np.quantile(p, np.linspace(0, 1, 11))
    q[-1] += 1e-12
    bins = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    out["calibration_deciles"] = [
        {"decile": int(d + 1), "mean_pred": float(p[bins == d].mean()),
         "observed": float(y[bins == d].mean()), "n": int((bins == d).sum())}
        for d in range(10)
    ]
    top = np.argsort(-p)[: int(0.01 * len(p))]
    out["top1pct_mean_pred"] = float(p[top].mean())
    out["top1pct_observed"] = float(y[top].mean())
    out["expected_vs_observed_total"] = [float(p.sum()), int(y.sum())]
    # Recommended configuration: drop patient demographics, then stratify the audit queue.
    from run_experiments import fairness_audit, stratified_score

    from fraudml.evaluate import summarize
    nd = df["LightGBM (no patient demographics)"].to_numpy()
    variants = {
        "LightGBM (ours)": s,
        "No demographics": nd,
        "No demographics + stratified α=0.5": stratified_score(df, nd, 0.5),
        "No demographics + stratified α=0.75": stratified_score(df, nd, 0.75),
        "No demographics + stratified α=1": stratified_score(df, nd, 1.0),
    }
    fa = fairness_audit(df, y, variants)
    out["recommended"] = {
        name: {"recall@1pct": summarize(y, sc)["recall@1pct"], "ap": summarize(y, sc)["ap"],
               "fpr_ratio": {g: v["fpr_ratio_vs_all"] for g, v in fa[name].items()},
               "max_fpr_ratio": max(v["fpr_ratio_vs_all"] for v in fa[name].values())}
        for name, sc in variants.items()
    }
    out["tradeoff_no_demographics"] = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        sa = stratified_score(df, nd, a)
        f = fairness_audit(df, y, {"m": sa})["m"]
        out["tradeoff_no_demographics"].append({
            "alpha": a, "recall@1pct": summarize(y, sa)["recall@1pct"],
            "max_fpr_ratio": max(v["fpr_ratio_vs_all"] for v in f.values())})
    for name, r in out["recommended"].items():
        print(f"{name:38s} recall@1%={r['recall@1pct']:.3f} AP={r['ap']:.4f} "
              f"worst FPR ratio={r['max_fpr_ratio']:.2f}")
    (REPORTS / "posthoc.json").write_text(json.dumps(out, indent=2))
    for name, rows in out["queues"].items():
        r = rows["1pct"]
        print(f"{name:15s} @1%: found {r['found']:3d} ({r['recall']:.1%}), "
              f"${r['dollars_found'] / 1e6:.1f}M ({r['dollar_share']:.0%} of excluded-provider $; "
              f"{r['dollar_share_excl_top2']:.0%} excluding the 2 largest)")
    print("calibration: expected", round(p.sum(), 1), "observed", int(y.sum()),
          "| top-1% predicted", round(out["top1pct_mean_pred"], 5),
          "observed", round(out["top1pct_observed"], 5))


if __name__ == "__main__":
    main()

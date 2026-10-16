"""Export aggregate curves for the interactive results page (docs/index.html).

Only aggregate statistics leave this script; no NPI or per-provider score is written.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import polars as pl

from fraudml.config import ARTIFACTS, REPORTS, ROOT

SMOKE = os.environ.get("SMOKE") == "1"
SRC = ARTIFACTS / "smoke" if SMOKE else ARTIFACTS
RES = (ARTIFACTS / "smoke" if SMOKE else REPORTS)
MODELS = ["LightGBM (ours)", "LightGBM + LR blend", "Logistic regression", "Isolation Forest",
          "Peer payment rule"]
BUDGETS = np.unique(np.round(np.logspace(np.log10(0.0005), np.log10(0.10), 120), 6))


def strata(df: pd.DataFrame) -> np.ndarray:
    def q(x):
        edges = np.nanquantile(x, [0.25, 0.5, 0.75])
        return np.where(np.isnan(x), 4, np.searchsorted(edges, x, side="right"))
    return q(df["bene_dual_share"].to_numpy()) * 5 + q(df["bene_under65_share"].to_numpy())


def curves(y, s, pay):
    order = np.argsort(-s, kind="stable")
    hits = np.cumsum(y[order])
    dollars = np.cumsum((pay * y)[order])
    n, P, D = len(y), y.sum(), (pay * y).sum()
    out = []
    for b in BUDGETS:
        k = max(1, int(round(b * n)))
        out.append({"b": float(b), "k": k, "hits": int(hits[k - 1]),
                    "recall": float(hits[k - 1] / P), "precision": float(hits[k - 1] / k),
                    "lift": float(hits[k - 1] / k / (P / n)),
                    "dollars": float(dollars[k - 1]), "dollar_share": float(dollars[k - 1] / D)})
    return out


ALPHAS = np.linspace(0, 1, 11)


def fairness_sweep(df, y, s, alphas=ALPHAS, budget=0.01):
    st = strata(df)
    g_rank = pd.Series(s).rank(pct=True).to_numpy()
    w_rank = pd.Series(s).groupby(st).rank(pct=True).to_numpy()
    dual = df["bene_dual_share"].to_numpy()
    u65 = df["bene_under65_share"].to_numpy()
    ruca = df["ruca_code"].to_numpy()
    groups = {
        "Low-income patients (top-quartile dual-eligible share)": dual >= np.nanquantile(dual, .75),
        "Disabled patients (top-quartile under-65 share)": u65 >= np.nanquantile(u65, .75),
        "Rural providers": ruca >= 4,
        "Urban providers": ruca < 4,
    }
    k = int(round(budget * len(y)))
    base_neg = y == 0
    rows = []
    for a in alphas:
        sc = (1 - a) * g_rank + a * w_rank + 1e-9 * g_rank
        flag = np.zeros(len(y), bool)
        flag[np.argpartition(-sc, k - 1)[:k]] = True
        fpr_all = flag[base_neg].mean()
        rows.append({
            "alpha": round(float(a), 2),
            "recall": float(flag[y == 1].mean()),
            "groups": {g: float(flag[m & base_neg].mean() / fpr_all) for g, m in groups.items()},
        })
    return rows


def main() -> None:
    df = pl.read_parquet(SRC / "test_scores.parquet").to_pandas()
    res = json.loads((RES / "results.json").read_text())
    y = df["label"].to_numpy()
    pay = np.exp(df["log_payment"].to_numpy()) - 1
    data = {
        "n": int(len(y)), "positives": int(y.sum()),
        "total_paid": float(pay.sum()), "paid_to_future_excluded": float(pay[y == 1].sum()),
        "models": {m: res["models"][m] | {"curve": curves(y, df[m].to_numpy(), pay)}
                   for m in MODELS if m in df},
        "fairness": fairness_sweep(df, y, df["LightGBM (ours)"].to_numpy()),
        "ablation": res["ablation_ap"],
        "shap_groups": res.get("shap_group_importance", {}),
        "impact": res["impact"]["models"].get("LightGBM (ours)", {}),
    }
    for extra in ("rolling.json", "hybrid.json", "deep_results.json"):
        p = RES / extra
        if p.exists():
            data[extra.removesuffix(".json")] = json.loads(p.read_text())
    out = ROOT / "docs" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.RESULTS = " + json.dumps(data, default=float) + ";\n")
    print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

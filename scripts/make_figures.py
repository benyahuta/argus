"""Render README figures from artifacts/ and reports/results.json."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraudml.config import ARTIFACTS, FIGURES, INTERIM, REPORTS  # noqa: E402

if os.environ.get("SMOKE") == "1":
    ARTIFACTS = REPORTS = FIGURES = ARTIFACTS / "smoke"  # noqa: F811

# Reference palette (dataviz skill): our model is the one saturated hue; baselines recede.
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
ORANGE = "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SURFACE = "#fcfcfb"
BASELINE_STYLE = {
    "Logistic regression": dict(color="#52514e", ls="-"),
    "Isolation Forest": dict(color="#8a8984", ls="--"),
    "Peer payment rule": dict(color="#b5b4ae", ls=":"),
}
OURS = "LightGBM (ours)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def recall_at_budget(df: pd.DataFrame, res: dict):
    y = df["label"].to_numpy()
    budgets = np.logspace(-4, -1, 60)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for name in [*BASELINE_STYLE, OURS]:
        s = df[name].to_numpy()
        order = np.argsort(-s)
        cum = np.cumsum(y[order]) / y.sum()
        rec = [cum[max(0, int(b * len(y)) - 1)] for b in budgets]
        style = BASELINE_STYLE.get(name, dict(color=BLUE, ls="-"))
        ax.plot(budgets * 100, rec, lw=2.4 if name == OURS else 1.8, label=name, **style)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1, 10], ["0.01%", "0.1%", "1%", "10%"])
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    r1 = res["models"][OURS]["recall@1pct"]
    ax.annotate(f"Audit 1% of providers\n→ catch {r1:.0%} of future exclusions",
                (1, r1), xytext=(-150, 18), textcoords="offset points", fontsize=9.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("Share of providers sent for review (log scale)")
    ax.set_ylabel("Share of future exclusions caught")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Recall vs. audit budget  ·  2021 cohort, exclusions 2022–24 "
                 f"(n={len(y):,}, {int(y.sum())} positives)")
    save(fig, "recall_at_budget.png")


def pr_curves(df: pd.DataFrame, res: dict):
    y = df["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for name in [*BASELINE_STYLE, OURS]:
        p, r, _ = precision_recall_curve(y, df[name].to_numpy())
        style = BASELINE_STYLE.get(name, dict(color=BLUE, ls="-"))
        ap = res["models"][name]["ap"]
        ax.plot(r, p, lw=2.4 if name == OURS else 1.6, label=f"{name}  (AP {ap:.3f})", **style)
    ax.axhline(y.mean(), color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(0.99, y.mean() * 1.25, f"random = {y.mean():.4f}", ha="right", fontsize=8.5,
            color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision (log scale)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Precision–recall on the held-out future cohort")
    save(fig, "pr_curves.png")


def ablation(res: dict):
    abl = res["ablation_ap"]
    full = abl["all features"]
    items = [(k, v) for k, v in abl.items() if k != "all features"]
    items.sort(key=lambda kv: kv[1])
    labels = [k.replace("- ", "without ") for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(labels, vals, color=BLUE_LIGHT, height=0.6)
    ax.axvline(full, color=BLUE, lw=2)
    ax.text(full, -0.9, f" all features: {full:.3f}", color=INK, fontsize=9, va="center")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f} ({(v - full) / full:+.0%})", va="center", fontsize=8.5,
                color=INK2)
    ax.set_xlabel("Test average precision (single seed)")
    ax.set_title("Feature-group ablation: what the model relies on", pad=14)
    ax.set_ylim(-1.3, len(items) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals + [full]) * 1.3)
    save(fig, "ablation.png")


def shap_summary():
    import shap
    sv = np.load(ARTIFACTS / "shap_values.npy")
    Xs = pd.read_parquet(ARTIFACTS / "shap_sample.parquet").drop(columns=["label", "score"])
    if "specialty_cat" in Xs:
        Xs["specialty_cat"] = Xs["specialty_cat"].cat.codes
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=15, show=False, plot_size=(7.5, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP: top-15 features (test sample)", loc="left")
    save(fig, "shap_summary.png")


def fairness(res: dict):
    fa = res["fairness"]
    groups = [g for g in fa[OURS] if g != "all providers"]
    models = [OURS, "LightGBM (no patient demographics)", "LightGBM + stratified audit",
              "Peer payment rule"]
    models = [m for m in models if m in fa]
    colors = [BLUE, ORANGE, "#1baf7a", "#b5b4ae"][:len(models)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.8 / len(models)
    yv = np.arange(len(groups))
    for i, (m, c) in enumerate(zip(models, colors, strict=True)):
        vals = [fa[m][g]["fpr_ratio_vs_all"] for g in groups]
        ax.barh(yv + (i - (len(models) - 1) / 2) * w, vals, height=w - 0.03, color=c, label=m)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(yv, groups)
    ax.invert_yaxis()
    ax.set_xlabel("False-positive rate relative to all providers (1.0 = parity)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Who gets wrongly flagged? (1% audit budget)")
    ax.grid(axis="y", visible=False)
    save(fig, "fairness.png")


def fairness_tradeoff(res: dict):
    rows = res["fairness_mitigation"]
    x = [r["max_fpr_ratio"] for r in rows]
    y = [r["recall@1pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    for r, xi, yi in zip(rows, x, y, strict=True):
        lab = {0.0: "α=0 (raw model)", 1.0: "α=1 (fully stratified)"}.get(r["alpha"],
                                                                        f"α={r['alpha']:g}")
        ax.annotate(lab, (xi, yi), xytext=(8, 4), textcoords="offset points", fontsize=9,
                    color=INK2)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.set_xlabel("Worst group false-positive rate ÷ overall (1.0 = parity)")
    ax.set_ylabel("Recall at 1% audit budget")
    ax.set_xlim(0.9, max(x) * 1.25)
    ax.set_title("Fairness–recall trade-off of stratified audit allocation")
    save(fig, "fairness_tradeoff.png")


def rolling(res_path):
    if not res_path.exists():
        return
    rows = json.loads(res_path.read_text())
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for name, style in ((OURS, dict(color=BLUE, lw=2.2)),
                        ("Peer payment rule", dict(color="#8a8984", lw=1.6, ls=":"))):
        rr = [r for r in rows if r["model"] == name]
        yrs = [r["test_year"] for r in rr]
        rec = [r["recall@1pct"] for r in rr]
        lo = [r["ci"]["recall@1pct"][0] for r in rr]
        hi = [r["ci"]["recall@1pct"][1] for r in rr]
        ax.fill_between(yrs, lo, hi, color=style["color"], alpha=0.12, lw=0)
        ax.plot(yrs, rec, marker="o", ms=7, mec=SURFACE, mew=2, label=name, **style)
    ax.set_xticks(yrs, [f"{y}\n(excl. {y + 1}–{str(y + 3)[2:]})" for y in yrs])
    ax.set_ylabel("Recall at 1% audit budget")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.set_title("Rolling-origin evaluation: one model per test year (95% CI)")
    save(fig, "rolling.png")


def lead_time(df: pd.DataFrame):
    y = df["label"].to_numpy()
    s = df[OURS].to_numpy()
    k = int(0.01 * len(y))
    flagged = np.zeros(len(y), bool)
    flagged[np.argpartition(-s, k - 1)[:k]] = True
    d = pd.to_datetime(df["excl_date"].dropna().astype(int).astype(str))
    pos = df[df["label"] == 1].assign(date=d, flagged=flagged[df["label"].to_numpy() == 1])
    months = (pos["date"] - pd.Timestamp("2021-12-31")).dt.days / 30.4
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.arange(0, 37, 3)
    ax.hist(months[~pos["flagged"]], bins=bins, color=GRID, label="missed", rwidth=0.9)
    ax.hist(months[pos["flagged"]], bins=bins, color=BLUE, label="flagged in top 1%",
            rwidth=0.9)
    ax.set_xlabel("Months between end of the scored year (Dec 2021) and OIG exclusion")
    ax.set_ylabel("Providers")
    ax.legend()
    ax.set_title("Early warning: how far ahead of exclusion the flag fires")
    save(fig, "lead_time.png")


def em_upcoding():
    ds = pl.read_parquet(INTERIM / "dataset.parquet",
                         columns=["label", "split", "em_est_share_level45", "specialty"])
    ds = ds.filter((pl.col("split") == "test") & pl.col("em_est_share_level45").is_not_null()
                   & pl.col("specialty").is_in(["Internal Medicine", "Family Practice"]))
    neg = ds.filter(pl.col("label") == 0)["em_est_share_level45"].to_numpy()
    pos = ds.filter(pl.col("label") == 1)["em_est_share_level45"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 26)
    ax.hist(neg, bins=bins, density=True, color=BLUE_LIGHT, label=f"all others (n={len(neg):,})")
    ax.hist(pos, bins=bins, density=True, histtype="step", lw=2.2, color=ORANGE,
            label=f"excluded 2022–24 (n={len(pos)})")
    ax.set_xlabel("Share of established-patient office visits billed at level 4 or 5")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Upcoding signal: internal & family medicine, 2021")
    save(fig, "em_upcoding.png")


def main():
    res = json.loads((REPORTS / "results.json").read_text())
    df = pl.read_parquet(ARTIFACTS / "test_scores.parquet").to_pandas()
    recall_at_budget(df, res)
    pr_curves(df, res)
    ablation(res)
    fairness(res)
    fairness_tradeoff(res)
    rolling(REPORTS / "rolling.json")
    lead_time(df)
    em_upcoding()
    shap_summary()


if __name__ == "__main__":
    main()

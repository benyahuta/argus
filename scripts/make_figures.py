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
    ax.text(full, -0.9, f" all features: {full:.4f}", color=INK, fontsize=9, va="center")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f} ({(v - full) / full:+.0%})", va="center", fontsize=8.5,
                color=INK2)
    ax.set_xlabel("Test average precision (3-seed LightGBM; random ranking = 0.0003)")
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
    Xs.columns = [PRETTY.get(c, c) for c in Xs.columns]
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=15, show=False, plot_size=(7.5, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP: top-15 features (test sample)", loc="left")
    save(fig, "shap_summary.png")


def fairness(res: dict):
    ph = json.loads((REPORTS / "posthoc.json").read_text())["recommended"]
    rule = res["fairness"]["Peer payment rule"]
    groups = [g for g in rule if g != "all providers"]
    models = {
        "LightGBM (ours)": (ph["LightGBM (ours)"], BLUE),
        "No patient demographics": (ph["No demographics"], BLUE_LIGHT),
        "No demographics + stratified α=0.75": (ph["No demographics + stratified α=0.75"],
                                               "#1baf7a"),
        "Peer payment rule": ({"fpr_ratio": {g: v["fpr_ratio_vs_all"] for g, v in rule.items()},
                               "recall@1pct": res["models"]["Peer payment rule"]["recall@1pct"]},
                              "#b5b4ae"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    w = 0.8 / len(models)
    yv = np.arange(len(groups))
    for i, (m, (r, c)) in enumerate(models.items()):
        vals = [r["fpr_ratio"][g] for g in groups]
        ax.barh(yv + (i - (len(models) - 1) / 2) * w, vals, height=w - 0.03, color=c,
                label=f"{m}  (recall@1% {r['recall@1pct']:.0%})")
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(yv, groups)
    ax.invert_yaxis()
    ax.set_xlabel("False-positive rate relative to all providers (1.0 = parity)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Who gets wrongly flagged? (1% audit budget)")
    ax.grid(axis="y", visible=False)
    save(fig, "fairness.png")


def fairness_tradeoff(res: dict):
    ph = json.loads((REPORTS / "posthoc.json").read_text())
    curves = {
        "all features": ([r["max_fpr_ratio"] for r in res["fairness_mitigation"]],
                         [r["recall@1pct"] for r in res["fairness_mitigation"]],
                         [r["alpha"] for r in res["fairness_mitigation"]], BLUE),
        "no patient demographics": ([r["max_fpr_ratio"] for r in ph["tradeoff_no_demographics"]],
                                    [r["recall@1pct"] for r in ph["tradeoff_no_demographics"]],
                                    [r["alpha"] for r in ph["tradeoff_no_demographics"]],
                                    "#1baf7a"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, (x, y, a, c) in curves.items():
        ax.plot(x, y, color=c, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
        for xi, yi, ai in zip(x, y, a, strict=True):
            if ai in (0.0, 0.5, 1.0):
                ax.annotate(f"α={ai:g}", (xi, yi), xytext=(7, 5), textcoords="offset points",
                            fontsize=8.5, color=INK2)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.set_xlabel("Worst-group false-positive rate ÷ overall (1.0 = parity)")
    ax.set_ylabel("Recall at 1% audit budget")
    ax.legend(loc="lower right", title="LightGBM trained on", fontsize=9, title_fontsize=9)
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
    ax.hist([months[pos["flagged"]], months[~pos["flagged"]]], bins=bins, stacked=True,
            color=[BLUE, GRID], label=["flagged in top 1%", "missed"], rwidth=0.9)
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
    ax.set_title("E/M level mix, internal & family medicine (2021): later-excluded providers\n"
                 "cluster at the extremes, not just at heavy level 4–5 billing", fontsize=11)
    save(fig, "em_upcoding.png")


PRETTY = {
    "peer_z_log_pay_per_bene": "payment per patient vs. peers (z)",
    "peer_z_srv_per_bene": "services per patient vs. peers (z)",
    "svc_intensity_excess": "per-code intensity vs. peers (log ratio)",
    "em_est_share_level45": "share of office visits billed level 4–5",
    "peer_z_em_est_share_level45": "level 4–5 visit share vs. peers (z)",
    "rx_opioid_share": "opioid share of prescriptions",
    "peer_z_rx_opioid_share": "opioid share vs. peers (z)",
    "mismatch_surprise": "code mix unlike declared specialty",
    "charge_markup": "charge / allowed markup",
    "log_pay_per_bene": "log payment per patient",
    "specialty_cat": "specialty",
    "svc_pay_excess": "allowed price per code vs. peers",
    "svc_facility_share": "share of payments in facility settings",
    "rx_lis_claim_share": "low-income-subsidy share of prescriptions",
    "rx_claims_per_bene": "prescriptions per patient",
    "bene_avg_age": "average patient age",
    "bene_dual_share": "dual-eligible patient share",
    "bene_under65_share": "under-65 (disabled) patient share",
    "bene_risk_score": "patient risk score",
    "is_organization": "is an organization",
    "log_benes": "log number of patients",
    "log_payment": "log total payment",
    "log_services": "log total services",
    "srv_per_bene": "services per patient",
    "svc_top_code_share": "share of payment from top code",
    "svc_pay_hhi": "payment concentration (HHI)",
    "svc_rare_code_pay_share": "payment share from rare codes",
    "rx_log_cost_per_claim": "log drug cost per prescription",
    "rx_brand_claim_share": "brand-name share of prescriptions",
    "rx_opioid_la_share": "long-acting opioid share",
    "rx_present": "prescribes under Part D",
    "peer_group_log_size": "log size of specialty peer group",
    "ruca_code": "rurality (RUCA)",
    "drug_pay_share": "drug share of Part B payment",
    "codes_per_bene": "distinct codes per patient",
    "rx_opioid_days_per_bene": "opioid days supplied per patient",
    "bene_female_share": "female patient share",
    "cc_cancer6": "patients with cancer (%)",
    "cc_alz": "patients with dementia (%)",
    "cc_depress": "patients with depression (%)",
    "cc_diabetes": "patients with diabetes (%)",
    "rx_antipsychotic_65_share": "antipsychotic share (65+ patients)",
    "peer_z_charge_markup": "charge markup vs. peers (z)",
    "peer_z_codes_per_bene": "codes per patient vs. peers (z)",
    "rx_antibiotic_share": "antibiotic share of prescriptions",
    "rx_days_per_claim": "days supplied per prescription",
    "svc_srv_per_bene_day": "services per patient-day",
    "svc_n_lines": "number of billed code lines",
    "mismatch_entropy": "specialty-prediction entropy",
    "em_est_mean_level": "mean office-visit level",
    "peer_z_bene_risk_score": "patient risk vs. peers (z)",
}


def case_study():
    """Anonymized explanation for the highest-ranked test provider who was later excluded."""
    import shap
    sv = np.load(ARTIFACTS / "shap_values.npy")
    S = pd.read_parquet(ARTIFACTS / "shap_sample.parquet")
    X = S.drop(columns=["label", "score"])
    import lightgbm as lgb
    pos = np.flatnonzero(S["label"].to_numpy() == 1)
    i = pos[np.argmax(S["score"].to_numpy()[pos])]
    data = X.iloc[i].astype(object).to_numpy()
    names = [PRETTY.get(c, c) for c in X.columns]
    base = np.mean([shap.TreeExplainer(lgb.Booster(model_file=str(f))).expected_value
                    for f in sorted(ARTIFACTS.glob("lgbm_seed*.txt"))])
    # Values rounded to 2 significant figures so the public files can't be searched for an exact
    # match (re-identification).
    data = np.array([v if isinstance(v, str) else float(f"{float(v):.2g}") for v in data],
                    dtype=object)
    ex = shap.Explanation(values=sv[i], base_values=float(base), data=data, feature_names=names)
    plt.figure()
    shap.plots.waterfall(ex, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.axes[0].set_title("Why one later-excluded provider ranked near the top "
                          "(log-odds contributions)", loc="left", fontsize=11)
    save(fig, "case_study.png")


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
    case_study()


if __name__ == "__main__":
    main()

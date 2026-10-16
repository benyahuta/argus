"""Render reports/RESULTS.md (markdown tables) from the JSON results."""

from __future__ import annotations

import json

from fraudml.config import REPORTS


def ci(r, k):
    lo, hi = r["ci"][k]
    return f"{r[k]:.3f} ({lo:.3f}–{hi:.3f})"


def main() -> None:
    res = json.loads((REPORTS / "results.json").read_text())
    d = res["data"]
    out = ["# Results", "",
           f"Test cohort: {d['test']['n']:,} providers billing Medicare in "
           f"{d['test']['years'][0]}, "
           f"of whom {d['test']['pos']} were excluded by OIG in the following three years "
           f"(prevalence {d['test']['pos'] / d['test']['n']:.4%}). 95% CIs from a stratified "
           "bootstrap.", "",
           "## Headline comparison", "",
           "| Model | AP | ROC-AUC | Recall @0.1% | Recall @1% | Lift @1% | Recall @5% |",
           "|---|---|---|---|---|---|---|"]
    order = ["Peer payment rule", "Isolation Forest", "Logistic regression", "LightGBM (ours)",
             "LightGBM + LR blend",
             "LightGBM (no patient demographics)", "LightGBM + stratified audit"]
    for m in order:
        if m not in res["models"]:
            continue
        r = res["models"][m]
        out.append(f"| {m} | {ci(r, 'ap')} | {ci(r, 'roc_auc')} | {r['recall@0.1pct']:.3f} | "
                   f"{ci(r, 'recall@1pct')} | {r['lift@1pct']:.1f}× | {r['recall@5pct']:.3f} |")
    deep = REPORTS / "deep_results.json"
    if deep.exists():
        dr = json.loads(deep.read_text())
        out.append(f"| ClaimSet Transformer (end-to-end) | {dr['test_ap']:.3f} | "
                   f"{dr['test_auc']:.3f} | | | | |")
    hyb = REPORTS / "hybrid.json"
    if hyb.exists():
        h = json.loads(hyb.read_text())
        out.append(f"| LightGBM + ClaimSet embeddings (3 seeds) | {h['hybrid']['ap']:.3f} | "
                   f"{h['hybrid']['roc_auc']:.3f} | {h['hybrid']['recall@0.1pct']:.3f} | "
                   f"{h['hybrid']['recall@1pct']:.3f} | {h['hybrid']['lift@1pct']:.1f}× | "
                   f"{h['hybrid']['recall@5pct']:.3f} |")
        dd = h["delta_hybrid_minus_base"]
        out += ["", f"Hybrid vs. same-seed LightGBM baseline (AP {h['base']['ap']:.3f}): "
                f"ΔAP = {dd['delta_ap']:+.4f} (95% CI {dd['lo']:+.4f} to {dd['hi']:+.4f})."]

    out += ["", "## Paired bootstrap: LightGBM (ours) minus baseline, ΔAP", "",
            "| Baseline | ΔAP | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|"]
    for m, s in res["significance"].items():
        out.append(f"| {m} | {s['delta_ap']:+.4f} | {s['lo']:+.4f} to {s['hi']:+.4f} | "
                   f"{s['p_le_0']:.3f} |")

    out += ["", "## Feature-group ablation (test AP, 3-seed LightGBM)", "",
            "| Variant | AP | Δ vs. all |", "|---|---|---|"]
    full = res["ablation_ap"]["all features"]
    for k, v in sorted(res["ablation_ap"].items(), key=lambda kv: kv[1]):
        out.append(f"| {k} | {v:.4f} | {(v - full) / full:+.0%} |")

    imp = res["impact"]
    out += ["", "## Dollars", "",
            f"Standardized Medicare payments in the test year: "
            f"${imp['total_payments_test_year'] / 1e9:.1f}B in total, "
            f"${imp['payments_to_future_excluded'] / 1e6:.1f}M to providers later excluded.", "",
            "| Budget | Providers reviewed | Later-excluded found | Their payments "
            "| Share of those $ |",
            "|---|---|---|---|---|"]
    for b, r in imp["models"]["LightGBM (ours)"].items():
        if "providers_reviewed" not in r:
            continue
        out.append(f"| {b.replace('pct', '%')} | {r['providers_reviewed']:,} | "
                   f"{r['future_excluded_found']} | ${r['their_payments']/1e6:.1f}M | "
                   f"{r['share_of_future_excluded_payments']:.0%} |")
    lt = imp["models"]["LightGBM (ours)"].get("lead_time_months_at_1pct")
    if lt:
        out += ["", f"Lead time for providers caught at 1%: median {lt['median']:.0f} months "
                f"(IQR {lt['q25']:.0f}–{lt['q75']:.0f}) between the end of the scored year and "
                "the exclusion."]

    out += ["", "## Fairness at a 1% audit budget", "",
            "False-positive rate of each group relative to all providers (1.0 = parity).", "",
            "| Group | " + " | ".join(res["fairness"]) + " |",
            "|---|" + "---|" * len(res["fairness"])]
    groups = list(next(iter(res["fairness"].values())))
    for g in groups:
        out.append(f"| {g} | " + " | ".join(
            f"{res['fairness'][m][g]['fpr_ratio_vs_all']:.2f}" for m in res["fairness"]) + " |")
    out += ["", "Stratified-allocation trade-off (α = weight on within-stratum ranking):", "",
            "| α | AP | Recall @1% | Worst-group FPR ratio |", "|---|---|---|---|"]
    for r in res["fairness_mitigation"]:
        out.append(f"| {r['alpha']:.2f} | {r['ap']:.4f} | {r['recall@1pct']:.3f} | "
                   f"{r['max_fpr_ratio']:.2f} |")

    roll = REPORTS / "rolling.json"
    if roll.exists():
        rows = json.loads(roll.read_text())
        out += ["", "## Rolling-origin evaluation (3-year label window)", "",
                "| Test year | Train years | Positives | Model | AP | ROC-AUC | Recall @1% |",
                "|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r['test_year']} | {r['train_years'][0]}–{r['train_years'][-1]} | "
                       f"{r['n_pos']} | {r['model']} | {ci(r, 'ap')} | {r['roc_auc']:.3f} | "
                       f"{ci(r, 'recall@1pct')} |")

    out += ["", "## Per-specialty (specialties with ≥ 8 test positives)", "",
            "| Specialty | Providers | Positives | ROC-AUC | Recall @ global 1% |",
            "|---|---|---|---|---|"]
    for r in res["by_specialty"]:
        out.append(f"| {r['specialty']} | {r['n']:,} | {r['pos']} | {r['roc_auc']:.3f} | "
                   f"{r['recall@1pct_global']:.2f} |")

    out += ["", "## Top SHAP features", "", "| Feature | mean abs SHAP |", "|---|---|"]
    for k, v in list(res["shap_importance"].items())[:15]:
        out.append(f"| `{k}` | {v:.4f} |")
    (REPORTS / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("wrote", REPORTS / "RESULTS.md")


if __name__ == "__main__":
    main()

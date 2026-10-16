"""Ranking metrics framed around a fixed audit budget."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> dict[str, float]:
    """Precision / recall / lift if investigators review the top ``budget`` share of providers."""
    k = max(1, int(round(budget * len(y))))
    top = np.argpartition(-score, k - 1)[:k]
    hits = y[top].sum()
    prevalence = y.mean()
    precision = hits / k
    return {
        "k": k,
        "hits": int(hits),
        "precision": float(precision),
        "recall": float(hits / max(1, y.sum())),
        "lift": float(precision / prevalence) if prevalence > 0 else np.nan,
    }


def summarize(y: np.ndarray, score: np.ndarray, budgets=(0.001, 0.005, 0.01, 0.05)) -> dict:
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    for b in budgets:
        r = at_budget(y, score, b)
        tag = f"{b * 100:g}pct"
        out[f"recall@{tag}"] = r["recall"]
        out[f"precision@{tag}"] = r["precision"]
        out[f"lift@{tag}"] = r["lift"]
    return out


def bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 500, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap CIs for AP, ROC-AUC and recall@1%.

    Positives and negatives are resampled separately so every replicate keeps the
    original prevalence (otherwise AP is dominated by noise in the positive count).
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    stats = {"ap": [], "roc_auc": [], "recall@1pct": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, sb = y[idx], score[idx]
        stats["ap"].append(average_precision_score(yb, sb))
        stats["roc_auc"].append(roc_auc_score(yb, sb))
        stats["recall@1pct"].append(at_budget(yb, sb, 0.01)["recall"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in stats.items()}


def paired_bootstrap_delta(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n_boot: int = 500,
                           seed: int = 0) -> dict[str, float]:
    """AP(a) - AP(b) with a paired stratified bootstrap; returns mean, CI and P(delta <= 0)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas.append(average_precision_score(y[idx], s_a[idx])
                      - average_precision_score(y[idx], s_b[idx]))
    d = np.asarray(deltas)
    return {"delta_ap": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_le_0": float((d <= 0).mean())}

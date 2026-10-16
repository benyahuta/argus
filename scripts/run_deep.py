"""Train the ClaimSet Transformer on raw billing lines and score the test cohort.

Phase 1: train on the train cohorts, pick the epoch count on the 2019 validation cohort.
Phase 2: refit on train+valid with that epoch count (3 seeds), score 2021.
Writes artifacts/deep_test_scores.npy (aligned with the test rows of dataset.parquet)
and reports/deep_results.json.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudml.config import ARTIFACTS, EXP, INTERIM, REPORTS
from fraudml.data import load_service
from fraudml.deep import LineTensors, Trainer, build_vocab, line_tensors

PROV_COLS = ["log_benes", "log_payment", "log_pay_per_bene", "log_n_codes", "bene_risk_score",
             "is_organization", "rx_present", "rx_log_claims", "rx_opioid_share",
             "charge_markup"]
N_AUX = 80
# Regularised for ~700 positives: small model, heavy dropout, few positive repeats.
MODEL_KW = dict(d=64, layers=2, dropout=0.3)
FIT_KW = dict(pos_repeat=3, neg_per_epoch=300_000, lr=5e-4)
EPOCHS = int(os.environ.get("EPOCHS", 10))
PHASE1_ONLY = os.environ.get("PHASE1_ONLY") == "1"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def prepare() -> SimpleNamespace:
    """Line tensors + per-row metadata for every row of dataset.parquet (grouped by year)."""
    ds = pl.read_parquet(INTERIM / "dataset.parquet")
    years = sorted(ds["year"].unique().to_list())
    train_years = [c.feature_year for c in EXP.train]
    log("building vocabulary from training years")
    svcs = {y: load_service(y) for y in years}
    vocab = build_vocab([svcs[y] for y in train_years])
    log(f"vocab: {len(vocab):,} HCPCS codes")

    parts = []
    for y in years:
        sub = ds.filter(pl.col("year") == y)
        parts.append(line_tensors(svcs[y], sub["npi"].to_list(), vocab))
        log(f"  lines {y}: {sub.height:,} providers")
    del svcs
    lt = LineTensors(*(np.concatenate([getattr(p, f) for p in parts])
                       for f in ("codes", "pos", "num", "mask")))
    # ds rows are grouped by year in the same order as `years`.
    ds = pl.concat([ds.filter(pl.col("year") == y) for y in years])

    specialties = sorted(ds["specialty"].unique().to_list())
    spec_id = {s: i for i, s in enumerate(specialties)}
    spec = np.array([spec_id[s] for s in ds["specialty"]], np.int64)
    top = (ds.filter(pl.col("split") == "train")["specialty"].value_counts()
           .sort("count", descending=True).head(N_AUX - 1)["specialty"].to_list())
    aux_id = {s: i for i, s in enumerate(top)}
    aux = np.array([aux_id.get(s, N_AUX - 1) for s in ds["specialty"]], np.int64)

    prov = ds.select(PROV_COLS).to_numpy().astype(np.float32)
    prov = np.nan_to_num(prov)
    tr_mask = (ds["split"] == "train").to_numpy()
    mu, sd = prov[tr_mask].mean(0), prov[tr_mask].std(0) + 1e-6
    prov = np.clip((prov - mu) / sd, -10, 10)

    y = ds["label"].to_numpy().astype(np.int64)
    split = ds["split"].to_numpy()
    idx = {k: np.flatnonzero(split == k) for k in ("train", "valid", "test")}
    return SimpleNamespace(ds=ds, lt=lt, spec=spec, aux=aux, prov=prov, y=y, idx=idx,
                           n_codes=len(vocab) + 3, n_spec=len(specialties), vocab=vocab)


def main() -> None:
    d = prepare()
    ds, lt, spec, aux, prov, y, idx = d.ds, d.lt, d.spec, d.aux, d.prov, d.y, d.idx
    n_codes, specialties = d.n_codes, range(d.n_spec)
    out = {"vocab_size": len(d.vocab), "phase1": [], "phase2": []}

    log("phase 1: select epochs on validation")
    tr = Trainer(lt, spec, aux, prov, n_codes, len(specialties), N_AUX, seed=0, log=log,
                 **MODEL_KW)
    hist = tr.fit(idx["train"], y, idx["valid"], y[idx["valid"]], epochs=EPOCHS, **FIT_KW)
    out["phase1"] = hist
    best_epoch = int(max(hist, key=lambda r: r["valid_ap"])["epoch"])
    out["best_epoch"] = best_epoch
    log(f"best epoch {best_epoch}")
    if PHASE1_ONLY:
        return

    log("phase 2: refit on train+valid, 3 seeds")
    trva = np.concatenate([idx["train"], idx["valid"]])
    preds = []
    for seed in range(3):
        t = Trainer(lt, spec, aux, prov, n_codes, len(specialties), N_AUX, seed=seed, log=log,
                    **MODEL_KW)
        out["phase2"].append(t.fit(trva, y, epochs=max(best_epoch, 2), **FIT_KW))
        preds.append(t.predict(idx["test"]))
        p = preds[-1]
        log(f"  seed {seed}: test AP={average_precision_score(y[idx['test']], p):.4f}")
    score = np.mean(preds, 0)
    yt = y[idx["test"]]
    out["test_ap"] = float(average_precision_score(yt, score))
    out["test_auc"] = float(roc_auc_score(yt, score))
    log(f"ensemble test AP={out['test_ap']:.4f} AUC={out['test_auc']:.3f}")

    # Save aligned with the test rows of dataset.parquet (same order as filter(split==test)).
    test_npis = ds["npi"].to_numpy()[idx["test"]]
    pl.DataFrame({"npi": test_npis, "deep_score": score}).write_parquet(
        ARTIFACTS / "deep_test_scores.parquet")
    (REPORTS / "deep_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

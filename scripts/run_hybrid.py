"""Hybrid: self-supervised ClaimSet embeddings as extra LightGBM features.

1. Train the ClaimSet Transformer encoder *only* on the specialty task (no fraud labels)
   over every train+valid provider-year: ~2.2M labelled examples instead of ~700.
2. Embed every provider-year; reduce to 16 PCA dims (fit on training rows) and add the
   network's specialty-surprise score.
3. Fit LightGBM (frozen tuned params, 3 seeds) with and without these features and
   compare on the 2021 test cohort with a paired bootstrap.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

from fraudml.config import ARTIFACTS, EXP, INTERIM, REPORTS
from fraudml.evaluate import paired_bootstrap_delta, summarize
from fraudml.features import ALL_FEATURES
from fraudml.models import fit_gbm_ensemble, predict_ensemble, to_pandas

T0 = time.time()
N_PCA = 16


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def encode(d, trva: np.ndarray):
    from run_deep import MODEL_KW, N_AUX

    from fraudml.deep import Trainer

    log("self-supervised encoder (specialty task only)")
    t = Trainer(d.lt, d.spec, d.aux, d.prov, d.n_codes, d.n_spec, N_AUX, seed=0, log=log,
                **(MODEL_KW | {"d": 96, "layers": 3, "dropout": 0.1}))
    # fraud_weight=0: the fraud labels never touch the encoder; sample providers uniformly.
    hist = t.fit(trva, np.zeros_like(d.y), epochs=6, neg_per_epoch=600_000, pos_repeat=0,
                 lr=1e-3, fraud_weight=0.0)
    log("embedding all provider-years")
    h, surprise = t.embed(np.arange(len(d.y)))
    return h, surprise, hist


def main() -> None:
    cache = ARTIFACTS / "nn_embeddings.npz"
    # PyTorch is only imported when the encoder must be trained: loading its OpenMP runtime
    # next to LightGBM's in one process segfaults on macOS.
    if not cache.exists():
        from run_deep import prepare

        d = prepare()
        trva = np.concatenate([d.idx["train"], d.idx["valid"]])
        h, surprise, hist = encode(d, trva)
        np.savez(cache, h=h, surprise=surprise, hist=np.array(hist, dtype=object))
        del d
    # Rows in the same (year-grouped) order prepare() used, without the line tensors.
    ds = pl.read_parquet(INTERIM / "dataset.parquet")
    ds = pl.concat([ds.filter(pl.col("year") == y) for y in sorted(ds["year"].unique())])
    split = ds["split"].to_numpy()
    idx = {k: np.flatnonzero(split == k) for k in ("train", "valid", "test")}
    d = SimpleNamespace(ds=ds, idx=idx, y=ds["label"].to_numpy().astype(np.int64))
    trva = np.concatenate([idx["train"], idx["valid"]])
    c = np.load(cache, allow_pickle=True)
    h, surprise, hist = c["h"].astype(np.float32), c["surprise"], list(c["hist"])
    pca = PCA(N_PCA, random_state=0).fit(h[trva])

    z = pca.transform(h)
    emb_cols = [f"nn_emb_{i}" for i in range(N_PCA)] + ["nn_specialty_surprise"]
    emb = pd.DataFrame(np.column_stack([z, surprise]), columns=emb_cols)
    # Specialty accuracy of the encoder on held-out 2021 rows (sanity check).
    acc = float((surprise[d.idx["test"]] < np.log(2)).mean())
    log(f"share of 2021 providers whose specialty gets p>0.5 from code mix alone: {acc:.3f}")

    specialties = sorted(d.ds["specialty"].unique().to_list())
    X = to_pandas(d.ds, ALL_FEATURES, specialties)
    num = X.columns.drop("specialty_cat")
    X[num] = X[num].astype(np.float32)
    del h
    Xh = pd.concat([X, emb.astype(np.float32)], axis=1)
    tuned = json.loads((REPORTS / "results.json").read_text())["tuning"]
    params = tuned["params"] | {"verbose": -1}
    n_iter = int(tuned["best_iter"] * 1.2)
    y = d.y
    te = d.idx["test"]
    seeds = (0, 1, 2)
    log("LightGBM base vs. base + embeddings")
    base = predict_ensemble(fit_gbm_ensemble(params, n_iter, X.iloc[trva], y[trva], seeds),
                            X.iloc[te])
    hyb = predict_ensemble(fit_gbm_ensemble(params, n_iter, Xh.iloc[trva], y[trva], seeds),
                           Xh.iloc[te])
    res = {
        "encoder_history": hist,
        "specialty_p_gt_half_2021": acc,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "base": summarize(y[te], base, EXP.audit_budgets),
        "hybrid": summarize(y[te], hyb, EXP.audit_budgets),
        "delta_hybrid_minus_base": paired_bootstrap_delta(y[te], hyb, base, n_boot=300),
        "nn_surprise_alone_ap": float(average_precision_score(y[te], surprise[te])),
    }
    log(f"base AP={res['base']['ap']:.4f}  hybrid AP={res['hybrid']['ap']:.4f}  "
        f"delta={res['delta_hybrid_minus_base']}")
    (REPORTS / "hybrid.json").write_text(json.dumps(res, indent=2, default=float))
    np.save(ARTIFACTS / "hybrid_test_scores.npy", hyb)


if __name__ == "__main__":
    main()

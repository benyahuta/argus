import numpy as np
import polars as pl
import pytest

from fraudml.config import Cohort
from fraudml.evaluate import at_budget, summarize
from fraudml.features import ALL_FEATURES, add_peer_features, build_year, service_features
from fraudml.labels import attach_labels


def make_svc(rows):
    cols = ["npi", "specialty", "hcpcs", "pos", "Tot_Benes", "Tot_Srvcs", "Avg_Mdcr_Stdzd_Amt"]
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.with_columns(
        pl.lit("N").alias("drug"),
        pl.col("Tot_Srvcs").alias("Tot_Bene_Day_Srvcs"),
        (pl.col("Avg_Mdcr_Stdzd_Amt") * 2).alias("Avg_Sbmtd_Chrg"),
        (pl.col("Avg_Mdcr_Stdzd_Amt") * 1.1).alias("Avg_Mdcr_Alowd_Amt"),
        pl.lit(2020).alias("year"),
    )


def test_em_upcoding_features():
    svc = make_svc([
        ("A", "IM", "99213", "O", 10, 80.0, 70.0),
        ("A", "IM", "99215", "O", 10, 20.0, 150.0),
        ("B", "IM", "99215", "O", 10, 100.0, 150.0),
    ])
    f = service_features(svc).sort("npi")
    a, b = f.row(0, named=True), f.row(1, named=True)
    assert a["em_est_mean_level"] == pytest.approx(0.8 * 3 + 0.2 * 5)
    assert a["em_est_share_level5"] == pytest.approx(0.2)
    assert b["em_est_share_level5"] == pytest.approx(1.0)
    assert b["svc_pay_hhi"] == pytest.approx(1.0)


def test_peer_features_are_within_specialty():
    df = pl.DataFrame({
        "year": [2020] * 6,
        "specialty": ["X"] * 3 + ["Y"] * 3,
        "m": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })
    out = add_peer_features(df, metrics=["m"])
    # Same relative position in each specialty -> same peer z-score despite 100x scale.
    z = out["peer_z_m"].to_list()
    assert z[:3] == pytest.approx(z[3:], rel=1e-3)
    assert out["peer_pct_m"].to_list()[2] == pytest.approx(1.0)


def test_labels_forward_window_and_prior_exclusions():
    feats = pl.DataFrame({"npi": ["1", "2", "3", "4", "5"], "x": [0, 0, 0, 0, 0]})
    leie = pl.DataFrame({
        "npi": ["1", "2", "3", "4"],
        "excl_type": ["1128a1", "1128a1", "1128b14", "1128a3"],
        "excl_date": [20160615, 20180301, 20180301, 20230101],
    })
    out = attach_labels(feats, leie, Cohort("train", 2016, window_years=3)).sort("npi")
    assert out["npi"].to_list() == ["2", "3", "4", "5"]  # 1 was excluded before the window
    assert dict(zip(out["npi"], out["label"], strict=True)) == {"2": 1, "3": 0, "4": 0, "5": 0}


def test_budget_metrics():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = at_budget(y, s, 0.2)
    assert r["hits"] == 1 and r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["lift"] == pytest.approx(2.5)
    assert summarize(y, s)["roc_auc"] == pytest.approx(15 / 16)


def test_build_year_end_to_end():
    rng = np.random.default_rng(0)
    codes = [f"{c:05d}" for c in range(10000, 10040)] + ["99213", "99214", "99215"]
    prov_rows, svc_rows = [], []
    for i in range(300):
        spec = ["IM", "Derm", "Ortho"][i % 3]
        npi = str(1_000_000_000 + i)
        my_codes = rng.choice(codes[(i % 3) * 13:(i % 3) * 13 + 13] + codes[-3:], 6, False)
        tot = 0.0
        for c in my_codes:
            n, amt = float(rng.integers(11, 200)), float(rng.uniform(20, 200))
            svc_rows.append((npi, spec, c, "O", 11, n, amt))
            tot += n * amt
        prov_rows.append({"npi": npi, "specialty": spec, "state": "NC", "entity": "I",
                          "ruca": "1", "participating": "Y", "credentials": "MD",
                          "Tot_Benes": 50.0, "Tot_Srvcs": 300.0, "Tot_HCPCS_Cds": 6.0,
                          "Tot_Sbmtd_Chrg": tot * 2, "Tot_Mdcr_Alowd_Amt": tot,
                          "Tot_Mdcr_Stdzd_Amt": tot, "Drug_Mdcr_Stdzd_Amt": 0.0,
                          "year": 2020})
    prov = pl.DataFrame(prov_rows)
    for c in ["Bene_Age_LT_65_Cnt", "Bene_Age_GT_84_Cnt", "Bene_Dual_Cnt", "Bene_Feml_Cnt",
              "Bene_Avg_Age", "Bene_Avg_Risk_Scre", "Bene_CC_PH_Diabetes_V2_Pct",
              "Bene_CC_PH_Hypertension_V2_Pct", "Bene_CC_PH_HF_NonIHD_V2_Pct",
              "Bene_CC_BH_Depress_V1_Pct", "Bene_CC_PH_CKD_V2_Pct", "Bene_CC_PH_COPD_V2_Pct",
              "Bene_CC_BH_Alz_NonAlzdem_V2_Pct", "Bene_CC_PH_Cancer6_V2_Pct",
              "Bene_CC_PH_IschemicHeart_V2_Pct"]:
        prov = prov.with_columns(pl.lit(rng.uniform(1, 40)).alias(c))
    out = build_year(prov, make_svc(svc_rows))
    assert out.height == 300
    missing = set(ALL_FEATURES) - {"specialty_cat"} - set(out.columns)
    assert not missing
    # Specialty is recoverable from the synthetic code mix, so surprise should be low.
    assert out["mismatch_surprise"].median() < 1.0

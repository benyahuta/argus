"""Provider-year feature engineering.

Every feature is computed within a single year, so peer baselines move with
annual coding changes (e.g. the 2021 E/M office-visit overhaul) instead of
drifting between training and test cohorts.

Feature groups (used for ablations, see ``FEATURE_GROUPS``):
  volume    - raw volume / payment / markup from the provider summary file
  bene      - beneficiary demographics, risk score and chronic-condition mix
  peer      - the same volume metrics expressed relative to same-specialty peers
  service   - code-mix shape: concentration, per-code intensity vs peers, rare codes
  em        - evaluation & management (E/M) level distribution: an upcoding signal
  mismatch  - does the code mix look like the declared specialty?
  meta      - specialty / entity / geography descriptors
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EPS = 1.0

EST_EM = ["99211", "99212", "99213", "99214", "99215"]
NEW_EM = ["99201", "99202", "99203", "99204", "99205"]


def _log(c: str) -> pl.Expr:
    return (pl.col(c).fill_null(0) + EPS).log()


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b > 0).then(a / b).otherwise(None)


# ---------------------------------------------------------------------------
# Provider summary file
# ---------------------------------------------------------------------------

CC_COLS = [
    "Bene_CC_PH_Diabetes_V2_Pct", "Bene_CC_PH_Hypertension_V2_Pct",
    "Bene_CC_PH_HF_NonIHD_V2_Pct", "Bene_CC_BH_Depress_V1_Pct", "Bene_CC_PH_CKD_V2_Pct",
    "Bene_CC_PH_COPD_V2_Pct", "Bene_CC_BH_Alz_NonAlzdem_V2_Pct",
    "Bene_CC_PH_Cancer6_V2_Pct", "Bene_CC_PH_IschemicHeart_V2_Pct",
]


def provider_features(prov: pl.DataFrame) -> pl.DataFrame:
    benes = pl.col("Tot_Benes")
    df = prov.with_columns(
        # volume
        _log("Tot_Benes").alias("log_benes"),
        _log("Tot_Srvcs").alias("log_services"),
        _log("Tot_Mdcr_Stdzd_Amt").alias("log_payment"),
        _log("Tot_HCPCS_Cds").alias("log_n_codes"),
        (_safe_div(pl.col("Tot_Mdcr_Stdzd_Amt"), benes) + EPS).log().alias("log_pay_per_bene"),
        _safe_div(pl.col("Tot_Srvcs"), benes).alias("srv_per_bene"),
        _safe_div(pl.col("Tot_Sbmtd_Chrg"), pl.col("Tot_Mdcr_Alowd_Amt")).alias("charge_markup"),
        _safe_div(pl.col("Drug_Mdcr_Stdzd_Amt").fill_null(0), pl.col("Tot_Mdcr_Stdzd_Amt"))
        .alias("drug_pay_share"),
        _safe_div(pl.col("Tot_HCPCS_Cds"), benes).alias("codes_per_bene"),
        # beneficiary mix
        _safe_div(pl.col("Bene_Age_LT_65_Cnt"), benes).alias("bene_under65_share"),
        _safe_div(pl.col("Bene_Age_GT_84_Cnt"), benes).alias("bene_over84_share"),
        _safe_div(pl.col("Bene_Dual_Cnt"), benes).alias("bene_dual_share"),
        _safe_div(pl.col("Bene_Feml_Cnt"), benes).alias("bene_female_share"),
        pl.col("Bene_Avg_Age").alias("bene_avg_age"),
        pl.col("Bene_Avg_Risk_Scre").alias("bene_risk_score"),
        *[(pl.col(c) / 100).alias("cc_" + c.split("_")[3].lower()) for c in CC_COLS],
        # meta
        (pl.col("entity") == "O").cast(pl.Int8).alias("is_organization"),
        (pl.col("participating") == "Y").cast(pl.Int8).alias("is_participating"),
        pl.col("ruca").cast(pl.Float64, strict=False).alias("ruca_code"),
    )
    return df


PEER_METRICS = ["log_pay_per_bene", "srv_per_bene", "charge_markup", "log_benes",
                "codes_per_bene", "bene_risk_score", "drug_pay_share"]


def add_peer_features(df: pl.DataFrame, metrics: list[str] = PEER_METRICS) -> pl.DataFrame:
    """Robust z-score and percentile of each metric within (year, specialty)."""
    grp = ["year", "specialty"]
    exprs = []
    for m in metrics:
        med = pl.col(m).median().over(grp)
        mad = (pl.col(m) - med).abs().median().over(grp)
        exprs.append(((pl.col(m) - med) / (1.4826 * mad + 1e-6)).clip(-50, 50).alias(f"peer_z_{m}"))
        exprs.append(
            (pl.col(m).rank("average").over(grp) / pl.col(m).count().over(grp))
            .alias(f"peer_pct_{m}")
        )
    exprs.append(pl.len().over(grp).log().alias("peer_group_log_size"))
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Provider x service file
# ---------------------------------------------------------------------------

def service_features(svc: pl.DataFrame) -> pl.DataFrame:
    svc = svc.with_columns(
        (pl.col("Tot_Srvcs") * pl.col("Avg_Mdcr_Stdzd_Amt")).alias("pay"),
        (pl.col("Tot_Srvcs") / pl.col("Tot_Benes")).alias("spb"),
    )
    code_grp = ["year", "specialty", "hcpcs"]
    svc = svc.with_columns(
        pl.col("spb").median().over(code_grp).alias("spb_med"),
        pl.col("Avg_Sbmtd_Chrg").median().over(code_grp).alias("chg_med"),
        pl.col("Avg_Mdcr_Stdzd_Amt").median().over(code_grp).alias("pay_med"),
        pl.len().over(code_grp).alias("code_n_prov"),
        pl.col("npi").n_unique().over(["year", "specialty"]).alias("spec_n_prov"),
    ).with_columns(
        (pl.col("spb") / pl.col("spb_med")).log().alias("lr_intensity"),
        (pl.col("Avg_Sbmtd_Chrg") / pl.col("chg_med")).log().alias("lr_charge"),
        (pl.col("Avg_Mdcr_Stdzd_Amt") / pl.col("pay_med")).log().alias("lr_pay"),
        ((pl.col("code_n_prov") / pl.col("spec_n_prov")) < 0.01).alias("is_rare"),
        (pl.col("Tot_Srvcs") / pl.col("Tot_Bene_Day_Srvcs")).alias("srv_per_day"),
    )

    w = pl.col("pay")
    wsum = w.sum()
    agg = svc.group_by(["npi", "year"]).agg(
        # concentration of payment across codes
        ((w / wsum) ** 2).sum().alias("svc_pay_hhi"),
        (w.max() / wsum).alias("svc_top_code_share"),
        pl.len().alias("svc_n_lines"),
        # payment-weighted intensity and pricing relative to same-specialty peers
        ((pl.col("lr_intensity") * w).sum() / wsum).alias("svc_intensity_excess"),
        ((pl.col("lr_charge") * w).sum() / wsum).alias("svc_charge_excess"),
        ((pl.col("lr_pay") * w).sum() / wsum).alias("svc_pay_excess"),
        pl.col("lr_intensity").max().alias("svc_max_intensity_excess"),
        (pl.col("lr_intensity") > np.log(3)).mean().alias("svc_share_codes_3x_peer"),
        (w.filter(pl.col("is_rare")).sum() / wsum).alias("svc_rare_code_pay_share"),
        ((pl.col("srv_per_day") * w).sum() / wsum).alias("svc_srv_per_bene_day"),
        (w.filter(pl.col("pos") == "F").sum() / wsum).alias("svc_facility_share"),
        # E/M
        *[pl.col("Tot_Srvcs").filter(pl.col("hcpcs") == c).sum().alias(f"n_{c}")
          for c in EST_EM + NEW_EM],
    )
    return agg.pipe(_em_features).fill_nan(None)


def _em_features(df: pl.DataFrame) -> pl.DataFrame:
    def block(codes: list[str], prefix: str) -> list[pl.Expr]:
        tot = pl.sum_horizontal([pl.col(f"n_{c}") for c in codes])
        lvl = pl.sum_horizontal([pl.col(f"n_{c}") * (i + 1) for i, c in enumerate(codes)])
        return [
            tot.alias(f"{prefix}_volume"),
            _safe_div(lvl, tot).alias(f"{prefix}_mean_level"),
            _safe_div(pl.col(f"n_{codes[-1]}"), tot).alias(f"{prefix}_share_level5"),
            _safe_div(pl.col(f"n_{codes[-2]}") + pl.col(f"n_{codes[-1]}"), tot)
            .alias(f"{prefix}_share_level45"),
        ]

    df = df.with_columns(*block(EST_EM, "em_est"), *block(NEW_EM, "em_new"))
    df = df.with_columns((pl.col("em_est_volume") + EPS).log().alias("em_est_log_volume"))
    return df.drop([f"n_{c}" for c in EST_EM + NEW_EM] + ["em_est_volume", "em_new_volume"])


def add_em_peer_features(df: pl.DataFrame) -> pl.DataFrame:
    """E/M level relative to same-specialty peers who also bill office visits."""
    grp = ["year", "specialty"]
    out = []
    for m in ["em_est_mean_level", "em_est_share_level45", "em_est_share_level5",
              "em_new_mean_level"]:
        med = pl.col(m).median().over(grp)
        sd = pl.col(m).std().over(grp)
        out.append(((pl.col(m) - med) / (sd + 1e-6)).clip(-20, 20).alias(f"peer_z_{m}"))
    return df.with_columns(out)


# ---------------------------------------------------------------------------
# Specialty mismatch: does this provider's code mix look like its specialty?
# ---------------------------------------------------------------------------

def specialty_mismatch(svc: pl.DataFrame, prov_specialty: pl.DataFrame, n_components: int = 48,
                       top_k_specialties: int = 60, max_fit: int = 150_000,
                       seed: int = 0) -> pl.DataFrame:
    """Out-of-fold surprise -log p(declared specialty | code mix), fitted within one year.

    Code-mix vectors are TF-IDF weighted payment shares over HCPCS codes, reduced
    with truncated SVD. A multinomial logistic regression predicts specialty; for
    each provider we score it with a model that never saw that provider (2-fold).
    """
    pay = svc.select(
        "npi", "hcpcs", (pl.col("Tot_Srvcs") * pl.col("Avg_Mdcr_Stdzd_Amt")).alias("pay")
    ).filter(pl.col("pay") > 0)
    npis = prov_specialty.filter(pl.col("npi").is_in(pay["npi"].unique().implode()))
    npi_idx = {n: i for i, n in enumerate(npis["npi"].to_list())}
    codes = pay["hcpcs"].unique().sort().to_list()
    code_idx = {c: i for i, c in enumerate(codes)}
    pay = pay.filter(pl.col("npi").is_in(list(npi_idx)))
    rows = np.fromiter((npi_idx[n] for n in pay["npi"]), dtype=np.int64, count=pay.height)
    cols = np.fromiter((code_idx[c] for c in pay["hcpcs"]), dtype=np.int64, count=pay.height)
    X = sparse.csr_matrix((np.log1p(pay["pay"].to_numpy()), (rows, cols)),
                          shape=(len(npi_idx), len(codes)))
    X = TfidfTransformer(sublinear_tf=False).fit_transform(X)
    n_components = min(n_components, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(X)
    Z = StandardScaler().fit_transform(Z)

    spec = npis["specialty"].to_numpy()
    names, counts = np.unique(spec, return_counts=True)
    top = set(names[np.argsort(-counts)[:top_k_specialties]])
    y = np.array([s if s in top else "__other__" for s in spec])

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, size=len(y))
    surprise = np.full(len(y), np.nan)
    entropy = np.full(len(y), np.nan)
    for k in (0, 1):
        fit_idx = np.flatnonzero(fold != k)
        if len(fit_idx) > max_fit:
            fit_idx = rng.choice(fit_idx, max_fit, replace=False)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Z[fit_idx], y[fit_idx])
        pred_idx = np.flatnonzero(fold == k)
        P = clf.predict_proba(Z[pred_idx])
        col = {c: i for i, c in enumerate(clf.classes_)}
        own = np.array([P[i, col[s]] if s in col else 1e-6 for i, s in enumerate(y[pred_idx])])
        surprise[pred_idx] = -np.log(np.clip(own, 1e-6, 1))
        entropy[pred_idx] = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)

    return pl.DataFrame({
        "npi": npis["npi"], "mismatch_surprise": surprise, "mismatch_entropy": entropy,
    })


# ---------------------------------------------------------------------------
# Part D prescribing
# ---------------------------------------------------------------------------

def partd_features(rx: pl.DataFrame) -> pl.DataFrame:
    clms = pl.col("rx_Tot_Clms")
    return rx.select(
        "npi", "year",
        pl.lit(1, dtype=pl.Int8).alias("rx_present"),
        (clms.fill_null(0) + EPS).log().alias("rx_log_claims"),
        (pl.col("rx_Tot_Drug_Cst").fill_null(0) + EPS).log().alias("rx_log_cost"),
        (_safe_div(pl.col("rx_Tot_Drug_Cst"), clms) + EPS).log().alias("rx_log_cost_per_claim"),
        _safe_div(clms, pl.col("rx_Tot_Benes")).alias("rx_claims_per_bene"),
        _safe_div(pl.col("rx_Tot_Day_Suply"), clms).alias("rx_days_per_claim"),
        _safe_div(pl.col("rx_Brnd_Tot_Clms"), clms).alias("rx_brand_claim_share"),
        _safe_div(pl.col("rx_Brnd_Tot_Drug_Cst"), pl.col("rx_Tot_Drug_Cst"))
        .alias("rx_brand_cost_share"),
        _safe_div(pl.col("rx_LIS_Tot_Clms"), clms).alias("rx_lis_claim_share"),
        _safe_div(pl.col("rx_Opioid_Tot_Clms").fill_null(0), clms).alias("rx_opioid_share"),
        _safe_div(pl.col("rx_Opioid_LA_Tot_Clms").fill_null(0), clms)
        .alias("rx_opioid_la_share"),
        _safe_div(pl.col("rx_Opioid_Tot_Suply"), pl.col("rx_Opioid_Tot_Benes"))
        .alias("rx_opioid_days_per_bene"),
        _safe_div(pl.col("rx_Antpsyct_GE65_Tot_Clms"), pl.col("rx_GE65_Tot_Clms"))
        .alias("rx_antipsychotic_65_share"),
        _safe_div(pl.col("rx_Antbtc_Tot_Clms"), clms).alias("rx_antibiotic_share"),
    )


RX_PEER = ["rx_opioid_share", "rx_log_cost_per_claim", "rx_brand_claim_share",
           "rx_claims_per_bene", "rx_opioid_days_per_bene"]


# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "volume": ["log_benes", "log_services", "log_payment", "log_n_codes", "log_pay_per_bene",
               "srv_per_bene", "charge_markup", "drug_pay_share", "codes_per_bene"],
    "bene": ["bene_under65_share", "bene_over84_share", "bene_dual_share", "bene_female_share",
             "bene_avg_age", "bene_risk_score"]
    + ["cc_" + c.split("_")[3].lower() for c in CC_COLS],
    "peer": [f"peer_{k}_{m}" for m in PEER_METRICS for k in ("z", "pct")] + ["peer_group_log_size"],
    "service": ["svc_pay_hhi", "svc_top_code_share", "svc_n_lines", "svc_intensity_excess",
                "svc_charge_excess", "svc_pay_excess", "svc_max_intensity_excess",
                "svc_share_codes_3x_peer", "svc_rare_code_pay_share", "svc_srv_per_bene_day",
                "svc_facility_share"],
    "em": ["em_est_mean_level", "em_est_share_level5", "em_est_share_level45",
           "em_est_log_volume", "em_new_mean_level", "em_new_share_level5",
           "em_new_share_level45", "peer_z_em_est_mean_level", "peer_z_em_est_share_level45",
           "peer_z_em_est_share_level5", "peer_z_em_new_mean_level"],
    "mismatch": ["mismatch_surprise", "mismatch_entropy"],
    "rx": ["rx_present", "rx_log_claims", "rx_log_cost", "rx_log_cost_per_claim",
           "rx_claims_per_bene", "rx_days_per_claim", "rx_brand_claim_share",
           "rx_brand_cost_share", "rx_lis_claim_share", "rx_opioid_share", "rx_opioid_la_share",
           "rx_opioid_days_per_bene", "rx_antipsychotic_65_share", "rx_antibiotic_share"]
    + [f"peer_z_{m}" for m in RX_PEER],
    "meta": ["is_organization", "is_participating", "ruca_code", "specialty_cat"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def build_year(prov: pl.DataFrame, svc: pl.DataFrame, rx: pl.DataFrame | None = None,
               min_benes: int = 11, seed: int = 0) -> pl.DataFrame:
    """Full feature table for one year (one row per Part B provider)."""
    df = provider_features(prov).filter(pl.col("Tot_Benes") >= min_benes)
    df = add_peer_features(df)
    sf = service_features(svc)
    mm = specialty_mismatch(svc, df.select("npi", "specialty"), seed=seed)
    df = df.join(sf, on=["npi", "year"], how="left").join(mm, on="npi", how="left")
    df = add_em_peer_features(df)
    if rx is not None:
        df = df.join(partd_features(rx), on=["npi", "year"], how="left")
        df = df.with_columns(pl.col("rx_present").fill_null(0))
        df = add_peer_features(df, metrics=RX_PEER).drop(
            [f"peer_pct_{m}" for m in RX_PEER])
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c)
                             for c in FEATURE_GROUPS["rx"])
    keep = ["npi", "year", "specialty", "state"] + [f for f in ALL_FEATURES if f != "specialty_cat"]
    floats = [c for c, t in df.schema.items() if t == pl.Float64]
    # Zero-priced codes can make log-ratios infinite; treat them as missing.
    df = df.with_columns(pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in floats)
    return df.select(keep)

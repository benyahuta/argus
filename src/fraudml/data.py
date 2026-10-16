"""Load the raw CMS / OIG files into typed parquet tables.

Raw inputs (see scripts/download_data.sh):
  * ``{year}_prov.csv``  Medicare Physician & Other Practitioners - by Provider
  * ``{year}_svc.csv``   Medicare Physician & Other Practitioners - by Provider and Service
  * ``leie.csv``         OIG List of Excluded Individuals/Entities
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import INTERIM, RAW

PROV_NUMERIC = [
    "Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs", "Tot_Sbmtd_Chrg", "Tot_Mdcr_Alowd_Amt",
    "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt", "Drug_Tot_HCPCS_Cds", "Drug_Tot_Benes",
    "Drug_Tot_Srvcs", "Drug_Mdcr_Stdzd_Amt", "Med_Tot_HCPCS_Cds", "Med_Tot_Benes",
    "Med_Tot_Srvcs", "Med_Mdcr_Stdzd_Amt", "Bene_Avg_Age", "Bene_Age_LT_65_Cnt",
    "Bene_Age_65_74_Cnt", "Bene_Age_75_84_Cnt", "Bene_Age_GT_84_Cnt", "Bene_Feml_Cnt",
    "Bene_Male_Cnt", "Bene_Dual_Cnt", "Bene_Ndual_Cnt", "Bene_Avg_Risk_Scre",
    "Bene_CC_PH_Diabetes_V2_Pct", "Bene_CC_PH_Hypertension_V2_Pct",
    "Bene_CC_PH_HF_NonIHD_V2_Pct", "Bene_CC_BH_Depress_V1_Pct", "Bene_CC_PH_CKD_V2_Pct",
    "Bene_CC_PH_COPD_V2_Pct", "Bene_CC_BH_Alz_NonAlzdem_V2_Pct",
    "Bene_CC_PH_Cancer6_V2_Pct", "Bene_CC_PH_IschemicHeart_V2_Pct",
]
PROV_KEEP = [
    "Rndrng_NPI", "Rndrng_Prvdr_Ent_Cd", "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_RUCA",
    "Rndrng_Prvdr_Type", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "Rndrng_Prvdr_Crdntls",
]
SVC_KEEP = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "HCPCS_Drug_Ind", "Place_Of_Srvc"]
SVC_NUMERIC = ["Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs", "Avg_Sbmtd_Chrg",
               "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Stdzd_Amt"]


def _num(cols: list[str], present: set[str]) -> list[pl.Expr]:
    """Cast to float, creating all-null columns when a year lacks a field."""
    return [
        pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False).alias(c)
        if c in present else pl.lit(None, dtype=pl.Float64).alias(c)
        for c in cols
    ]


def _cached(path: Path, build, force: bool = False) -> pl.DataFrame:
    if path.exists() and not force:
        return pl.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.write_parquet(path)
    return df


def load_provider(year: int, raw: Path = RAW, force: bool = False) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        lf = pl.scan_csv(raw / f"{year}_prov.csv", infer_schema=False)
        present = set(lf.collect_schema().names())
        return (
            lf.select([pl.col(c) for c in PROV_KEEP] + _num(PROV_NUMERIC, present))
            .rename({"Rndrng_NPI": "npi", "Rndrng_Prvdr_Type": "specialty",
                     "Rndrng_Prvdr_State_Abrvtn": "state", "Rndrng_Prvdr_Ent_Cd": "entity",
                     "Rndrng_Prvdr_RUCA": "ruca", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "participating",
                     "Rndrng_Prvdr_Crdntls": "credentials"})
            .with_columns(pl.lit(year).alias("year"))
            .collect()
        )

    return _cached(INTERIM / f"prov_{year}.parquet", build, force)


def load_service(year: int, raw: Path = RAW, force: bool = False) -> pl.DataFrame:
    def build() -> pl.DataFrame:
        lf = pl.scan_csv(raw / f"{year}_svc.csv", infer_schema=False)
        present = set(lf.collect_schema().names())
        return (
            lf.select([pl.col(c) for c in SVC_KEEP] + _num(SVC_NUMERIC, present))
            .rename({"Rndrng_NPI": "npi", "Rndrng_Prvdr_Type": "specialty",
                     "HCPCS_Cd": "hcpcs", "HCPCS_Drug_Ind": "drug", "Place_Of_Srvc": "pos"})
            .with_columns(pl.lit(year).alias("year"))
            .collect()
        )

    return _cached(INTERIM / f"svc_{year}.parquet", build, force)


PARTD_NUMERIC = [
    "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Day_Suply", "Tot_Benes",
    "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst", "Gnrc_Tot_Clms", "LIS_Tot_Clms",
    "Opioid_Tot_Clms", "Opioid_Tot_Suply", "Opioid_Tot_Benes", "Opioid_Prscrbr_Rate",
    "Opioid_LA_Tot_Clms", "Antbtc_Tot_Clms", "Antpsyct_GE65_Tot_Clms", "GE65_Tot_Clms",
]


def load_partd(year: int, raw: Path = RAW, force: bool = False) -> pl.DataFrame:
    """Medicare Part D Prescribers - by Provider (prescribing, incl. opioids)."""
    def build() -> pl.DataFrame:
        lf = pl.scan_csv(raw / f"{year}_partd.csv", infer_schema=False)
        present = set(lf.collect_schema().names())
        return (
            lf.select([pl.col("Prscrbr_NPI").alias("npi")] + _num(PARTD_NUMERIC, present))
            .rename({c: f"rx_{c}" for c in PARTD_NUMERIC})
            .with_columns(pl.lit(year).alias("year"))
            .collect()
        )

    return _cached(INTERIM / f"partd_{year}.parquet", build, force)


def load_leie(raw: Path = RAW) -> pl.DataFrame:
    """Excluded individuals/entities that have an NPI on file."""
    return (
        pl.read_csv(raw / "leie.csv", infer_schema=False)
        .filter(pl.col("NPI").is_not_null() & (pl.col("NPI") != "0"))
        .select(
            pl.col("NPI").alias("npi"),
            pl.col("EXCLTYPE").str.strip_chars().alias("excl_type"),
            pl.col("EXCLDATE").cast(pl.Int64).alias("excl_date"),
            pl.col("GENERAL").alias("excl_general"),
        )
    )

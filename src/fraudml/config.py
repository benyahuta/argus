"""Central configuration: data paths, cohort years and label definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# LEIE exclusion authorities that indicate program fraud, abuse, or misconduct
# (as opposed to e.g. defaulted student loans, 1128b14). See
# https://oig.hhs.gov/exclusions/authorities.asp
FRAUD_EXCLUSION_TYPES: tuple[str, ...] = (
    "1128a1",  # conviction of program-related crimes
    "1128a2",  # conviction relating to patient abuse or neglect
    "1128a3",  # felony conviction relating to health care fraud
    "1128a4",  # felony conviction relating to controlled substances
    "1128b1",  # misdemeanor conviction relating to health care fraud
    "1128b2",  # conviction relating to obstruction of an investigation
    "1128b3",  # misdemeanor conviction relating to controlled substances
    "1128b4",  # license revocation, suspension, or surrender
    "1128b5",  # exclusion or suspension under federal or state health program
    "1128b6",  # claims for excessive charges or unnecessary services
    "1128b7",  # fraud, kickbacks, and other prohibited activities
    "1128b8",  # entities controlled by a sanctioned individual
    "1128Aa",  # civil monetary penalty
)


@dataclass(frozen=True)
class Cohort:
    """A provider-year snapshot whose label is 'excluded within the next window years'."""

    name: str
    feature_year: int
    window_years: int = 3

    @property
    def label_start(self) -> int:  # first day of feature_year + 1 (YYYYMMDD)
        return (self.feature_year + 1) * 10000 + 101

    @property
    def label_end(self) -> int:  # last day of the window (YYYYMMDD)
        return (self.feature_year + self.window_years) * 10000 + 1231


@dataclass(frozen=True)
class Experiment:
    # Strictly forward-in-time: every training label is observed before any test label.
    train: tuple[Cohort, ...] = (Cohort("train", 2016), Cohort("train", 2017))
    # 2-year window so validation labels (2020-21) end before test labels (2022-24) begin,
    # which lets the final model be refit on train + valid without leaking test outcomes.
    valid: Cohort = Cohort("valid", 2019, window_years=2)
    test: Cohort = Cohort("test", 2021)
    # Ignore providers with too little Medicare volume to say anything about.
    min_benes: int = 11
    seed: int = 7
    n_bootstrap: int = 1000
    audit_budgets: tuple[float, ...] = field(default=(0.001, 0.005, 0.01, 0.05))

    @property
    def years(self) -> list[int]:
        return sorted({c.feature_year for c in (*self.train, self.valid, self.test)})


EXP = Experiment()

# Every year downloaded (scripts/data_urls.txt); used by the rolling-origin evaluation.
ALL_YEARS = list(range(2016, 2023))

"""Locked protocol constants. Changing any value here is a protocol change and must be
recorded in docs/EVALUASI.md before results are regenerated."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Protocol:
    # data generation (Sparkov)
    n_customers: int = 1000
    start_date: str = "2019-01-01"
    end_date: str = "2020-12-31"
    seed: int = 42
    # time split (inclusive start, exclusive end), all UTC
    # Jan-Mar 2019 is feature warm-up only: 90-day habit features need history
    warmup: tuple[str, str] = ("2019-01-01", "2019-04-01")
    train: tuple[str, str] = ("2019-04-01", "2020-01-01")
    val_a: tuple[str, str] = ("2020-01-01", "2020-04-01")  # model selection, early stopping
    val_b: tuple[str, str] = ("2020-04-01", "2020-07-01")  # calibration and thresholds
    test: tuple[str, str] = ("2020-07-01", "2021-01-01")  # used once
    # review budgets as a share of transactions; primary is 1 percent
    budgets: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    primary_budget: float = 0.01
    # bootstrap
    n_boot: int = 1000
    boot_seed: int = 20261029
    # explainability
    shap_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    shap_sample: int = 5000
    workers: int = field(default_factory=lambda: int(os.environ.get("TXRISK_WORKERS", "4")))


PROTOCOL = Protocol()


def data_dir() -> Path:
    return Path(os.environ.get("TXRISK_DATA", ROOT / "data"))


def reports_dir() -> Path:
    return Path(os.environ.get("TXRISK_REPORTS", ROOT / "reports"))

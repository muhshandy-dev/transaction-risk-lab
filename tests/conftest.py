from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from txrisk.features import CATEGORIES


def make_clean(n_cards=6, n_per_card=80, seed=0) -> pd.DataFrame:
    """Small hand-made clean table with the same columns as ingest.clean()."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = 1_546_300_800  # 2019-01-01
    for c in range(n_cards):
        t = t0 + np.sort(rng.integers(0, 120 * 86400, n_per_card))
        for i in range(n_per_card):
            rows.append({"card_id": f"K{c + 1:04d}", "unix_time": int(t[i]),
                         "category": CATEGORIES[rng.integers(0, len(CATEGORIES))],
                         "amt": float(np.round(rng.gamma(2, 30), 2)), "distance_km": float(rng.uniform(1, 120)),
                         "is_fraud": int(rng.random() < 0.05), "audit_gender": "F" if c % 2 else "M",
                         "audit_age_band": "25-49", "audit_area": "kota"})
    df = pd.DataFrame(rows).sort_values(["unix_time", "card_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "tx_id", np.arange(len(df)))
    df.insert(2, "ts", pd.to_datetime(df["unix_time"], unit="s", utc=True))
    return df


@pytest.fixture
def clean_df():
    return make_clean()

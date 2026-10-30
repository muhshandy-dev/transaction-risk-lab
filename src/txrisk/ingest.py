"""Turn raw Sparkov output into a clean, PII-free transaction table.

Faker fills Sparkov rows with fake names, street addresses, card numbers and SSNs. None
of that is needed for modelling, so it is dropped here. Cards get a pseudonymous id
(``K0001`` ...) in order of first appearance. Home and merchant coordinates are reduced
to a distance in km, then dropped. Gender, age band and urban/rural are kept only as
audit columns; they are never model features (see features.FEATURES).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PII_COLUMNS = ["ssn", "cc_num", "first", "last", "street", "city", "state", "zip", "job", "acct_num",
               "trans_num", "merchant", "lat", "long", "merch_lat", "merch_long", "dob", "city_pop",
               "profile", "trans_date", "trans_time"]

CLEAN_COLUMNS = ["tx_id", "card_id", "ts", "unix_time", "category", "amt", "distance_km", "is_fraud",
                 "audit_gender", "audit_age_band", "audit_area"]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def age_band(age: np.ndarray) -> np.ndarray:
    return np.where(age < 25, "<25", np.where(age < 50, "25-49", "50+"))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["unix_time"] = df["unix_time"].astype("int64")
    df = df.sort_values(["unix_time", "cc_num", "trans_num"], kind="mergesort").reset_index(drop=True)
    first_seen = df.drop_duplicates("cc_num")["cc_num"].tolist()
    card_map = {c: f"K{i + 1:04d}" for i, c in enumerate(first_seen)}
    out = pd.DataFrame({
        "tx_id": np.arange(len(df), dtype=np.int64),
        "card_id": df["cc_num"].map(card_map).values,
        "ts": pd.to_datetime(df["unix_time"], unit="s", utc=True),
        "unix_time": df["unix_time"].values,
        "category": df["category"].values,
        "amt": df["amt"].astype(float).values,
        "distance_km": haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]).round(3),
        "is_fraud": df["is_fraud"].astype(int).values,
    })
    dob = pd.to_datetime(df["dob"])
    age = (out["ts"].dt.tz_localize(None) - dob).dt.days.values / 365.25
    out["audit_gender"] = df["gender"].values
    out["audit_age_band"] = age_band(age)
    out["audit_area"] = np.where(df["city_pop"].astype(int) >= 2500, "kota", "desa")
    return out[CLEAN_COLUMNS]


def assert_no_pii(df: pd.DataFrame) -> None:
    leaked = [c for c in PII_COLUMNS if c in df.columns]
    if leaked:
        raise AssertionError(f"PII columns present: {leaked}")

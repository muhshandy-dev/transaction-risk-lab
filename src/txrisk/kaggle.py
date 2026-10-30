"""Adapter for the published Kaggle copy of Sparkov data (kartik2112/fraud-detection, CC0).

That dataset was produced by an older Sparkov version with 1,000 customers over the same
dates (2019-01-01 to 2020-12-31), so the same protocol can be replayed on it as an
independent draw. Its `unix_time` column is shifted by several years, so time is taken from
`trans_date_trans_time` (treated as UTC). Not run in this repository's sessions: Kaggle is not
reachable there. See notebooks/kaggle_reproduce.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEEDED = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long", "city_pop", "dob",
          "trans_num", "merch_lat", "merch_long", "is_fraud"]


def load_kaggle(*csv_paths: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"not the Kaggle Sparkov layout, missing columns: {missing}")
    ts = pd.to_datetime(df["trans_date_trans_time"], utc=True)
    out = df[NEEDED[1:]].copy()
    out["unix_time"] = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
    # columns clean() drops anyway; filled so the raw table has the generator's shape
    for c in ("ssn", "first", "last", "street", "city", "state", "zip", "job", "acct_num", "profile",
              "trans_date", "trans_time", "merchant"):
        if c not in out.columns:
            out[c] = df[c] if c in df.columns else ""
    out = out.drop_duplicates("trans_num")
    return out.sort_values(["unix_time", "cc_num", "trans_num"], kind="mergesort").reset_index(drop=True)


def import_kaggle(csv_paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_transactions.parquet"
    load_kaggle(*csv_paths).to_parquet(path, index=False)
    return path

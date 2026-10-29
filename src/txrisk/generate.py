"""Deterministic driver around the vendored Sparkov generator."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SPARKOV_COMMIT = "b5eb45c89d36f2aa4ef16044a42945bed8b96d93"
PROFILE_DIR = Path(__file__).parent / "sparkov" / "profiles"
MAIN_CONFIG = PROFILE_DIR / "main_config.json"


def _seed_all(seed: int) -> None:
    from faker import Faker

    random.seed(seed)
    np.random.seed(seed % 2**32)
    Faker.seed(seed)


def task_seed(base: int, name: str) -> int:
    digest = hashlib.sha256(f"{base}:{name}".encode()).hexdigest()
    return int(digest[:12], 16)


def _utc() -> None:
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()


def _customers(n: int, seed: int, out: Path) -> None:
    from .sparkov import customer

    _utc()
    _seed_all(seed)
    with open(out, "w") as fh, redirect_stdout(fh):
        customer.main(n, seed, MAIN_CONFIG, None)


def _profile_task(args: tuple[str, str, str, str, str, int]) -> str:
    customers_file, profile_name, start, end, out_file, seed = args
    from .sparkov import transaction

    _utc()
    _seed_all(seed)
    start_d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    with open(out_file, "w") as fh, redirect_stdout(fh):
        transaction.main(customers_file, PROFILE_DIR / profile_name, start_d, end_d, None)
    return out_file


def generate(n_customers: int, start: str, end: str, seed: int, out_dir: Path, workers: int = 4) -> Path:
    """Run Sparkov and write one combined raw parquet file (still contains fake PII)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "sparkov_work"
    work.mkdir(exist_ok=True)
    cust_file = work / "customers.csv"
    _customers(n_customers, seed, cust_file)
    profiles = list(json.loads(MAIN_CONFIG.read_text()).keys())
    tasks = [
        (str(cust_file), p, start, end, str(work / p.replace(".json", ".csv")), task_seed(seed, p))
        for p in profiles
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            files = list(ex.map(_profile_task, tasks))
    else:
        files = [_profile_task(t) for t in tasks]
    frames = [pd.read_csv(f, sep="|", dtype=str, keep_default_na=False) for f in files]
    raw = pd.concat([f for f in frames if len(f)], ignore_index=True)
    raw["unix_time"] = raw["unix_time"].astype("int64")
    raw = raw.sort_values(["unix_time", "cc_num", "trans_num"], kind="mergesort").reset_index(drop=True)
    out = out_dir / "raw_transactions.parquet"
    raw.to_parquet(out, index=False)
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/txr")
    print(generate(n, "2019-01-01", "2019-03-31", 42, out, 4))

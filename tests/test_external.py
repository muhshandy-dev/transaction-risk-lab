import dataclasses

import numpy as np
import pandas as pd
import pytest

from txrisk.config import PROTOCOL
from txrisk.external import INPUTS, V_COLS, load, prepare, run_external, sha256_file


def fake_ulb(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    d.insert(0, "Time", np.sort(rng.integers(0, 172800, n)).astype(float))
    d["Amount"] = rng.gamma(2.0, 40.0, n).round(2)
    y = (rng.random(n) < 0.01).astype(int)
    d.loc[y == 1, "V14"] -= 4
    d.loc[y == 1, "V4"] += 2
    d["Class"] = y
    return d


def test_split_is_chronological_and_complete():
    d, masks, info = prepare(fake_ulb())
    t = d["Time"].to_numpy()
    order = ["train", "val_a", "val_b", "test"]
    assert sum(m.sum() for m in masks.values()) == len(d)
    for a, b in zip(order, order[1:], strict=False):
        assert t[masks[a]].max() < t[masks[b]].min()
    assert abs(masks["test"].mean() - 0.20) < 0.01
    assert info["test_rows_with_exact_copy_before_test"] == 0


def test_load_checks_columns(tmp_path):
    p = tmp_path / "x.csv"
    fake_ulb(50).drop(columns=["V3"]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load(p)
    assert len(sha256_file(p)) == 64


def test_external_protocol_runs_and_is_consistent():
    p = dataclasses.replace(PROTOCOL, n_boot=20, shap_sample=300)
    res = run_external(fake_ulb(), p, threads=2)
    g, lg = res["models"]["lgbm"], res["models"]["logistic"]
    assert res["explain"]["checks"]["lgbm_additivity_max_abs"] < 1e-6
    assert res["explain"]["checks"]["logistic_additivity_max_abs"] < 1e-9
    assert {e["feature"] for e in res["explain"]["global"]} == set(INPUTS)
    # the planted signal is found by both models
    assert g["ap"] > 0.3 and lg["ap"] > 0.3
    assert res["explain"]["global"][0]["feature"] in ("V14", "V4")
    for k in ("logistic", "lgbm"):
        c = res["ci"][k]["ap"]
        assert 0 <= c["lo"] <= c["hi"] <= 1
    assert res["paired_lgbm_minus_logistic"]["ap"]["n_boot"] == 20

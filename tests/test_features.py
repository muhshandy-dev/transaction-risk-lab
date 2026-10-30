import numpy as np
import pandas as pd

from txrisk.features import FEATURES, NUMERIC, build_features

from .conftest import make_clean


def test_all_features_present(clean_df):
    f = build_features(clean_df)
    for name in FEATURES:
        assert name in f.columns
    assert (f["tx_id"].to_numpy() == clean_df["tx_id"].to_numpy()).all()


def test_features_do_not_see_the_future():
    """Change and delete everything after a cut-off: earlier features must not move."""
    df = make_clean(seed=3)
    full = build_features(df)
    cut = df["unix_time"].quantile(0.6)
    later = df["unix_time"] > cut
    changed = df.copy()
    changed.loc[later, "amt"] = changed.loc[later, "amt"] * 50 + 999
    changed.loc[later, "category"] = "travel"
    changed.loc[later, "is_fraud"] = 1 - changed.loc[later, "is_fraud"]
    truncated = df[~later | (df.index % 3 == 0)]  # also drop many future rows
    for other in (build_features(changed), build_features(truncated)):
        a = full.set_index("tx_id").loc[df.loc[~later, "tx_id"], NUMERIC]
        b = other.set_index("tx_id").loc[df.loc[~later, "tx_id"], NUMERIC]
        pd.testing.assert_frame_equal(a, b)


def test_labels_are_not_used():
    df = make_clean(seed=5)
    flipped = df.assign(is_fraud=1 - df["is_fraud"])
    a = build_features(df)[NUMERIC]
    b = build_features(flipped)[NUMERIC]
    pd.testing.assert_frame_equal(a, b)


def test_velocity_and_habit_values_by_hand():
    t0 = 1_546_300_800
    rows = [  # one card, times in seconds after t0
        (0, "grocery_pos", 10.0), (600, "grocery_pos", 30.0), (4000, "travel", 100.0),
        (90000, "grocery_pos", 20.0), (90000 + 91 * 86400, "grocery_pos", 40.0),
    ]
    df = pd.DataFrame({"tx_id": range(5), "card_id": "K0001", "unix_time": [t0 + r[0] for r in rows],
                       "category": [r[1] for r in rows], "amt": [r[2] for r in rows], "distance_km": 1.0,
                       "is_fraud": 0, "audit_gender": "F", "audit_age_band": "<25", "audit_area": "kota"})
    df["ts"] = pd.to_datetime(df["unix_time"], unit="s", utc=True)
    f = build_features(df)
    assert f["n_1h"].tolist() == [0, 1, 1, 0, 0]  # third tx: only the 600 s one is within 1 h
    assert f["n_24h"].tolist() == [0, 1, 2, 1, 0]  # fourth tx at 25 h: only the 4000 s one is within 24 h
    assert np.isnan(f["secs_since_prev"].iloc[0]) and f["secs_since_prev"].iloc[1] == 600
    assert f["amt_24h"].iloc[2] == 40.0
    assert np.isclose(f["amt_ratio_card"].iloc[2], 100.0 / 20.0)
    assert np.isnan(f["amt_ratio_card_cat"].iloc[2])  # first travel tx
    assert np.isclose(f["cat_share_card"].iloc[3], 2 / 3)
    # last tx is 91 days later: the 90-day habit window is empty again
    assert np.isnan(f["amt_ratio_card"].iloc[4]) and f["n_7d"].iloc[4] == 0

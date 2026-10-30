import pandas as pd

from txrisk.generate import generate
from txrisk.ingest import PII_COLUMNS, assert_no_pii, clean


def test_generator_is_deterministic_and_pii_is_dropped(tmp_path):
    a = generate(6, "2019-01-01", "2019-02-15", 42, tmp_path / "a", workers=2)
    b = generate(6, "2019-01-01", "2019-02-15", 42, tmp_path / "b", workers=1)
    ra, rb = pd.read_parquet(a), pd.read_parquet(b)
    pd.testing.assert_frame_equal(ra, rb)
    assert ra["is_fraud"].isin(["0", "1"]).all()
    c = clean(ra)
    assert_no_pii(c)
    assert not set(PII_COLUMNS) & set(c.columns)
    raw_text = c.astype(str).to_numpy().ravel()
    for value in ra["first"].unique().tolist() + ra["cc_num"].unique().tolist():
        assert value not in raw_text
    assert c["card_id"].str.match(r"^K\d{4}$").all()
    assert (c["distance_km"] >= 0).all()


def test_category_does_not_depend_on_date(tmp_path):
    """Guards patch P5: legitimate categories must not rotate with the calendar."""
    raw = pd.read_parquet(generate(30, "2019-01-01", "2019-12-31", 5, tmp_path / "c", workers=2))
    legit = raw[raw["is_fraud"] == "0"]
    per_month = legit.groupby(legit["trans_date"].str[:7])["category"].nunique()
    assert per_month.min() >= 12, per_month.to_dict()
    hours = legit["trans_time"].str[:2].astype(int)
    am_share = (hours < 12).groupby(legit["trans_date"].str[:7]).mean()
    assert am_share.max() - am_share.min() < 0.15, am_share.to_dict()

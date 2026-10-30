import pandas as pd

from txrisk.generate import generate
from txrisk.ingest import clean
from txrisk.kaggle import load_kaggle


def test_kaggle_layout_round_trip(tmp_path):
    """Write generator output in the Kaggle CSV layout and read it back through the adapter."""
    raw = pd.read_parquet(generate(4, "2019-01-01", "2019-01-20", 3, tmp_path / "g", workers=1))
    ts = pd.to_datetime(raw["unix_time"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    kag = raw.assign(trans_date_trans_time=ts, unix_time=raw["unix_time"] - 10**8)  # Kaggle unix_time is shifted
    half = len(kag) // 2
    kag.iloc[:half].to_csv(tmp_path / "fraudTrain.csv", index=True)
    kag.iloc[half:].to_csv(tmp_path / "fraudTest.csv", index=True)
    back = load_kaggle(tmp_path / "fraudTrain.csv", tmp_path / "fraudTest.csv")
    assert (back["unix_time"].to_numpy() == raw["unix_time"].to_numpy()).all()
    a, b = clean(raw), clean(back)
    pd.testing.assert_frame_equal(a, b)

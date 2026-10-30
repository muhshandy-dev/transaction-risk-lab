import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from txrisk.metrics import Ranked, at_threshold, calibration, card_bootstrap_weights, threshold_for_budget


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    s = np.round(rng.normal(y * 1.5, 1), 1)  # rounding creates many ties
    return y, s


def test_ap_and_auc_match_sklearn_with_ties():
    y, s = _data()
    r = Ranked(y, s)
    assert np.isclose(r.average_precision(), average_precision_score(y, s))
    assert np.isclose(r.roc_auc(), roc_auc_score(y, s))


def test_integer_weights_equal_row_replication():
    y, s = _data(1500, 1)
    w = np.random.default_rng(2).integers(0, 4, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    r = Ranked(y, s)
    assert np.isclose(r.average_precision(w), average_precision_score(y[rep], s[rep]))
    assert np.isclose(r.roc_auc(w), roc_auc_score(y[rep], s[rep]))


def test_recall_at_budget_top_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    amt = np.array([10, 1, 30, 1, 1, 1, 1, 1, 1, 60], dtype=float)
    res = Ranked(y, s, amt).at_budget([0.1, 0.3])
    assert res[0.1]["recall"] == 1 / 3 and res[0.1]["precision"] == 1.0
    assert np.isclose(res[0.3]["recall"], 2 / 3) and np.isclose(res[0.3]["precision"], 2 / 3)
    assert np.isclose(res[0.3]["amount_recall"], 40 / 100)


def test_threshold_and_operational_metrics():
    y, s = _data(10000, 4)
    thr = threshold_for_budget(s, 0.05)
    op = at_threshold(y, s, thr)
    assert 0.03 < op["alert_rate"] < 0.07


def test_calibration_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 200000)
    y = (rng.random(len(p)) < p).astype(int)
    good = calibration(y, p)
    bad = calibration(y, np.clip(p * 3, 0, 1))
    assert good["ece"] < 0.003 and bad["ece"] > 0.05
    assert abs(good["mean_pred"] - good["frac_pos"]) < 0.002


def test_card_bootstrap_resamples_whole_cards():
    card = np.repeat(np.arange(50), 7)
    for w in card_bootstrap_weights(card, 5, 0):
        per_card = w.reshape(50, 7)
        assert (per_card == per_card[:, :1]).all()  # every row of a card shares the weight
        assert per_card[:, 0].sum() == 50

import numpy as np
import pandas as pd

from txrisk.features import build_features
from txrisk.models import LogisticModel, Platt, fit_lgbm, lgbm_matrix, lgbm_params

from .conftest import make_clean


def _frame():
    f = build_features(make_clean(n_cards=10, n_per_card=150, seed=11))
    # plant a learnable signal so the fit is not degenerate
    f["is_fraud"] = ((f["amt"] > 90) & (f["hour"] >= 12)).astype(int)
    return f


def test_logistic_contributions_are_exact_and_match_shap():
    import shap

    f = _frame()
    m = LogisticModel(0.1).fit(f, f["is_fraud"].to_numpy())
    contrib, base, names = m.contributions(f)
    assert np.allclose(contrib.sum(1) + base, m.margin(f), atol=1e-9)
    Z = m.spec.transform(f)
    ex = shap.LinearExplainer((m.clf.coef_[0], m.clf.intercept_[0]), shap.maskers.Independent(Z, max_samples=len(Z)))
    sv = ex(Z).values
    groups = m.spec.groups()
    grouped = pd.DataFrame(sv, columns=groups).T.groupby(level=0, sort=False).sum().T[names].to_numpy()
    assert np.allclose(grouped, contrib, atol=1e-8)


def test_lgbm_deterministic():
    f = _frame()
    y = f["is_fraud"].to_numpy()
    a = fit_lgbm(f, y, lgbm_params(7, 5), 30).predict(lgbm_matrix(f), raw_score=True)
    b = fit_lgbm(f, y, lgbm_params(7, 5), 30).predict(lgbm_matrix(f), raw_score=True)
    assert np.array_equal(a, b)


def test_platt_recovers_scaling():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 2, 50000)
    y = (rng.random(len(m)) < 1 / (1 + np.exp(-(0.5 * m - 1)))).astype(int)
    p = Platt().fit(m, y)
    assert abs(p.a - 0.5) < 0.05 and abs(p.b + 1) < 0.08

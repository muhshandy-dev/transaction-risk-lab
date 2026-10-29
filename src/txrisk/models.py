"""Logistic baseline and LightGBM, both exportable to a plain JSON spec for the browser."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import CATEGORIES, NUMERIC

LOG_FEATURES = ["amt", "distance_km", "secs_since_prev", "n_1h", "n_24h", "n_7d", "amt_24h",
                "amt_ratio_card", "amt_ratio_card_cat"]
LINEAR_NUMERIC = [f for f in NUMERIC if f not in ("hour", "dow")]
ONEHOT = {"category": CATEGORIES, "hour": list(range(24)), "dow": list(range(7))}


class LinearSpec:
    """Hand-written preprocessing so the exact same transform can run in TypeScript.

    numeric: optional log1p, impute train median (plus a missing indicator), standardise.
    one-hot: category, hour, dow."""

    def __init__(self, interactions: bool = False):
        self.interactions = interactions

    def fit(self, X: pd.DataFrame) -> LinearSpec:
        self.numeric = []
        for f in LINEAR_NUMERIC:
            x = self._raw(X, f)
            med = float(np.nanmedian(x))
            has_nan = bool(np.isnan(x).any())
            xi = np.where(np.isnan(x), med, x)
            self.numeric.append({"name": f, "log1p": f in LOG_FEATURES, "median": med,
                                 "missing_indicator": has_nan, "mean": float(xi.mean()),
                                 "std": float(xi.std()) or 1.0})
        self.onehot = {k: list(v) for k, v in ONEHOT.items()}
        self.columns = []
        for n in self.numeric:
            self.columns.append(("num", n["name"]))
            if n["missing_indicator"]:
                self.columns.append(("miss", n["name"]))
        for k, vals in self.onehot.items():
            for v in vals:
                self.columns.append(("oh", f"{k}={v}"))
        if self.interactions:
            for v in self.onehot["category"]:
                self.columns.append(("ix", f"category={v}*amt"))
        return self

    @staticmethod
    def _raw(X, f):
        x = X[f].to_numpy(dtype=float)
        if f in LOG_FEATURES:
            x = np.log1p(np.maximum(x, 0))
        return x

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for n in self.numeric:
            x = self._raw(X, n["name"])
            miss = np.isnan(x)
            cols.append((np.where(miss, n["median"], x) - n["mean"]) / n["std"])
            if n["missing_indicator"]:
                cols.append(miss.astype(float))
        for k, vals in self.onehot.items():
            raw = X[k].astype(str).to_numpy() if k == "category" else X[k].to_numpy()
            for v in vals:
                cols.append((raw == v).astype(float))
        if self.interactions:
            z_amt = cols[0]  # first numeric column is amt (log1p, standardised)
            cat = X["category"].astype(str).to_numpy()
            for v in self.onehot["category"]:
                cols.append((cat == v) * z_amt)
        return np.column_stack(cols)

    def groups(self) -> list[str]:
        """Original feature each design column belongs to (for grouped attributions)."""
        out = []
        for kind, name in self.columns:
            if kind == "oh":
                out.append(name.split("=")[0])
            elif kind == "ix":
                out.append("category*amt")
            else:
                out.append(name)
        return out


class LogisticModel:
    def __init__(self, C: float, interactions: bool = False):
        self.C = C
        self.interactions = interactions

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogisticModel:
        self.spec = LinearSpec(self.interactions).fit(X)
        assert self.spec.numeric[0]["name"] == "amt"
        Z = self.spec.transform(X)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, tol=1e-8)
        self.clf.fit(Z, y)
        self.design_mean = Z.mean(axis=0)
        return self

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.decision_function(self.spec.transform(X))

    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, float, list[str]]:
        """Exact additive attributions per original feature: coef * (z - mean_train(z)).

        Equals SHAP for a linear model with an independent (interventional) background equal
        to the training mean. Returns (n x features), base value, feature names."""
        Z = self.spec.transform(X)
        coef = self.clf.coef_[0]
        contrib = (Z - self.design_mean) * coef
        groups = self.spec.groups()
        names = list(dict.fromkeys(groups))
        G = np.zeros((len(groups), len(names)))
        for j, g in enumerate(groups):
            G[j, names.index(g)] = 1
        base = float(self.clf.intercept_[0] + self.design_mean @ coef)
        return contrib @ G, base, names

    def export(self) -> dict:
        return {"type": "logistic", "C": self.C, "numeric": self.spec.numeric, "onehot": self.spec.onehot,
                "columns": [f"{k}:{n}" for k, n in self.spec.columns],
                "coef": self.clf.coef_[0].tolist(), "intercept": float(self.clf.intercept_[0]),
                "design_mean": self.design_mean.tolist()}


LGBM_FEATURES = NUMERIC + ["category"]


def lgbm_matrix(X: pd.DataFrame) -> pd.DataFrame:
    M = X[LGBM_FEATURES].copy()
    M["category"] = pd.Categorical(X["category"].astype(str), categories=CATEGORIES)
    return M


def lgbm_params(num_leaves: int, min_child_samples: int, seed: int = 0, threads: int = 4) -> dict:
    return {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_cat_to_onehot": 16,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
        "verbosity": -1,
    }


def fit_lgbm(Xtr, ytr, params, num_boost_round, Xval=None, yval=None, early_stopping=None):
    import lightgbm as lgb

    dtr = lgb.Dataset(lgbm_matrix(Xtr), label=ytr, categorical_feature=["category"], free_raw_data=False)
    callbacks = []
    valid_sets = []
    if Xval is not None:
        dval = lgb.Dataset(lgbm_matrix(Xval), label=yval, reference=dtr, categorical_feature=["category"])
        valid_sets = [dval]
        if early_stopping:
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    p = dict(params, metric="average_precision")
    return lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class Platt:
    """p = sigmoid(a * margin + b), fitted on validation B."""

    def fit(self, margin: np.ndarray, y: np.ndarray) -> Platt:
        lr = LogisticRegression(C=1e6, max_iter=1000, tol=1e-10)
        lr.fit(np.asarray(margin).reshape(-1, 1), y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def __call__(self, margin):
        return sigmoid(self.a * np.asarray(margin) + self.b)

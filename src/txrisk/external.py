"""External check on real card transactions: the ULB / Worldline credit card fraud dataset.

284,807 European card transactions from September 2013 (two days), 492 frauds. V1-V28 are
PCA components released by the bank, so only Time and Amount are original. There is no card
id, which rules out the per-card behavioural features of the main protocol. This module
therefore tests a narrower claim: does the modelling and evaluation protocol (logistic
baseline vs LightGBM, selection on validation A, Platt and budget thresholds on validation
B, one test pass, bootstrap CI, TreeSHAP) hold up on real fraud, not synthetic fraud.

Locked before any result was looked at (docs/EVALUASI.md, kriteria K16-K19):
  split   rows in time order: 50% train, 15% validation A, 15% validation B, 20% test
  inputs  V1-V28 and log1p(Amount); no time-of-day (only two days, it would extrapolate)
  CI      1000 bootstrap resamples of 10-minute blocks (frauds arrive in bursts)
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROTOCOL, Protocol
from .metrics import Ranked, at_threshold, calibration, card_bootstrap_weights, ci, threshold_for_budget
from .models import Platt, lgbm_params, sigmoid

ULB_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
ULB_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
SPLIT = (0.50, 0.65, 0.80)
BLOCK_SECONDS = 600
V_COLS = [f"V{i}" for i in range(1, 29)]
INPUTS = V_COLS + ["log_amount"]
LOGIT_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LGBM_GRID = ((15, 50), (15, 200), (31, 50), (31, 200), (63, 50), (63, 200))


def _r(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _log(msg):
    print(f"[ulb] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path, url: str = ULB_URL, expected: str | None = ULB_SHA256) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        _log(f"downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    digest = sha256_file(dest)
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch for {dest}: {digest} != {expected}")
    return dest


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = ["Time", *V_COLS, "Amount", "Class"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return d


def prepare(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    d = d.sort_values("Time", kind="mergesort").reset_index(drop=True)
    d["log_amount"] = np.log1p(d["Amount"].clip(lower=0))
    n = len(d)
    t = d["Time"].to_numpy(float)
    cuts = [t[int(n * q)] for q in SPLIT]
    names = ("train", "val_a", "val_b", "test")
    edges = [-np.inf, *cuts, np.inf]
    masks = {nm: (t >= edges[i]) & (t < edges[i + 1]) for i, nm in enumerate(names)}
    # exact duplicate rows of the test period that also occur earlier (possible leakage)
    key = pd.util.hash_pandas_object(d[["Time", *V_COLS, "Amount"]], index=False).to_numpy()
    seen_before = np.isin(key[masks["test"]], key[~masks["test"]])
    info = {"rows": int(n), "fraud": int(d["Class"].sum()), "time_cuts_seconds": [float(c) for c in cuts],
            "duplicate_rows": int(d.duplicated().sum()),
            "test_rows_with_exact_copy_before_test": int(seen_before.sum()),
            "splits": {nm: {"rows": int(m.sum()), "fraud": int(d.loc[m, "Class"].sum()),
                            "fraud_rate": _r(d.loc[m, "Class"].mean()),
                            "time_from_h": _r(t[m].min() / 3600, 3), "time_to_h": _r(t[m].max() / 3600, 3)}
                       for nm, m in masks.items()}}
    return d, masks, info


class Scaled:
    """Standardised inputs + logistic regression; exact additive contributions like the main model."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1
        Z = (X - self.mu) / self.sd
        self.clf = LogisticRegression(C=self.C, max_iter=5000, tol=1e-8).fit(Z, y)
        self.zmean = Z.mean(0)
        return self

    def z(self, X):
        return (X - self.mu) / self.sd

    def margin(self, X):
        return self.clf.decision_function(self.z(X))

    def contributions(self, X):
        coef = self.clf.coef_[0]
        return (self.z(X) - self.zmean) * coef, float(self.clf.intercept_[0] + self.zmean @ coef)


def _lgb_train(Xtr, ytr, params, rounds, Xva=None, yva=None, early=None):
    import lightgbm as lgb

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=INPUTS, free_raw_data=False)
    vs, cb = [], []
    if Xva is not None:
        vs = [lgb.Dataset(Xva, label=yva, reference=dtr)]
        if early:
            cb.append(lgb.early_stopping(early, verbose=False))
    return lgb.train(dict(params, metric="average_precision"), dtr, num_boost_round=rounds, valid_sets=vs,
                     callbacks=cb)


def run_external(d: pd.DataFrame, p: Protocol = PROTOCOL, threads: int = 4, n_boot: int | None = None,
                 source: dict | None = None) -> dict:
    t0 = time.time()
    n_boot = n_boot or p.n_boot
    d, masks, info = prepare(d)
    X = d[INPUTS].to_numpy(float)
    y = d["Class"].to_numpy(int)
    Xtr, ytr = X[masks["train"]], y[masks["train"]]
    Xa, ya = X[masks["val_a"]], y[masks["val_a"]]
    Xb, yb = X[masks["val_b"]], y[masks["val_b"]]
    Xt, yt = X[masks["test"]], y[masks["test"]]
    amt_t = d.loc[masks["test"], "Amount"].to_numpy(float)

    # 2. selection on validation A
    grid = {"logistic": [], "lgbm": []}
    for C in LOGIT_GRID:
        m = Scaled(C).fit(Xtr, ytr)
        grid["logistic"].append({"C": C, "val_a_ap": _r(Ranked(ya, m.margin(Xa)).average_precision())})
        _log(f"logistic C={C} val A AP={grid['logistic'][-1]['val_a_ap']}")
    for nl, mcs in LGBM_GRID:
        b = _lgb_train(Xtr, ytr, lgbm_params(nl, mcs, 0, threads), 5000, Xa, ya, early=200)
        s = b.predict(Xa, raw_score=True, num_iteration=b.best_iteration)
        grid["lgbm"].append({"num_leaves": nl, "min_child_samples": mcs, "best_iteration": int(b.best_iteration),
                             "val_a_ap": _r(Ranked(ya, s).average_precision())})
        _log(f"lgbm {nl}/{mcs} iters={b.best_iteration} val A AP={grid['lgbm'][-1]['val_a_ap']}")
    best = {k: max(v, key=lambda e: (e["val_a_ap"], -v.index(e))) for k, v in grid.items()}

    logit = Scaled(best["logistic"]["C"]).fit(Xtr, ytr)
    bl = best["lgbm"]
    booster = _lgb_train(Xtr, ytr, lgbm_params(bl["num_leaves"], bl["min_child_samples"], 0, threads),
                         max(bl["best_iteration"], 1))

    # 3. calibration and thresholds on validation B
    mb = {"logistic": logit.margin(Xb), "lgbm": booster.predict(Xb, raw_score=True)}
    mt = {"logistic": logit.margin(Xt), "lgbm": booster.predict(Xt, raw_score=True)}
    platt = {k: Platt().fit(mb[k], yb) for k in mb}
    pb = {k: platt[k](mb[k]) for k in mb}
    pt = {k: platt[k](mt[k]) for k in mt}
    thr = {k: {b: threshold_for_budget(pb[k], b) for b in p.budgets} for k in mb}

    # 4. single test pass
    scores = {"rule_amount": d.loc[masks["test"], "log_amount"].to_numpy(float), **mt}
    labels = {"rule_amount": "Aturan: nominal terbesar", "logistic": "Regresi logistik", "lgbm": "LightGBM"}
    models = {}
    for k, s in scores.items():
        rk = Ranked(yt, s, amt_t)
        e = {"label": labels[k], "ap": _r(rk.average_precision()), "roc_auc": _r(rk.roc_auc()),
             "at_budget": {str(b): {m: _r(v) for m, v in dd.items()} for b, dd in rk.at_budget(p.budgets).items()},
             "pr_curve": rk.pr_curve(150)}
        if k in platt:
            e["calibration"] = {"raw": calibration(yt, sigmoid(mt[k])), "platt": calibration(yt, pt[k])}
            e["operational"] = {str(b): {"threshold": _r(thr[k][b], 9), **at_threshold(yt, pt[k], thr[k][b],
                                                                                    amt=amt_t)}
                                for b in p.budgets}
        models[k] = e

    blocks = pd.factorize((d.loc[masks["test"], "Time"].to_numpy() // BLOCK_SECONDS).astype(int))[0]
    ranked = {k: Ranked(yt, s, amt_t) for k, s in scores.items()}
    samp = {k: {"ap": [], "auc": [], **{f"recall@{b}": [] for b in p.budgets}} for k in scores}
    for k in platt:
        samp[k].update({"op_recall": [], "op_alert_rate": [], "brier": [], "ece": []})
    diff = {"ap": [], f"recall@{p.primary_budget}": [], "op_recall": []}
    for w in card_bootstrap_weights(blocks, n_boot, p.boot_seed):
        for k, rk in ranked.items():
            s = samp[k]
            s["ap"].append(rk.average_precision(w))
            s["auc"].append(rk.roc_auc(w))
            ab = rk.at_budget(p.budgets, w)
            for b in p.budgets:
                s[f"recall@{b}"].append(ab[b]["recall"])
            if k in platt:
                op = at_threshold(yt, pt[k], thr[k][p.primary_budget], w)
                s["op_recall"].append(op["recall"])
                s["op_alert_rate"].append(op["alert_rate"])
                cal = calibration(yt, pt[k], w)
                s["brier"].append(cal["brier"])
                s["ece"].append(cal["ece"])
        G, L = samp["lgbm"], samp["logistic"]
        for m in diff:
            diff[m].append(G[m][-1] - L[m][-1])
    cis = {k: {m: ci(v) for m, v in dd.items()} for k, dd in samp.items()}
    paired = {m: {"mean": _r(np.nanmean(v)), **ci(v), "share_gt_0": _r(np.nanmean(np.asarray(v) > 0))}
              for m, v in diff.items()}

    # 5. explanations (TreeSHAP on a test sample plus every test fraud)
    import shap

    rng = np.random.default_rng(p.seed + 1)
    pick = np.union1d(rng.choice(len(Xt), size=min(p.shap_sample, len(Xt)), replace=False), np.flatnonzero(yt))
    Xs = Xt[pick]
    ex = shap.TreeExplainer(booster)
    sv = np.asarray(ex.shap_values(Xs))
    if sv.ndim == 3:
        sv = sv[-1]
    base = float(np.ravel(ex.expected_value)[-1])
    add = float(np.max(np.abs(sv.sum(1) + base - booster.predict(Xs, raw_score=True))))
    lc, lbase = logit.contributions(Xs)
    ladd = float(np.max(np.abs(lc.sum(1) + lbase - logit.margin(Xs))))
    fr = yt[pick] == 1
    imp = sorted(({"feature": f, "mean_abs_shap": _r(np.abs(sv[:, j]).mean()),
                   "mean_abs_shap_fraud": _r(np.abs(sv[fr, j]).mean()),
                   "mean_abs_logistic": _r(np.abs(lc[:, j]).mean())} for j, f in enumerate(INPUTS)),
                 key=lambda e: -e["mean_abs_shap"])

    return {
        "dataset": {"name": "Credit Card Fraud Detection (ULB Machine Learning Group, Worldline)",
                    "period": "September 2013, dua hari", "url": ULB_URL, "sha256": ULB_SHA256,
                    "licence": "Database Contents License (DbCL) v1.0 menurut halaman Kaggle mlg-ulb/creditcardfraud",
                    "citation": "Dal Pozzolo, Caelen, Johnson, Bontempi (2015). Calibrating Probability with "
                                "Undersampling for Unbalanced Classification. IEEE SSCI.",
                    **(source or {})},
        "protocol": {"split_row_shares": list(SPLIT), "inputs": INPUTS, "block_seconds": BLOCK_SECONDS,
                     "n_boot": n_boot, "boot_seed": p.boot_seed, "budgets": list(p.budgets),
                     "primary_budget": p.primary_budget, "n_blocks_test": int(blocks.max() + 1)},
        "data": info,
        "selection": {"grid": grid, "best": best},
        "models": models,
        "ci": cis,
        "paired_lgbm_minus_logistic": paired,
        "platt": {k: {"a": v.a, "b": v.b} for k, v in platt.items()},
        "explain": {"sample_size": int(len(pick)), "test_frauds_in_sample": int(fr.sum()), "lgbm_base_value": base,
                    "checks": {"lgbm_additivity_max_abs": add, "logistic_additivity_max_abs": ladd},
                    "global": imp},
        "seconds": round(time.time() - t0, 1),
    }

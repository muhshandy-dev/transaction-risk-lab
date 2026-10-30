"""Stress tests on the synthetic data (kriteria K20). Hyper-parameters stay locked at the
run-2 selection; nothing here is tuned on the test period.

  label_missing_q   fraud labels missing for a share q of fraud cards in train and validation
                    (chargebacks never reported); retrained, judged on clean test labels
  novel_pattern     online-category fraud (shopping_net, misc_net) removed from train and
                    validation; recall reported separately for those categories in test
  amount_up_25      every test-period amount x1.25; features rebuilt; models not retrained
  clock_shift_3h    every test-period timestamp +3 hours; features rebuilt; not retrained
  cold_start        history before the test period deleted, so every card starts new on
                    1 July 2020; features rebuilt; not retrained

The base models are refitted with the locked settings and checked against run 2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROTOCOL, ROOT, Protocol, data_dir, reports_dir
from .features import build_features, split_masks
from .ingest import clean
from .metrics import Ranked, at_threshold, card_bootstrap_weights, ci, threshold_for_budget
from .models import LogisticModel, Platt, fit_lgbm, lgbm_matrix, lgbm_params
from .pipeline import build_data, log, r, write_json

LABEL_MISSING = (0.1, 0.3, 0.5)
NOVEL_CATEGORIES = ("shopping_net", "misc_net")
AMOUNT_FACTOR = 1.25
CLOCK_SHIFT_SECONDS = 3 * 3600
MODELS = ("logistic", "lgbm")


def locked_settings() -> dict:
    best = json.loads((ROOT / "reports" / "results.json").read_text())["selection_best"]
    return {"C": best["logistic"]["C"], "num_leaves": best["lgbm"]["num_leaves"],
            "min_child_samples": best["lgbm"]["min_child_samples"], "rounds": best["lgbm"]["best_iteration"]}


class Fitted:
    """Logistic + LightGBM with Platt and budget thresholds from validation B."""

    def __init__(self, tr: pd.DataFrame, vb: pd.DataFrame, s: dict, p: Protocol, threads: int):
        y = tr["is_fraud"].to_numpy()
        self.logit = LogisticModel(s["C"]).fit(tr, y)
        self.booster = fit_lgbm(tr, y, lgbm_params(s["num_leaves"], s["min_child_samples"], 0, threads),
                                s["rounds"])
        mb = self.margins(vb)
        yb = vb["is_fraud"].to_numpy()
        self.platt = {k: Platt().fit(mb[k], yb) for k in MODELS}
        self.thr = {k: threshold_for_budget(self.platt[k](mb[k]), p.primary_budget) for k in MODELS}

    def margins(self, X):
        return {"logistic": self.logit.margin(X), "lgbm": self.booster.predict(lgbm_matrix(X), raw_score=True)}


def _test_frame(f: pd.DataFrame, test_ids: np.ndarray) -> pd.DataFrame:
    te = f[f["tx_id"].isin(test_ids)].sort_values("tx_id")
    assert len(te) == len(test_ids)
    return te


class Scorer:
    """Scores every scenario on the same test transactions with the same card resamples,
    so differences against the base scenario are paired."""

    def __init__(self, te: pd.DataFrame, p: Protocol):
        self.p = p
        self.ids = te["tx_id"].to_numpy()
        self.y = te["is_fraud"].to_numpy()
        self.novel = (self.y == 1) & te["category"].astype(str).isin(NOVEL_CATEGORIES).to_numpy()
        self.weights = list(card_bootstrap_weights(pd.factorize(te["card_id"])[0], p.n_boot, p.boot_seed))
        self.base = {}

    def metrics(self, rk: Ranked, prob, thr, w=None) -> dict:
        b = self.p.primary_budget
        ww = np.ones(len(self.y)) if w is None else w
        op = at_threshold(self.y, prob, thr, ww)
        out = {"ap": rk.average_precision(ww), "recall_top": rk.at_budget([b], ww)[b]["recall"],
               "op_alert_rate": op["alert_rate"], "op_recall": op["recall"], "op_precision": op["precision"]}
        if self.novel.any():
            alert = prob >= thr
            other = (self.y == 1) & ~self.novel
            out["op_recall_novel"] = float(np.sum(ww * self.novel * alert) / np.sum(ww * self.novel))
            out["op_recall_other"] = float(np.sum(ww * other * alert) / np.sum(ww * other))
        return out

    def score(self, name: str, fitted: Fitted, te: pd.DataFrame, note: str) -> dict:
        assert (te["tx_id"].to_numpy() == self.ids).all()
        m = fitted.margins(te)
        res = {"scenario": name, "note": note, "models": {}}
        for k in MODELS:
            prob = fitted.platt[k](m[k])
            rk = Ranked(self.y, m[k])
            point = self.metrics(rk, prob, fitted.thr[k])
            samples = {key: [] for key in point}
            for w in self.weights:
                for key, v in self.metrics(rk, prob, fitted.thr[k], w).items():
                    samples[key].append(v)
            entry = {key: {"value": r(point[key]), **ci(samples[key])} for key in point}
            if name == "base":
                self.base[k] = (point, samples)
            else:
                bp, bs = self.base[k]
                entry["delta_vs_base"] = {key: {"value": r(point[key] - bp[key]),
                                                **ci(np.subtract(samples[key], bs[key]))}
                                          for key in ("ap", "recall_top", "op_recall")}
            res["models"][k] = entry
            log(f"stress {name} {k}: AP={point['ap']:.4f} op_recall={point['op_recall']:.4f} "
                f"alert={point['op_alert_rate']:.4f}")
        return res


def run_stress(p: Protocol = PROTOCOL, ddir: Path | None = None, out: Path | None = None, threads: int = 4,
               settings: dict | None = None) -> dict:
    t0 = time.time()
    ddir = ddir or data_dir()
    out = out or reports_dir()
    s = settings or locked_settings()
    f = build_data(p, ddir)
    masks = split_masks(f, p)
    tr, vb = f[masks["train"]], f[masks["val_b"]]
    test_ids = np.sort(f.loc[masks["test"], "tx_id"].to_numpy())
    te = _test_frame(f, test_ids)
    sc = Scorer(te, p)
    results = []

    base = Fitted(tr, vb, s, p, threads)
    results.append(sc.score("base", base, te, "Model dengan pengaturan terkunci run 2, data apa adanya."))

    # (a) missing fraud labels for whole cards in train and validation
    pre = f["ts"] < pd.Timestamp(p.test[0], tz="UTC")
    fraud_cards = np.sort(f.loc[pre & (f["is_fraud"] == 1), "card_id"].unique())
    rng = np.random.default_rng(p.seed + 7)
    perm = rng.permutation(fraud_cards)
    for q in LABEL_MISSING:
        hidden = set(perm[: int(round(q * len(perm)))])
        g = f.copy()
        hide = pre & g["card_id"].isin(hidden) & (g["is_fraud"] == 1)
        g.loc[hide, "is_fraud"] = 0
        fit = Fitted(g[masks["train"]], g[masks["val_b"]], s, p, threads)
        results.append(sc.score(f"label_missing_{int(q * 100)}", fit, te,
                                f"Label fraud {int(q * 100)}% kartu fraud ({len(hidden)} kartu, {int(hide.sum())} "
                                "transaksi) dianggap sah di latih dan validasi; model dilatih ulang."))

    # (b) a fraud pattern never seen in training
    cat = f["category"].astype(str)
    drop = pre & (f["is_fraud"] == 1) & cat.isin(NOVEL_CATEGORIES)
    g = f[~drop]
    gm = split_masks(g, p)
    fit = Fitted(g[gm["train"]], g[gm["val_b"]], s, p, threads)
    results.append(sc.score("novel_pattern", fit, te,
                            f"{int(drop.sum())} transaksi fraud kategori {', '.join(NOVEL_CATEGORIES)} dihapus dari "
                            "latih dan validasi; model dilatih ulang."))

    # (c)-(e) shifts in the test period; base models, Platt and thresholds unchanged
    c = clean(pd.read_parquet(ddir / "raw_transactions.parquet"))
    t_test = pd.Timestamp(p.test[0], tz="UTC")
    in_test = c["tx_id"].isin(test_ids)

    def shifted(df):
        return _test_frame(build_features(df), test_ids)

    d = c.copy()
    d.loc[in_test, "amt"] = d.loc[in_test, "amt"] * AMOUNT_FACTOR
    results.append(sc.score("amount_up_25", base, shifted(d),
                            "Semua nominal periode uji dikali 1,25 (inflasi); fitur dihitung ulang; "
                            "tanpa latih ulang."))

    d = c.copy()
    d.loc[in_test, "unix_time"] = d.loc[in_test, "unix_time"] + CLOCK_SHIFT_SECONDS
    d.loc[in_test, "ts"] = d.loc[in_test, "ts"] + pd.Timedelta(seconds=CLOCK_SHIFT_SECONDS)
    results.append(sc.score("clock_shift_3h", base, shifted(d),
                            "Semua waktu periode uji digeser +3 jam (zona waktu pemroses berubah); fitur dihitung "
                            "ulang; tanpa latih ulang."))

    d = c[c["ts"] >= t_test]
    results.append(sc.score("cold_start", base, shifted(d),
                            "Riwayat sebelum 1 Juli 2020 dihapus, jadi setiap kartu mulai tanpa riwayat; fitur "
                            "dihitung ulang; tanpa latih ulang."))

    res = {"settings": s, "n_boot": p.n_boot, "primary_budget": p.primary_budget,
           "label_missing": list(LABEL_MISSING), "novel_categories": list(NOVEL_CATEGORIES),
           "test_rows": int(len(te)), "test_fraud": int(sc.y.sum()), "test_fraud_novel": int(sc.novel.sum()),
           "scenarios": results, "seconds": round(time.time() - t0, 1)}
    write_json(out / "stress.json", res)
    return res

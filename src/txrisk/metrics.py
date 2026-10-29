"""Metrics for rare-event ranking and calibration, with optional row weights.

Row weights let the card-level bootstrap reuse one sort per model: a bootstrap sample is
just the multiplicity of each card, broadcast to its rows. With unit weights the ranking
metrics equal scikit-learn's (tests/test_metrics.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed reliability bins (probability). Dense near zero because fraud is rare.
RELIABILITY_EDGES = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 1.0])


@dataclass
class Ranked:
    """Scores sorted once (descending, ties kept together)."""

    y: np.ndarray
    s: np.ndarray
    amt: np.ndarray | None = None

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.s = np.asarray(self.s, dtype=float)
        self.order = np.argsort(-self.s, kind="mergesort")
        ss = self.s[self.order]
        self.y_sorted = self.y[self.order]
        self.amt_sorted = None if self.amt is None else np.asarray(self.amt, dtype=float)[self.order]
        # index of the last element of each tie group
        self.group_end = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]

    def _cum(self, w):
        ws = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)[self.order]
        tp = np.cumsum(ws * self.y_sorted)
        fp = np.cumsum(ws * (1 - self.y_sorted))
        return ws, tp, fp

    def average_precision(self, w=None) -> float:
        _, tp, fp = self._cum(w)
        tp_g, fp_g = tp[self.group_end], fp[self.group_end]
        P = tp_g[-1]
        if P == 0:
            return float("nan")
        prec = tp_g / np.maximum(tp_g + fp_g, 1e-300)
        rec = tp_g / P
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))

    def roc_auc(self, w=None) -> float:
        _, tp, fp = self._cum(w)
        tp_g, fp_g = tp[self.group_end], fp[self.group_end]
        P, N = tp_g[-1], fp_g[-1]
        tp_prev = np.r_[0.0, tp_g[:-1]]
        d_fp = np.diff(np.r_[0.0, fp_g])
        d_tp = tp_g - tp_prev
        # negatives in a group are outranked by all earlier positives, ties count half
        return float(np.sum(d_fp * (tp_prev + 0.5 * d_tp)) / (P * N))

    def at_budget(self, budgets, w=None) -> dict:
        """Top-k metrics where k is a share of (weighted) transactions."""
        ws, tp, fp = self._cum(w)
        cw = np.cumsum(ws)
        W, P = cw[-1], tp[-1]
        if self.amt_sorted is not None:
            fraud_amt = np.cumsum(ws * self.y_sorted * self.amt_sorted)
        out = {}
        for b in budgets:
            k = int(np.searchsorted(cw, b * W, side="right"))  # rows with cumulative weight <= b*W
            if k == 0:
                rec = prec = amt_rec = 0.0
            else:
                rec = tp[k - 1] / P
                prec = tp[k - 1] / cw[k - 1]
                amt_rec = fraud_amt[k - 1] / fraud_amt[-1] if self.amt_sorted is not None else float("nan")
            out[b] = {"recall": float(rec), "precision": float(prec), "amount_recall": float(amt_rec)}
        return out

    def curve(self, budgets) -> list[dict]:
        res = self.at_budget(budgets)
        return [{"budget": float(b), **res[b]} for b in budgets]

    def pr_curve(self, max_points: int = 300) -> list[dict]:
        _, tp, fp = self._cum(None)
        tp_g, fp_g = tp[self.group_end], fp[self.group_end]
        P = tp_g[-1]
        rec = tp_g / P
        prec = tp_g / (tp_g + fp_g)
        thr = self.s[self.order][self.group_end]
        # keep points evenly spread in recall
        targets = np.linspace(0, 1, max_points)
        idx = np.unique(np.minimum(np.searchsorted(rec, targets, side="left"), len(rec) - 1))
        return [{"recall": float(rec[i]), "precision": float(prec[i]), "threshold": float(thr[i])}
                for i in idx]


def threshold_for_budget(scores: np.ndarray, budget: float) -> float:
    """Score threshold that flags about `budget` share of these (validation) scores."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - budget, method="higher"))


def at_threshold(y, s, thr, w=None, amt=None) -> dict:
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    alert = (np.asarray(s) >= thr).astype(float)
    P = np.sum(w * y)
    N = np.sum(w * (1 - y))
    tp = np.sum(w * y * alert)
    fp = np.sum(w * (1 - y) * alert)
    res = {
        "alert_rate": float(np.sum(w * alert) / np.sum(w)),
        "recall": float(tp / P) if P > 0 else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp > 0 else float("nan"),
        "fpr": float(fp / N) if N > 0 else float("nan"),
    }
    if amt is not None:
        a = np.asarray(amt, dtype=float)
        res["amount_recall"] = float(np.sum(w * y * alert * a) / np.sum(w * y * a))
    return res


def calibration(y, p, w=None, edges=RELIABILITY_EDGES) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    W = w.sum()
    brier = float(np.sum(w * (p - y) ** 2) / W)
    logloss = float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / W)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    wb = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    pb = np.bincount(idx, weights=w * p, minlength=len(edges) - 1)
    yb = np.bincount(idx, weights=w * y, minlength=len(edges) - 1)
    ece = float(np.sum(np.abs(pb - yb)) / W)
    # standard 10 equal-width bins, reported for comparability
    idx10 = np.clip((p * 10).astype(int), 0, 9)
    ece10 = float(np.sum(np.abs(np.bincount(idx10, weights=w * p, minlength=10)
                                - np.bincount(idx10, weights=w * y, minlength=10))) / W)
    # calibration where alerts happen: bins with predicted p >= 0.01, normalised by their weight
    hi = edges[:-1] >= 0.01
    w_hi = wb[hi].sum()
    ece_alert = float(np.sum(np.abs(pb[hi] - yb[hi])) / w_hi) if w_hi > 0 else float("nan")
    bins = []
    for i in range(len(edges) - 1):
        if wb[i] > 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": float(wb[i]),
                         "mean_pred": float(pb[i] / wb[i]), "frac_pos": float(yb[i] / wb[i])})
    return {"brier": brier, "log_loss": logloss, "ece": ece, "ece10": ece10, "ece_alert_region": ece_alert,
            "mean_pred": float(np.sum(w * p) / W), "frac_pos": float(np.sum(w * y) / W), "bins": bins}


def card_bootstrap_weights(card_idx: np.ndarray, n_boot: int, seed: int):
    """Yield row weights for a bootstrap that resamples whole cards with replacement."""
    rng = np.random.default_rng(seed)
    n_cards = int(card_idx.max()) + 1
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_cards, n_cards), minlength=n_cards).astype(float)
        yield counts[card_idx]


def ci(values, alpha: float = 0.05) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"lo": float(np.quantile(v, alpha / 2)), "hi": float(np.quantile(v, 1 - alpha / 2)),
            "n_boot": int(len(v))}

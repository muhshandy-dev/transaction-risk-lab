"""Causal per-card behavioural features.

Every feature of transaction i uses only transactions of the same card that come before
i in the fixed order (unix_time, then tx_id). Nothing from the future, and no labels, is
used. tests/test_features.py checks this by changing future rows and recomputing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
              "shopping_net", "shopping_pos", "travel"]

# name -> short Indonesian label used in reports and on the website
FEATURES: dict[str, str] = {
    "amt": "Nominal transaksi (USD)",
    "category": "Kategori merchant",
    "hour": "Jam transaksi (UTC)",
    "dow": "Hari dalam pekan",
    "distance_km": "Jarak rumah ke merchant (km)",
    "secs_since_prev": "Detik sejak transaksi sebelumnya",
    "n_1h": "Jumlah transaksi kartu 1 jam terakhir",
    "n_24h": "Jumlah transaksi kartu 24 jam terakhir",
    "n_7d": "Jumlah transaksi kartu 7 hari terakhir",
    "amt_24h": "Total nominal kartu 24 jam terakhir",
    "amt_ratio_card": "Nominal dibagi rata-rata kartu 90 hari terakhir",
    "amt_ratio_card_cat": "Nominal dibagi rata-rata kartu di kategori ini (90 hari)",
    "cat_share_card": "Porsi transaksi kartu di kategori ini (90 hari)",
    "hourband_share_card": "Porsi transaksi kartu di rentang 4 jam ini (90 hari)",
}
NUMERIC = [f for f in FEATURES if f != "category"]
AUDIT = ["audit_gender", "audit_age_band", "audit_area"]
HABIT_WINDOW = 90 * 86400  # seconds
_BIG = np.int64(10_000_000_000)  # larger than any time span, keeps groups apart in one sorted key


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b, NaN where b is 0 or NaN (no history yet)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    ok = np.isfinite(b) & (b != 0)
    np.divide(a, b, out=out, where=ok)
    return out


def _windowed(group_idx: np.ndarray, t: np.ndarray, v: np.ndarray, window: int):
    """Prior count and prior sum of v over (t - window, t] within each group, excluding the row.

    Rows must be sorted by (group, t, tie-breaker)."""
    key = group_idx.astype(np.int64) * _BIG + t.astype(np.int64)
    pos = np.arange(len(key))
    lo = np.searchsorted(key, key - window, side="right")
    lo = np.minimum(lo, pos)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    return (pos - lo).astype(float), cs[pos] - cs[lo]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (same row order as input) with feature columns added."""
    base = df.sort_values(["card_id", "unix_time", "tx_id"], kind="mergesort")
    order = base.index.to_numpy()
    card = pd.factorize(base["card_id"])[0]
    t = base["unix_time"].to_numpy(np.int64)
    amt = base["amt"].to_numpy(float)
    hour = base["ts"].dt.hour.to_numpy()
    out = pd.DataFrame(index=base.index)
    out["hour"] = hour
    out["dow"] = base["ts"].dt.day_of_week.to_numpy()

    prev_t = np.r_[np.nan, t[:-1].astype(float)]
    first = np.r_[True, card[1:] != card[:-1]]
    gap = t - prev_t
    gap[first] = np.nan
    out["secs_since_prev"] = gap

    for name, w in (("n_1h", 3600), ("n_24h", 86400), ("n_7d", 7 * 86400)):
        out[name] = _windowed(card, t, amt, w)[0]
    out["amt_24h"] = _windowed(card, t, amt, 86400)[1]

    # habit features over a bounded look-back so they mean the same thing in every period
    # (an all-history version drifted as cards accumulated history, see docs/EVALUASI.md)
    n_prior, s_prior = _windowed(card, t, amt, HABIT_WINDOW)
    out["amt_ratio_card"] = _safe_div(amt, _safe_div(s_prior, n_prior))

    cat_codes = pd.Categorical(base["category"], categories=CATEGORIES).codes
    if (cat_codes < 0).any():
        raise ValueError("unknown category")
    hb = hour // 4
    for codes, fname in ((cat_codes, "cat"), (hb, "hb")):
        # sort by card, code, then time order inside the card (position in `base`)
        sub_order = np.lexsort((np.arange(len(t)), codes, card))
        g = card[sub_order].astype(np.int64) * 100 + codes[sub_order]
        n_g, s_g = _windowed(g, t[sub_order], amt[sub_order], HABIT_WINDOW)
        n_g_full = np.empty_like(n_g)
        s_g_full = np.empty_like(s_g)
        n_g_full[sub_order] = n_g
        s_g_full[sub_order] = s_g
        share = _safe_div(n_g_full, n_prior)
        if fname == "cat":
            out["cat_share_card"] = share
            out["amt_ratio_card_cat"] = _safe_div(amt, _safe_div(s_g_full, n_g_full))
        else:
            out["hourband_share_card"] = share
    out = out.loc[df.index]
    res = df.copy()
    for c in out.columns:
        res[c] = out[c].to_numpy()
    res["category"] = pd.Categorical(res["category"], categories=CATEGORIES)
    assert (order == base.index.to_numpy()).all()
    return res


def split_masks(df: pd.DataFrame, protocol) -> dict[str, np.ndarray]:
    ts = df["ts"]
    masks = {}
    for name in ("train", "val_a", "val_b", "test"):
        lo, hi = getattr(protocol, name)
        masks[name] = ((ts >= pd.Timestamp(lo, tz="UTC")) & (ts < pd.Timestamp(hi, tz="UTC"))).to_numpy()
    return masks

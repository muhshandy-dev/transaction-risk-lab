"""Export reports, models and sample cases as static JSON for the website.

Nothing personal is exported: cases carry a transaction number, time, category, amount and
the behavioural features only. Precomputed Python values are included next to every case so
the browser can show, and the tests can check, that its live computation agrees.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import PROTOCOL, ROOT, data_dir, reports_dir
from .features import CATEGORIES, FEATURES, split_masks
from .models import LGBM_FEATURES, LinearSpec, LogisticModel, lgbm_matrix, sigmoid
from .pipeline import write_json

MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _num(x, nd=None):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd) if nd is not None else x


def compact_tree(node: dict) -> dict:
    cols = {k: [] for k in ("f", "t", "cats", "dl", "mt", "l", "r", "v", "c")}

    def visit(n) -> int:
        i = len(cols["f"])
        for k in cols:
            cols[k].append(None)
        if "leaf_value" in n:
            cols["f"][i] = -1
            cols["v"][i] = n["leaf_value"]
            cols["c"][i] = n.get("leaf_count", 0)
            cols["l"][i] = cols["r"][i] = -1
            cols["dl"][i] = cols["mt"][i] = 0
            return i
        cols["f"][i] = n["split_feature"]
        if n["decision_type"] == "==":
            cols["cats"][i] = [int(c) for c in str(n["threshold"]).split("||")]
        else:
            cols["t"][i] = float(n["threshold"])
        cols["dl"][i] = 1 if n["default_left"] else 0
        cols["mt"][i] = MISSING[n["missing_type"]]
        cols["v"][i] = None  # internal values are not needed for prediction or TreeSHAP
        cols["c"][i] = n["internal_count"]
        cols["l"][i] = visit(n["left_child"])
        cols["r"][i] = visit(n["right_child"])
        return i

    visit(node)
    if all(c is None for c in cols["cats"]):
        del cols["cats"]
    return cols


def export_lgbm(booster) -> dict:
    d = booster.dump_model()
    names = d["feature_names"]
    assert names == LGBM_FEATURES, names
    return {"type": "lgbm", "feature_names": names, "categories": CATEGORIES,
            "num_trees": len(d["tree_info"]), "trees": [compact_tree(t["tree_structure"]) for t in d["tree_info"]]}


def load_logistic(spec: dict) -> LogisticModel:
    from sklearn.linear_model import LogisticRegression

    m = LogisticModel(spec["C"])
    s = LinearSpec()
    s.interactions = False
    s.numeric = spec["numeric"]
    s.onehot = spec["onehot"]
    s.onehot["hour"] = [int(v) for v in s.onehot["hour"]]
    s.onehot["dow"] = [int(v) for v in s.onehot["dow"]]
    s.columns = [tuple(c.split(":", 1)) for c in spec["columns"]]
    m.spec = s
    m.clf = LogisticRegression()
    m.clf.coef_ = np.array([spec["coef"]])
    m.clf.intercept_ = np.array([spec["intercept"]])
    m.clf.classes_ = np.array([0, 1])
    m.design_mean = np.array(spec["design_mean"])
    return m


def feature_record(row) -> dict:
    out = {}
    for f in LGBM_FEATURES:
        v = row[f]
        out[f] = str(v) if f == "category" else _num(v)
    return out


def export(reports: Path | None = None, ddir: Path | None = None, web: Path | None = None) -> None:
    import lightgbm as lgb

    reports = reports or reports_dir()
    artifacts = reports.parent / "artifacts"
    web = web or ROOT / "web"
    pub = web / "public" / "data"
    pub.mkdir(parents=True, exist_ok=True)

    res = json.loads((reports / "results.json").read_text())
    booster = lgb.Booster(model_file=str(artifacts / "lgbm.txt"))
    logit_spec = json.loads((artifacts / "logistic.json").read_text())
    logit = load_logistic(json.loads(json.dumps(logit_spec)))
    calib = json.loads((artifacts / "calibration.json").read_text())
    thresholds = json.loads((artifacts / "thresholds.json").read_text())
    failures = json.loads((reports / "failures.json").read_text())

    # run date is stored in the report; file modification time changes with every checkout
    generated = res.get("evaluated") or datetime.fromtimestamp((reports / "results.json").stat().st_mtime,
                                                               UTC).strftime("%Y-%m-%d")
    manifest_path = (ddir or data_dir()) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    meta = {"version": __version__, "evaluated": generated, "protocol": res["protocol"], "data": res["data"],
            "manifest": manifest, "environment": res["environment"], "selection_best": res["selection_best"],
            "features": [{"name": k, "label": v} for k, v in FEATURES.items()]}
    write_json(pub / "meta.json", meta)
    write_json(pub / "metrics.json", {k: res[k] for k in ("models", "ci", "paired_lgbm_minus_logistic",
                                                          "paired_lgbm_minus_logistic_ix", "budget_curves")})
    for name in ("explain", "fairness", "failures", "ablations", "selection", "external_ulb", "stress"):
        p = reports / f"{name}.json"
        if p.exists():
            write_json(pub / f"{name}.json", json.loads(p.read_text()))
    models_dir = pub / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lgbm.json").write_text(json.dumps(export_lgbm(booster), separators=(",", ":"), allow_nan=False))
    write_json(models_dir / "logistic.json", logit_spec)
    write_json(models_dir / "calibration.json", {"platt": calib, "thresholds": thresholds,
                                                 "primary_budget": PROTOCOL.primary_budget})

    # cases and parity fixture come from the test period
    f = pd.read_parquet((ddir or data_dir()) / "features.parquet")
    f["category"] = pd.Categorical(f["category"].astype(str), categories=CATEGORIES)
    te = f[split_masks(f, PROTOCOL)["test"]].set_index("tx_id", drop=False)

    def py_values(rows: pd.DataFrame):
        X = lgbm_matrix(rows)
        margin = booster.predict(X, raw_score=True)
        contrib = booster.predict(X, pred_contrib=True)
        lc, lbase, lnames = logit.contributions(rows)
        lm = logit.margin(rows)
        return margin, contrib, lm, lc, lbase, lnames

    cases = []
    for bucket, ids in failures["case_ids"].items():
        rows = te.loc[ids]
        margin, contrib, lm, lc, lbase, lnames = py_values(rows)
        for i, (_, row) in enumerate(rows.iterrows()):
            amt = float(row["amt"])
            cases.append({
                "id": f"T-{int(row['tx_id']):07d}", "bucket": bucket, "ts": row["ts"].strftime("%Y-%m-%d %H:%M"),
                "category": str(row["category"]), "amt": amt, "is_fraud": int(row["is_fraud"]),
                "features": feature_record(row),
                "card_mean_90d": _num(amt / row["amt_ratio_card"]) if row["amt_ratio_card"] else None,
                "card_cat_mean_90d": _num(amt / row["amt_ratio_card_cat"]) if row["amt_ratio_card_cat"] else None,
                "python": {"lgbm_margin": float(margin[i]), "lgbm_shap": contrib[i, :-1].tolist(),
                           "lgbm_base": float(contrib[i, -1]), "logistic_margin": float(lm[i]),
                           "logistic_contrib": dict(zip(lnames, lc[i].tolist(), strict=True)),
                           "logistic_base": lbase,
                           "lgbm_prob": float(sigmoid(calib["lgbm"]["a"] * margin[i] + calib["lgbm"]["b"])),
                           "logistic_prob": float(sigmoid(calib["logistic"]["a"] * lm[i] + calib["logistic"]["b"]))},
            })
    cases.sort(key=lambda c: -c["python"]["lgbm_margin"])
    write_json(pub / "cases.json", {"model_version": __version__, "evaluated": generated, "cases": cases})

    rng = np.random.default_rng(99)
    pick = rng.choice(len(te), size=400, replace=False)
    rows = te.iloc[np.sort(pick)]
    margin, contrib, lm, _, _, _ = py_values(rows)
    fixture = {"rows": [feature_record(r) for _, r in rows.iterrows()], "lgbm_margin": margin.tolist(),
               "lgbm_contrib": contrib.tolist(), "logistic_margin": lm.tolist()}
    fix = web / "tests" / "fixtures"
    fix.mkdir(parents=True, exist_ok=True)
    (fix / "parity.json").write_text(json.dumps(fixture))
    print(f"exported {len(cases)} cases, {booster.num_trees()} trees to {pub}")

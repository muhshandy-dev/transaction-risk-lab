"""End-to-end protocol: data, selection, calibration, single test evaluation, analysis.

Order matters and is part of the protocol (docs/EVALUASI.md):
  1. data      generate (Sparkov) -> clean (drop PII) -> causal features
  2. select    hyper-parameters on validation A only
  3. calibrate Platt scaling and budget thresholds on validation B only
  4. test      every model scored once on the test period, card-level bootstrap
  5. analyse   SHAP, failures, group audit, ablations (ablations use validation B)
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import PROTOCOL, Protocol, data_dir, reports_dir
from .features import AUDIT, CATEGORIES, FEATURES, build_features, split_masks
from .generate import SPARKOV_COMMIT, generate
from .ingest import assert_no_pii, clean
from .metrics import Ranked, at_threshold, calibration, card_bootstrap_weights, ci, threshold_for_budget
from .models import LGBM_FEATURES, LogisticModel, Platt, fit_lgbm, lgbm_matrix, lgbm_params, sigmoid

LOGIT_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LGBM_GRID = ((15, 50), (15, 200), (31, 50), (31, 200), (63, 50), (63, 200))
MODEL_LABELS = {
    "random": "Acak (prevalensi)",
    "rule_amount": "Aturan: nominal terbesar",
    "logistic": "Regresi logistik",
    "logistic_ix": "Regresi logistik + interaksi kategori x nominal",
    "lgbm": "LightGBM",
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def frame_sha256(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def r(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# --------------------------------------------------------------------------- 1. data
def build_data(p: Protocol = PROTOCOL, ddir: Path | None = None, force: bool = False) -> pd.DataFrame:
    ddir = ddir or data_dir()
    feat_path = ddir / "features.parquet"
    raw_path = ddir / "raw_transactions.parquet"
    if feat_path.exists() and not force:
        f = pd.read_parquet(feat_path)
        f["category"] = pd.Categorical(f["category"].astype(str), categories=CATEGORIES)
        return f
    if not raw_path.exists() or force:
        log(f"generating Sparkov data: {p.n_customers} customers {p.start_date}..{p.end_date}")
        generate(p.n_customers, p.start_date, p.end_date, p.seed, ddir, p.workers)
    raw = pd.read_parquet(raw_path)
    c = clean(raw)
    assert_no_pii(c)
    f = build_features(c)
    f.to_parquet(feat_path, index=False)
    manifest = {
        "generator": "Sparkov_Data_Generation", "generator_commit": SPARKOV_COMMIT, "licence": "MIT",
        "n_customers": p.n_customers, "start_date": p.start_date, "end_date": p.end_date, "seed": p.seed,
        "rows": int(len(c)), "cards": int(c["card_id"].nunique()), "fraud_rows": int(c["is_fraud"].sum()),
        "raw_sha256": frame_sha256(raw.sort_values(["unix_time", "cc_num", "trans_num"]).reset_index(drop=True)),
        "clean_sha256": frame_sha256(c),
    }
    write_json(ddir / "manifest.json", manifest)
    f["category"] = pd.Categorical(f["category"].astype(str), categories=CATEGORIES)
    return f


def data_summary(f: pd.DataFrame, masks: dict) -> dict:
    out = {}
    for name, m in masks.items():
        sub = f[m]
        out[name] = {"rows": int(m.sum()), "fraud": int(sub["is_fraud"].sum()),
                     "fraud_rate": r(sub["is_fraud"].mean()), "cards": int(sub["card_id"].nunique()),
                     "fraud_cards": int(sub.loc[sub["is_fraud"] == 1, "card_id"].nunique()),
                     "start": str(sub["ts"].min()), "end": str(sub["ts"].max())}
    return out


# --------------------------------------------------------------------------- 2. select
def select_models(tr: pd.DataFrame, va: pd.DataFrame, threads: int) -> dict:
    ytr, yva = tr["is_fraud"].to_numpy(), va["is_fraud"].to_numpy()
    sel = {"logistic": [], "logistic_ix": [], "lgbm": []}
    for kind, ix in (("logistic", False), ("logistic_ix", True)):
        for C in LOGIT_GRID:
            t0 = time.time()
            m = LogisticModel(C, interactions=ix).fit(tr, ytr)
            ap = Ranked(yva, m.margin(va)).average_precision()
            sel[kind].append({"C": C, "val_a_ap": r(ap), "seconds": round(time.time() - t0, 1)})
            log(f"{kind} C={C}: val A AP={ap:.4f}")
    for nl, mcs in LGBM_GRID:
        t0 = time.time()
        b = fit_lgbm(tr, ytr, lgbm_params(nl, mcs, 0, threads), 5000, va, yva, early_stopping=200)
        ap = Ranked(yva, b.predict(lgbm_matrix(va), raw_score=True, num_iteration=b.best_iteration))
        ap = ap.average_precision()
        sel["lgbm"].append({"num_leaves": nl, "min_child_samples": mcs, "best_iteration": int(b.best_iteration),
                            "val_a_ap": r(ap), "seconds": round(time.time() - t0, 1)})
        log(f"lgbm leaves={nl} mcs={mcs}: iters={b.best_iteration} val A AP={ap:.4f}")
    best = {k: max(v, key=lambda d: (d["val_a_ap"], -v.index(d))) for k, v in sel.items()}
    return {"grid": sel, "best": best}


def fit_final(tr: pd.DataFrame, best: dict, threads: int, seed: int = 0):
    ytr = tr["is_fraud"].to_numpy()
    logit = LogisticModel(best["logistic"]["C"]).fit(tr, ytr)
    logit_ix = LogisticModel(best["logistic_ix"]["C"], interactions=True).fit(tr, ytr)
    b = best["lgbm"]
    booster = fit_lgbm(tr, ytr, lgbm_params(b["num_leaves"], b["min_child_samples"], seed, threads),
                       b["best_iteration"])
    return logit, logit_ix, booster


def margins(models, X: pd.DataFrame) -> dict[str, np.ndarray]:
    logit, logit_ix, booster = models
    return {
        "rule_amount": np.log(X["amt"].to_numpy(float)),
        "logistic": logit.margin(X),
        "logistic_ix": logit_ix.margin(X),
        "lgbm": booster.predict(lgbm_matrix(X), raw_score=True),
    }


# --------------------------------------------------------------------------- 4. test
def bootstrap_test(y, amt, card_idx, scores: dict, probs: dict, thresholds: dict, p: Protocol) -> dict:
    ranked = {k: Ranked(y, s, amt) for k, s in scores.items()}
    keys = list(scores)
    samples = {k: {"ap": [], "auc": [], **{f"recall@{b}": [] for b in p.budgets},
                   **{f"precision@{b}": [] for b in p.budgets}, "op_recall": [], "op_alert_rate": [],
                   "op_precision": [], "brier": [], "ece": []} for k in keys}
    diff = {"ap": [], f"recall@{p.primary_budget}": [], "brier": [], "op_recall": []}
    diff_ix = {"ap": [], f"recall@{p.primary_budget}": []}
    for w in card_bootstrap_weights(card_idx, p.n_boot, p.boot_seed):
        for k in keys:
            rk = ranked[k]
            s = samples[k]
            s["ap"].append(rk.average_precision(w))
            s["auc"].append(rk.roc_auc(w))
            ab = rk.at_budget(p.budgets, w)
            for b in p.budgets:
                s[f"recall@{b}"].append(ab[b]["recall"])
                s[f"precision@{b}"].append(ab[b]["precision"])
            if k in probs:
                op = at_threshold(y, probs[k], thresholds[k][p.primary_budget], w)
                s["op_recall"].append(op["recall"])
                s["op_alert_rate"].append(op["alert_rate"])
                s["op_precision"].append(op["precision"])
                cal = calibration(y, probs[k], w)
                s["brier"].append(cal["brier"])
                s["ece"].append(cal["ece"])
        L, G, X = samples["logistic"], samples["lgbm"], samples["logistic_ix"]
        diff["ap"].append(G["ap"][-1] - L["ap"][-1])
        diff[f"recall@{p.primary_budget}"].append(G[f"recall@{p.primary_budget}"][-1]
                                                  - L[f"recall@{p.primary_budget}"][-1])
        diff["brier"].append(G["brier"][-1] - L["brier"][-1])
        diff["op_recall"].append(G["op_recall"][-1] - L["op_recall"][-1])
        diff_ix["ap"].append(G["ap"][-1] - X["ap"][-1])
        diff_ix[f"recall@{p.primary_budget}"].append(G[f"recall@{p.primary_budget}"][-1]
                                                     - X[f"recall@{p.primary_budget}"][-1])
    cis = {k: {m: ci(v) for m, v in d.items() if len(v)} for k, d in samples.items()}

    def summarise(d):
        return {m: {"mean": r(np.mean(v)), **ci(v), "share_gt_0": r(np.mean(np.asarray(v) > 0))}
                for m, v in d.items()}

    return {"ci": cis, "paired_lgbm_minus_logistic": summarise(diff),
            "paired_lgbm_minus_logistic_ix": summarise(diff_ix)}


def evaluate(f: pd.DataFrame, masks: dict, models, p: Protocol) -> tuple[dict, dict]:
    vb, te = f[masks["val_b"]], f[masks["test"]]
    yb, yt = vb["is_fraud"].to_numpy(), te["is_fraud"].to_numpy()
    mb, mt = margins(models, vb), margins(models, te)
    platt = {k: Platt().fit(mb[k], yb) for k in ("logistic", "logistic_ix", "lgbm")}
    from sklearn.isotonic import IsotonicRegression

    iso = {k: IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(mb[k], yb)
           for k in platt}
    probs_b = {k: platt[k](mb[k]) for k in platt}
    probs_t = {k: platt[k](mt[k]) for k in platt}
    thresholds = {k: {b: threshold_for_budget(probs_b[k], b) for b in p.budgets} for k in platt}
    amt = te["amt"].to_numpy(float)
    card_idx = pd.factorize(te["card_id"])[0]

    rng = np.random.default_rng(p.seed)
    scores = dict(mt)
    scores["random"] = rng.random(len(te))
    res = {"models": {}}
    for k, s in scores.items():
        rk = Ranked(yt, s, amt)
        entry = {"label": MODEL_LABELS[k], "ap": r(rk.average_precision()), "roc_auc": r(rk.roc_auc()),
                 "at_budget": {str(b): {m: r(v) for m, v in d.items()} for b, d in rk.at_budget(p.budgets).items()}}
        if k in platt:
            entry["calibration"] = {
                "raw": calibration(yt, sigmoid(mt[k])),
                "platt": calibration(yt, probs_t[k]),
                "isotonic": calibration(yt, iso[k].predict(mt[k])),
            }
            entry["platt"] = {"a": platt[k].a, "b": platt[k].b}
            entry["operational"] = {
                str(b): {"threshold": r(thresholds[k][b], 9), **at_threshold(yt, probs_t[k], thresholds[k][b],
                                                                            amt=amt)}
                for b in p.budgets}
            entry["pr_curve"] = rk.pr_curve(200)
        res["models"][k] = entry
    log("bootstrap on test (cards resampled)")
    boot = bootstrap_test(yt, amt, card_idx, {k: scores[k] for k in ("rule_amount", "logistic", "logistic_ix",
                                                                    "lgbm")},
                          probs_t, thresholds, p)
    res.update(boot)
    budget_grid = np.unique(np.r_[np.geomspace(0.0005, 0.05, 60), list(p.budgets)])
    res["budget_curves"] = {k: Ranked(yt, scores[k], amt).curve(budget_grid)
                            for k in ("rule_amount", "logistic", "lgbm", "random")}
    ctx = {"probs_t": probs_t, "margins_t": mt, "thresholds": thresholds, "platt": platt, "test": te}
    return res, ctx


# --------------------------------------------------------------------------- 5. analyse
def explain(models, f: pd.DataFrame, masks: dict, best: dict, p: Protocol, threads: int) -> dict:
    import shap

    logit, _, booster = models
    te = f[masks["test"]]
    rng = np.random.default_rng(p.seed + 1)
    idx = np.sort(rng.choice(len(te), size=min(p.shap_sample, len(te)), replace=False))
    sample = te.iloc[idx]
    Xs = lgbm_matrix(sample)
    explainer = shap.TreeExplainer(booster)
    sv = np.asarray(explainer.shap_values(Xs))
    if sv.ndim == 3:  # some shap versions return [neg, pos]
        sv = sv[-1]
    base = float(np.ravel(explainer.expected_value)[-1])
    raw = booster.predict(Xs, raw_score=True)
    additivity = float(np.max(np.abs(sv.sum(1) + base - raw)))
    contrib = booster.predict(Xs, pred_contrib=True)
    vs_lgbm_contrib = float(np.max(np.abs(contrib[:, :-1] - sv)))
    lc, lbase, lnames = logit.contributions(sample)
    logit_additivity = float(np.max(np.abs(lc.sum(1) + lbase - logit.margin(sample))))

    imp_lgbm = dict(zip(LGBM_FEATURES, np.abs(sv).mean(0), strict=True))
    imp_logit = dict(zip(lnames, np.abs(lc).mean(0), strict=True))

    # stability across training seeds (bagging and feature sub-sampling change)
    seeds = []
    b = best["lgbm"]
    tr = f[masks["train"]]
    for seed in p.shap_seeds:
        bs = fit_lgbm(tr, tr["is_fraud"].to_numpy(), lgbm_params(b["num_leaves"], b["min_child_samples"], seed,
                                                                threads), b["best_iteration"])
        c = bs.predict(Xs, pred_contrib=True)[:, :-1]
        seeds.append(np.abs(c).mean(0))
    S = np.array(seeds)
    ranks = np.argsort(np.argsort(-S, axis=1), axis=1)
    from scipy.stats import spearmanr

    rhos = [spearmanr(S[i], S[j]).statistic for i in range(len(S)) for j in range(i + 1, len(S))]
    top5 = [set(np.argsort(-S[i])[:5]) for i in range(len(S))]
    overlap = [len(top5[i] & top5[j]) / 5 for i in range(len(S)) for j in range(i + 1, len(S))]

    dep = {}
    order = sorted(imp_lgbm, key=imp_lgbm.get, reverse=True)
    pick = rng.choice(len(sample), size=min(700, len(sample)), replace=False)
    for feat in order[:6]:
        j = LGBM_FEATURES.index(feat)
        col = sample[feat]
        vals = col.astype(str).to_numpy() if feat == "category" else col.to_numpy(float)
        dep[feat] = [{"x": (vals[i] if feat == "category" else r(vals[i], 4)), "shap": r(sv[i, j], 5),
                      "fraud": int(sample["is_fraud"].iloc[i])} for i in pick]
    return {
        "sample_size": int(len(sample)),
        "lgbm_base_value": base,
        "logistic_base_value": lbase,
        "checks": {"lgbm_additivity_max_abs": additivity, "shap_vs_lightgbm_pred_contrib_max_abs": vs_lgbm_contrib,
                   "logistic_additivity_max_abs": logit_additivity},
        "global": {
            "lgbm": [{"feature": k, "label": FEATURES[k], "mean_abs_shap": r(v)} for k, v in
                     sorted(imp_lgbm.items(), key=lambda kv: -kv[1])],
            "logistic": [{"feature": k, "label": FEATURES[k], "mean_abs_contrib": r(v)} for k, v in
                         sorted(imp_logit.items(), key=lambda kv: -kv[1])],
        },
        "seed_stability": {
            "seeds": list(p.shap_seeds),
            "spearman_min": r(min(rhos)), "spearman_mean": r(np.mean(rhos)),
            "top5_overlap_min": r(min(overlap)), "top5_overlap_mean": r(np.mean(overlap)),
            "rank_by_seed": {LGBM_FEATURES[j]: ranks[:, j].tolist() for j in range(len(LGBM_FEATURES))},
        },
        "dependence": dep,
        "logistic_coefficients": [{"column": c, "coef": r(v)} for c, v in
                                  zip(logit.export()["columns"], logit.export()["coef"], strict=True)],
    }


def failures(ctx: dict, p: Protocol, booster) -> dict:
    te = ctx["test"].copy()
    b = p.primary_budget
    te["alert"] = ctx["probs_t"]["lgbm"] >= ctx["thresholds"]["lgbm"][b]
    te["alert_logit"] = ctx["probs_t"]["logistic"] >= ctx["thresholds"]["logistic"][b]
    te["night"] = (te["hour"] >= 22) | (te["hour"] < 4)
    fraud = te[te["is_fraud"] == 1]
    legit = te[te["is_fraud"] == 0]

    def by(df, col, flag):
        g = df.groupby(col, observed=True)[flag].agg(["mean", "size"])
        return [{"group": str(k), "rate": r(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()]

    amt_band = pd.cut(te["amt"], [0, 20, 100, 300, 800, 1e9], labels=["<20", "20-100", "100-300", "300-800",
                                                                       ">800"])
    te["amt_band"] = amt_band
    fraud = te[te["is_fraud"] == 1]
    legit = te[te["is_fraud"] == 0]

    # episodes: consecutive fraud rows of one card (Sparkov makes one window per card)
    ep = fraud.sort_values(["card_id", "unix_time"]).groupby("card_id")
    episodes = []
    for _, g in ep:
        hit = np.flatnonzero(g["alert"].to_numpy())
        episodes.append({"n_fraud": len(g), "detected": bool(len(hit)), "first_alert_pos": int(hit[0]) + 1
                         if len(hit) else None, "amount": float(g["amt"].sum()),
                         "amount_caught": float(g.loc[g["alert"], "amt"].sum())})
    E = pd.DataFrame(episodes)

    def case_rows(df, n, seed):
        if len(df) == 0:
            return []
        pick = df.sample(n=min(n, len(df)), random_state=seed)
        return pick["tx_id"].astype(int).tolist()

    return {
        "budget": b,
        "model": "lgbm",
        "threshold": ctx["thresholds"]["lgbm"][b],
        "fraud_missed": int((~fraud["alert"]).sum()), "fraud_total": int(len(fraud)),
        "false_alerts": int(legit["alert"].sum()), "alerts": int(te["alert"].sum()),
        "missed_by_lgbm_caught_by_logistic": int(((~fraud["alert"]) & fraud["alert_logit"]).sum()),
        "caught_by_lgbm_missed_by_logistic": int((fraud["alert"] & ~fraud["alert_logit"]).sum()),
        "recall_by_category": by(fraud, "category", "alert"),
        "recall_by_amount_band": by(fraud, "amt_band", "alert"),
        "recall_night_vs_day": by(fraud.assign(waktu=np.where(fraud["night"], "malam 22-04", "siang 04-22")),
                                  "waktu", "alert"),
        "false_alert_rate_by_category": by(legit, "category", "alert"),
        "false_alert_rate_by_amount_band": by(legit, "amt_band", "alert"),
        "episodes": {
            "n": int(len(E)), "detected": int(E["detected"].sum()), "detected_share": r(E["detected"].mean()),
            "median_first_alert_pos": r(E["first_alert_pos"].median()),
            "share_caught_by_2nd_tx": r(((E["first_alert_pos"] <= 2) & E["detected"]).mean()),
            "amount_caught_share": r(E["amount_caught"].sum() / E["amount"].sum()),
        },
        "case_ids": {
            "missed": case_rows(fraud[~fraud["alert"]], 12, 1),
            "false_alert": case_rows(legit[legit["alert"]], 12, 2),
            "caught": case_rows(fraud[fraud["alert"]], 12, 3),
            "quiet": case_rows(legit[~legit["alert"]], 8, 4),
        },
    }


def fairness(ctx: dict, p: Protocol) -> dict:
    te = ctx["test"]
    y = te["is_fraud"].to_numpy()
    b = p.primary_budget
    out = {"budget": b, "note": "Atribut kelompok tidak dipakai sebagai fitur model; hanya untuk audit.",
           "groups": {}}
    for attr in AUDIT:
        rows = []
        for gval in sorted(te[attr].unique()):
            m = (te[attr] == gval).to_numpy()
            sub_y = y[m]
            card_idx = pd.factorize(te.loc[m, "card_id"])[0]
            entry = {"group": str(gval), "n": int(m.sum()), "fraud": int(sub_y.sum()),
                     "cards": int(card_idx.max() + 1)}
            for k in ("logistic", "lgbm"):
                pk = ctx["probs_t"][k][m]
                thr = ctx["thresholds"][k][b]
                point = at_threshold(sub_y, pk, thr)
                tprs, fprs = [], []
                for w in card_bootstrap_weights(card_idx, 400, p.boot_seed + 7):
                    o = at_threshold(sub_y, pk, thr, w)
                    tprs.append(o["recall"])
                    fprs.append(o["fpr"])
                entry[k] = {"tpr": r(point["recall"]), "tpr_ci": ci(tprs), "fpr": r(point["fpr"], 7),
                            "fpr_ci": ci(fprs), "alert_rate": r(point["alert_rate"], 7)}
            rows.append(entry)
        out["groups"][attr] = rows
    return out


def ablations(f: pd.DataFrame, masks: dict, best: dict, threads: int) -> dict:
    """What the boosted model leans on. Scored on validation B (diagnostic, not selection)."""
    tr, vb = f[masks["train"]], f[masks["val_b"]]
    b = best["lgbm"]
    sets = {
        "semua fitur": [],
        "tanpa jam": ["hour", "hourband_share_card"],
        "tanpa nominal": ["amt", "amt_24h", "amt_ratio_card", "amt_ratio_card_cat"],
        "tanpa velocity": ["n_1h", "n_24h", "n_7d", "secs_since_prev", "amt_24h"],
        "tanpa kebiasaan kartu": ["amt_ratio_card", "amt_ratio_card_cat", "cat_share_card", "hourband_share_card"],
    }
    out = []
    for name, drop in sets.items():
        Xtr, Xvb = tr.copy(), vb.copy()
        for c in drop:
            Xtr[c] = 0.0 if c != "hour" else 0
            Xvb[c] = 0.0 if c != "hour" else 0
        bs = fit_lgbm(Xtr, tr["is_fraud"].to_numpy(), lgbm_params(b["num_leaves"], b["min_child_samples"], 0,
                                                                  threads), b["best_iteration"])
        s = bs.predict(lgbm_matrix(Xvb), raw_score=True)
        rk = Ranked(vb["is_fraud"].to_numpy(), s)
        out.append({"set": name, "dropped": drop, "val_b_ap": r(rk.average_precision()),
                    "val_b_recall_at_1pct": r(rk.at_budget([0.01])[0.01]["recall"])})
        log(f"ablation {name}: AP={out[-1]['val_b_ap']}")
    return {"split": "val_b", "model": "lgbm", "results": out}


# --------------------------------------------------------------------------- run all
def run(p: Protocol = PROTOCOL, out: Path | None = None, ddir: Path | None = None, threads: int = 4,
        skip_ablations: bool = False) -> dict:
    out = out or reports_dir()
    t0 = time.time()
    f = build_data(p, ddir)
    masks = split_masks(f, p)
    summary = data_summary(f, masks)
    log(f"data ready: {len(f)} rows; " + ", ".join(f"{k}={v['rows']}" for k, v in summary.items()))
    sel = select_models(f[masks["train"]], f[masks["val_a"]], threads)
    write_json(out / "selection.json", sel)
    models = fit_final(f[masks["train"]], sel["best"], threads)
    res, ctx = evaluate(f, masks, models, p)
    res["data"] = summary
    res["protocol"] = {k: v for k, v in p.__dict__.items() if k != "workers"}
    res["selection_best"] = sel["best"]
    res["environment"] = {"txrisk": __version__, "python": platform.python_version(),
                          "numpy": np.__version__, "pandas": pd.__version__,
                          "lightgbm": __import__("lightgbm").__version__,
                          "sklearn": __import__("sklearn").__version__, "shap": __import__("shap").__version__}
    write_json(out / "results.json", res)
    ex = explain(models, f, masks, sel["best"], p, threads)
    write_json(out / "explain.json", ex)
    fl = failures(ctx, p, models[2])
    write_json(out / "failures.json", fl)
    fa = fairness(ctx, p)
    write_json(out / "fairness.json", fa)
    if not skip_ablations:
        write_json(out / "ablations.json", ablations(f, masks, sel["best"], threads))
    artifacts = out.parent / "artifacts" if out.name == "reports" else out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    models[2].save_model(str(artifacts / "lgbm.txt"))
    write_json(artifacts / "logistic.json", models[0].export())
    write_json(artifacts / "calibration.json", {k: {"a": v.a, "b": v.b} for k, v in ctx["platt"].items()})
    write_json(artifacts / "thresholds.json", {k: {str(b): t for b, t in v.items()} for k, v in
                                               ctx["thresholds"].items()})
    log(f"done in {time.time() - t0:.0f}s")
    return res

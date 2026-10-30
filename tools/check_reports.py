"""Sanity checks on committed reports: protocol fields present, numbers in range, CIs bracket
the point estimate, and the website data was exported from these reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(reports: str) -> int:
    rp = Path(reports)
    r = json.loads((rp / "results.json").read_text())
    errors = []
    for k, m in r["models"].items():
        for key in ("ap", "roc_auc"):
            if not 0 <= m[key] <= 1:
                errors.append(f"{k}.{key} out of range")
        ci = r["ci"].get(k)
        if ci and not (ci["ap"]["lo"] - 0.02 <= m["ap"] <= ci["ap"]["hi"] + 0.02):
            errors.append(f"{k}: AP {m['ap']} outside its CI {ci['ap']}")
    ex = json.loads((rp / "explain.json").read_text())
    if ex["checks"]["lgbm_additivity_max_abs"] > 1e-6:
        errors.append("TreeSHAP additivity check failed")
    ext_path = rp / "external_ulb.json"
    if ext_path.exists():
        x = json.loads(ext_path.read_text())
        for k, m in x["models"].items():
            c = x["ci"][k]["ap"]
            if not (0 <= m["ap"] <= 1 and c["lo"] - 0.02 <= m["ap"] <= c["hi"] + 0.02):
                errors.append(f"external {k}: AP {m['ap']} outside range or CI {c}")
        if x["explain"]["checks"]["lgbm_additivity_max_abs"] > 1e-6:
            errors.append("external TreeSHAP additivity check failed")
        if x["data"]["test_rows_with_exact_copy_before_test"] != 0:
            errors.append("external test rows duplicated before the test period")
    st_path = rp / "stress.json"
    if st_path.exists():
        st = json.loads(st_path.read_text())
        base = st["scenarios"][0]
        if base["scenario"] != "base" or base["models"]["lgbm"]["ap"]["value"] != r["models"]["lgbm"]["ap"]:
            errors.append("stress base scenario does not reproduce run 2 LightGBM AP")
    web = rp.parent / "web" / "public" / "data" / "metrics.json"
    if web.exists():
        w = json.loads(web.read_text())
        if w["models"]["lgbm"]["ap"] != r["models"]["lgbm"]["ap"]:
            errors.append("web/public/data is not exported from these reports")
    for e in errors:
        print("GAGAL:", e)
    print("ok" if not errors else f"{len(errors)} masalah")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reports"))

"""Compare headline numbers of two report folders (committed vs fresh run).

Exit code 1 if any headline metric differs by more than the tolerance. Different CPUs can
change LightGBM's floating point summation order slightly, so exact equality is only
expected on the same machine; the tolerance covers that."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 0.005


def headline(path: Path) -> dict[str, float]:
    r = json.loads((path / "results.json").read_text())
    out = {}
    for k, m in r["models"].items():
        out[f"{k}.ap"] = m["ap"]
        out[f"{k}.roc_auc"] = m["roc_auc"]
        for b, v in m["at_budget"].items():
            out[f"{k}.recall@{b}"] = v["recall"]
        if "calibration" in m:
            out[f"{k}.brier_platt"] = m["calibration"]["platt"]["brier"]
    for k, v in r["paired_lgbm_minus_logistic"].items():
        out[f"paired.{k}.lo"] = v["lo"]
        out[f"paired.{k}.hi"] = v["hi"]
    return out


def main(a: str, b: str) -> int:
    ha, hb = headline(Path(a)), headline(Path(b))
    worst = 0.0
    bad = 0
    for k in sorted(ha):
        if k.startswith("random."):
            continue  # random scores are reported for orientation only
        d = abs(ha[k] - hb.get(k, float("nan")))
        worst = max(worst, d)
        flag = "" if d <= TOL else "  <-- beda"
        bad += bool(flag)
        print(f"{k:40s} {ha[k]:.6f} {hb.get(k, float('nan')):.6f} {d:.2e}{flag}")
    print(f"selisih terbesar {worst:.2e}; melewati toleransi {TOL}: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

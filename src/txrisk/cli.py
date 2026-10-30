"""Command line: `txrisk run` reproduces every number in reports/ from nothing."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import PROTOCOL


def small_protocol():
    """Tiny configuration for tests and CI smoke runs. Never used for reported results."""
    return dataclasses.replace(PROTOCOL, n_customers=60, seed=7, n_boot=30, shap_sample=500,
                               shap_seeds=(0, 1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="txrisk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("data", "run"):
        s = sub.add_parser(name)
        s.add_argument("--small", action="store_true", help="tiny smoke configuration")
        s.add_argument("--data", type=Path, default=None)
        s.add_argument("--out", type=Path, default=None)
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--force", action="store_true")
        s.add_argument("--skip-ablations", action="store_true")
    k = sub.add_parser("import-kaggle", help="convert Kaggle fraudTrain.csv/fraudTest.csv into raw input")
    k.add_argument("csv", type=Path, nargs="+")
    k.add_argument("--data", type=Path, required=True)
    e = sub.add_parser("external", help="evaluate the same protocol on the real ULB credit card dataset")
    e.add_argument("--csv", type=Path, default=None, help="creditcard.csv (downloaded and checked if absent)")
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--n-boot", type=int, default=None)
    st = sub.add_parser("stress", help="stress tests on the synthetic data with locked hyper-parameters")
    st.add_argument("--small", action="store_true")
    st.add_argument("--data", type=Path, default=None)
    st.add_argument("--out", type=Path, default=None)
    st.add_argument("--threads", type=int, default=4)
    w = sub.add_parser("export-web")
    w.add_argument("--reports", type=Path, default=None)
    w.add_argument("--data", type=Path, default=None)
    w.add_argument("--web", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "data":
        from .pipeline import build_data

        build_data(small_protocol() if args.small else PROTOCOL, args.data, force=args.force)
    elif args.cmd == "run":
        from .pipeline import run

        run(small_protocol() if args.small else PROTOCOL, args.out, args.data, args.threads,
            skip_ablations=args.skip_ablations)
    elif args.cmd == "import-kaggle":
        from .kaggle import import_kaggle

        print(import_kaggle(args.csv, args.data))
    elif args.cmd == "external":
        from .config import data_dir, reports_dir
        from .external import download, load, run_external, sha256_file
        from .pipeline import write_json

        csv = args.csv or download(data_dir() / "ulb" / "creditcard.csv")
        res = run_external(load(csv), PROTOCOL, args.threads, args.n_boot, {"file_sha256": sha256_file(csv)})
        write_json((args.out or reports_dir()) / "external_ulb.json", res)
    elif args.cmd == "stress":
        from .stress import run_stress

        run_stress(small_protocol() if args.small else PROTOCOL, args.data, args.out, args.threads)
    elif args.cmd == "export-web":
        from .export_web import export

        export(args.reports, args.data, args.web)


if __name__ == "__main__":
    main()

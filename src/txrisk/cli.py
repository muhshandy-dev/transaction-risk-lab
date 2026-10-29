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
    elif args.cmd == "export-web":
        from .export_web import export

        export(args.reports, args.data, args.web)


if __name__ == "__main__":
    main()

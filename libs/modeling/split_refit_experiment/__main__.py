"""Command-line validation for split/refit campaign artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import DEFAULT_SOURCE_CSV, materialize_split, verify_split
from .registry import validate_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m libs.modeling.split_refit_experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize-split", "verify-split"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        command.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
        command.add_argument("--strict", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--through", default="split")
    validate.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "materialize-split":
        result = materialize_split(args.campaign, source_csv=args.source_csv).as_dict()
    elif args.command == "verify-split":
        result = verify_split(
            args.campaign, source_csv=args.source_csv, strict=args.strict
        ).as_dict()
    else:
        registry = validate_registry(
            args.campaign, strict=args.strict, through=args.through
        )
        result = {"registry": registry.as_dict()}
        if args.through == "split":
            result["split"] = verify_split(
                args.campaign, source_csv=DEFAULT_SOURCE_CSV, strict=False
            ).as_dict()
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

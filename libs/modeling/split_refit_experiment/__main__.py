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
    preflight = commands.add_parser("preflight-evaluation")
    preflight.add_argument("--campaign", type=Path, required=True)
    preflight.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    fit = commands.add_parser("fit-evaluation")
    fit.add_argument("--campaign", type=Path, required=True)
    fit.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    fit.add_argument("--timeout-seconds", type=int, default=3300)
    child = commands.add_parser("fit-child", help=argparse.SUPPRESS)
    child.add_argument("--campaign", type=Path, required=True)
    child.add_argument("--source-csv", type=Path, required=True)
    score = commands.add_parser("score-evaluation")
    score.add_argument("--campaign", type=Path, required=True)
    score.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    verify = commands.add_parser("verify-evaluation")
    verify.add_argument("--campaign", type=Path, required=True)
    verify.add_argument("--recompute-all", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "materialize-split":
        result = materialize_split(args.campaign, source_csv=args.source_csv).as_dict()
    elif args.command == "verify-split":
        result = verify_split(
            args.campaign, source_csv=args.source_csv, strict=args.strict
        ).as_dict()
    elif args.command == "validate":
        registry = validate_registry(
            args.campaign, strict=args.strict, through=args.through
        )
        result = {"registry": registry.as_dict()}
        if args.through == "split":
            result["split"] = verify_split(
                args.campaign, source_csv=DEFAULT_SOURCE_CSV, strict=False
            ).as_dict()
    elif args.command == "preflight-evaluation":
        from .runner import durable_preflight

        result = durable_preflight(args.campaign, source_csv=args.source_csv)
    elif args.command == "fit-evaluation":
        from .runner import launch_fit

        result = launch_fit(
            args.campaign,
            source_csv=args.source_csv,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "fit-child":
        from .runner import fit_child

        result = fit_child(args.campaign, source_csv=args.source_csv)
    elif args.command == "score-evaluation":
        from .runner import score_evaluation

        result = score_evaluation(args.campaign, source_csv=args.source_csv)
    else:
        from .verification import verify_evaluation

        result = verify_evaluation(args.campaign, recompute_all=args.recompute_all)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

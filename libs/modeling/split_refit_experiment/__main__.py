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
    validate.add_argument("--through", default="final")
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
    preflight_refit = commands.add_parser("preflight-refit")
    preflight_refit.add_argument("--campaign", type=Path, required=True)
    preflight_refit.add_argument("--source-csv", type=Path, required=True)
    preflight_refit.add_argument("--strict", action="store_true")
    fit_refit = commands.add_parser("fit-refit")
    fit_refit.add_argument("--campaign", type=Path, required=True)
    fit_refit.add_argument("--source-csv", type=Path, required=True)
    fit_refit.add_argument("--timeout-seconds", type=int, default=3900)
    refit_child = commands.add_parser("refit-child", help=argparse.SUPPRESS)
    refit_child.add_argument("--campaign", type=Path, required=True)
    refit_child.add_argument("--source-csv", type=Path, required=True)
    recover_refit = commands.add_parser("recover-refit-evidence")
    recover_refit.add_argument("--campaign", type=Path, required=True)
    correct_refit = commands.add_parser("correct-refit-lineage")
    correct_refit.add_argument("--campaign", type=Path, required=True)
    verify_attempt = commands.add_parser("verify-refit-attempt")
    verify_attempt.add_argument("--campaign", type=Path, required=True)
    verify_attempt.add_argument("--strict", action="store_true")
    verify_refit = commands.add_parser("verify-refit")
    verify_refit.add_argument("--campaign", type=Path, required=True)
    verify_refit.add_argument("--recompute-lineage", action="store_true")
    write_report = commands.add_parser("write-report")
    write_report.add_argument("--campaign", type=Path, required=True)
    verify_report = commands.add_parser("verify-report")
    verify_report.add_argument("--campaign", type=Path, required=True)
    verify_report.add_argument("--strict", action="store_true")
    verify_branches = commands.add_parser("verify-branches")
    verify_branches.add_argument("--campaign", type=Path, required=True)
    verify_branches.add_argument("--strict", action="store_true")
    verify_handoffs = commands.add_parser("verify-artifact-handoffs")
    verify_handoffs.add_argument("--campaign", type=Path, required=True)
    verify_handoffs.add_argument("--strict", action="store_true")
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
        if args.through == "final":
            from .verification import validate_final_campaign

            result = validate_final_campaign(args.campaign, strict=args.strict)
        else:
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
    elif args.command == "verify-evaluation":
        from .verification import verify_evaluation

        result = verify_evaluation(args.campaign, recompute_all=args.recompute_all)
    elif args.command == "preflight-refit":
        if not args.strict:
            raise ValueError("full-data refit preflight requires --strict")
        from .refit import durable_refit_preflight

        result = durable_refit_preflight(args.campaign, source_csv=args.source_csv)
    elif args.command == "fit-refit":
        from .refit import launch_refit

        result = launch_refit(
            args.campaign,
            source_csv=args.source_csv,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "refit-child":
        from .refit import refit_child

        result = refit_child(args.campaign, source_csv=args.source_csv)
    elif args.command == "recover-refit-evidence":
        from .refit import recover_refit_evidence

        result = recover_refit_evidence(args.campaign)
    elif args.command == "correct-refit-lineage":
        from .refit import correct_refit_lineage

        result = correct_refit_lineage(args.campaign)
    elif args.command == "verify-refit-attempt":
        from .refit import verify_refit_attempt

        result = verify_refit_attempt(args.campaign, strict=args.strict)
    elif args.command == "verify-refit":
        from .verification import verify_refit

        result = verify_refit(args.campaign, recompute_lineage=args.recompute_lineage)
    elif args.command == "write-report":
        from .report import write_final_report

        result = write_final_report(args.campaign)
    elif args.command == "verify-report":
        from .verification import verify_report

        result = verify_report(args.campaign, strict=args.strict)
    elif args.command == "verify-branches":
        from .verification import verify_branches

        result = verify_branches(args.campaign, repo=Path.cwd(), strict=args.strict)
    else:
        from .verification import verify_artifact_handoffs

        result = verify_artifact_handoffs(args.campaign, strict=args.strict)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

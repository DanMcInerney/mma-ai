"""Command-line seams for campaign bootstrap and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import AccessLedger
from .validation import validate_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m libs.modeling.experiment_campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--expect-terminal-through", type=int)
    validate.add_argument("--require-unsealed", action="store_true")
    validate.add_argument("--require-gate-closed", action="store_true")
    validate.add_argument("--require-sealed", action="store_true")
    validate.add_argument("--require-gate-accesses", type=int)
    verify_run = subparsers.add_parser("verify-run")
    verify_run.add_argument("--campaign", type=Path, required=True)
    verify_run.add_argument("--experiment", required=True)
    verify_run.add_argument("--recompute-all", action="store_true")
    verify_lineage = subparsers.add_parser("verify-feature-lineage")
    verify_lineage.add_argument("--campaign", type=Path, required=True)
    verify_lineage.add_argument("--experiment", required=True)
    verify_lineage.add_argument("--strict", action="store_true")
    audit_safety = subparsers.add_parser("audit-safety")
    audit_safety.add_argument("--campaign", type=Path, required=True)
    audit_safety.add_argument("--through")
    audit_safety.add_argument("--require-gate-closed", action="store_true")
    audit_safety.add_argument("--final", action="store_true")
    replay = subparsers.add_parser("replay-decisions")
    replay.add_argument("--campaign", type=Path, required=True)
    replay.add_argument("--through", required=True)
    replay.add_argument("--require-development-only", action="store_true")
    replay.add_argument("--require-gate-independent", action="store_true")
    verify_results = subparsers.add_parser("verify-results")
    verify_results.add_argument("--campaign", type=Path, required=True)
    verify_results.add_argument("--recompute-all", action="store_true")
    verify_report = subparsers.add_parser("verify-report")
    verify_report.add_argument("--campaign", type=Path, required=True)
    verify_report.add_argument("--strict", action="store_true")
    gate = subparsers.add_parser("gate-status")
    gate.add_argument("--campaign", type=Path, required=True)
    gate.add_argument("--require-closed", action="store_true")
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--campaign", type=Path, required=True)
    bootstrap.add_argument("--artifact-root", type=Path, required=True)
    bootstrap.add_argument("--source-root", type=Path, required=True)
    bootstrap.add_argument("--source-revision", required=True)
    return parser


def _bootstrap(args: argparse.Namespace) -> dict:
    from libs.modeling.training_profiles import (
        WIN_V8_HYBRID_NO_RECENCY_PROFILE,
        WIN_V8_HYBRID_WORKING_PROFILE,
    )

    from .baseline import BaselineSources, bootstrap_experiment_zero

    source_root = args.source_root.resolve()
    accepted_evidence = source_root / ".orch" / "runs" / "20260815T125602Z-new-feature-hybrid-retrain"
    no_recency_evidence = source_root / ".orch" / "runs" / "20260815T202035Z-no-recency-weight-experiment"
    result = bootstrap_experiment_zero(
        args.campaign,
        args.artifact_root,
        sources=BaselineSources(
            frozen_csv=source_root / "data" / "training_data.csv",
            accepted_model=source_root / "AutogluonModels" / "ag-20260815_090928-win-hybrid",
            no_recency_model=source_root / "AutogluonModels" / "ag-20260815_163858-win-hybrid",
            accepted_evidence=accepted_evidence,
            no_recency_evidence=no_recency_evidence,
        ),
        source_revision=args.source_revision,
        working_profile=dict(WIN_V8_HYBRID_WORKING_PROFILE),
        no_recency_profile=dict(WIN_V8_HYBRID_NO_RECENCY_PROFILE),
        expected_population={"total": 3267, "pre_2025": 2807, "from_2025": 460, "gate": 178},
        expected_source_hashes={
            "frozen_csv": "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA",
            "accepted_evidence/direct-evaluation.json": "6665DF5DE0A9CABEFAE52304B8ADC135F064446A9FDB3763C0833D7D09E8ED69",
            "accepted_evidence/final-reverification.md": "41CB2A246A0C4BE936C1295A5D3882981F9DD03472ED719FA10AB32F26D82C54",
            "no_recency_evidence/direct-evaluation.json": "D4B769DF541DEB65B72673930806CEC243C7E95634B4F19C5FE25C9AD3C870F7",
            "no_recency_evidence/final-verification.md": "7CC2285DB55AE6BBCF1E9897D4C752DDA7004B70D7C7F39ABCB79CCE04B7D0C4",
        },
        expected_model_identities={
            "accepted": {
                "source_name": "ag-20260815_090928-win-hybrid",
                "file_count": 56,
                "complete_tree_sha256": "55445E804973B96B43AB6EC86E856A37390FF4937EAC968DC01106E71A257091",
                "native_tree_sha256": "2B90CD505809E7624B8A8701A170BCA41220A937F6C2C24513F30C073D8D2346",
            },
            "no_recency": {
                "source_name": "ag-20260815_163858-win-hybrid",
                "file_count": 56,
                "complete_tree_sha256": "368DD8B9EA70340AC4330B40D671E2FE01B35A9BCE35C158A8D6B0A5507C2BC9",
                "native_tree_sha256": "83975AF832458E3B67677D0AB24AD3D52FCEBE2AEDB36BE7D04D5FEDC9C5B16D",
            },
        },
    )
    return result.__dict__


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        if args.require_sealed:
            from .gate import validate_final_registry

            result = validate_final_registry(args.campaign)
            if args.expect_terminal_through != 10:
                raise SystemExit("final seal validation requires --expect-terminal-through 10")
            if args.require_gate_accesses is not None and result["protected_access_count"] != args.require_gate_accesses:
                raise SystemExit("protected gate access count differs from requirement")
        elif args.expect_terminal_through is not None or args.require_unsealed or args.require_gate_closed:
            from .runner import validate_terminal_campaign

            result = validate_terminal_campaign(
                args.campaign,
                expect_terminal_through=args.expect_terminal_through,
                require_gate_closed=args.require_gate_closed,
            )
            if args.require_unsealed:
                gate = AccessLedger(args.campaign).gate_status()
                if gate["state"] != "closed" or gate["candidate_id"] is not None:
                    raise SystemExit("campaign has a sealed candidate")
        else:
            result = validate_campaign(args.campaign, strict=args.strict).__dict__
    elif args.command == "verify-run":
        from .runner import verify_family_run

        result = verify_family_run(
            args.campaign,
            args.experiment,
            recompute_all=args.recompute_all,
        )
    elif args.command == "verify-feature-lineage":
        from .runner import verify_feature_lineage

        result = verify_feature_lineage(
            args.campaign,
            args.experiment,
            strict=args.strict,
        )
    elif args.command == "audit-safety":
        if args.final:
            from .report import audit_final_safety

            result = audit_final_safety(args.campaign)
        else:
            from .runner import audit_campaign_safety

            if args.through is None:
                raise SystemExit("non-final safety audit requires --through")
            result = audit_campaign_safety(
                args.campaign,
                through=args.through,
                require_gate_closed=args.require_gate_closed,
            )
    elif args.command == "replay-decisions":
        if args.require_gate_independent:
            from .report import replay_final_decisions

            result = replay_final_decisions(args.campaign, require_gate_independent=True)
        else:
            from .runner import replay_campaign_decisions

            result = replay_campaign_decisions(args.campaign, through=args.through)
        if args.require_development_only and not args.require_gate_independent:
            gate = AccessLedger(args.campaign).gate_status()
            if gate["state"] != "closed" or gate["protected_access_count"] != 0:
                raise SystemExit("decision replay is not development-only")
    elif args.command == "gate-status":
        result = AccessLedger(args.campaign).gate_status()
        if args.require_closed and (
            result["state"] != "closed" or result["protected_access_count"] != 0
        ):
            raise SystemExit("gate is not closed with zero protected accesses")
    elif args.command == "verify-results":
        from .report import verify_results

        result = verify_results(args.campaign, recompute_all=args.recompute_all)
    elif args.command == "verify-report":
        from .report import verify_report

        result = verify_report(args.campaign, strict=args.strict)
    else:
        result = _bootstrap(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

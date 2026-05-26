"""Command line entry point for the MMA AI web dashboard."""

from __future__ import annotations

import argparse
import os

import uvicorn


def _default_port() -> int:
    return int(os.getenv("MMA_AI_PORT") or os.getenv("MMA_AI_WEB_PORT") or "8000")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the MMA AI web dashboard.")
    parser.add_argument("--host", default=os.getenv("MMA_AI_HOST", "0.0.0.0"), help="Host interface to bind.")
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help="Port to bind. Defaults to MMA_AI_PORT, then MMA_AI_WEB_PORT, then 8000.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("MMA_AI_RELOAD") == "1",
        help="Enable uvicorn reload for local development.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    uvicorn.run("libs.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

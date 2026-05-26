"""Script wrapper for dashboard model evaluation reports."""

from __future__ import annotations

from libs.web.evaluations import cli


def main(argv: list[str] | None = None) -> int:
    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())

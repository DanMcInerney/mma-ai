"""Command line entry point for scraping UFCStats raw CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from libs.paths import raw_ufcstats_dir
from libs.scraping.ufcstats import scrape_ufcstats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape UFCStats fighters and fights into data/raw/ufcstats.")
    parser.add_argument("--output-dir", default=str(raw_ufcstats_dir()), help="Directory for competitions.csv and individuals.csv.")
    parser.add_argument("--fighters-only", action="store_true", help="Scrape only fighter profile data.")
    parser.add_argument("--fights-only", action="store_true", help="Scrape only completed fight data.")
    parser.add_argument("--force-full", action="store_true", help="Ignore existing CSVs and rebuild them from the crawl.")
    parser.add_argument("--log-level", default="INFO", help="Scrapy log level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fighters_only and args.fights_only:
        raise SystemExit("Choose at most one of --fighters-only or --fights-only.")

    counts = scrape_ufcstats(
        output_dir=args.output_dir,
        fighters=not args.fights_only,
        fights=not args.fighters_only,
        force_full=args.force_full,
        log_level=args.log_level,
    )

    for label, count in counts.items():
        print(f"{label}: {count} total rows")


if __name__ == "__main__":
    main()

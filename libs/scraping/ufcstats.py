"""In-repo UFCStats Scrapy spiders and CSV merge helpers.

This module adapts the standalone UFCScraper project into importable code so
the main repository can scrape raw UFCStats CSVs without a sibling checkout.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


FIGHTER_FIELDS = ["name", "nickname", "url", "dob", "weight", "reach", "height", "stance"]

COMPETITION_METADATA_FIELDS = [
    "result",
    "player1",
    "player2",
    "player1_url",
    "player2_url",
    "weightclass",
    "method",
    "round",
    "time",
    "time_format",
    "referee",
    "details",
    "player1_nickname",
    "player2_nickname",
    "event_date",
    "event_location",
    "event_url",
]

TOTAL_SECTION_STATS = ["KD", "Sig_str", "Total_str", "Td", "Sub_att", "Rev", "Ctrl"]
SIG_STR_SECTION_STATS = ["Head", "Body", "Leg", "Distance", "Clinch", "Ground"]

COMPETITION_FIELDS = (
    COMPETITION_METADATA_FIELDS
    + [f"{fighter}_rd{round_no}_{stat}" for fighter in ("p1", "p2") for stat in TOTAL_SECTION_STATS for round_no in range(1, 6)]
    + [f"{fighter}_rd{round_no}_{stat}" for fighter in ("p1", "p2") for stat in SIG_STR_SECTION_STATS for round_no in range(1, 6)]
)


def _normalize_text(value: str | None) -> str:
    return value.strip() if value else ""


def _nickname(value: str | None) -> str:
    cleaned = _normalize_text(value).replace('"', "")
    return cleaned if cleaned else "--"


def _read_existing_values(path: Path, column: str) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(path, usecols=[column], dtype=str)
    except (ValueError, pd.errors.EmptyDataError):
        return set()

    return set(df[column].dropna().astype(str))


def _read_csv_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return None


def _valid_key_mask(frame: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in key_columns:
        mask &= frame[column].notna()
        mask &= frame[column].astype("string").str.strip().fillna("").ne("")
    return mask


def _merge_csv(
    existing_path: Path,
    new_path: Path,
    fieldnames: list[str],
    key_columns: list[str],
    replace: bool = False,
) -> int:
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_frame = None if replace else _read_csv_frame(existing_path)
    new_frame = _read_csv_frame(new_path)

    if not replace and existing_frame is not None and (new_frame is None or new_frame.empty):
        return len(existing_frame)

    frames = []
    if existing_frame is not None:
        frames.append(existing_frame)
    if new_frame is not None:
        frames.append(new_frame)

    if not frames:
        with existing_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return 0

    merged = pd.concat(frames, ignore_index=True)
    for field in fieldnames:
        if field not in merged.columns:
            merged[field] = pd.NA
    merged = merged[fieldnames]
    merged = merged[_valid_key_mask(merged, key_columns)]
    merged = merged.drop_duplicates(subset=key_columns, keep="last" if replace else "first")
    merged.to_csv(existing_path, index=False)
    return len(merged)


class CsvWriterPipeline:
    """Write each spider to its own CSV using fixed field order."""

    def __init__(self):
        self._files = {}
        self._writers = {}
        self._crawler = None

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline._crawler = crawler
        return pipeline

    def _spider(self, spider=None):
        if spider is not None:
            return spider
        if self._crawler is not None:
            return self._crawler.spider
        raise RuntimeError("Scrapy did not provide a spider instance to CsvWriterPipeline")

    def open_spider(self, spider=None):
        spider = self._spider(spider)
        output_path = Path(spider.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=spider.output_fields, extrasaction="ignore")
        writer.writeheader()
        self._files[id(spider)] = handle
        self._writers[id(spider)] = writer

    def close_spider(self, spider=None):
        spider = self._spider(spider)
        handle = self._files.pop(id(spider), None)
        self._writers.pop(id(spider), None)
        if handle:
            handle.close()

    def process_item(self, item, spider=None):
        spider = self._spider(spider)
        self._writers[id(spider)].writerow(dict(item))
        return item


try:
    import numpy as np
    import scrapy
except ImportError:  # pragma: no cover - handled at runtime by scrape_ufcstats()
    np = None
    scrapy = None


if scrapy is not None:

    class UfcFighterSpider(scrapy.Spider):
        name = "ufc_fighter_scraper"
        allowed_domains = ["ufcstats.com"]
        output_fields = FIGHTER_FIELDS

        def __init__(self, existing_urls: Iterable[str] | None = None, output_path: str | Path | None = None, **kwargs):
            super().__init__(**kwargs)
            self.existing_urls = set(existing_urls or [])
            self.output_path = str(output_path)

        def start_requests(self):
            base_url = "http://ufcstats.com/statistics/fighters?char="
            for char in "abcdefghijklmnopqrstuvwxyz":
                yield scrapy.Request(f"{base_url}{char}&page=all", callback=self.parse)

        async def start(self):
            for request in self.start_requests():
                yield request

        def parse(self, response):
            for fighter_link in response.css("td.b-statistics__table-col a::attr(href)").getall():
                if fighter_link not in self.existing_urls:
                    yield scrapy.Request(fighter_link, callback=self.parse_fighter)

        def parse_fighter(self, response):
            yield {
                "name": _normalize_text(response.css("span.b-content__title-highlight::text").get()),
                "nickname": _nickname(response.xpath("/html/body/section/div/p/text()").get()),
                "url": response.url,
                "dob": _normalize_text(response.xpath('//i[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "dob:")]/following-sibling::text()').get()),
                "weight": _normalize_text(response.xpath('//i[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "weight:")]/following-sibling::text()').get()),
                "reach": _normalize_text(response.xpath('//i[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "reach:")]/following-sibling::text()').get()),
                "height": _normalize_text(response.xpath('//i[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "height:")]/following-sibling::text()').get()),
                "stance": _normalize_text(response.xpath('//i[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "stance:")]/following-sibling::text()').get()) or "--",
            }


    class UfcFightSpider(scrapy.Spider):
        name = "ufc_scraper"
        allowed_domains = ["ufcstats.com"]
        start_urls = ["http://ufcstats.com/statistics/events/completed?page=all"]
        output_fields = COMPETITION_FIELDS

        def __init__(self, existing_event_urls: Iterable[str] | None = None, output_path: str | Path | None = None, **kwargs):
            super().__init__(**kwargs)
            self.existing_event_urls = set(existing_event_urls or [])
            self.output_path = str(output_path)

        def parse(self, response):
            for event in response.css("td.b-statistics__table-col"):
                if event.xpath('.//img[contains(@src, "/next.png")]'):
                    continue

                event_link = event.css("i.b-statistics__table-content a::attr(href)").get()
                if event_link and event_link not in self.existing_event_urls:
                    yield scrapy.Request(event_link, callback=self.parse_event)

        def parse_event(self, response):
            event_date = _normalize_text(response.xpath('//i[contains(text(), "Date:")]/following-sibling::text()').get())
            event_location = _normalize_text(response.xpath('//i[contains(text(), "Location:")]/following-sibling::text()').get())

            for fight in response.css("tr.b-fight-details__table-row.b-fight-details__table-row__hover.js-fight-details-click::attr(onclick)"):
                onclick = fight.get()
                if not onclick:
                    continue
                fight_link = onclick.split("'")[1]
                yield scrapy.Request(
                    fight_link,
                    callback=self.parse_fight,
                    meta={"event_date": event_date, "event_location": event_location, "event_url": response.url},
                )

        def parse_fight(self, response):
            stats_section = response.xpath("/html/body/section/div/div/section").get() or ""
            if "not currently available" in stats_section:
                return

            fighters = [_normalize_text(fighter) for fighter in response.css("h3.b-fight-details__person-name a::text").getall()]
            if len(fighters) < 2:
                return

            weight_class = ""
            weight_nodes = response.xpath("/html/body/section/div/div/div[2]/div[1]/i/text()")
            for node in weight_nodes:
                weight_class = _normalize_text(node.get())
                if weight_class:
                    break

            details = ""
            detail_text = response.xpath('//i[contains(text(), "Details:")]/ancestor::p[1]/text()').getall()
            if len(detail_text) > 1:
                details = _normalize_text(detail_text[1])
            if not details:
                detail_nodes = response.css('i:contains("Details:") ~ i.b-fight-details__text-item')
                details = " ".join(
                    text.strip()
                    for node in detail_nodes
                    for text in node.css("::text").getall()
                    if text.strip()
                )

            round_no = _normalize_text(response.xpath('//i[contains(text(), "Round:")]/following-sibling::text()').get())
            fight_data = {
                "result": _normalize_text(response.xpath("/html/body/section/div/div/div[1]/div[1]/i/text()").get()),
                "player1": fighters[0],
                "player2": fighters[1],
                "player1_url": _normalize_text(response.xpath("/html/body/section/div/div/div[1]/div[1]/div/h3/a/@href").get()),
                "player2_url": _normalize_text(response.xpath("/html/body/section/div/div/div[1]/div[2]/div/h3/a/@href").get()),
                "weightclass": weight_class,
                "method": _normalize_text(response.xpath('//i[contains(text(), "Method:")]/following-sibling::i[@style="font-style: normal"]/text()').get()),
                "round": round_no,
                "time": _normalize_text(response.xpath('//i[contains(text(), "Time:")]/following-sibling::text()').get()),
                "time_format": _normalize_text(response.xpath('//i[contains(text(), "Time format:")]/following-sibling::text()').get()),
                "referee": _normalize_text(response.xpath('//i[contains(text(), "Referee:")]/following-sibling::span/text()').get()),
                "details": details,
                "player1_nickname": _nickname(response.xpath("/html/body/section/div/div/div[1]/div[1]/div/p/text()").get()),
                "player2_nickname": _nickname(response.xpath("/html/body/section/div/div/div[1]/div[2]/div/p/text()").get()),
                "event_date": response.meta.get("event_date"),
                "event_location": response.meta.get("event_location"),
                "event_url": response.meta.get("event_url"),
            }

            fight_data.update(self.parse_section("/html/body/section/div/div/section[3]/table", response))
            fight_data.update(self.parse_section("/html/body/section/div/div/section[5]/table", response))

            yield fight_data

        def parse_section(self, section_xpath, response):
            section = response.xpath(section_xpath)
            row_elements = section.xpath('.//tr[@class="b-fight-details__table-row"]')
            if not row_elements:
                return {}

            column_names = row_elements[0].xpath("./th/text()").getall()
            column_names = [name.strip() for name in column_names if name.strip()][1:]
            if column_names.count("Td %") > 1:
                column_names[column_names.index("Td %")] = "Td"

            fight_data = {}
            for index, row_element in enumerate(row_elements[1:]):
                fighter_names = row_element.xpath("./td/p/a/text()").getall()
                data_columns = row_element.xpath("./td/p/text()").getall()[2:]
                data_columns = [data.strip() for data in data_columns if data.strip()]

                for fighter_index, _fighter in enumerate(fighter_names):
                    for stat_index, stat in enumerate(data_columns[fighter_index::2]):
                        key = f"{'p1' if fighter_index == 0 else 'p2'}_rd{index + 1}_{column_names[stat_index]}"
                        key = key.replace(".", "").replace(" ", "_")
                        if "%" not in key:
                            fight_data[key] = stat

            round_one_keys = [key for key in fight_data if "_rd1_" in key]
            completed = {}
            for key in round_one_keys:
                for round_no in range(1, 6):
                    round_key = key.replace("rd1", f"rd{round_no}")
                    completed[round_key] = fight_data.get(round_key, np.nan)

            return completed


def scrape_ufcstats(
    output_dir: str | Path,
    fighters: bool = True,
    fights: bool = True,
    force_full: bool = False,
    log_level: str = "INFO",
) -> dict[str, int]:
    """Run UFCStats spiders and merge results into stable CSV files."""
    if scrapy is None:
        raise RuntimeError("Scrapy is not installed. Run `uv sync` or install the project dependencies first.")

    from scrapy.crawler import CrawlerProcess

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    competitions_path = output_dir / "competitions.csv"
    individuals_path = output_dir / "individuals.csv"
    temp_dir = output_dir / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_competitions = temp_dir / "competitions_new.csv"
    temp_individuals = temp_dir / "individuals_new.csv"

    existing_fighter_urls = set() if force_full else _read_existing_values(individuals_path, "url")
    existing_event_urls = set() if force_full else _read_existing_values(competitions_path, "event_url")

    process = CrawlerProcess(
        settings={
            "BOT_NAME": "mma-ai",
            "ROBOTSTXT_OBEY": True,
            "DOWNLOAD_DELAY": 0.25,
            "FEED_EXPORT_ENCODING": "utf-8",
            "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
            "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
            "ITEM_PIPELINES": {"libs.scraping.ufcstats.CsvWriterPipeline": 300},
            "LOG_LEVEL": log_level,
        }
    )

    if fighters:
        process.crawl(UfcFighterSpider, existing_urls=existing_fighter_urls, output_path=temp_individuals)
    if fights:
        process.crawl(UfcFightSpider, existing_event_urls=existing_event_urls, output_path=temp_competitions)

    process.start()

    counts = {}
    if fighters:
        counts["fighters"] = _merge_csv(individuals_path, temp_individuals, FIGHTER_FIELDS, ["url"], replace=force_full)
    if fights:
        counts["fights"] = _merge_csv(
            competitions_path,
            temp_competitions,
            COMPETITION_FIELDS,
            ["event_url", "player1_url", "player2_url"],
            replace=force_full,
        )

    for temp_path in (temp_competitions, temp_individuals):
        if temp_path.exists():
            temp_path.unlink()

    return counts


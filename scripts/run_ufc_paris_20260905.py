"""Reproduce the supported UFC Paris predictions from the 2026-09-05 card.

The official card contains debutants and replacement fighters that are absent
from the pinned inference dataset. This runner invokes the existing predictor
only for bouts where both normalized fighter names have history, consolidates
the per-bout CSVs, and records the excluded bouts instead of fabricating rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
MODEL_PATH = REPO_ROOT / "AutogluonModels" / "ag-20260815_163858-win-hybrid"
PREDICTION_DATA = REPO_ROOT / "data" / "prediction_data.csv"
TRAINING_DATA = MODEL_PATH / "training_data.csv"
EVENT_NAME = "UFC Fight Night: Hooker vs Parnasse"
EVENT_DATE = "2026-09-05"
SOURCE_RETRIEVED_AT_UTC = "2026-09-05T12:11:00Z"
SOURCE_URLS = [
    "https://www.ufc.com/event/ufc-fight-night-september-05-2026",
    "https://www.ufc.com/news/ufc-fight-night-paris-official-weigh-in-results-hooker-parnasse",
    "https://www.ufc.com/news/updates-ufc-fight-night-paris-2026",
]
CARD_NOTE = "The Sept. 1 update replaces Mairon Santos with Pavel Andrusca."
PREVIOUS_RESULTS_URL = (
    "https://www.ufc.com/news/ufc-shanghai-official-scorecards-nurmagomedov-vs-song"
)

# Names are the repository's existing normalized aliases; odds are the
# American prices shown on the official event page at retrieval time.
CARD = [
    ("dan hooker", "salahdine parnasse", 400, -550),
    ("fares ziam", "axel sola", -150, 125),
    ("michael page", "nursulton ruziboev", -160, 130),
    ("daniil donchenko", "punahele soriano", -245, 200),
    ("kurtis campbell", "trevor peek", -390, 310),
    ("losene keita", "muhammad naimov", -360, 280),
    ("morgan charriere", "felipe lima", 155, -185),
    ("mario pinto", "ryan spann", -305, 245),
    ("oumar sy", "modestas bukauskas", -230, 190),
    ("nathaniel wood", "pavel andrusca", -315, 250),
    ("michael aljarouj", "fabia sintes", -125, 105),
    ("nora cornolle", "klaudia sygula", -120, 100),
    ("matthieu duclos", "luis felipe dias", -115, -105),
    ("delphine benouaich", "sofia montenegro", -145, 125),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_fighter_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "fighter_name" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no fighter_name column")
        for row in reader:
            name = (row.get("fighter_name") or "").strip().lower()
            if name:
                names.add(name)
    return names


def slug(name: str) -> str:
    return "_".join(name.lower().replace("'", "").split())


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        first_line = stream.readline()
        if not first_line.startswith("# Fight Predictions"):
            raise ValueError(f"Unexpected prediction preamble in {path}")
        return list(csv.DictReader(stream))


def consolidate_csvs(per_fight_dirs: list[Path], output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fight_dir in per_fight_dirs:
        fight_rows = read_prediction_rows(fight_dir / "fight_predictions.csv")
        if len(fight_rows) != 1:
            raise ValueError(f"Expected one prediction in {fight_dir}, got {len(fight_rows)}")
        rows.extend(fight_rows)

    rows.sort(key=lambda row: float(row["Confidence"]), reverse=True)
    fieldnames = list(rows[0])
    output_path = output_dir / "fight_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# Fight Predictions using ORIGINAL model predictions\n")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def consolidate_stats(per_fight_dirs: list[Path], output_dir: Path) -> None:
    output_path = output_dir / "prediction_stats.csv"
    writer = None
    with output_path.open("w", encoding="utf-8", newline="") as output_stream:
        for fight_dir in per_fight_dirs:
            with (fight_dir / "prediction_stats.csv").open("r", encoding="utf-8", newline="") as input_stream:
                reader = csv.DictReader(input_stream)
                if writer is None:
                    writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames, lineterminator="\n")
                    writer.writeheader()
                elif reader.fieldnames != writer.fieldnames:
                    raise ValueError(f"Prediction stats schema mismatch in {fight_dir}")
                writer.writerows(reader)


def output_hashes(output_dir: Path, manifest_path: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.resolve() != manifest_path.resolve()
            and path.name != "run_manifest.json"
        ):
            hashes[path.relative_to(output_dir).as_posix()] = sha256_file(path)
    return hashes


def website_event(rows: list[dict[str, str]]) -> dict[str, object]:
    predictions = []
    for row in rows:
        fighter1 = row["Fighter1"].lower()
        fighter2 = row["Fighter2"].lower()
        pick = row["AI_Pick"].lower()
        picked_odds = row["Fighter1_Odds"] if pick == fighter1 else row["Fighter2_Odds"]
        ev = float(row["EV"])
        predictions.append(
            {
                "fight": f"{fighter1} vs {fighter2}",
                "prediction": pick,
                "ai_win_pct": round(float(row["Confidence"]), 1),
                "ai_odds": int(row["AI_Odds"]),
                "vegas_odds": float(picked_odds),
                "ev": round(ev, 2),
                "result": "Pending" if ev > 0 else "Pending - no bet",
            }
        )
    return {
        "name": EVENT_NAME,
        "date": "-".join(str(int(part)) for part in EVENT_DATE.split("-")),
        "predictions": predictions,
    }


def model_file_hashes() -> dict[str, str]:
    return {
        path.relative_to(MODEL_PATH).as_posix(): sha256_file(path)
        for path in sorted(MODEL_PATH.rglob("*"))
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--screenshots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-fight artifacts and retained zero-row eligibility probes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(f"Project runtime not found: {PROJECT_PYTHON}")

    available_names = normalized_fighter_names(PREDICTION_DATA)
    supported = []
    excluded = []
    for fighter1, fighter2, odds1, odds2 in CARD:
        missing = [name for name in (fighter1, fighter2) if name not in available_names]
        record = {
            "fighter1": fighter1,
            "fighter2": fighter2,
            "fighter1_odds": odds1,
            "fighter2_odds": odds2,
        }
        if missing:
            record["reason"] = "missing prediction history: " + ", ".join(missing)
            excluded.append(record)
        elif odds1 is None or odds2 is None:
            record["reason"] = "current American odds unavailable at source retrieval"
            excluded.append(record)
        else:
            supported.append(record)

    commands = []
    execution = []
    per_fight_dirs = []
    predicted = []
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "42"
    environment["PYTHONUNBUFFERED"] = "1"
    for index, fight in enumerate(supported, start=1):
        fighter1 = fight["fighter1"]
        fighter2 = fight["fighter2"]
        fight_dir = output_dir / f"{slug(fighter1)}_vs_{slug(fighter2)}"
        manual_odds = json.dumps(
            {fighter1: fight["fighter1_odds"], fighter2: fight["fighter2_odds"]},
            sort_keys=True,
        )
        command = [
            str(PROJECT_PYTHON),
            str(REPO_ROOT / "predict.py"),
            "--model-path",
            str(MODEL_PATH),
            "--training-data-csv",
            str(TRAINING_DATA),
            "--prediction-data-csv",
            str(PREDICTION_DATA),
            "--fighter1",
            fighter1,
            "--fighter2",
            fighter2,
            "--fight-date",
            EVENT_DATE,
            "--odds",
            "--manual-odds-json",
            manual_odds,
            "--no-manual-odds",
            "--output-dir",
            str(fight_dir),
        ]
        if args.no_shap:
            command.append("--no-shap")
        command.append("--screenshots" if args.screenshots else "--no-screenshots")
        commands.append(command)
        prediction_path = fight_dir / "fight_predictions.csv"
        existing_rows = read_prediction_rows(prediction_path) if prediction_path.is_file() else []
        prefix = f"shap_{fighter1.split()[0]}_{fighter2.split()[0]}"
        shap_paths = [fight_dir / f"{prefix}.html", fight_dir / f"{prefix}.json"]
        if args.screenshots and not args.no_shap:
            shap_paths.append(fight_dir / "screenshots" / f"{prefix}.png")
        reusable = args.resume and prediction_path.is_file()
        if reusable and len(existing_rows) == 1 and not args.no_shap:
            reusable = all(path.is_file() for path in shap_paths)

        if reusable:
            print(
                f"[card] resume {index}/{len(supported)} {fighter1} vs {fighter2} "
                f"({len(existing_rows)} rows)",
                flush=True,
            )
            execution.append({"command": command, "mode": "resumed", "exit_code": 0})
        else:
            print(f"[card] {index}/{len(supported)} {fighter1} vs {fighter2}", flush=True)
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                timeout=600,
            )
            print(f"[card] exit {completed.returncode}: {fighter1} vs {fighter2}", flush=True)
            execution.append(
                {"command": command, "mode": "executed", "exit_code": completed.returncode}
            )
            if completed.returncode != 0:
                return completed.returncode
            existing_rows = read_prediction_rows(prediction_path)

        if len(existing_rows) == 0:
            excluded_fight = dict(fight)
            excluded_fight["reason"] = (
                "existing inference builder produced no feature row under its "
                "minimum-history eligibility policy"
            )
            excluded.append(excluded_fight)
            print(f"[card] excluded after feature build: {fighter1} vs {fighter2}", flush=True)
            continue
        if len(existing_rows) != 1:
            raise ValueError(f"Expected one prediction in {fight_dir}, got {len(existing_rows)}")
        if not args.no_shap:
            for path in shap_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing SHAP artifact: {path}")
        if args.screenshots:
            # Save each completed bout immediately, even if a later bout fails.
            screenshots_dir = output_dir / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)
            kinds = ["grouped", "ind"] + ([] if args.no_shap else ["shap"])
            for kind in kinds:
                png = fight_dir / "screenshots" / f"{kind}_{fighter1.split()[0]}_{fighter2.split()[0]}.png"
                if not png.is_file():
                    raise FileNotFoundError(f"Missing required chart screenshot: {png}")
                shutil.copy2(png, screenshots_dir / png.name)
        predicted.append(fight)
        per_fight_dirs.append(fight_dir)

    rows = consolidate_csvs(per_fight_dirs, output_dir)
    consolidate_stats(per_fight_dirs, output_dir)
    website_record = website_event(rows)
    (output_dir / "website_event.json").write_text(
        json.dumps(website_record, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": {
            "name": EVENT_NAME,
            "date": EVENT_DATE,
            "timezone": "America/New_York",
            "official_card_bouts": len(CARD),
            "predicted_bouts": len(predicted),
            "excluded_bouts": excluded,
        },
        "sources": {
            "retrieved_at_utc": SOURCE_RETRIEVED_AT_UTC,
            "urls": SOURCE_URLS,
            "card_note": CARD_NOTE,
            "previous_results": {
                "event": "UFC Fight Night: Nurmagomedov vs Song",
                "url": PREVIOUS_RESULTS_URL,
                "verified_winners": [
                    "rei tsuruya",
                    "sumudaerji",
                    "song yadong",
                    "kai asakura",
                    "sean woodson",
                ],
            },
        },
        "inputs": {
            "prediction_data": {
                "path": PREDICTION_DATA.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(PREDICTION_DATA),
            },
            "training_data": {
                "path": TRAINING_DATA.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(TRAINING_DATA),
            },
            "model_path": MODEL_PATH.relative_to(REPO_ROOT).as_posix(),
            "model_files": model_file_hashes(),
        },
        "runtime": {
            "project_python": str(PROJECT_PYTHON),
            "python_version": "3.12.4",
            "autogluon_tabular": "1.6.1",
            "pandas": "2.3.2",
            "shap": "0.48.0",
            "pythonhashseed": 42,
        },
        "pipeline": {
            "launcher": str(Path(sys.executable).resolve()),
            "runner": Path(__file__).relative_to(REPO_ROOT).as_posix(),
            "commands": commands,
            "execution": execution,
            "uncalibrated_predictions": True,
            "shap_enabled": not args.no_shap,
            "shap_settings": {
                "algorithm": "shap.KernelExplainer",
                "approximate": True,
                "nsamples": 500,
                "background_rows": 100,
                "background_sample_random_state": 42,
                "kernel_random_seed": 42,
                "caveat": "Kernel SHAP remains an approximate explanation; fixed seeds reduce sampling drift but do not make it exact.",
            },
        },
        "degrees_of_freedom": {
            "coverage": "Predict only bouts where both normalized names exist in the pinned prediction dataset.",
            "odds": "Use official event-page American odds captured at retrieval time; the existing predictor de-vigs them for market probabilities and uses picked-fighter book odds for wager EV.",
            "website_eligibility": "Positive numeric EV (> 0) is Pending; every other predicted bout is Pending - no bet.",
        },
        "predictions": rows,
        "outputs": output_hashes(output_dir, manifest_path),
    }
    if EVENT_DATE != "2026-09-05":
        manifest["sources"].pop("previous_results")
        manifest["degrees_of_freedom"]["odds"] = "Frozen American odds from sbaodds.com/mma-odds/ at the recorded retrieval time; picked-fighter book odds determine wager EV."
    manifest["pipeline"]["screenshots_enabled"] = args.screenshots
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(serialized, encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(serialized, encoding="utf-8")
    print(f"[card] wrote {len(rows)} predictions and manifest {manifest_path}", flush=True)
    print(f"[card] declared {len(excluded)} coverage gaps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run predict.py for one bout with the upcoming bout's official division.

The model's weightclass input is fighter1's division from their last fight.
When the red-corner fighter (always --fighter1) is changing division, pass
--upcoming-weightclass-code instead of swapping corners. Only that one
categorical value in the upcoming feature row changes; every other input is
verified unchanged and the override is recorded in the output directory.

Usage:
  python scripts/predict_with_event_context.py --upcoming-weightclass-code 4 \
      --upcoming-weightclass featherweight -- <predict.py arguments>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import predict

COLUMN = "fighter1_weightclass_encoded"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upcoming-weightclass-code", type=int, required=True)
    parser.add_argument("--upcoming-weightclass", required=True, help="Official division name, for the record.")
    parser.add_argument("predict_args", nargs=argparse.REMAINDER)
    own = parser.parse_args()
    forwarded = own.predict_args[1:] if own.predict_args[:1] == ["--"] else own.predict_args
    sys.argv = [str(ROOT / "predict.py"), *forwarded]
    args = predict.parse_args()
    require(args.fighter1 and args.fighter2, "Context overrides are for single-bout --fighter1/--fighter2 runs")
    require(args.output_dir, "--output-dir is required so the override can be recorded")
    pair = (args.fighter1.strip().lower(), args.fighter2.strip().lower())
    out = Path(args.output_dir)

    class EventContextBuilder(predict.InferenceDataBuilder):
        def build(self):
            frames = super().build()
            require(len(frames) == 1, "Expected exactly one eligible bout")
            key = next(iter(frames))
            frame = frames[key]
            require(len(frame) == 1 and not frame.columns.duplicated().any(), "Invalid upcoming feature row")
            row = frame.iloc[0]
            require((row["fighter_name"], row["opponent"]) == pair and bool(row["fighter1"]),
                    "Feature row is not the requested fighter1/fighter2 pair")
            original = frame.copy(deep=True)
            old_value = float(original.iloc[0][COLUMN])
            frame = frame.copy(deep=True)
            frame.loc[frame.index[0], COLUMN] = own.upcoming_weightclass_code
            pd.testing.assert_frame_equal(original.drop(columns=[COLUMN]), frame.drop(columns=[COLUMN]), check_exact=True)
            frames[key] = frame
            record = {
                "applied_at_utc": datetime.now(timezone.utc).isoformat(),
                "fighter1": pair[0], "fighter2": pair[1],
                "fight_date": args.fight_date,
                "raw_feature": COLUMN,
                "historical_value": old_value,
                "new_value": float(own.upcoming_weightclass_code),
                "weight_class": own.upcoming_weightclass,
                "non_context_columns_unchanged": len(frame.columns) - 1,
            }
            out.mkdir(parents=True, exist_ok=True)
            (out / "event_context_correction.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"[event-context] {pair[0]} vs {pair[1]}: {COLUMN} {old_value} -> "
                  f"{own.upcoming_weightclass_code} ({own.upcoming_weightclass})", flush=True)
            return frames

    predict.InferenceDataBuilder = EventContextBuilder
    predict.cli()


if __name__ == "__main__":
    main()

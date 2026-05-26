"""Run a dashboard training request and emit a structured result marker."""

from __future__ import annotations

import json
import sys

from libs.web.models import TrainingRequest
from libs.web.services import TRAINING_RESULT_BEGIN, TRAINING_RESULT_END, run_training_impl


def main() -> int:
    raw_request = sys.stdin.read().strip()
    if not raw_request:
        print("Expected a JSON TrainingRequest on stdin.", file=sys.stderr)
        return 2

    request = TrainingRequest(**json.loads(raw_request))
    result = run_training_impl(request)
    print(TRAINING_RESULT_BEGIN)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(TRAINING_RESULT_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

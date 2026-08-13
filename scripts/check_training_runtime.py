"""Read-only command that verifies the production AutoGluon CUDA runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.modeling.train import training_runtime_preflight


def main() -> int:
    print(json.dumps(training_runtime_preflight(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

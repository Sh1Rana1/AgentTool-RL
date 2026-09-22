"""Generate deterministic AgentTool-RL dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.generator import write_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()

    manifest = write_dataset(
        args.output,
        seed=args.seed,
        split_sizes={
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify generated splits, checksums, validation rules, and reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.generator import SUPPORTED_SPLITS, write_dataset  # noqa: E402
from agenttool_rl.validation import load_jsonl, validate_scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "generated"
    )
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_sizes: dict[str, int] = {}
    for split in SUPPORTED_SPLITS:
        metadata = manifest["splits"][split]
        path = args.dataset / metadata["file"]
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(f"{split}: checksum mismatch")
        count = validate_scenarios(load_jsonl(path))
        if count != metadata["count"]:
            raise RuntimeError(f"{split}: expected {metadata['count']} rows, found {count}")
        split_sizes[split] = count

    with tempfile.TemporaryDirectory() as temporary_directory:
        reproduced = write_dataset(
            Path(temporary_directory),
            seed=manifest["seed"],
            split_sizes=split_sizes,
        )
    if reproduced != manifest:
        raise RuntimeError("dataset cannot be reproduced from the recorded seed and version")

    total = sum(split_sizes.values())
    print(
        f"Verified {total} scenarios across {len(SUPPORTED_SPLITS)} disjoint, "
        "reproducible splits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

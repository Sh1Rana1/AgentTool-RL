from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.generator import ROOT_CAUSES, ScenarioGenerator, write_dataset
from agenttool_rl.validation import load_jsonl, validate_scenarios


class ScenarioGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_for_a_seed(self) -> None:
        first = ScenarioGenerator(7).generate("train", 10)
        second = ScenarioGenerator(7).generate("train", 10)
        self.assertEqual(first, second)

    def test_splits_have_disjoint_ids_and_balanced_root_causes(self) -> None:
        generator = ScenarioGenerator(42)
        train = generator.generate("train", 10)
        test = generator.generate("test", 10)
        self.assertFalse(
            {item["task_id"] for item in train}
            & {item["task_id"] for item in test}
        )
        self.assertEqual(
            {item["hidden_state"]["root_cause"] for item in train}, set(ROOT_CAUSES)
        )

    def test_writer_emits_valid_splits_and_checksum_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            manifest = write_dataset(
                output,
                seed=9,
                split_sizes={"train": 10, "validation": 5, "test": 5},
            )
            self.assertEqual(manifest["splits"]["train"]["count"], 10)
            saved_manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertEqual(saved_manifest, manifest)
            train_bytes = (output / "train.jsonl").read_bytes()
            self.assertNotIn(b"\r\n", train_bytes)
            self.assertEqual(validate_scenarios(load_jsonl(output / "test.jsonl")), 5)


if __name__ == "__main__":
    unittest.main()

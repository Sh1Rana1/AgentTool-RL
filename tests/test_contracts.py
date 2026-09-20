from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.contracts import TOOL_CONTRACTS, exported_tool_schemas
from agenttool_rl.validation import load_jsonl, validate_scenarios


class ToolContractTests(unittest.TestCase):
    def test_required_tools_are_present(self) -> None:
        self.assertEqual(
            set(TOOL_CONTRACTS),
            {
                "query_logs",
                "query_metrics",
                "get_dependencies",
                "get_recent_deployments",
                "submit_diagnosis",
            },
        )

    def test_contracts_are_json_serializable_and_closed(self) -> None:
        schemas = exported_tool_schemas()
        json.dumps(schemas)
        for schema in schemas:
            self.assertFalse(schema["parameters"]["additionalProperties"])
            self.assertTrue(schema["parameters"]["properties"])


class ScenarioTests(unittest.TestCase):
    def test_all_example_scenarios_are_valid(self) -> None:
        path = PROJECT_ROOT / "data/examples/scenarios.jsonl"
        scenarios = load_jsonl(path)
        self.assertEqual(validate_scenarios(scenarios), 5)

    def test_hidden_answers_are_not_embedded_in_user_requests(self) -> None:
        path = PROJECT_ROOT / "data/examples/scenarios.jsonl"
        for scenario in load_jsonl(path):
            root_cause = scenario["hidden_state"]["root_cause"]
            self.assertNotIn(root_cause, scenario["user_request"])


if __name__ == "__main__":
    unittest.main()


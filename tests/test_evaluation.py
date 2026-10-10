from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.agents import RandomAgent, RuleBasedAgent
from agenttool_rl.evaluation import evaluate_agents
from agenttool_rl.validation import load_jsonl


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_jsonl(PROJECT_ROOT / "data" / "examples" / "scenarios.jsonl")

    def test_unified_report_aggregates_agents(self) -> None:
        report = evaluate_agents(
            {
                "random": lambda index: RandomAgent(seed=100 + index),
                "rule-based": lambda index: RuleBasedAgent(),
            },
            self.scenarios,
            dataset="examples",
            max_steps=6,
        )
        self.assertEqual(len(report.reports), 2)
        rule_report = next(item for item in report.reports if item.agent == "rule-based")
        self.assertEqual(rule_report.task_success_rate, 1.0)
        self.assertEqual(rule_report.schema_valid_rate, 1.0)
        self.assertEqual(rule_report.semantic_argument_accuracy, 1.0)
        self.assertEqual(rule_report.evidence_validity, 1.0)
        self.assertEqual(rule_report.average_evidence_coverage, 1.0)
        self.assertIsNone(rule_report.error_recovery_rate)
        self.assertEqual(rule_report.average_tool_calls, 3.0)

    def test_agent_failure_is_isolated_to_its_episode(self) -> None:
        class BrokenAgent:
            name = "broken"

            def reset(self, observation: dict) -> None:
                self.observation = observation

            def act(self):
                raise RuntimeError("backend unavailable")

            def observe(self, action, result) -> None:
                raise AssertionError("observe should not be reached")

        report = evaluate_agents(
            {"broken": lambda index: BrokenAgent()},
            self.scenarios[:2],
            dataset="failure-isolation",
            max_steps=6,
        )
        broken = report.reports[0]
        self.assertEqual(broken.agent_error_rate, 1.0)
        self.assertEqual(broken.average_tool_calls, 0.0)
        self.assertTrue(all(item.agent_error for item in broken.episodes))

    def test_report_writes_machine_and_human_readable_outputs(self) -> None:
        report = evaluate_agents(
            {"rule-based": lambda index: RuleBasedAgent()},
            self.scenarios[:1],
            dataset="one-example",
            max_steps=6,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "report.json"
            report.write(json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = json_path.with_suffix(".md").read_text(encoding="utf-8")
            self.assertEqual(payload["reports"][0]["task_success_rate"], 1.0)
            self.assertIn("AgentTool-RL Baseline Report", markdown)
            self.assertIn("rule-based", markdown)

    def test_frozen_report_matches_current_evaluator(self) -> None:
        test_scenarios = load_jsonl(
            PROJECT_ROOT / "data" / "generated" / "test.jsonl"
        )
        report = evaluate_agents(
            {
                "random": lambda index: RandomAgent(seed=42 + index),
                "rule-based": lambda index: RuleBasedAgent(),
            },
            test_scenarios,
            dataset="data/generated/test.jsonl",
            max_steps=6,
        )
        frozen = json.loads(
            (PROJECT_ROOT / "reports" / "v0.3_baselines.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report.as_dict(), frozen)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.environment import EpisodeFinishedError, IncidentEnvironment
from agenttool_rl.validation import load_jsonl


def load_scenario(task_id: str) -> dict:
    path = PROJECT_ROOT / "data" / "examples" / "scenarios.jsonl"
    return next(item for item in load_jsonl(path) if item["task_id"] == task_id)


class IncidentEnvironmentTests(unittest.TestCase):
    def test_reset_does_not_expose_hidden_state(self) -> None:
        scenario = load_scenario("deploy_regression_001")
        observation = IncidentEnvironment(max_steps=5).reset(scenario)
        serialized = json.dumps(observation)
        self.assertNotIn("hidden_state", observation)
        self.assertNotIn("task_id", observation)
        self.assertNotIn(scenario["hidden_state"]["required_evidence"][0], serialized)
        self.assertNotIn(
            scenario["hidden_state"]["tool_data"]["logs"][0]["message"], serialized
        )

    def test_successful_episode_collects_and_submits_real_evidence(self) -> None:
        env = IncidentEnvironment(load_scenario("deploy_regression_001"), max_steps=5)
        deploy = env.step(
            "get_recent_deployments",
            {"service": "payment-api", "since": "2026-09-20T09:00:00Z"},
        )
        self.assertTrue(deploy.info["valid"])
        self.assertIn("deploy:payment-api:v2026.09.20.2", deploy.info["evidence_ids"])

        logs = env.step(
            "query_logs",
            {
                "service": "payment-api",
                "keyword": "null",
                "start_time": "2026-09-20T09:30:00Z",
                "end_time": "2026-09-20T10:00:00Z",
            },
        )
        self.assertEqual(logs.observation["result"]["count"], 1)

        final = env.step(
            "submit_diagnosis",
            {
                "root_cause": "deployment_regression",
                "evidence_ids": [
                    "deploy:payment-api:v2026.09.20.2",
                    "log:payment-api:null-pointer-checkout",
                ],
                "mitigation": "rollback v2026.09.20.2",
            },
        )
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)
        self.assertTrue(final.info["success"])
        self.assertAlmostEqual(final.info["episode_reward"], 2.04)
        self.assertAlmostEqual(deploy.reward + logs.reward + final.reward, 2.04)
        with self.assertRaises(EpisodeFinishedError):
            env.step("get_dependencies", {"service": "payment-api"})

    def test_invalid_parameter_returns_structured_recoverable_error(self) -> None:
        env = IncidentEnvironment(load_scenario("db_pool_001"), max_steps=5)
        invalid = env.step(
            "query_metrics",
            {
                "service": "orders-db",
                "metric": "pool_usage",
                "start_time": "2026-09-20T10:00:00Z",
                "end_time": "2026-09-20T10:30:00Z",
            },
        )
        self.assertFalse(invalid.info["valid"])
        self.assertEqual(
            invalid.observation["error"]["error_type"], "invalid_enum_value"
        )
        self.assertTrue(invalid.observation["error"]["recoverable"])

        recovered = env.step(
            "query_metrics",
            {
                "service": "orders-db",
                "metric": "connection_pool_usage",
                "start_time": "2026-09-20T10:00:00Z",
                "end_time": "2026-09-20T10:30:00Z",
            },
        )
        self.assertTrue(recovered.info["valid"])
        self.assertEqual(env.reward_breakdown().recovery, 0.15)

    def test_unrelated_valid_call_is_not_counted_as_recovery(self) -> None:
        env = IncidentEnvironment(load_scenario("db_pool_001"), max_steps=5)
        env.step(
            "query_metrics",
            {
                "service": "orders-db",
                "metric": "pool_usage",
                "start_time": "2026-09-20T10:00:00Z",
                "end_time": "2026-09-20T10:30:00Z",
            },
        )
        env.step("get_dependencies", {"service": "order-api", "direction": "upstream"})
        self.assertEqual(env.reward_breakdown().recovery, 0.0)

    def test_unknown_service_error_does_not_reveal_hidden_topology(self) -> None:
        env = IncidentEnvironment(load_scenario("db_pool_001"), max_steps=4)
        result = env.step(
            "get_dependencies", {"service": "invented-service", "direction": "all"}
        )
        serialized = json.dumps(result.observation)
        self.assertEqual(result.observation["error"]["error_type"], "unknown_service")
        self.assertNotIn("orders-db", serialized)

    def test_unseen_evidence_reference_is_rejected(self) -> None:
        env = IncidentEnvironment(load_scenario("certificate_expired_001"), max_steps=4)
        result = env.step(
            "submit_diagnosis",
            {
                "root_cause": "certificate_expired",
                "evidence_ids": ["log:invented:evidence"],
                "mitigation": "rotate the expired client certificate",
            },
        )
        self.assertFalse(result.terminated)
        self.assertEqual(
            result.observation["error"]["error_type"], "invalid_evidence_reference"
        )
        self.assertEqual(env.reward_breakdown().invalid_evidence, -0.25)

    def test_budget_exhaustion_truncates_episode(self) -> None:
        env = IncidentEnvironment(load_scenario("cache_outage_001"), max_steps=2)
        arguments = {"service": "catalog-api", "direction": "upstream"}
        env.step("get_dependencies", arguments)
        result = env.step("get_dependencies", arguments)
        self.assertTrue(result.truncated)
        self.assertTrue(result.observation["budget_exhausted"])
        self.assertEqual(env.reward_breakdown().redundancy, -0.05)


if __name__ == "__main__":
    unittest.main()

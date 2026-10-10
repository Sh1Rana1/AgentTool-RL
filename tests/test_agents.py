from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.agents import BaseLLMAgent, RandomAgent, RuleBasedAgent
from agenttool_rl.evaluation import run_episode
from agenttool_rl.llm import ChatMessage
from agenttool_rl.validation import load_jsonl


class ScriptedBackend:
    def __init__(self, responses: list[ChatMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatMessage:
        self.calls.append((list(messages), list(tools)))
        return self.responses.pop(0)


class BaselineAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_jsonl(PROJECT_ROOT / "data" / "examples" / "scenarios.jsonl")

    def test_rule_agent_solves_all_reviewed_examples(self) -> None:
        for scenario in self.scenarios:
            with self.subTest(task_id=scenario["task_id"]):
                metrics = run_episode(RuleBasedAgent(), scenario, max_steps=6)
                self.assertTrue(metrics.success)
                self.assertEqual(metrics.calls, 3)
                self.assertEqual(metrics.invalid_calls, 0)

    def test_random_agent_is_reproducible(self) -> None:
        scenario = self.scenarios[0]
        first = run_episode(RandomAgent(seed=17), scenario, max_steps=6)
        second = run_episode(RandomAgent(seed=17), scenario, max_steps=6)
        self.assertEqual(first, second)

    def test_base_llm_agent_supports_native_tool_calls(self) -> None:
        backend = ScriptedBackend(
            [
                ChatMessage(
                    tool_name="get_recent_deployments",
                    tool_arguments={
                        "service": "payment-api",
                        "since": "2026-09-20T09:00:00Z",
                    },
                    tool_call_id="call_1",
                ),
                ChatMessage(
                    tool_name="query_logs",
                    tool_arguments={
                        "service": "payment-api",
                        "keyword": "null",
                        "start_time": "2026-09-20T09:30:00Z",
                        "end_time": "2026-09-20T10:00:00Z",
                    },
                    tool_call_id="call_2",
                ),
                ChatMessage(
                    tool_name="submit_diagnosis",
                    tool_arguments={
                        "root_cause": "deployment_regression",
                        "evidence_ids": [
                            "deploy:payment-api:v2026.09.20.2",
                            "log:payment-api:null-pointer-checkout",
                        ],
                        "mitigation": "rollback v2026.09.20.2",
                    },
                    tool_call_id="call_3",
                ),
            ]
        )
        metrics = run_episode(BaseLLMAgent(backend), self.scenarios[0], max_steps=5)
        self.assertTrue(metrics.success)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(backend.calls[1][0][-1]["role"], "tool")

    def test_base_llm_agent_supports_json_text_fallback(self) -> None:
        backend = ScriptedBackend(
            [
                ChatMessage(
                    content=(
                        "```json\n"
                        '{"tool":"get_dependencies","arguments":'
                        '{"service":"order-api","direction":"upstream"}}\n'
                        "```"
                    )
                )
            ]
        )
        agent = BaseLLMAgent(backend)
        observation = {
            "incident": self.scenarios[1]["incident"],
            "available_tools": [],
            "budget_remaining": 5,
        }
        agent.reset(observation)
        action = agent.act()
        self.assertEqual(action.tool, "get_dependencies")
        self.assertEqual(action.arguments["service"], "order-api")

    def test_base_llm_agent_reprompts_non_action_output(self) -> None:
        backend = ScriptedBackend(
            [
                ChatMessage(content="I should inspect the dependencies next."),
                ChatMessage(
                    tool_name="get_dependencies",
                    tool_arguments={"service": "order-api", "direction": "upstream"},
                    tool_call_id="call_retry",
                ),
            ]
        )
        agent = BaseLLMAgent(backend)
        observation = {
            "incident": self.scenarios[1]["incident"],
            "available_tools": [],
            "budget_remaining": 5,
        }
        agent.reset(observation)

        action = agent.act()

        self.assertEqual(action.tool, "get_dependencies")
        self.assertEqual(action.raw_response["format_retries"], 1)
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("not a tool call", backend.calls[1][0][-1]["content"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.art_rollout import gather_art_rollout_groups, rollout_art
from agenttool_rl.validation import load_jsonl


def tool_choice(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    tool_call = SimpleNamespace(id=call_id, function=function)
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(message=message)


def text_choice(content: str) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))


class FakeTrajectory:
    def __init__(
        self,
        *,
        messages_and_choices,
        tools=None,
        metadata=None,
        reward=0.0,
    ) -> None:
        self.messages_and_choices = list(messages_and_choices)
        self.tools = tools
        self.metadata = metadata or {}
        self.reward = reward
        self.metrics = {}
        self.logs = []

    def messages(self):
        messages = []
        for item in self.messages_and_choices:
            if isinstance(item, SimpleNamespace):
                message = {
                    "role": "assistant",
                    "content": item.message.content or "",
                }
                if item.message.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in item.message.tool_calls
                    ]
                messages.append(message)
            else:
                messages.append(dict(item))
        return messages

    def finish(self):
        return self


class FakeTrajectoryGroup:
    def __new__(cls, trajectories, *, metadata=None):
        items = list(trajectories)

        async def collect():
            results = await asyncio.gather(*items, return_exceptions=True)
            group = object.__new__(cls)
            group.trajectories = [item for item in results if isinstance(item, FakeTrajectory)]
            group.exceptions = [item for item in results if isinstance(item, BaseException)]
            group.metadata = metadata or {}
            return group

        return collect()


class FakeArt:
    Trajectory = FakeTrajectory
    TrajectoryGroup = FakeTrajectoryGroup
    last_gather_options = None

    @classmethod
    async def gather_trajectory_groups(cls, groups, **options):
        cls.last_gather_options = options
        return list(await asyncio.gather(*groups))


class FakeCompletions:
    def __init__(self, choices, tracker=None) -> None:
        self.choices = list(choices)
        self.tracker = tracker
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.tracker is not None:
            self.tracker["active"] += 1
            self.tracker["max_active"] = max(
                self.tracker["max_active"], self.tracker["active"]
            )
            await asyncio.sleep(0.01)
            self.tracker["active"] -= 1
        return SimpleNamespace(choices=[self.choices.pop(0)])


class FakeModel:
    def __init__(self, response_sequences, tracker=None) -> None:
        self.response_sequences = list(response_sequences)
        self.tracker = tracker
        self.clients = []

    def openai_client(self):
        completions = FakeCompletions(self.response_sequences.pop(0), self.tracker)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.clients.append(client)
        return client

    def get_inference_name(self, step=None):
        del step
        return "test-policy"


class FailingModel:
    def openai_client(self):
        raise RuntimeError("inference endpoint unavailable")

    def get_inference_name(self, step=None):
        del step
        return "test-policy"


class ArtRolloutTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_jsonl(PROJECT_ROOT / "data" / "examples" / "scenarios.jsonl")

    async def test_rollout_builds_trainable_multiturn_trajectory(self) -> None:
        model = FakeModel(
            [
                [
                    tool_choice(
                        "get_recent_deployments",
                        {
                            "service": "payment-api",
                            "since": "2026-09-20T09:00:00Z",
                        },
                        "call_1",
                    ),
                    tool_choice(
                        "query_logs",
                        {
                            "service": "payment-api",
                            "keyword": "null",
                            "start_time": "2026-09-20T09:30:00Z",
                            "end_time": "2026-09-20T10:00:00Z",
                        },
                        "call_2",
                    ),
                    tool_choice(
                        "submit_diagnosis",
                        {
                            "root_cause": "deployment_regression",
                            "evidence_ids": [
                                "deploy:payment-api:v2026.09.20.2",
                                "log:payment-api:null-pointer-checkout",
                            ],
                            "mitigation": "rollback v2026.09.20.2",
                        },
                        "call_3",
                    ),
                ]
            ]
        )

        trajectory = await rollout_art(
            model,
            self.scenarios[0],
            max_steps=6,
            temperature=0.6,
            art_module=FakeArt,
        )

        self.assertTrue(trajectory.metrics["success"])
        self.assertEqual(trajectory.metrics["calls"], 3)
        self.assertEqual(trajectory.reward, 2.04)
        self.assertEqual(trajectory.metadata["scenario_id"], "deploy_regression_001")
        self.assertEqual(len(trajectory.tools), 5)
        self.assertEqual(
            sum(isinstance(item, SimpleNamespace) for item in trajectory.messages_and_choices),
            3,
        )
        tool_messages = [
            item
            for item in trajectory.messages_and_choices
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        self.assertEqual(
            [item["tool_call_id"] for item in tool_messages],
            ["call_1", "call_2", "call_3"],
        )
        self.assertEqual(model.clients[0].chat.completions.requests[0]["model"], "test-policy")

    async def test_rollout_records_format_correction_in_history(self) -> None:
        model = FakeModel(
            [
                [
                    text_choice("I should inspect deployments."),
                    tool_choice(
                        "get_recent_deployments",
                        {
                            "service": "payment-api",
                            "since": "2026-09-20T09:00:00Z",
                        },
                        "call_retry",
                    ),
                ]
            ]
        )

        trajectory = await rollout_art(
            model,
            self.scenarios[0],
            max_steps=1,
            art_module=FakeArt,
        )

        self.assertEqual(trajectory.metrics["format_retries"], 1)
        self.assertEqual(trajectory.metrics["calls"], 1)
        correction_messages = [
            item
            for item in trajectory.messages_and_choices
            if isinstance(item, dict) and "previous response" in item.get("content", "")
        ]
        self.assertEqual(len(correction_messages), 1)

    async def test_rollout_returns_auditable_partial_trajectory_on_client_error(self) -> None:
        trajectory = await rollout_art(
            FailingModel(),
            self.scenarios[0],
            art_module=FakeArt,
        )

        self.assertTrue(trajectory.metrics["agent_error"])
        self.assertEqual(trajectory.metrics["calls"], 0)
        self.assertEqual(trajectory.reward, 0.0)
        self.assertIn("inference endpoint unavailable", trajectory.logs[0])

    async def test_group_gather_runs_same_scenario_rollouts_concurrently(self) -> None:
        tracker = {"active": 0, "max_active": 0}
        action = tool_choice(
            "get_recent_deployments",
            {"service": "payment-api", "since": "2026-09-20T09:00:00Z"},
            "call_group",
        )
        model = FakeModel([[action] for _ in range(4)], tracker=tracker)

        groups = await gather_art_rollout_groups(
            model,
            self.scenarios[:2],
            rollouts_per_group=2,
            max_steps=1,
            art_module=FakeArt,
        )

        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group.trajectories) == 2 for group in groups))
        self.assertGreater(tracker["max_active"], 1)
        self.assertEqual(FakeArt.last_gather_options["max_exceptions"], 4)


if __name__ == "__main__":
    unittest.main()

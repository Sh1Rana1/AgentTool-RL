"""Run a deterministic episode that includes one recoverable parameter error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.environment import IncidentEnvironment  # noqa: E402
from agenttool_rl.validation import load_jsonl  # noqa: E402


def main() -> int:
    scenarios = load_jsonl(PROJECT_ROOT / "data" / "examples" / "scenarios.jsonl")
    scenario = next(item for item in scenarios if item["task_id"] == "db_pool_001")
    env = IncidentEnvironment(scenario, max_steps=6)

    actions = [
        ("get_dependencies", {"service": "order-api", "direction": "upstream"}),
        (
            "query_metrics",
            {
                "service": "orders-db",
                "metric": "pool_usage",
                "start_time": "2026-09-20T10:00:00Z",
                "end_time": "2026-09-20T10:30:00Z",
            },
        ),
        (
            "query_metrics",
            {
                "service": "orders-db",
                "metric": "connection_pool_usage",
                "start_time": "2026-09-20T10:00:00Z",
                "end_time": "2026-09-20T10:30:00Z",
            },
        ),
        (
            "submit_diagnosis",
            {
                "root_cause": "database_connection_pool_exhausted",
                "evidence_ids": [
                    "dep:order-api:orders-db",
                    "metric:orders-db:connection_pool_usage:99",
                ],
                "mitigation": "increase connection pool capacity",
            },
        ),
    ]

    for tool, arguments in actions:
        result = env.step(tool, arguments)
        print(
            json.dumps(
                {
                    "tool": tool,
                    "observation": result.observation,
                    "step_reward": result.reward,
                },
                ensure_ascii=False,
            )
        )

    print(json.dumps(env.episode_summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

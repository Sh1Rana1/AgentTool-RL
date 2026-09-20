"""Static validation for versioned example scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ROOT_CAUSES, TOOL_CONTRACTS

REQUIRED_TOP_LEVEL = {
    "task_id",
    "user_request",
    "incident",
    "hidden_state",
    "expected_path",
}
REQUIRED_INCIDENT = {"service", "alert", "started_at"}
REQUIRED_HIDDEN = {
    "root_cause",
    "relevant_services",
    "required_evidence",
    "acceptable_mitigations",
}


class ScenarioValidationError(ValueError):
    """Raised when a scenario violates the repository data contract."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScenarioValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ScenarioValidationError(f"{path}:{line_number}: expected a JSON object")
            scenarios.append(item)
    return scenarios


def _require_keys(value: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise ScenarioValidationError(f"{context}: missing keys {missing}")


def validate_scenario(scenario: dict[str, Any]) -> None:
    task_id = scenario.get("task_id", "<unknown>")
    _require_keys(scenario, REQUIRED_TOP_LEVEL, task_id)
    if not isinstance(scenario["task_id"], str) or not scenario["task_id"].strip():
        raise ScenarioValidationError(f"{task_id}: task_id must be a non-empty string")
    if not isinstance(scenario["user_request"], str) or not scenario["user_request"].strip():
        raise ScenarioValidationError(f"{task_id}: user_request must be a non-empty string")

    incident = scenario["incident"]
    hidden = scenario["hidden_state"]
    expected_path = scenario["expected_path"]
    if not isinstance(incident, dict):
        raise ScenarioValidationError(f"{task_id}: incident must be an object")
    if not isinstance(hidden, dict):
        raise ScenarioValidationError(f"{task_id}: hidden_state must be an object")
    if not isinstance(expected_path, list) or not expected_path:
        raise ScenarioValidationError(f"{task_id}: expected_path must be a non-empty list")

    _require_keys(incident, REQUIRED_INCIDENT, f"{task_id}.incident")
    _require_keys(hidden, REQUIRED_HIDDEN, f"{task_id}.hidden_state")
    if hidden["root_cause"] not in ROOT_CAUSES:
        raise ScenarioValidationError(
            f"{task_id}: unsupported root cause {hidden['root_cause']!r}"
        )
    if not hidden["required_evidence"]:
        raise ScenarioValidationError(f"{task_id}: required_evidence cannot be empty")
    if not hidden["acceptable_mitigations"]:
        raise ScenarioValidationError(f"{task_id}: acceptable_mitigations cannot be empty")

    for step_number, step in enumerate(expected_path, start=1):
        if not isinstance(step, dict) or "tool" not in step or "purpose" not in step:
            raise ScenarioValidationError(
                f"{task_id}.expected_path[{step_number}]: expected tool and purpose"
            )
        if step["tool"] not in TOOL_CONTRACTS:
            raise ScenarioValidationError(
                f"{task_id}.expected_path[{step_number}]: unknown tool {step['tool']!r}"
            )
    if expected_path[-1]["tool"] != "submit_diagnosis":
        raise ScenarioValidationError(f"{task_id}: expected_path must end with submit_diagnosis")


def validate_scenarios(scenarios: Iterable[dict[str, Any]]) -> int:
    seen_ids: set[str] = set()
    count = 0
    for scenario in scenarios:
        validate_scenario(scenario)
        task_id = scenario["task_id"]
        if task_id in seen_ids:
            raise ScenarioValidationError(f"duplicate task_id: {task_id}")
        seen_ids.add(task_id)
        count += 1
    if count == 0:
        raise ScenarioValidationError("scenario collection is empty")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AgentTool-RL JSONL scenarios.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/examples/scenarios.jsonl"),
    )
    args = parser.parse_args(argv)
    count = validate_scenarios(load_jsonl(args.path))
    print(f"Validated {count} scenarios and {len(TOOL_CONTRACTS)} tool contracts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


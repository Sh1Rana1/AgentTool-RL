"""Static validation for versioned example scenarios."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
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
    "tool_data",
}
REQUIRED_TOOL_DATA = {"logs", "metrics", "dependencies", "deployments"}
TOOL_DATA_FIELDS = {
    "logs": {"service", "timestamp", "level", "message", "evidence_id"},
    "metrics": {"service", "metric", "timestamp", "value", "unit", "evidence_id"},
    "dependencies": {"source", "target", "evidence_id"},
    "deployments": {"service", "version", "deployed_at", "status", "evidence_id"},
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


def _validate_timestamp(value: Any, context: str) -> None:
    if not isinstance(value, str):
        raise ScenarioValidationError(f"{context}: expected an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScenarioValidationError(f"{context}: invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ScenarioValidationError(f"{context}: timestamp must include a timezone")


def _validate_string_list(value: Any, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ScenarioValidationError(f"{context}: expected a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ScenarioValidationError(f"{context}: items must be non-empty strings")


def _validate_tool_data(tool_data: Any, task_id: str) -> set[str]:
    if not isinstance(tool_data, dict):
        raise ScenarioValidationError(f"{task_id}.hidden_state.tool_data: expected an object")
    _require_keys(tool_data, REQUIRED_TOOL_DATA, f"{task_id}.hidden_state.tool_data")

    evidence_ids: set[str] = set()
    allowed_metrics = next(
        parameter.enum
        for parameter in TOOL_CONTRACTS["query_metrics"].parameters
        if parameter.name == "metric"
    )
    for collection, required_fields in TOOL_DATA_FIELDS.items():
        records = tool_data[collection]
        if not isinstance(records, list):
            raise ScenarioValidationError(f"{task_id}.tool_data.{collection}: expected a list")
        for index, record in enumerate(records):
            context = f"{task_id}.tool_data.{collection}[{index}]"
            if not isinstance(record, dict):
                raise ScenarioValidationError(f"{context}: expected an object")
            _require_keys(record, required_fields, context)
            evidence_id = record["evidence_id"]
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ScenarioValidationError(f"{context}.evidence_id: expected a string")
            if evidence_id in evidence_ids:
                raise ScenarioValidationError(f"{task_id}: duplicate evidence ID {evidence_id!r}")
            evidence_ids.add(evidence_id)

            if collection in {"logs", "metrics"}:
                _validate_timestamp(record["timestamp"], f"{context}.timestamp")
            elif collection == "deployments":
                _validate_timestamp(record["deployed_at"], f"{context}.deployed_at")
            if collection == "metrics" and record["metric"] not in allowed_metrics:
                raise ScenarioValidationError(
                    f"{context}.metric: unsupported metric {record['metric']!r}"
                )
    return evidence_ids


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
    for key in REQUIRED_INCIDENT:
        if not isinstance(incident[key], str) or not incident[key].strip():
            raise ScenarioValidationError(f"{task_id}.incident.{key}: expected a string")
    _validate_timestamp(incident["started_at"], f"{task_id}.incident.started_at")
    if hidden["root_cause"] not in ROOT_CAUSES:
        raise ScenarioValidationError(
            f"{task_id}: unsupported root cause {hidden['root_cause']!r}"
        )
    _validate_string_list(hidden["relevant_services"], f"{task_id}.relevant_services")
    _validate_string_list(hidden["required_evidence"], f"{task_id}.required_evidence")
    _validate_string_list(
        hidden["acceptable_mitigations"], f"{task_id}.acceptable_mitigations"
    )
    available_evidence = _validate_tool_data(hidden["tool_data"], task_id)
    missing_evidence = sorted(set(hidden["required_evidence"]) - available_evidence)
    if missing_evidence:
        raise ScenarioValidationError(
            f"{task_id}: required evidence is unavailable: {missing_evidence}"
        )

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

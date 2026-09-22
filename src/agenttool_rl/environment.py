"""Stateful, dependency-free incident diagnosis environment."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from .contracts import TOOL_CONTRACTS, ToolContract, exported_tool_schemas
from .models import StepResult, ToolError, TrajectoryStep
from .reward import RewardBreakdown, RewardConfig, score_trajectory
from .validation import validate_scenario


class EpisodeFinishedError(RuntimeError):
    """Raised when an action is attempted after termination or truncation."""


class _InvalidCall(ValueError):
    def __init__(self, error: ToolError) -> None:
        super().__init__(error.message)
        self.error = error


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise _InvalidCall(
            ToolError(
                "invalid_timestamp",
                f"{field} must be an ISO-8601 timestamp with a timezone.",
                details={"field": field, "value": value},
            )
        ) from exc
    if parsed.tzinfo is None:
        raise _InvalidCall(
            ToolError(
                "invalid_timestamp",
                f"{field} must include a timezone.",
                details={"field": field, "value": value},
            )
        )
    return parsed


class IncidentEnvironment:
    """Execute validated tool calls against one hidden incident scenario."""

    def __init__(
        self,
        scenario: dict[str, Any] | None = None,
        *,
        max_steps: int = 8,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        self._scenario: dict[str, Any] | None = None
        self._trajectory: list[TrajectoryStep] = []
        self._collected_evidence: set[str] = set()
        self._submitted_diagnosis: dict[str, Any] | None = None
        self._action_keys: set[str] = set()
        self._terminated = False
        self._truncated = False
        self._last_score = 0.0
        if scenario is not None:
            self.reset(scenario)

    @property
    def trajectory(self) -> tuple[TrajectoryStep, ...]:
        return tuple(self._trajectory)

    @property
    def collected_evidence(self) -> frozenset[str]:
        return frozenset(self._collected_evidence)

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - len(self._trajectory)

    def reset(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Start a fresh episode and return a hidden-state-free observation."""

        validate_scenario(scenario)
        self._scenario = deepcopy(scenario)
        self._trajectory.clear()
        self._collected_evidence.clear()
        self._submitted_diagnosis = None
        self._action_keys.clear()
        self._terminated = False
        self._truncated = False
        self._last_score = 0.0
        return {
            "type": "incident",
            "user_request": scenario["user_request"],
            "incident": deepcopy(scenario["incident"]),
            "available_tools": exported_tool_schemas(),
            "budget_remaining": self.max_steps,
        }

    def step(self, tool: str, arguments: dict[str, Any]) -> StepResult:
        """Validate and execute one call, returning an incremental reward."""

        scenario = self._require_scenario()
        if self.done:
            raise EpisodeFinishedError("episode is already finished; call reset() first")

        action_key = json.dumps(
            {"tool": tool, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
        repeated = action_key in self._action_keys
        self._action_keys.add(action_key)

        try:
            normalized = self._validate_call(tool, arguments)
            payload, evidence_ids = self._execute(tool, normalized)
            self._collected_evidence.update(evidence_ids)
            observation = {
                "type": "tool_result",
                "tool": tool,
                "result": payload,
                "budget_remaining": self.remaining_steps - 1,
            }
            step = TrajectoryStep(
                index=len(self._trajectory),
                tool=tool,
                arguments=deepcopy(normalized),
                valid=True,
                observation=deepcopy(observation),
                evidence_ids=tuple(evidence_ids),
                repeated=repeated,
            )
            if tool == "submit_diagnosis":
                self._submitted_diagnosis = deepcopy(normalized)
                self._terminated = True
        except _InvalidCall as exc:
            observation = {
                "type": "tool_error",
                "tool": tool,
                "error": exc.error.as_dict(),
                "budget_remaining": self.remaining_steps - 1,
            }
            step = TrajectoryStep(
                index=len(self._trajectory),
                tool=tool,
                arguments=deepcopy(arguments) if isinstance(arguments, dict) else {},
                valid=False,
                observation=deepcopy(observation),
                error=exc.error,
                repeated=repeated,
            )

        self._trajectory.append(step)
        if not self._terminated and len(self._trajectory) >= self.max_steps:
            self._truncated = True
            observation["budget_exhausted"] = True
            observation["budget_remaining"] = 0
            step.observation["budget_exhausted"] = True
            step.observation["budget_remaining"] = 0

        breakdown = self.reward_breakdown()
        incremental_reward = round(breakdown.total - self._last_score, 6)
        self._last_score = breakdown.total
        info: dict[str, Any] = {
            "step_index": step.index,
            "valid": step.valid,
            "repeated": step.repeated,
            "evidence_ids": list(step.evidence_ids),
            "episode_reward": breakdown.total,
        }
        if self.done:
            info["reward_breakdown"] = breakdown.as_dict()
            info["success"] = breakdown.success
        return StepResult(
            observation=observation,
            reward=incremental_reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def reward_breakdown(self) -> RewardBreakdown:
        scenario = self._require_scenario()
        return score_trajectory(
            scenario,
            self._trajectory,
            self._collected_evidence,
            self._submitted_diagnosis,
            self.reward_config,
        )

    def episode_summary(self) -> dict[str, Any]:
        scenario = self._require_scenario()
        return {
            "task_id": scenario["task_id"],
            "terminated": self._terminated,
            "truncated": self._truncated,
            "steps": len(self._trajectory),
            "collected_evidence": sorted(self._collected_evidence),
            "trajectory": [step.as_dict() for step in self._trajectory],
            "reward": self.reward_breakdown().as_dict(),
        }

    def _require_scenario(self) -> dict[str, Any]:
        if self._scenario is None:
            raise RuntimeError("environment has not been reset with a scenario")
        return self._scenario

    def _validate_call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool not in TOOL_CONTRACTS:
            raise _InvalidCall(
                ToolError(
                    "unknown_tool",
                    f"Unknown tool {tool!r}.",
                    details={"available_tools": sorted(TOOL_CONTRACTS)},
                )
            )
        if not isinstance(arguments, dict):
            raise _InvalidCall(
                ToolError("invalid_arguments", "Tool arguments must be a JSON object.")
            )

        contract = TOOL_CONTRACTS[tool]
        normalized = self._validate_schema(contract, arguments)
        self._validate_semantics(tool, normalized)
        return normalized

    @staticmethod
    def _validate_schema(contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        parameters = {item.name: item for item in contract.parameters}
        unexpected = sorted(set(arguments) - parameters.keys())
        if unexpected:
            raise _InvalidCall(
                ToolError(
                    "unexpected_argument",
                    f"Unexpected arguments for {contract.name}: {unexpected}.",
                    details={"unexpected": unexpected},
                )
            )

        missing = sorted(
            item.name
            for item in contract.parameters
            if item.required and item.name not in arguments
        )
        if missing:
            raise _InvalidCall(
                ToolError(
                    "missing_argument",
                    f"Missing required arguments for {contract.name}: {missing}.",
                    details={"missing": missing},
                )
            )

        normalized = deepcopy(arguments)
        for parameter in contract.parameters:
            if parameter.name not in normalized:
                if parameter.default is not None:
                    normalized[parameter.name] = parameter.default
                continue
            value = normalized[parameter.name]
            expected_type = str if parameter.json_type == "string" else list
            if not isinstance(value, expected_type):
                raise _InvalidCall(
                    ToolError(
                        "invalid_argument_type",
                        f"{parameter.name} must be {parameter.json_type}.",
                        details={"field": parameter.name, "expected": parameter.json_type},
                    )
                )
            if isinstance(value, str) and not value.strip():
                raise _InvalidCall(
                    ToolError(
                        "empty_argument",
                        f"{parameter.name} cannot be empty.",
                        details={"field": parameter.name},
                    )
                )
            if parameter.items_type == "string" and any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise _InvalidCall(
                    ToolError(
                        "invalid_array_item",
                        f"Every item in {parameter.name} must be a non-empty string.",
                        details={"field": parameter.name},
                    )
                )
            if parameter.enum and value not in parameter.enum:
                raise _InvalidCall(
                    ToolError(
                        "invalid_enum_value",
                        f"Unsupported value for {parameter.name}: {value!r}.",
                        details={"field": parameter.name, "allowed": list(parameter.enum)},
                    )
                )
        return normalized

    def _validate_semantics(self, tool: str, arguments: dict[str, Any]) -> None:
        if "service" in arguments and arguments["service"] not in self._known_services():
            raise _InvalidCall(
                ToolError(
                    "unknown_service",
                    f"Unknown service {arguments['service']!r}.",
                    details={"field": "service", "value": arguments["service"]},
                )
            )

        if tool in {"query_logs", "query_metrics"}:
            start = _parse_timestamp(arguments["start_time"], "start_time")
            end = _parse_timestamp(arguments["end_time"], "end_time")
            if start >= end:
                raise _InvalidCall(
                    ToolError(
                        "invalid_time_window",
                        "start_time must be earlier than end_time.",
                    )
                )
            if (end - start).total_seconds() > 24 * 60 * 60:
                raise _InvalidCall(
                    ToolError(
                        "time_window_too_large",
                        "The maximum query window is 24 hours.",
                    )
                )
        elif tool == "get_recent_deployments":
            _parse_timestamp(arguments["since"], "since")
        elif tool == "submit_diagnosis":
            evidence_ids = arguments["evidence_ids"]
            if not evidence_ids:
                raise _InvalidCall(
                    ToolError(
                        "missing_evidence",
                        "At least one evidence ID is required.",
                    )
                )
            if len(evidence_ids) != len(set(evidence_ids)):
                raise _InvalidCall(
                    ToolError(
                        "duplicate_evidence_reference",
                        "evidence_ids must not contain duplicates.",
                    )
                )
            unseen = sorted(set(evidence_ids) - self._collected_evidence)
            if unseen:
                raise _InvalidCall(
                    ToolError(
                        "invalid_evidence_reference",
                        "Diagnosis references evidence not returned in this episode.",
                        details={"unseen_evidence_ids": unseen},
                    )
                )

    def _known_services(self) -> set[str]:
        scenario = self._require_scenario()
        hidden = scenario["hidden_state"]
        services = set(hidden["relevant_services"])
        services.add(scenario["incident"]["service"])
        data = hidden["tool_data"]
        for item in data.get("logs", []):
            services.add(item["service"])
        for item in data.get("metrics", []):
            services.add(item["service"])
        for item in data.get("deployments", []):
            services.add(item["service"])
        for item in data.get("dependencies", []):
            services.update((item["source"], item["target"]))
        return services

    def _execute(
        self, tool: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        executors: dict[str, Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]] = {
            "query_logs": self._query_logs,
            "query_metrics": self._query_metrics,
            "get_dependencies": self._get_dependencies,
            "get_recent_deployments": self._get_recent_deployments,
            "submit_diagnosis": self._submit_diagnosis,
        }
        return executors[tool](arguments)

    def _query_logs(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        data = self._require_scenario()["hidden_state"]["tool_data"]["logs"]
        start = _parse_timestamp(arguments["start_time"], "start_time")
        end = _parse_timestamp(arguments["end_time"], "end_time")
        keyword = arguments["keyword"].casefold()
        matches = []
        for record in data:
            searchable = " ".join(
                (record["message"], *record.get("keywords", []))
            ).casefold()
            timestamp = _parse_timestamp(record["timestamp"], "record.timestamp")
            if (
                record["service"] == arguments["service"]
                and start <= timestamp < end
                and keyword in searchable
            ):
                matches.append(
                    {
                        key: deepcopy(record[key])
                        for key in ("timestamp", "level", "message", "evidence_id")
                    }
                )
        evidence = [item["evidence_id"] for item in matches]
        return {"entries": matches, "count": len(matches)}, evidence

    def _query_metrics(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        data = self._require_scenario()["hidden_state"]["tool_data"]["metrics"]
        start = _parse_timestamp(arguments["start_time"], "start_time")
        end = _parse_timestamp(arguments["end_time"], "end_time")
        matches = []
        for record in data:
            timestamp = _parse_timestamp(record["timestamp"], "record.timestamp")
            if (
                record["service"] == arguments["service"]
                and record["metric"] == arguments["metric"]
                and start <= timestamp < end
            ):
                matches.append(
                    {
                        key: deepcopy(record[key])
                        for key in ("timestamp", "value", "unit", "evidence_id")
                    }
                )
        evidence = [item["evidence_id"] for item in matches]
        return {
            "service": arguments["service"],
            "metric": arguments["metric"],
            "points": matches,
            "count": len(matches),
        }, evidence

    def _get_dependencies(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        data = self._require_scenario()["hidden_state"]["tool_data"]["dependencies"]
        service = arguments["service"]
        direction = arguments["direction"]
        matches = []
        for record in data:
            if record["source"] == service and direction in {"upstream", "all"}:
                matches.append(
                    {
                        "service": record["target"],
                        "relationship": "upstream",
                        "evidence_id": record["evidence_id"],
                    }
                )
            elif record["target"] == service and direction in {"downstream", "all"}:
                matches.append(
                    {
                        "service": record["source"],
                        "relationship": "downstream",
                        "evidence_id": record["evidence_id"],
                    }
                )
        evidence = [item["evidence_id"] for item in matches]
        return {"dependencies": matches, "count": len(matches)}, evidence

    def _get_recent_deployments(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        data = self._require_scenario()["hidden_state"]["tool_data"]["deployments"]
        since = _parse_timestamp(arguments["since"], "since")
        matches = []
        for record in data:
            deployed_at = _parse_timestamp(record["deployed_at"], "record.deployed_at")
            if record["service"] == arguments["service"] and deployed_at >= since:
                matches.append(
                    {
                        key: deepcopy(record[key])
                        for key in ("version", "deployed_at", "status", "evidence_id")
                    }
                )
        evidence = [item["evidence_id"] for item in matches]
        return {"deployments": matches, "count": len(matches)}, evidence

    @staticmethod
    def _submit_diagnosis(
        arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        return {
            "accepted": True,
            "message": "Diagnosis recorded; episode terminated.",
        }, []

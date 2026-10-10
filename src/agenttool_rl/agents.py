"""Agent interfaces and deterministic baselines for the v0.3 evaluator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from .contracts import ROOT_CAUSES
from .llm import ChatBackend
from .models import StepResult


DEFAULT_SYSTEM_PROMPT = (
    "You are an incident diagnosis agent. On every turn, call exactly one of the "
    "available tools; never reply with analysis or prose. Use investigation tools "
    "to collect evidence before calling submit_diagnosis. Never invent evidence IDs. "
    "When a tool returns an empty result or a recoverable error, choose another valid "
    "tool call or correct the arguments and retry. Stay within the call budget. The "
    "final answer must also be a submit_diagnosis tool call. If native tool calls are "
    "unavailable, return only a JSON object with keys tool and arguments."
)


@dataclass(frozen=True)
class AgentAction:
    """One tool action proposed by an agent."""

    tool: str
    arguments: dict[str, Any]
    raw_response: Any = None


class ToolAgent(Protocol):
    """Minimal stateful agent interface used by the evaluator."""

    name: str

    def reset(self, observation: dict[str, Any]) -> None: ...

    def act(self) -> AgentAction: ...

    def observe(self, action: AgentAction, result: StepResult) -> None: ...


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


@dataclass
class RandomAgent:
    """A seeded baseline that samples schema-shaped calls without hidden state."""

    seed: int = 42
    name: str = "random"
    _rng: random.Random = field(init=False, repr=False)
    _incident: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    _known_services: list[str] = field(init=False, repr=False, default_factory=list)
    _evidence: list[str] = field(init=False, repr=False, default_factory=list)
    _budget_remaining: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, observation: dict[str, Any]) -> None:
        self._rng = random.Random(self.seed)
        self._incident = dict(observation["incident"])
        self._known_services = [self._incident["service"]]
        self._evidence = []
        self._budget_remaining = observation["budget_remaining"]

    def act(self) -> AgentAction:
        if self._budget_remaining <= 1:
            return AgentAction(
                "submit_diagnosis",
                {
                    "root_cause": self._rng.choice(ROOT_CAUSES),
                    "evidence_ids": list(dict.fromkeys(self._evidence)),
                    "mitigation": "restart the affected service",
                },
            )

        tool = self._rng.choice(
            ("query_logs", "query_metrics", "get_dependencies", "get_recent_deployments")
        )
        service = self._rng.choice(self._known_services)
        started_at = _parse_timestamp(self._incident["started_at"])
        start = _iso(started_at - timedelta(minutes=15))
        end = _iso(started_at + timedelta(minutes=45))
        if tool == "query_logs":
            return AgentAction(
                tool,
                {
                    "service": service,
                    "keyword": self._rng.choice(
                        ("error", "timeout", "connection", "null", "expired", "cache")
                    ),
                    "start_time": start,
                    "end_time": end,
                },
            )
        if tool == "query_metrics":
            return AgentAction(
                tool,
                {
                    "service": service,
                    "metric": self._rng.choice(
                        (
                            "error_rate",
                            "latency_p95",
                            "connection_pool_usage",
                            "cache_hit_rate",
                            "disk_usage",
                            "memory_usage",
                            "tls_handshake_errors",
                        )
                    ),
                    "start_time": start,
                    "end_time": end,
                },
            )
        if tool == "get_dependencies":
            return AgentAction(
                tool,
                {"service": service, "direction": self._rng.choice(("upstream", "all"))},
            )
        return AgentAction(tool, {"service": service, "since": start})

    def observe(self, action: AgentAction, result: StepResult) -> None:
        del action
        self._budget_remaining = result.observation["budget_remaining"]
        for evidence_id in result.info["evidence_ids"]:
            if evidence_id not in self._evidence:
                self._evidence.append(evidence_id)
        payload = result.observation.get("result", {})
        for dependency in payload.get("dependencies", []):
            service = dependency.get("service")
            if isinstance(service, str) and service not in self._known_services:
                self._known_services.append(service)


@dataclass
class RuleBasedAgent:
    """An interpretable upper sanity-check for the five v0.3 scenario templates."""

    name: str = "rule-based"
    _incident: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    _kind: str = field(init=False, repr=False, default="")
    _phase: int = field(init=False, repr=False, default=0)
    _evidence: list[str] = field(init=False, repr=False, default_factory=list)
    _dependency: str | None = field(init=False, repr=False, default=None)
    _version: str | None = field(init=False, repr=False, default=None)

    def reset(self, observation: dict[str, Any]) -> None:
        self._incident = dict(observation["incident"])
        alert = self._incident["alert"].casefold()
        if "error_rate" in alert:
            self._kind = "deployment"
        elif "tls_handshake" in alert:
            self._kind = "certificate"
        elif "504" in alert:
            self._kind = "upstream"
        elif "3s" in alert:
            self._kind = "database"
        else:
            self._kind = "cache"
        self._phase = 0
        self._evidence = []
        self._dependency = None
        self._version = None

    def act(self) -> AgentAction:
        service = self._incident["service"]
        started_at = _parse_timestamp(self._incident["started_at"])
        start = _iso(started_at - timedelta(minutes=15))
        end = _iso(started_at + timedelta(minutes=60))

        plans = {
            "deployment": (
                ("get_recent_deployments", {"service": service, "since": start}),
                (
                    "query_logs",
                    {
                        "service": service,
                        "keyword": "null",
                        "start_time": start,
                        "end_time": end,
                    },
                ),
            ),
            "database": (
                ("get_dependencies", {"service": service, "direction": "upstream"}),
                (
                    "query_metrics",
                    {
                        "service": self._dependency or service,
                        "metric": "connection_pool_usage",
                        "start_time": start,
                        "end_time": end,
                    },
                ),
            ),
            "cache": (
                ("get_dependencies", {"service": service, "direction": "upstream"}),
                (
                    "query_metrics",
                    {
                        "service": service,
                        "metric": "cache_hit_rate",
                        "start_time": start,
                        "end_time": end,
                    },
                ),
            ),
            "upstream": (
                ("get_dependencies", {"service": service, "direction": "upstream"}),
                (
                    "query_metrics",
                    {
                        "service": self._dependency or service,
                        "metric": "latency_p95",
                        "start_time": start,
                        "end_time": end,
                    },
                ),
            ),
            "certificate": (
                (
                    "query_logs",
                    {
                        "service": service,
                        "keyword": "expired",
                        "start_time": start,
                        "end_time": end,
                    },
                ),
                ("get_dependencies", {"service": service, "direction": "upstream"}),
            ),
        }
        plan = plans[self._kind]
        if self._phase < len(plan):
            tool, arguments = plan[self._phase]
            return AgentAction(tool, arguments)
        return AgentAction("submit_diagnosis", self._diagnosis())

    def observe(self, action: AgentAction, result: StepResult) -> None:
        for evidence_id in result.info["evidence_ids"]:
            if evidence_id not in self._evidence:
                self._evidence.append(evidence_id)
        payload = result.observation.get("result", {})
        dependencies = payload.get("dependencies", [])
        if dependencies:
            self._dependency = dependencies[0]["service"]
        deployments = payload.get("deployments", [])
        if deployments:
            self._version = deployments[0]["version"]
        if action.tool != "submit_diagnosis":
            self._phase += 1

    def _diagnosis(self) -> dict[str, Any]:
        root_causes = {
            "deployment": "deployment_regression",
            "database": "database_connection_pool_exhausted",
            "cache": "cache_unavailable",
            "upstream": "upstream_timeout",
            "certificate": "certificate_expired",
        }
        if self._kind == "deployment":
            mitigation = f"rollback {self._version}" if self._version else "rollback deployment"
        elif self._kind == "database":
            mitigation = "increase connection pool capacity"
        elif self._kind == "cache":
            mitigation = f"restore {self._dependency}" if self._dependency else "restore cache"
        elif self._kind == "upstream":
            mitigation = f"scale {self._dependency}" if self._dependency else "scale upstream"
        else:
            mitigation = "rotate the expired client certificate"
        return {
            "root_cause": root_causes[self._kind],
            "evidence_ids": list(self._evidence),
            "mitigation": mitigation,
        }


@dataclass
class BaseLLMAgent:
    """A tool-calling baseline backed by an OpenAI-compatible chat endpoint."""

    backend: ChatBackend
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_format_retries: int = 2
    name: str = "base-llm"
    _messages: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _tools: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _pending_tool_call_id: str | None = field(init=False, repr=False, default=None)
    _last_mode: str = field(init=False, repr=False, default="native")

    def reset(self, observation: dict[str, Any]) -> None:
        self._tools = list(observation["available_tools"])
        public_observation = {
            key: value for key, value in observation.items() if key != "available_tools"
        }
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "Initial environment observation:\n"
                + json.dumps(public_observation, ensure_ascii=False),
            },
        ]
        self._pending_tool_call_id = None
        self._last_mode = "native"

    def act(self) -> AgentAction:
        attempts: list[dict[str, Any]] = []
        for retry_index in range(self.max_format_retries + 1):
            response = self.backend.complete(self._messages, self._tools)
            attempts.append(response.as_dict())
            if response.tool_name is not None:
                self._last_mode = "native"
                tool_call_id = response.tool_call_id or f"call_{len(self._messages)}"
                self._pending_tool_call_id = tool_call_id
                raw_arguments = response.tool_arguments
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else dict(raw_arguments or {})
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    arguments = {}
                    tool_name = response.tool_name
                else:
                    tool_name = response.tool_name
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": response.tool_name,
                                    "arguments": raw_arguments
                                    if isinstance(raw_arguments, str)
                                    else json.dumps(raw_arguments or {}, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
                return AgentAction(
                    tool_name,
                    arguments,
                    raw_response={"attempts": attempts, "format_retries": retry_index},
                )

            self._last_mode = "json"
            content = response.content or ""
            self._messages.append({"role": "assistant", "content": content})
            try:
                payload = parse_json_action(content)
                return AgentAction(
                    payload["tool"],
                    payload["arguments"],
                    raw_response={"attempts": attempts, "format_retries": retry_index},
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                if retry_index < self.max_format_retries:
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not a tool call. Call exactly "
                                "one available tool now. Do not explain your choice."
                            ),
                        }
                    )

        return AgentAction(
            "__invalid_model_output__",
            {},
            raw_response={"attempts": attempts, "format_retries": self.max_format_retries},
        )

    def observe(self, action: AgentAction, result: StepResult) -> None:
        content = json.dumps(result.observation, ensure_ascii=False)
        if self._last_mode == "native" and self._pending_tool_call_id is not None:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._pending_tool_call_id,
                    "name": action.tool,
                    "content": content,
                }
            )
        else:
            self._messages.append(
                {"role": "user", "content": "Environment response:\n" + content}
            )
        self._pending_tool_call_id = None


def parse_json_action(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("model action must be a JSON object")
    if not isinstance(payload.get("tool"), str):
        raise ValueError("model action requires a tool name")
    if not isinstance(payload.get("arguments"), dict):
        raise ValueError("model action arguments must be an object")
    return payload

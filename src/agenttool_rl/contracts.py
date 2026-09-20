"""Versioned, dependency-free contracts for the incident diagnosis tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

JsonType = Literal["string", "array"]


@dataclass(frozen=True)
class ToolParameter:
    """One input parameter exposed to the agent."""

    name: str
    json_type: JsonType
    description: str
    required: bool = True
    items_type: str | None = None
    enum: tuple[str, ...] = ()
    default: Any = None

    def as_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": self.json_type,
            "description": self.description,
        }
        if self.items_type is not None:
            schema["items"] = {"type": self.items_type}
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True)
class ToolContract:
    """Machine-readable tool definition used by agents and validators."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    version: str = "1.0"

    def as_json_schema(self) -> dict[str, Any]:
        required = [parameter.name for parameter in self.parameters if parameter.required]
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter.name: parameter.as_json_schema()
                    for parameter in self.parameters
                },
                "required": required,
                "additionalProperties": False,
            },
            "x-contract-version": self.version,
        }


ROOT_CAUSES = (
    "deployment_regression",
    "database_connection_pool_exhausted",
    "cache_unavailable",
    "upstream_timeout",
    "disk_full",
    "memory_leak",
    "configuration_error",
    "certificate_expired",
)


TOOL_CONTRACTS: dict[str, ToolContract] = {
    "query_logs": ToolContract(
        name="query_logs",
        description="Query service logs within an ISO-8601 time window.",
        parameters=(
            ToolParameter("service", "string", "Canonical service name."),
            ToolParameter("keyword", "string", "Keyword or error signature to search."),
            ToolParameter("start_time", "string", "Inclusive ISO-8601 start time."),
            ToolParameter("end_time", "string", "Exclusive ISO-8601 end time."),
        ),
    ),
    "query_metrics": ToolContract(
        name="query_metrics",
        description="Query one metric for a service within a time window.",
        parameters=(
            ToolParameter("service", "string", "Canonical service name."),
            ToolParameter(
                "metric",
                "string",
                "Metric to inspect.",
                enum=(
                    "error_rate",
                    "latency_p95",
                    "connection_pool_usage",
                    "cache_hit_rate",
                    "disk_usage",
                    "memory_usage",
                    "tls_handshake_errors",
                ),
            ),
            ToolParameter("start_time", "string", "Inclusive ISO-8601 start time."),
            ToolParameter("end_time", "string", "Exclusive ISO-8601 end time."),
        ),
    ),
    "get_dependencies": ToolContract(
        name="get_dependencies",
        description="Return upstream or downstream dependencies for a service.",
        parameters=(
            ToolParameter("service", "string", "Canonical service name."),
            ToolParameter(
                "direction",
                "string",
                "Dependency direction.",
                required=False,
                enum=("upstream", "downstream", "all"),
                default="all",
            ),
        ),
    ),
    "get_recent_deployments": ToolContract(
        name="get_recent_deployments",
        description="Return deployments for a service since the supplied timestamp.",
        parameters=(
            ToolParameter("service", "string", "Canonical service name."),
            ToolParameter("since", "string", "Inclusive ISO-8601 timestamp."),
        ),
    ),
    "submit_diagnosis": ToolContract(
        name="submit_diagnosis",
        description="Submit the final root cause, supporting evidence, and mitigation.",
        parameters=(
            ToolParameter("root_cause", "string", "Canonical root-cause label.", enum=ROOT_CAUSES),
            ToolParameter(
                "evidence_ids",
                "array",
                "Evidence IDs returned by earlier calls in this episode.",
                items_type="string",
            ),
            ToolParameter("mitigation", "string", "Concrete mitigation or rollback action."),
        ),
    ),
}


def exported_tool_schemas() -> list[dict[str, Any]]:
    """Return deterministic JSON schemas in tool-name order."""

    return [TOOL_CONTRACTS[name].as_json_schema() for name in sorted(TOOL_CONTRACTS)]


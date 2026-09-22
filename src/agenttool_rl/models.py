"""Shared data models for environment transitions and trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolError:
    """A machine-readable error that can be returned to an agent."""

    error_type: str
    message: str
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryStep:
    """One attempted tool call and the environment response."""

    index: int
    tool: str
    arguments: dict[str, Any]
    valid: bool
    observation: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    error: ToolError | None = None
    repeated: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        return value


@dataclass(frozen=True)
class StepResult:
    """Gym-like transition result without requiring Gym as a dependency."""

    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

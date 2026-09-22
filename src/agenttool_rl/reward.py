"""Deterministic, programmatic reward for incident-diagnosis trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Collection, Sequence

from .models import TrajectoryStep


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients frozen for the v0.2 environment contract."""

    task_success: float = 1.0
    diagnosis_correct: float = 0.30
    evidence_coverage: float = 0.40
    mitigation_correct: float = 0.20
    valid_call_ratio: float = 0.20
    recovery: float = 0.15
    invalid_call: float = -0.15
    invalid_evidence_reference: float = -0.25
    redundant_call: float = -0.05
    call_cost: float = -0.02
    incorrect_diagnosis: float = -0.50


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable reward components for one trajectory prefix or episode."""

    total: float
    task_success: float
    diagnosis: float
    evidence: float
    mitigation: float
    valid_calls: float
    recovery: float
    invalid_calls: float
    invalid_evidence: float
    redundancy: float
    cost: float
    evidence_coverage: float
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def score_trajectory(
    scenario: dict[str, Any],
    trajectory: Sequence[TrajectoryStep],
    collected_evidence: Collection[str],
    submitted_diagnosis: dict[str, Any] | None,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score a trajectory using only environment state and deterministic rules."""

    config = config or RewardConfig()
    hidden = scenario["hidden_state"]
    required = set(hidden["required_evidence"])
    collected = set(collected_evidence)
    coverage = len(required & collected) / len(required) if required else 1.0

    diagnosis_matches = False
    mitigation_matches = False
    submitted_evidence: set[str] = set()
    if submitted_diagnosis is not None:
        diagnosis_matches = submitted_diagnosis["root_cause"] == hidden["root_cause"]
        submitted_evidence = set(submitted_diagnosis["evidence_ids"])
        candidate = _normalize_text(submitted_diagnosis["mitigation"])
        acceptable = {_normalize_text(item) for item in hidden["acceptable_mitigations"]}
        mitigation_matches = candidate in acceptable

    required_submitted = required.issubset(submitted_evidence)
    success = bool(
        submitted_diagnosis is not None
        and diagnosis_matches
        and mitigation_matches
        and required_submitted
    )

    attempted = len(trajectory)
    valid_count = sum(step.valid for step in trajectory)
    invalid_count = attempted - valid_count
    redundant_count = sum(step.repeated for step in trajectory)
    bad_evidence_count = sum(
        step.error is not None and step.error.error_type == "invalid_evidence_reference"
        for step in trajectory
    )
    recovery_count = sum(
        current.valid and not previous.valid and current.tool == previous.tool
        for previous, current in zip(trajectory, trajectory[1:])
    )

    task_component = config.task_success if success else 0.0
    if submitted_diagnosis is None:
        diagnosis_component = 0.0
    elif diagnosis_matches:
        diagnosis_component = config.diagnosis_correct
    else:
        diagnosis_component = config.incorrect_diagnosis

    mitigation_component = config.mitigation_correct if mitigation_matches else 0.0
    evidence_component = config.evidence_coverage * coverage
    valid_component = config.valid_call_ratio * (valid_count / attempted) if attempted else 0.0
    recovery_component = config.recovery * recovery_count
    invalid_component = config.invalid_call * invalid_count
    bad_evidence_component = config.invalid_evidence_reference * bad_evidence_count
    redundancy_component = config.redundant_call * redundant_count
    cost_component = config.call_cost * attempted

    total = sum(
        (
            task_component,
            diagnosis_component,
            evidence_component,
            mitigation_component,
            valid_component,
            recovery_component,
            invalid_component,
            bad_evidence_component,
            redundancy_component,
            cost_component,
        )
    )
    return RewardBreakdown(
        total=round(total, 6),
        task_success=round(task_component, 6),
        diagnosis=round(diagnosis_component, 6),
        evidence=round(evidence_component, 6),
        mitigation=round(mitigation_component, 6),
        valid_calls=round(valid_component, 6),
        recovery=round(recovery_component, 6),
        invalid_calls=round(invalid_component, 6),
        invalid_evidence=round(bad_evidence_component, 6),
        redundancy=round(redundancy_component, 6),
        cost=round(cost_component, 6),
        evidence_coverage=round(coverage, 6),
        success=success,
    )

"""Unified episode runner, metrics, and reports for agent baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .agents import AgentAction, ToolAgent
from .environment import IncidentEnvironment

SCHEMA_ERROR_TYPES = {
    "unknown_tool",
    "invalid_arguments",
    "unexpected_argument",
    "missing_argument",
    "invalid_argument_type",
    "empty_argument",
    "invalid_array_item",
    "invalid_enum_value",
}


@dataclass(frozen=True)
class EpisodeMetrics:
    agent: str
    task_id: str
    root_cause: str
    success: bool
    terminated: bool
    truncated: bool
    diagnosis_correct: bool
    reward: float
    evidence_coverage: float
    calls: int
    valid_calls: int
    schema_valid_calls: int
    invalid_calls: int
    repeated_calls: int
    submitted_evidence: int
    valid_evidence_references: int
    recoverable_errors: int
    recovered_errors: int
    agent_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentReport:
    agent: str
    episode_count: int
    task_success_rate: float
    diagnosis_accuracy: float
    completion_rate: float
    truncation_rate: float
    schema_valid_rate: float
    semantic_argument_accuracy: float
    average_evidence_coverage: float
    evidence_validity: float | None
    error_recovery_rate: float | None
    invalid_call_rate: float
    average_tool_calls: float
    average_reward: float
    agent_error_rate: float
    success_by_root_cause: dict[str, float]
    episodes: tuple[EpisodeMetrics, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["episodes"] = [episode.as_dict() for episode in self.episodes]
        return value


@dataclass(frozen=True)
class BenchmarkReport:
    dataset: str
    max_steps: int
    reports: tuple[AgentReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "max_steps": self.max_steps,
            "reports": [report.as_dict() for report in self.reports],
        }

    def to_markdown(self) -> str:
        lines = [
            "# AgentTool-RL Baseline Report",
            "",
            f"Dataset: `{self.dataset}`  ",
            f"Maximum tool calls per episode: `{self.max_steps}`",
            "",
            "| Agent | Success | Diagnosis | Schema valid | Semantic valid | Coverage | "
            "Evidence valid | Recovery | Invalid | Avg calls | Avg reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for report in self.reports:
            lines.append(
                "| {agent} | {success:.1%} | {diagnosis:.1%} | {schema:.1%} | "
                "{semantic:.1%} | {coverage:.1%} | {evidence} | {recovery} | "
                "{invalid:.1%} | {calls:.2f} | {reward:.3f} |".format(
                    agent=report.agent,
                    success=report.task_success_rate,
                    diagnosis=report.diagnosis_accuracy,
                    schema=report.schema_valid_rate,
                    semantic=report.semantic_argument_accuracy,
                    coverage=report.average_evidence_coverage,
                    evidence=_format_percent(report.evidence_validity),
                    recovery=_format_percent(report.error_recovery_rate),
                    invalid=report.invalid_call_rate,
                    calls=report.average_tool_calls,
                    reward=report.average_reward,
                )
            )
        lines.extend(
            [
                "",
                "## Metric definitions",
                "",
                "- **Success:** correct diagnosis, required evidence, and accepted mitigation.",
                (
                    "- **Schema valid:** calls that pass tool name, type, required-field, "
                    "and enum checks."
                ),
                "- **Semantic valid:** executable calls divided by schema-valid calls.",
                "- **Coverage:** required evidence collected before the episode ends.",
                (
                    "- **Evidence valid:** submitted references that were observed earlier "
                    "in the episode."
                ),
                "- **Recovery:** recoverable errors immediately corrected with the same tool.",
                "- **Invalid:** calls rejected by schema or semantic validation.",
                "",
                "The rule-based agent encodes heuristics for the five known template "
                "families and is "
                "an environment sanity-check, not a learned generalization result.",
                "",
            ]
        )
        return "\n".join(lines)

    def write(self, json_path: Path, markdown_path: Path | None = None) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        target = markdown_path or json_path.with_suffix(".md")
        target.write_text(self.to_markdown(), encoding="utf-8", newline="\n")


AgentFactory = Callable[[int], ToolAgent]


def run_episode(
    agent: ToolAgent,
    scenario: dict[str, Any],
    *,
    max_steps: int = 8,
) -> EpisodeMetrics:
    """Run one isolated episode and compute metrics from its recorded trajectory."""

    env = IncidentEnvironment(max_steps=max_steps)
    initial_observation = env.reset(scenario)
    agent_error: str | None = None
    try:
        agent.reset(initial_observation)
        while not env.done:
            action = agent.act()
            if not isinstance(action, AgentAction):
                raise TypeError("agent.act() must return AgentAction")
            result = env.step(action.tool, action.arguments)
            agent.observe(action, result)
    except Exception as exc:  # evaluator isolates backend and agent failures per episode
        agent_error = f"{type(exc).__name__}: {exc}"[:500]

    trajectory = env.trajectory
    breakdown = env.reward_breakdown()
    calls = len(trajectory)
    valid_calls = sum(step.valid for step in trajectory)
    schema_valid_calls = sum(
        step.error is None or step.error.error_type not in SCHEMA_ERROR_TYPES
        for step in trajectory
    )
    repeated_calls = sum(step.repeated for step in trajectory)
    recoverable_errors = sum(
        step.error is not None and step.error.recoverable for step in trajectory
    )
    recovered_errors = sum(
        current.valid
        and previous.error is not None
        and previous.error.recoverable
        and current.tool == previous.tool
        for previous, current in zip(trajectory, trajectory[1:])
    )

    seen_evidence: set[str] = set()
    submitted_evidence = 0
    valid_evidence_references = 0
    diagnosis_correct = False
    for step in trajectory:
        if step.tool == "submit_diagnosis":
            references = step.arguments.get("evidence_ids", [])
            if isinstance(references, list):
                submitted_evidence += len(references)
                valid_evidence_references += sum(item in seen_evidence for item in references)
            if step.valid:
                diagnosis_correct = (
                    step.arguments.get("root_cause") == scenario["hidden_state"]["root_cause"]
                )
        seen_evidence.update(step.evidence_ids)

    return EpisodeMetrics(
        agent=agent.name,
        task_id=scenario["task_id"],
        root_cause=scenario["hidden_state"]["root_cause"],
        success=breakdown.success,
        terminated=env.done and bool(trajectory) and trajectory[-1].tool == "submit_diagnosis"
        and trajectory[-1].valid,
        truncated=env.done and not (
            bool(trajectory)
            and trajectory[-1].tool == "submit_diagnosis"
            and trajectory[-1].valid
        ),
        diagnosis_correct=diagnosis_correct,
        reward=breakdown.total,
        evidence_coverage=breakdown.evidence_coverage,
        calls=calls,
        valid_calls=valid_calls,
        schema_valid_calls=schema_valid_calls,
        invalid_calls=calls - valid_calls,
        repeated_calls=repeated_calls,
        submitted_evidence=submitted_evidence,
        valid_evidence_references=valid_evidence_references,
        recoverable_errors=recoverable_errors,
        recovered_errors=recovered_errors,
        agent_error=agent_error,
    )


def evaluate_agent(
    name: str,
    factory: AgentFactory,
    scenarios: Sequence[dict[str, Any]],
    *,
    max_steps: int = 8,
) -> AgentReport:
    episodes = tuple(
        run_episode(factory(index), scenario, max_steps=max_steps)
        for index, scenario in enumerate(scenarios)
    )
    if not episodes:
        raise ValueError("at least one scenario is required")

    total_calls = sum(item.calls for item in episodes)
    total_schema_valid = sum(item.schema_valid_calls for item in episodes)
    total_valid = sum(item.valid_calls for item in episodes)
    total_submitted_evidence = sum(item.submitted_evidence for item in episodes)
    total_valid_evidence = sum(item.valid_evidence_references for item in episodes)
    total_recoverable = sum(item.recoverable_errors for item in episodes)
    total_recovered = sum(item.recovered_errors for item in episodes)

    root_causes = sorted({item.root_cause for item in episodes})
    success_by_root_cause = {}
    for root_cause in root_causes:
        matching = [item for item in episodes if item.root_cause == root_cause]
        success_by_root_cause[root_cause] = _ratio(
            sum(item.success for item in matching), len(matching)
        )

    return AgentReport(
        agent=name,
        episode_count=len(episodes),
        task_success_rate=_ratio(sum(item.success for item in episodes), len(episodes)),
        diagnosis_accuracy=_ratio(
            sum(item.diagnosis_correct for item in episodes), len(episodes)
        ),
        completion_rate=_ratio(sum(item.terminated for item in episodes), len(episodes)),
        truncation_rate=_ratio(sum(item.truncated for item in episodes), len(episodes)),
        schema_valid_rate=_ratio(total_schema_valid, total_calls),
        semantic_argument_accuracy=_ratio(total_valid, total_schema_valid),
        average_evidence_coverage=round(
            fmean(item.evidence_coverage for item in episodes), 6
        ),
        evidence_validity=_optional_ratio(total_valid_evidence, total_submitted_evidence),
        error_recovery_rate=_optional_ratio(total_recovered, total_recoverable),
        invalid_call_rate=_ratio(sum(item.invalid_calls for item in episodes), total_calls),
        average_tool_calls=round(fmean(item.calls for item in episodes), 6),
        average_reward=round(fmean(item.reward for item in episodes), 6),
        agent_error_rate=_ratio(
            sum(item.agent_error is not None for item in episodes), len(episodes)
        ),
        success_by_root_cause=success_by_root_cause,
        episodes=episodes,
    )


def evaluate_agents(
    factories: Mapping[str, AgentFactory],
    scenarios: Sequence[dict[str, Any]],
    *,
    dataset: str,
    max_steps: int = 8,
) -> BenchmarkReport:
    reports = tuple(
        evaluate_agent(name, factory, scenarios, max_steps=max_steps)
        for name, factory in factories.items()
    )
    return BenchmarkReport(dataset=dataset, max_steps=max_steps, reports=reports)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _optional_ratio(
    numerator: int | float, denominator: int | float
) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"

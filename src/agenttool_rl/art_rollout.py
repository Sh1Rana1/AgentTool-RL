"""OpenPipe ART adapters for multi-turn, tool-using environment rollouts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence

from .agents import AgentAction, DEFAULT_SYSTEM_PROMPT, parse_json_action
from .environment import IncidentEnvironment


class ArtModel(Protocol):
    """The subset of ``art.Model`` used by the rollout adapter."""

    def openai_client(self) -> Any: ...

    def get_inference_name(self, step: int | None = None) -> str: ...


async def rollout_art(
    model: ArtModel,
    scenario: dict[str, Any],
    *,
    max_steps: int = 6,
    temperature: float = 0.7,
    max_format_retries: int = 2,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    art_module: Any | None = None,
) -> Any:
    """Run one environment episode and return an ART ``Trajectory``.

    Assistant ``Choice`` objects are retained as trainable outputs. Environment
    observations are appended as tool messages, so the complete multi-turn context is
    available to ART's tokenizer and GRPO trainer.
    """

    if max_format_retries < 0:
        raise ValueError("max_format_retries must be non-negative")
    art = _load_art(art_module)
    env = IncidentEnvironment(max_steps=max_steps)
    initial = env.reset(scenario)
    tool_schemas = [
        {"type": "function", "function": schema}
        for schema in initial["available_tools"]
    ]
    public_initial = {key: value for key, value in initial.items() if key != "available_tools"}
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Initial environment observation:\n"
                + json.dumps(public_initial, ensure_ascii=False),
            },
        ],
        tools=tool_schemas,
        metadata={"scenario_id": scenario["task_id"]},
    )
    total_format_retries = 0

    try:
        client = model.openai_client()
        while not env.done:
            action: AgentAction | None = None
            tool_call_id: str | None = None
            native_tool_call = False
            for retry_index in range(max_format_retries + 1):
                completion = await client.chat.completions.create(
                    model=model.get_inference_name(),
                    messages=trajectory.messages(),
                    tools=tool_schemas,
                    tool_choice="required",
                    temperature=temperature,
                )
                choice = completion.choices[0]
                trajectory.messages_and_choices.append(choice)
                action, tool_call_id, native_tool_call = _choice_action(choice)
                if action is not None:
                    total_format_retries += retry_index
                    break
                if retry_index < max_format_retries:
                    trajectory.messages_and_choices.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not a tool call. Call exactly "
                                "one available tool now. Do not explain your choice."
                            ),
                        }
                    )
            if action is None:
                total_format_retries += max_format_retries
                action = AgentAction("__invalid_model_output__", {})

            result = env.step(action.tool, action.arguments)
            observation = json.dumps(result.observation, ensure_ascii=False)
            if native_tool_call and tool_call_id is not None:
                trajectory.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": observation,
                    }
                )
            else:
                trajectory.messages_and_choices.append(
                    {"role": "user", "content": "Environment response:\n" + observation}
                )
    except Exception as exc:
        trajectory.logs.append(f"rollout_error: {type(exc).__name__}: {exc}")
        trajectory.metrics["agent_error"] = True

    breakdown = env.reward_breakdown()
    trajectory.reward = breakdown.total
    trajectory.metrics.update(
        _trajectory_metrics(env, scenario, total_format_retries=total_format_retries)
    )
    return trajectory.finish()


async def gather_art_rollout_groups(
    model: ArtModel,
    scenarios: Sequence[dict[str, Any]],
    *,
    rollouts_per_group: int = 4,
    max_steps: int = 6,
    temperature: float = 0.7,
    max_format_retries: int = 2,
    max_exceptions: int | None = None,
    art_module: Any | None = None,
) -> list[Any]:
    """Collect same-scenario rollout groups through ART's concurrent gather API."""

    if rollouts_per_group < 2:
        raise ValueError("rollouts_per_group must be at least 2 for grouped policy updates")
    if not scenarios:
        raise ValueError("at least one scenario is required")
    art = _load_art(art_module)
    allowed_exceptions = (
        rollouts_per_group * len(scenarios)
        if max_exceptions is None
        else max_exceptions
    )
    groups = [
        art.TrajectoryGroup(
            (
                rollout_art(
                    model,
                    scenario,
                    max_steps=max_steps,
                    temperature=temperature,
                    max_format_retries=max_format_retries,
                    art_module=art,
                )
                for _ in range(rollouts_per_group)
            ),
            metadata={"scenario_id": scenario["task_id"]},
        )
        for scenario in scenarios
    ]
    return await art.gather_trajectory_groups(
        groups,
        pbar_desc="agenttool-rollouts",
        max_exceptions=allowed_exceptions,
    )


def _choice_action(choice: Any) -> tuple[AgentAction | None, str | None, bool]:
    message = choice.message
    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) == 1:
        tool_call = tool_calls[0]
        function = tool_call.function
        raw_arguments = function.arguments
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else dict(raw_arguments or {})
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            arguments = {}
        return (
            AgentAction(function.name, arguments),
            getattr(tool_call, "id", None),
            True,
        )
    if tool_calls:
        return None, None, False

    content = getattr(message, "content", None) or ""
    try:
        payload = parse_json_action(content)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None, False
    return AgentAction(payload["tool"], payload["arguments"]), None, False


def _trajectory_metrics(
    env: IncidentEnvironment,
    scenario: Mapping[str, Any],
    *,
    total_format_retries: int,
) -> dict[str, float | int | bool]:
    trajectory = env.trajectory
    valid_calls = sum(step.valid for step in trajectory)
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
    diagnosis_correct = any(
        step.valid
        and step.tool == "submit_diagnosis"
        and step.arguments.get("root_cause") == scenario["hidden_state"]["root_cause"]
        for step in trajectory
    )
    breakdown = env.reward_breakdown()
    terminated = (
        bool(trajectory)
        and trajectory[-1].valid
        and trajectory[-1].tool == "submit_diagnosis"
    )
    return {
        "success": breakdown.success,
        "diagnosis_correct": diagnosis_correct,
        "terminated": terminated,
        "truncated": env.done and not terminated,
        "calls": len(trajectory),
        "valid_calls": valid_calls,
        "invalid_calls": len(trajectory) - valid_calls,
        "recoverable_errors": recoverable_errors,
        "recovered_errors": recovered_errors,
        "evidence_coverage": breakdown.evidence_coverage,
        "format_retries": total_format_retries,
    }


def _load_art(art_module: Any | None) -> Any:
    if art_module is not None:
        return art_module
    try:
        import art
    except ImportError as exc:
        raise RuntimeError(
            "OpenPipe ART is not installed; install the project with the 'art' extra."
        ) from exc
    return art

"""Run one Base LLM episode and export its complete, auditable trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.agents import BaseLLMAgent  # noqa: E402
from agenttool_rl.environment import IncidentEnvironment  # noqa: E402
from agenttool_rl.llm import OpenAICompatibleBackend  # noqa: E402
from agenttool_rl.validation import load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "generated" / "test.jsonl"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenarios = load_jsonl(args.dataset)
    if args.index < 0 or args.index >= len(scenarios):
        parser.error(f"--index must be between 0 and {len(scenarios) - 1}")

    backend = OpenAICompatibleBackend(base_url=args.base_url, model=args.model)
    agent = BaseLLMAgent(backend=backend, name=f"base-llm:{args.model}")
    env = IncidentEnvironment(max_steps=args.max_steps)
    initial_observation = env.reset(scenarios[args.index])
    agent.reset(initial_observation)
    actions: list[dict[str, object]] = []

    while not env.done:
        action = agent.act()
        result = env.step(action.tool, action.arguments)
        actions.append(
            {
                "tool": action.tool,
                "arguments": action.arguments,
                "model_response": action.raw_response,
                "environment_response": result.as_dict(),
            }
        )
        agent.observe(action, result)

    payload = {
        "agent": agent.name,
        "dataset": _portable_path(args.dataset),
        "scenario_index": args.index,
        "actions": actions,
        "episode": env.episode_summary(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"Trajectory: {args.output}")
    return 0


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())

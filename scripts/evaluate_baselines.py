"""Evaluate Random, Rule-based, and optional Base LLM agents with one protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.agents import BaseLLMAgent, RandomAgent, RuleBasedAgent  # noqa: E402
from agenttool_rl.evaluation import AgentFactory, evaluate_agents  # noqa: E402
from agenttool_rl.llm import OpenAICompatibleBackend  # noqa: E402
from agenttool_rl.validation import load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "generated" / "test.jsonl"
    )
    parser.add_argument(
        "--agents",
        default="random,rule",
        help="Comma-separated subset of: random, rule, base-llm.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "v0.3_baselines.json",
    )
    args = parser.parse_args()

    selected = [item.strip() for item in args.agents.split(",") if item.strip()]
    unsupported = sorted(set(selected) - {"random", "rule", "base-llm"})
    if unsupported:
        parser.error(f"unsupported agents: {unsupported}")
    if "base-llm" in selected and (not args.base_url or not args.model):
        parser.error("base-llm requires --base-url and --model")

    scenarios = load_jsonl(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        scenarios = scenarios[: args.limit]

    factories: dict[str, AgentFactory] = {}
    if "random" in selected:
        factories["random"] = lambda index: RandomAgent(seed=args.seed + index)
    if "rule" in selected:
        factories["rule-based"] = lambda index: RuleBasedAgent()
    if "base-llm" in selected:
        backend = OpenAICompatibleBackend(
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get(args.api_key_env),
        )
        factories[f"base-llm:{args.model}"] = lambda index: BaseLLMAgent(
            backend=backend, name=f"base-llm:{args.model}"
        )

    report = evaluate_agents(
        factories,
        scenarios,
        dataset=_portable_path(args.dataset),
        max_steps=args.max_steps,
    )
    report.write(args.output)
    print(report.to_markdown())
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.output.with_suffix('.md')}")
    return 0


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())

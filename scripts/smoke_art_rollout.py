"""Run one real OpenPipe ART trajectory against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.art_rollout import rollout_art  # noqa: E402
from agenttool_rl.validation import load_jsonl  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    try:
        import art
    except ImportError as exc:
        raise SystemExit("Install the ART extra first: pip install -e \".[art]\"") from exc

    scenarios = load_jsonl(args.dataset)
    if args.index < 0 or args.index >= len(scenarios):
        raise SystemExit(f"--index must be between 0 and {len(scenarios) - 1}")
    model = art.Model(
        name=args.model,
        project="agenttool-rl",
        inference_api_key=args.api_key,
        inference_base_url=args.base_url,
        inference_model_name=args.model,
        report_metrics=[],
    )
    trajectory = await rollout_art(
        model,
        scenarios[args.index],
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
    payload = trajectory.for_logging()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Trajectory: {args.output}")
    print(f"Reward: {trajectory.reward:.3f}")
    print(f"Metrics: {json.dumps(trajectory.metrics, sort_keys=True)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "generated" / "test.jsonl"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports" / "v0.4_art_rollout_smoke.json"
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

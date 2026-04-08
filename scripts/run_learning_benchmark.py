#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_benchmark.agents import create_agent
from learning_benchmark.reporting import save_learning_track_report, save_results_json
from learning_benchmark.runner import run_benchmark


ALL_SUITES = [
    "belief_update",
    "concept_rule_learning",
    "long_context_retention",
    "reward_skill_learning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learning benchmark")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "concept_rule_learning", "long_context_retention", "belief_update", "reward_skill_learning"],
        help="Suite to run",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per suite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent", type=str, default="dummy", choices=["dummy", "anthropic"], help="Agent adapter")
    parser.add_argument("--out", type=str, default="learning_benchmark/results/results.json", help="Output results JSON path")
    parser.add_argument(
        "--report-md",
        type=str,
        default="learning_benchmark/results/learning_track_report.md",
        help="Output markdown report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    suites = ALL_SUITES if args.suite == "all" else [args.suite]
    agent = create_agent(args.agent, seed=args.seed)

    result = run_benchmark(
        agent=agent,
        suites=suites,
        split=args.split,
        n_episodes=args.episodes,
        seed=args.seed,
        compute_ood_reference=True,
        ood_reference_episodes=min(50, args.episodes),
    )

    save_results_json(args.out, result)
    save_learning_track_report(args.report_md, result)

    payload = result.to_dict()
    summary = {
        "output": args.out,
        "report": args.report_md,
        "weighted_learning_score": payload["global_metrics"].get("WeightedLearningScore"),
        "core_belief_update_score": payload["global_metrics"].get("CoreBeliefUpdateScore"),
        "ood_gap": payload["global_metrics"].get("OODGap"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

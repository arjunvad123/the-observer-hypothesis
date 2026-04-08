from __future__ import annotations

from typing import List

from .agents import AgentAdapter
from .engine import BenchmarkRunner
from .types import BenchmarkResult, RunConfig


def run_benchmark(
    agent: AgentAdapter,
    suites: List[str],
    split: str = "test",
    n_episodes: int = 200,
    seed: int = 42,
    compute_ood_reference: bool = True,
    ood_reference_episodes: int = 50,
) -> BenchmarkResult:
    runner = BenchmarkRunner(agent=agent)
    config = RunConfig(
        suites=suites,
        split=split,
        n_episodes=n_episodes,
        seed=seed,
        compute_ood_reference=compute_ood_reference,
        ood_reference_episodes=ood_reference_episodes,
    )
    return runner.run_all(config)

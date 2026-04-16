from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EpisodePlan:
    """Deterministic plan metadata for one generated episode."""

    suite: str
    split: str
    episode_index: int
    episode_seed: int
    template_family: str
    complexity: int
    leakage_key: str


@dataclass
class EpisodeResult:
    """Raw per-episode outcome."""

    episode_id: str
    suite: str
    split: str
    template_family: str
    complexity: int
    turns: List[Dict[str, str]] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    """Suite-level aggregation."""

    suite_id: str
    split: str
    n_episodes: int
    episode_results: List[EpisodeResult]
    aggregate_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "split": self.split,
            "n_episodes": self.n_episodes,
            "episode_results": [e.to_dict() for e in self.episode_results],
            "aggregate_metrics": self.aggregate_metrics,
        }


@dataclass
class BenchmarkResult:
    """Top-level results payload matching the benchmark schema."""

    metadata: Dict[str, Any]
    suite_results: List[SuiteResult]
    global_metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "suite_results": [s.to_dict() for s in self.suite_results],
            "global_metrics": self.global_metrics,
            "diagnostics": self.diagnostics,
        }


@dataclass
class RunConfig:
    """Configuration for BenchmarkRunner.run_all."""

    suites: List[str]
    split: str = "test"
    n_episodes: int = 200
    seed: int = 42
    compute_ood_reference: bool = True
    ood_reference_episodes: int = 50


BENCHMARK_VERSION = "0.1.0"


def make_metadata(agent_id: str, seed: int, split: str, suites: List[str]) -> Dict[str, Any]:
    return {
        "benchmark_name": "Interactive Belief Updating Benchmark",
        "benchmark_version": BENCHMARK_VERSION,
        "core_suite": "belief_update",
        "suite_mode": "core_plus_extensions",
        "seed": seed,
        "split": split,
        "suites": suites,
        "agent_id": agent_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "learning_thesis": (
            "Measures whether an agent can incorporate new information during interaction "
            "and improve subsequent decisions under distribution shift."
        ),
    }

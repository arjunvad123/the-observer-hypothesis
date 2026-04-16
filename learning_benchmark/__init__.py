"""Learning benchmark package for Measuring AGI (Learning Track).

Core thesis:
    Can an agent incorporate new information during interaction and
    improve future decisions under shift?
"""

from .agents import AgentAdapter, DummyRandomAgent, AnthropicAgent, RuleBasedAdaptiveAgent, StubbornAgent, create_agent
from .engine import BenchmarkRunner
from .runner import run_benchmark
from .types import BenchmarkResult, RunConfig, SuiteResult, EpisodeResult

__all__ = [
    "AgentAdapter",
    "DummyRandomAgent",
    "AnthropicAgent",
    "RuleBasedAdaptiveAgent",
    "StubbornAgent",
    "create_agent",
    "BenchmarkRunner",
    "run_benchmark",
    "BenchmarkResult",
    "RunConfig",
    "SuiteResult",
    "EpisodeResult",
]

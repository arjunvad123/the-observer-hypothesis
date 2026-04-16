# Interactive Belief Updating Benchmark

A deterministic, text-only benchmark package for the Kaggle "Measuring AGI" Learning Track.

Core thesis:

> Can an agent incorporate new information during interaction and improve future decisions under shift?

## Core vs Extensions

- Core suite: `belief_update`
- Extension suites: `concept_rule_learning`, `long_context_retention`, `reward_skill_learning`

This structure keeps benchmark identity focused while still providing triangulation.

## Run

```bash
python scripts/run_learning_benchmark.py --suite all --split test --episodes 200 --seed 42 --agent dummy
```

Outputs:

- `learning_benchmark/results/results.json`
- `learning_benchmark/results/learning_track_report.md`

## Public Interfaces

- `AgentAdapter`
  - `start_episode(episode_meta) -> None`
  - `respond(turn_input) -> str`
  - `end_episode() -> None`
- `BenchmarkRunner`
  - `run_suite(suite_name, split, n_episodes, seed, agent=None) -> SuiteResult`
  - `run_all(config) -> BenchmarkResult`

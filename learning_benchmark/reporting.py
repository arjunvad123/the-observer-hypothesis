from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .types import BenchmarkResult


def save_results_json(path: str, result: BenchmarkResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)


def save_learning_track_report(path: str, result: BenchmarkResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = render_learning_track_report(result)
    out.write_text(content)


def render_learning_track_report(result: BenchmarkResult) -> str:
    data = result.to_dict()
    metadata = data["metadata"]
    global_metrics = data["global_metrics"]
    diagnostics = data["diagnostics"]
    suite_map = {s["suite_id"]: s for s in data["suite_results"]}

    failure_counts = diagnostics.get("failure_mode_counts", {})

    benchmark_name = metadata.get("benchmark_name", "Interactive Belief Updating Benchmark")
    capability = metadata.get("learning_thesis", "Interactive belief updating under shift")

    core = suite_map.get("belief_update", {})
    core_metrics = core.get("aggregate_metrics", {})

    extension_lines = []
    for suite_id in ["concept_rule_learning", "long_context_retention", "reward_skill_learning"]:
        suite = suite_map.get(suite_id)
        if not suite:
            continue
        extension_lines.append(
            f"- `{suite_id}`: {json.dumps(suite.get('aggregate_metrics', {}), sort_keys=True)}"
        )

    qualitative_examples = _collect_failure_examples(data)

    return f"""# Learning Track Submission Draft

## Benchmark Name
{benchmark_name}

## Intelligence Capability Being Tested
{capability}

## Problem Description
This benchmark measures whether an agent can incorporate new information during interaction and improve future decisions, rather than relying only on prior memorization. The centerpiece is `belief_update`, which introduces explicit contradictions and corrections, then checks whether the agent updates and stays updated.

## Task Structure
- **Core suite (`belief_update`)**: fact -> query -> delayed correction -> repeated post-correction queries.
- **Extension suite (`concept_rule_learning`)**: infer a transformation from few examples and generalize.
- **Extension suite (`long_context_retention`)**: retain facts across distractor-heavy dialogue.
- **Extension suite (`reward_skill_learning`)**: online bandit-style adaptation with reward feedback and distribution shift.

## Generalization Strategy
- Procedural episode generation with seeded reproducibility.
- Train/dev/test splits with disjoint template families (holdout lexical forms).
- Increasing complexity at test time (longer delays, stronger distractors, harder corrections, reward mapping shifts).
- Chance-normalized scores and anti-shortcut diagnostics.

## Evaluation Metrics
Deterministic definitions:
- `FewShotGeneralization`: correct concept query fraction.
- `RetentionFidelity`: correct recall fraction after distractors.
- `BeliefRevisionRate`: correct post-correction answers / post-correction queries.
- `PerseverationRate`: stale-value answers / post-correction queries.
- `LearningCurveAUC`: normalized trapezoidal AUC over turn-level correctness signals.
- `RewardRegret`: optimal expected reward minus observed reward.
- `AdaptationEfficiency`: turns until stable-correct behavior after correction or shift.
- `OODGap`: train-reference weighted score minus evaluation weighted score.

Current run metrics:
- `WeightedLearningScore`: {global_metrics.get('WeightedLearningScore')}
- `CoreBeliefUpdateScore`: {global_metrics.get('CoreBeliefUpdateScore')}
- `OODGap`: {global_metrics.get('OODGap')}
- `Belief update aggregate`: {json.dumps(core_metrics, sort_keys=True)}

Extension suite aggregates:
{chr(10).join(extension_lines) if extension_lines else '- (No extension suites in this run)'}

## Why This Measures AGI
Static accuracy can hide brittleness. This benchmark targets **online learning behavior**: whether the agent changes its internal policy when evidence changes, and whether the change persists. This isolates adaptive intelligence beyond pretraining recall.

## Expected Model Failures
Observed failure modes from this run:
- `drift`: {failure_counts.get('drift', 0)}
- `perseveration`: {failure_counts.get('perseveration', 0)}
- `shortcut_suspected`: {failure_counts.get('shortcut_suspected', 0)}
- `episode_failed`: {failure_counts.get('episode_failed', 0)}

## Reflection and Limitations
What we learned while designing this benchmark:
1. Breadth can dilute interpretability, so the benchmark is centered on one core mechanism (`belief_update`) with extensions as supporting evidence.
2. Deterministic grading improves reproducibility but constrains task expressiveness.
3. Text-only reward learning is feasible but easier to make artificial; it is best treated as an extension, not the core claim.

Limitations:
- Structured token-level grading may undercount partial reasoning quality.
- Real-world learning often requires richer action spaces than text responses.
- Some shortcuts may remain despite anti-shortcut heuristics.

Ablation placeholders:
- [ ] Remove correction delays and compare `BeliefRevisionRate`.
- [ ] Disable distractors and compare `RetentionFidelity`.
- [ ] Freeze reward mapping shifts and compare `RewardRegret`.

Qualitative failure examples:
{qualitative_examples}
"""


def _collect_failure_examples(result_dict: Dict[str, object], limit: int = 5) -> str:
    lines: List[str] = []
    count = 0
    for suite in result_dict.get("suite_results", []):
        suite_id = suite.get("suite_id")
        for ep in suite.get("episode_results", []):
            tags = ep.get("diagnostics", [])
            if not tags:
                continue
            lines.append(
                f"- `{suite_id}` / `{ep.get('episode_id')}` tags={tags} "
                f"metrics={json.dumps(ep.get('metrics', {}), sort_keys=True)}"
            )
            count += 1
            if count >= limit:
                return "\n".join(lines)
    if not lines:
        return "- No flagged failures in this run."
    return "\n".join(lines)

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .types import EpisodeResult, SuiteResult


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def chance_normalize(score: float, chance_level: float, max_score: float = 1.0) -> float:
    """Chance-normalized score in [-1, 1] with deterministic clamping."""
    if max_score <= chance_level:
        return 0.0
    normalized = (score - chance_level) / (max_score - chance_level)
    return clamp(normalized, -1.0, 1.0)


def learning_curve_auc(binary_or_unit_series: List[float]) -> float:
    """Normalized area under a [0,1]-valued curve.

    Formula: trapezoidal AUC divided by (n-1), returning 0 when n < 2.
    """
    if len(binary_or_unit_series) < 2:
        return float(binary_or_unit_series[0]) if binary_or_unit_series else 0.0
    total = 0.0
    for i in range(len(binary_or_unit_series) - 1):
        total += (binary_or_unit_series[i] + binary_or_unit_series[i + 1]) / 2.0
    return clamp(total / (len(binary_or_unit_series) - 1), 0.0, 1.0)


def aggregate_suite_metrics(episode_results: List[EpisodeResult]) -> Dict[str, float]:
    if not episode_results:
        return {}

    keys = sorted({k for ep in episode_results for k in ep.metrics.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = mean(ep.metrics.get(key, 0.0) for ep in episode_results)

    # Shared aggregates
    out["episode_count"] = float(len(episode_results))
    out["failure_rate"] = mean(1.0 if "episode_failed" in ep.diagnostics else 0.0 for ep in episode_results)
    return out


def build_global_metrics(
    suite_results: List[SuiteResult],
    train_reference: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    suite_map = {s.suite_id: s.aggregate_metrics for s in suite_results}

    few_shot = suite_map.get("concept_rule_learning", {}).get("FewShotGeneralization", 0.0)
    retention = suite_map.get("long_context_retention", {}).get("RetentionFidelity", 0.0)
    revision = suite_map.get("belief_update", {}).get("BeliefRevisionRate", 0.0)
    perseveration = suite_map.get("belief_update", {}).get("PerseverationRate", 1.0)
    reward_skill = suite_map.get("reward_skill_learning", {}).get("RewardSkillScore", 0.0)

    belief_combo = clamp((revision + (1.0 - perseveration)) / 2.0, 0.0, 1.0)
    weighted = (
        0.30 * few_shot
        + 0.25 * retention
        + 0.25 * belief_combo
        + 0.20 * reward_skill
    )

    core_belief_score = clamp(
        0.5 * revision
        + 0.3 * (1.0 - perseveration)
        + 0.2 * suite_map.get("belief_update", {}).get("AdaptationEfficiencyScore", 0.0),
        0.0,
        1.0,
    )

    generalization_gaps: Dict[str, float] = {}
    ood_gap = None
    if train_reference:
        ref_few_shot = train_reference.get("concept_rule_learning", {}).get("FewShotGeneralization", few_shot)
        ref_retention = train_reference.get("long_context_retention", {}).get("RetentionFidelity", retention)
        ref_revision = train_reference.get("belief_update", {}).get("BeliefRevisionRate", revision)
        ref_persev = train_reference.get("belief_update", {}).get("PerseverationRate", perseveration)
        ref_reward = train_reference.get("reward_skill_learning", {}).get("RewardSkillScore", reward_skill)

        ref_belief_combo = clamp((ref_revision + (1.0 - ref_persev)) / 2.0, 0.0, 1.0)
        ref_weighted = (
            0.30 * ref_few_shot
            + 0.25 * ref_retention
            + 0.25 * ref_belief_combo
            + 0.20 * ref_reward
        )
        ood_gap = float(ref_weighted - weighted)

        for suite_id in set(train_reference.keys()).intersection(suite_map.keys()):
            eval_primary = _primary_metric_for_suite(suite_id, suite_map[suite_id])
            ref_primary = _primary_metric_for_suite(suite_id, train_reference[suite_id])
            generalization_gaps[suite_id] = float(ref_primary - eval_primary)

    return {
        "WeightedLearningScore": float(weighted),
        "CoreBeliefUpdateScore": float(core_belief_score),
        "components": {
            "FewShotGeneralization": float(few_shot),
            "RetentionFidelity": float(retention),
            "BeliefRevisionAndLowPerseveration": float(belief_combo),
            "RewardSkillScore": float(reward_skill),
        },
        "OODGap": ood_gap,
        "generalization_gaps": generalization_gaps,
    }


def collect_diagnostics(suite_results: List[SuiteResult]) -> Dict[str, object]:
    counts: Dict[str, int] = {
        "drift": 0,
        "perseveration": 0,
        "shortcut_suspected": 0,
        "episode_failed": 0,
    }

    by_suite: Dict[str, Dict[str, int]] = {}
    for suite in suite_results:
        suite_counts = {k: 0 for k in counts.keys()}
        for ep in suite.episode_results:
            for tag in ep.diagnostics:
                if tag in suite_counts:
                    suite_counts[tag] += 1
                    counts[tag] += 1
        by_suite[suite.suite_id] = suite_counts

    return {
        "failure_mode_counts": counts,
        "by_suite": by_suite,
        "total_episodes": int(sum(s.n_episodes for s in suite_results)),
    }


def _primary_metric_for_suite(suite_id: str, metrics: Dict[str, float]) -> float:
    if suite_id == "concept_rule_learning":
        return metrics.get("FewShotGeneralization", 0.0)
    if suite_id == "long_context_retention":
        return metrics.get("RetentionFidelity", 0.0)
    if suite_id == "belief_update":
        return metrics.get("BeliefRevisionRate", 0.0)
    if suite_id == "reward_skill_learning":
        return metrics.get("RewardSkillScore", 0.0)
    return 0.0

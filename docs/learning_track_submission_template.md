# Learning Track Submission Template

Use this as the final narrative for Kaggle "Measuring AGI" Learning Track.

## Benchmark Name
{{benchmark_name}}

## Intelligence Capability Being Tested
{{learning_thesis}}

## Problem Description
This benchmark focuses on **interactive learning** rather than static recall:

> Can the model incorporate new information during interaction and improve future decisions under shift?

The benchmark is intentionally centered on `belief_update` as the core suite. Other suites are extensions for triangulation, not the main claim.

## Task Structure
- Core suite: `belief_update` (fact, contradiction, correction, repeated post-correction queries)
- Extension: `concept_rule_learning`
- Extension: `long_context_retention`
- Extension: `reward_skill_learning`

## Generalization Strategy
- Procedural generation per episode with fixed seed reproducibility
- Train/dev/test split separation via holdout template families
- Increased difficulty on test split (delay, distractor load, correction depth, mapping shift)
- Anti-shortcut controls: lexical variation, value randomization, chance-normalized scoring

## Evaluation Metrics
All metrics are deterministic and machine-computable:

- `FewShotGeneralization = correct_concept_queries / total_concept_queries`
- `RetentionFidelity = correct_recall_queries / total_recall_queries`
- `BeliefRevisionRate = correct_post_correction_answers / total_post_correction_queries`
- `PerseverationRate = stale_value_answers / total_post_correction_queries`
- `LearningCurveAUC = trapezoidal_auc(correctness_series) / (n - 1)`
- `RewardRegret = sum(optimal_mean_reward_t - observed_reward_t)`
- `AdaptationEfficiency = first_turn_where_behavior_is_stably_correct`
- `OODGap = train_reference_weighted_score - evaluation_weighted_score`

## Why This Measures AGI
{{why_agi_relevant}}

## Expected Model Failures
- Drift under long context
- Perseveration on invalidated beliefs
- Shortcut behavior (surface-form patterning without update)

Observed counts in current run:
- `drift`: {{drift_count}}
- `perseveration`: {{perseveration_count}}
- `shortcut_suspected`: {{shortcut_count}}
- `episode_failed`: {{episode_failed_count}}

## Reflection and Limitations
What we learned while designing this benchmark:
1. {{reflection_point_1}}
2. {{reflection_point_2}}
3. {{reflection_point_3}}

Limitations:
- {{limitation_1}}
- {{limitation_2}}
- {{limitation_3}}

## Ablation Placeholders
- [ ] Remove correction delay and compare `BeliefRevisionRate`
- [ ] Remove distractors and compare `RetentionFidelity`
- [ ] Remove reward mapping shifts and compare `RewardRegret`

## Qualitative Failure Examples
{{qualitative_failure_examples}}

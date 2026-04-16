from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

from .agents import AgentAdapter
from .scoring import (
    aggregate_suite_metrics,
    build_global_metrics,
    chance_normalize,
    collect_diagnostics,
    clamp,
    learning_curve_auc,
    mean,
)
from .splits import build_episode_plans
from .task_generation import generate_episode, reward_mean_for_trial
from .types import BenchmarkResult, EpisodePlan, EpisodeResult, RunConfig, SuiteResult, make_metadata


class BenchmarkRunner:
    """Runner for the learning benchmark."""

    def __init__(self, agent: AgentAdapter):
        self.agent = agent

    def run_suite(
        self,
        suite_name: str,
        split: str,
        n_episodes: int,
        seed: int,
        agent: Optional[AgentAdapter] = None,
    ) -> SuiteResult:
        agent = agent or self.agent
        plans = build_episode_plans(suite=suite_name, split=split, n_episodes=n_episodes, seed=seed)

        episode_results: List[EpisodeResult] = []
        for plan in plans:
            scenario = generate_episode(plan)
            if suite_name == "concept_rule_learning":
                episode_results.append(self._run_concept_episode(agent, plan, scenario))
            elif suite_name == "long_context_retention":
                episode_results.append(self._run_retention_episode(agent, plan, scenario))
            elif suite_name == "belief_update":
                episode_results.append(self._run_belief_episode(agent, plan, scenario))
            elif suite_name == "reward_skill_learning":
                episode_results.append(self._run_reward_episode(agent, plan, scenario))
            else:
                raise ValueError(f"Unknown suite: {suite_name}")

        aggregate = aggregate_suite_metrics(episode_results)
        return SuiteResult(
            suite_id=suite_name,
            split=split,
            n_episodes=n_episodes,
            episode_results=episode_results,
            aggregate_metrics=aggregate,
        )

    def run_all(self, config: RunConfig) -> BenchmarkResult:
        suite_results = [
            self.run_suite(
                suite_name=suite,
                split=config.split,
                n_episodes=config.n_episodes,
                seed=config.seed + idx * 1000,
            )
            for idx, suite in enumerate(config.suites)
        ]

        train_reference: Optional[Dict[str, Dict[str, float]]] = None
        if config.compute_ood_reference and config.split != "train":
            train_reference = {}
            ref_n = max(10, min(config.ood_reference_episodes, config.n_episodes))
            for idx, suite in enumerate(config.suites):
                train_suite = self.run_suite(
                    suite_name=suite,
                    split="train",
                    n_episodes=ref_n,
                    seed=config.seed + idx * 1000,
                )
                train_reference[suite] = train_suite.aggregate_metrics

        global_metrics = build_global_metrics(suite_results, train_reference=train_reference)
        diagnostics = collect_diagnostics(suite_results)
        if train_reference is not None:
            diagnostics["train_reference"] = train_reference

        metadata = make_metadata(
            agent_id=self.agent.agent_id,
            seed=config.seed,
            split=config.split,
            suites=config.suites,
        )
        metadata["episodes_per_suite"] = config.n_episodes

        return BenchmarkResult(
            metadata=metadata,
            suite_results=suite_results,
            global_metrics=global_metrics,
            diagnostics=diagnostics,
        )

    def _run_concept_episode(self, agent: AgentAdapter, plan: EpisodePlan, scenario: Dict[str, object]) -> EpisodeResult:
        turns: List[Dict[str, str]] = []
        responses: List[str] = []
        diagnostics: List[str] = []

        agent.start_episode({"suite": plan.suite, "episode_id": scenario["episode_id"], "split": plan.split})
        try:
            intro = str(scenario["intro"])
            examples = scenario["examples"]
            header = [intro, "Examples:"]
            for inp, out in examples:
                header.append(f"EXAMPLE {inp} -> {out}")
            header.append("Answer each query with one transformed token.")
            prompt = "\n".join(header)
            turns.append({"role": "user", "content": prompt})
            responses.append(agent.respond(prompt))

            answers = scenario["answers"]
            correct_flags: List[float] = []

            for query, gold in zip(scenario["queries"], answers):
                q = f"TRANSFORM_QUERY {query}"
                turns.append({"role": "user", "content": q})
                raw = agent.respond(q)
                parsed = _normalize_token(raw)
                responses.append(raw)
                is_correct = 1.0 if parsed == gold else 0.0
                correct_flags.append(is_correct)
                turns.append({"role": "grader", "content": f"gold={gold} pred={parsed} correct={int(is_correct)}"})
                if parsed == query and gold != query:
                    diagnostics.append("shortcut_suspected")

            few_shot = mean(correct_flags)
            metrics = {
                "FewShotGeneralization": few_shot,
                "FewShotGeneralizationChanceNorm": chance_normalize(few_shot, float(scenario.get("chance_level", 0.0))),
                "LearningCurveAUC": learning_curve_auc(correct_flags),
            }

            if few_shot < 0.1:
                diagnostics.append("episode_failed")

            return EpisodeResult(
                episode_id=str(scenario["episode_id"]),
                suite=plan.suite,
                split=plan.split,
                template_family=plan.template_family,
                complexity=plan.complexity,
                turns=turns,
                responses=responses,
                metrics=metrics,
                diagnostics=sorted(set(diagnostics)),
                metadata={"rule_name": scenario["rule_name"]},
            )
        finally:
            agent.end_episode()

    def _run_retention_episode(self, agent: AgentAdapter, plan: EpisodePlan, scenario: Dict[str, object]) -> EpisodeResult:
        turns: List[Dict[str, str]] = []
        responses: List[str] = []
        diagnostics: List[str] = []

        agent.start_episode({"suite": plan.suite, "episode_id": scenario["episode_id"], "split": plan.split})
        try:
            facts = scenario["facts"]
            intro = str(scenario["intro"])
            intro_lines = [intro, "Facts to store:"]
            for key, value in facts.items():
                intro_lines.append(f"FACT {key} = {value}")
            message = "\n".join(intro_lines)
            turns.append({"role": "user", "content": message})
            responses.append(agent.respond(message))

            for d in scenario["distractors"]:
                turns.append({"role": "user", "content": f"DISTRACTOR {d}"})
                responses.append(agent.respond(f"DISTRACTOR {d}"))

            query_keys = scenario["query_keys"]
            answers = scenario["answers"]
            fact_values = set(facts.values())

            correct_flags: List[float] = []
            drift_count = 0
            for key, gold in zip(query_keys, answers):
                q = f"QUERY {key}"
                turns.append({"role": "user", "content": q})
                raw = agent.respond(q)
                parsed = _normalize_token(raw)
                responses.append(raw)
                is_correct = 1.0 if parsed == gold else 0.0
                correct_flags.append(is_correct)
                if parsed not in fact_values:
                    drift_count += 1
                    diagnostics.append("drift")
                turns.append({"role": "grader", "content": f"gold={gold} pred={parsed} correct={int(is_correct)}"})

            fidelity = mean(correct_flags)
            metrics = {
                "RetentionFidelity": fidelity,
                "RetentionChanceNorm": chance_normalize(fidelity, float(scenario.get("chance_level", 0.0))),
                "LearningCurveAUC": learning_curve_auc(correct_flags),
                "DriftRate": float(drift_count / max(1, len(correct_flags))),
            }

            if fidelity < 0.2:
                diagnostics.append("episode_failed")

            return EpisodeResult(
                episode_id=str(scenario["episode_id"]),
                suite=plan.suite,
                split=plan.split,
                template_family=plan.template_family,
                complexity=plan.complexity,
                turns=turns,
                responses=responses,
                metrics=metrics,
                diagnostics=sorted(set(diagnostics)),
                metadata={"fact_count": len(facts), "distractor_count": len(scenario["distractors"])},
            )
        finally:
            agent.end_episode()

    def _run_belief_episode(self, agent: AgentAdapter, plan: EpisodePlan, scenario: Dict[str, object]) -> EpisodeResult:
        turns: List[Dict[str, str]] = []
        responses: List[str] = []
        diagnostics: List[str] = []

        agent.start_episode({"suite": plan.suite, "episode_id": scenario["episode_id"], "split": plan.split})
        try:
            intro = str(scenario["intro"])
            turns.append({"role": "user", "content": intro})
            responses.append(agent.respond(intro))

            stale_values = set()
            current_value = None
            opportunities = 0
            revised_correct = 0
            perseveration = 0
            post_correction_flags: List[float] = []
            corrections_seen = 0

            for item in scenario["timeline"]:
                kind = item["type"]
                if kind == "fact":
                    current_value = item["value"]
                    text = f"FACT {item['key']} = {item['value']}"
                    turns.append({"role": "user", "content": text})
                    responses.append(agent.respond(text))
                elif kind == "distractor":
                    text = f"DISTRACTOR {item['text']}"
                    turns.append({"role": "user", "content": text})
                    responses.append(agent.respond(text))
                elif kind == "correction":
                    if current_value is not None:
                        stale_values.add(current_value)
                    current_value = item["value"]
                    corrections_seen += 1
                    text = f"CORRECTION {item['key']} = {item['value']}"
                    turns.append({"role": "user", "content": text})
                    responses.append(agent.respond(text))
                elif kind == "query":
                    text = f"QUERY {item['key']}"
                    turns.append({"role": "user", "content": text})
                    raw = agent.respond(text)
                    parsed = _normalize_token(raw)
                    responses.append(raw)
                    expected = item["expected"]
                    is_correct = 1.0 if parsed == expected else 0.0

                    if corrections_seen > 0:
                        opportunities += 1
                        revised_correct += int(is_correct)
                        post_correction_flags.append(is_correct)
                        if parsed in stale_values:
                            perseveration += 1
                            diagnostics.append("perseveration")

                    turns.append(
                        {
                            "role": "grader",
                            "content": (
                                f"expected={expected} pred={parsed} correct={int(is_correct)} "
                                f"post_correction={int(corrections_seen > 0)}"
                            ),
                        }
                    )

            revision_rate = float(revised_correct / opportunities) if opportunities else 0.0
            perseveration_rate = float(perseveration / opportunities) if opportunities else 0.0
            adapt_turns = _turns_to_stable_correct(post_correction_flags)
            adapt_score = _adaptation_efficiency_score(adapt_turns, len(post_correction_flags))

            metrics = {
                "BeliefRevisionRate": revision_rate,
                "PerseverationRate": perseveration_rate,
                "AdaptationEfficiency": float(adapt_turns),
                "AdaptationEfficiencyScore": adapt_score,
                "LearningCurveAUC": learning_curve_auc(post_correction_flags),
            }

            if perseveration_rate > 0.0:
                diagnostics.append("perseveration")
            if revision_rate < 0.5:
                diagnostics.append("episode_failed")

            return EpisodeResult(
                episode_id=str(scenario["episode_id"]),
                suite=plan.suite,
                split=plan.split,
                template_family=plan.template_family,
                complexity=plan.complexity,
                turns=turns,
                responses=responses,
                metrics=metrics,
                diagnostics=sorted(set(diagnostics)),
                metadata={
                    "opportunities": opportunities,
                    "initial_value": scenario["initial_value"],
                    "final_value": scenario["final_value"],
                },
            )
        finally:
            agent.end_episode()

    def _run_reward_episode(self, agent: AgentAdapter, plan: EpisodePlan, scenario: Dict[str, object]) -> EpisodeResult:
        turns: List[Dict[str, str]] = []
        responses: List[str] = []
        diagnostics: List[str] = []

        rng = random.Random(plan.episode_seed + 777)
        actions: List[str] = list(scenario["actions"])

        agent.start_episode({"suite": plan.suite, "episode_id": scenario["episode_id"], "split": plan.split})
        try:
            intro = str(scenario["intro"])
            turns.append({"role": "user", "content": intro})
            responses.append(agent.respond(intro))

            total_reward = 0.0
            optimal_expected = 0.0
            optimal_flags: List[float] = []
            action_history: List[str] = []

            n_trials = int(scenario["n_trials"])
            for t in range(n_trials):
                prompt = f"BANDIT_TRIAL {t + 1}/{n_trials}. Options: [{', '.join(actions)}]"
                turns.append({"role": "user", "content": prompt})
                raw = agent.respond(prompt)
                pred_action = _extract_action(raw, actions)
                responses.append(raw)
                action_history.append(pred_action)

                means = {a: reward_mean_for_trial(scenario, t, a) for a in actions}
                best_action = max(means, key=means.get)
                optimal_flags.append(1.0 if pred_action == best_action else 0.0)

                observed = clamp(rng.gauss(means[pred_action], 0.08), 0.0, 1.0)
                total_reward += observed
                optimal_expected += means[best_action]

                feedback = f"REWARD {pred_action} = {observed:.3f}"
                turns.append({"role": "user", "content": feedback})
                responses.append(agent.respond(feedback))

            regret = max(0.0, optimal_expected - total_reward)
            regret_ratio = regret / max(optimal_expected, 1e-9)
            reward_skill_score = clamp(1.0 - regret_ratio, 0.0, 1.0)

            shift_trial = scenario.get("shift_trial")
            if shift_trial is None:
                adapt_turns = 1.0
                adapt_score = 1.0
            else:
                post_shift_flags = optimal_flags[int(shift_trial) :]
                adapt_turns = float(_turns_to_stable_correct(post_shift_flags, min_consecutive=3))
                adapt_score = _adaptation_efficiency_score(int(adapt_turns), len(post_shift_flags))

            if len(set(action_history)) == 1:
                diagnostics.append("shortcut_suspected")
            if reward_skill_score < 0.2:
                diagnostics.append("episode_failed")

            metrics = {
                "RewardRegret": float(regret),
                "RewardSkillScore": float(reward_skill_score),
                "LearningCurveAUC": learning_curve_auc(optimal_flags),
                "AdaptationEfficiency": float(adapt_turns),
                "AdaptationEfficiencyScore": float(adapt_score),
            }

            return EpisodeResult(
                episode_id=str(scenario["episode_id"]),
                suite=plan.suite,
                split=plan.split,
                template_family=plan.template_family,
                complexity=plan.complexity,
                turns=turns,
                responses=responses,
                metrics=metrics,
                diagnostics=sorted(set(diagnostics)),
                metadata={
                    "n_trials": n_trials,
                    "optimal_expected": float(optimal_expected),
                    "total_reward": float(total_reward),
                    "shift_trial": int(shift_trial) if shift_trial is not None else None,
                },
            )
        finally:
            agent.end_episode()


def _normalize_token(text: str) -> str:
    text = text.strip().lower()
    match = re.search(r"([a-z0-9_\-]+)", text)
    return match.group(1) if match else ""


def _extract_action(text: str, valid_actions: List[str]) -> str:
    stripped = text.strip().upper()
    for action in valid_actions:
        if action in stripped.split():
            return action
    token_match = re.search(r"([A-Z])", stripped)
    if token_match and token_match.group(1) in valid_actions:
        return token_match.group(1)
    return valid_actions[0]


def _turns_to_stable_correct(flags: List[float], min_consecutive: int = 1) -> int:
    if not flags:
        return 1
    n = len(flags)
    for i in range(n):
        if min_consecutive == 1:
            if all(v >= 1.0 for v in flags[i:]):
                return i + 1
        else:
            if i + min_consecutive <= n and all(v >= 1.0 for v in flags[i : i + min_consecutive]):
                return i + 1
    return n + 1


def _adaptation_efficiency_score(turns_to_stable: int, sequence_len: int) -> float:
    if sequence_len <= 0:
        return 0.0
    if turns_to_stable > sequence_len:
        return 0.0
    score = 1.0 - (turns_to_stable - 1) / max(1, sequence_len)
    return clamp(score, 0.0, 1.0)

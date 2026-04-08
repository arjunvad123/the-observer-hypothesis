from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from learning_benchmark.agents import DummyRandomAgent, RuleBasedAdaptiveAgent, StubbornAgent
from learning_benchmark.engine import BenchmarkRunner
from learning_benchmark.reporting import render_learning_track_report, save_results_json
from learning_benchmark.scoring import chance_normalize, learning_curve_auc
from learning_benchmark.splits import build_and_validate_split_plans, build_episode_plans
from learning_benchmark.task_generation import generate_episode
from learning_benchmark.types import RunConfig


class LearningBenchmarkTests(unittest.TestCase):
    def test_generator_determinism(self) -> None:
        plans1 = build_episode_plans("belief_update", "test", n_episodes=5, seed=123)
        plans2 = build_episode_plans("belief_update", "test", n_episodes=5, seed=123)
        self.assertEqual([p.__dict__ for p in plans1], [p.__dict__ for p in plans2])

        ep1 = generate_episode(plans1[0])
        ep2 = generate_episode(plans2[0])
        self.assertEqual(ep1, ep2)

    def test_split_leakage_check(self) -> None:
        report = build_and_validate_split_plans("concept_rule_learning", n_per_split=20, seed=99)
        self.assertTrue(report["ok"])
        self.assertTrue(all(v == 0 for v in report["overlaps"].values()))

    def test_metric_invariants(self) -> None:
        auc = learning_curve_auc([0.0, 0.0, 1.0, 1.0])
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

        norm = chance_normalize(0.75, 0.25)
        self.assertGreaterEqual(norm, -1.0)
        self.assertLessEqual(norm, 1.0)

    def test_integration_outputs_and_diagnostics(self) -> None:
        runner = BenchmarkRunner(agent=DummyRandomAgent(seed=7))
        cfg = RunConfig(
            suites=["belief_update", "concept_rule_learning", "long_context_retention", "reward_skill_learning"],
            split="test",
            n_episodes=12,
            seed=7,
            compute_ood_reference=True,
            ood_reference_episodes=10,
        )
        result = runner.run_all(cfg)
        payload = result.to_dict()

        self.assertIn("metadata", payload)
        self.assertIn("suite_results", payload)
        self.assertIn("global_metrics", payload)
        self.assertIn("diagnostics", payload)
        self.assertGreaterEqual(len(payload["suite_results"]), 4)
        self.assertIn("by_suite", payload["diagnostics"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "results.json"
            save_results_json(str(out), result)
            self.assertTrue(out.exists())
            parsed = json.loads(out.read_text())
            self.assertIn("global_metrics", parsed)

            report = render_learning_track_report(result)
            self.assertIn("Benchmark Name", report)
            self.assertIn("Reflection and Limitations", report)

    def test_adaptive_baseline_beats_stubborn_on_belief_update(self) -> None:
        cfg = RunConfig(
            suites=["belief_update"],
            split="test",
            n_episodes=20,
            seed=21,
            compute_ood_reference=False,
        )

        adaptive = BenchmarkRunner(agent=RuleBasedAdaptiveAgent(seed=21)).run_all(cfg)
        stubborn = BenchmarkRunner(agent=StubbornAgent(seed=21)).run_all(cfg)

        adaptive_score = adaptive.to_dict()["suite_results"][0]["aggregate_metrics"]["BeliefRevisionRate"]
        stubborn_score = stubborn.to_dict()["suite_results"][0]["aggregate_metrics"]["BeliefRevisionRate"]
        self.assertGreater(adaptive_score, stubborn_score)

    def test_non_adaptive_train_vs_test_gap(self) -> None:
        runner = BenchmarkRunner(agent=StubbornAgent(seed=33))

        train_cfg = RunConfig(
            suites=["belief_update", "concept_rule_learning", "long_context_retention", "reward_skill_learning"],
            split="train",
            n_episodes=25,
            seed=33,
            compute_ood_reference=False,
        )
        test_cfg = RunConfig(
            suites=["belief_update", "concept_rule_learning", "long_context_retention", "reward_skill_learning"],
            split="test",
            n_episodes=25,
            seed=33,
            compute_ood_reference=False,
        )

        train_result = runner.run_all(train_cfg).to_dict()
        test_result = runner.run_all(test_cfg).to_dict()

        train_score = train_result["global_metrics"]["WeightedLearningScore"]
        test_score = test_result["global_metrics"]["WeightedLearningScore"]
        self.assertGreaterEqual(train_score, test_score)


if __name__ == "__main__":
    unittest.main()

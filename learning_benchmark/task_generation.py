from __future__ import annotations

import random
import string
from typing import Callable, Dict, List, Tuple

from .types import EpisodePlan


def generate_episode(plan: EpisodePlan) -> Dict[str, object]:
    rng = random.Random(plan.episode_seed)
    if plan.suite == "concept_rule_learning":
        return generate_concept_rule_episode(plan, rng)
    if plan.suite == "long_context_retention":
        return generate_retention_episode(plan, rng)
    if plan.suite == "belief_update":
        return generate_belief_update_episode(plan, rng)
    if plan.suite == "reward_skill_learning":
        return generate_reward_episode(plan, rng)
    raise ValueError(f"Unknown suite: {plan.suite}")


def generate_concept_rule_episode(plan: EpisodePlan, rng: random.Random) -> Dict[str, object]:
    templates = {
        "c0": "Infer the hidden mapping from examples, then answer each TRANSFORM_QUERY.",
        "c1": "Learn the secret rule from EXAMPLE pairs. Reply with only the transformed token.",
        "c2": "Generalize a rule from few demonstrations. Do not explain; output transformed tokens.",
        "c3": "Use provided examples to learn a symbolic transformation and apply it to new strings.",
        "c4": "Few-shot concept learning task: infer function from EXAMPLE lines and answer queries.",
    }

    rule_name, rule_fn = _sample_rule(plan.complexity, rng)
    suffix = _random_token(rng, min_len=2, max_len=3)

    n_examples = 3 + (1 if plan.complexity >= 3 else 0)
    n_queries = 4 + (1 if plan.complexity >= 4 else 0)

    examples = []
    for _ in range(n_examples):
        inp = _random_word(rng, 4, 6)
        out = rule_fn(inp, suffix)
        examples.append((inp, out))

    queries = []
    answers = []
    for _ in range(n_queries):
        q = _random_word(rng, 4, 7)
        queries.append(q)
        answers.append(rule_fn(q, suffix))

    return {
        "suite": plan.suite,
        "episode_id": f"{plan.suite}_{plan.split}_{plan.episode_index}",
        "template_family": plan.template_family,
        "complexity": plan.complexity,
        "intro": templates[plan.template_family],
        "rule_name": rule_name,
        "rule_params": {"suffix": suffix},
        "examples": examples,
        "queries": queries,
        "answers": answers,
        "chance_level": 0.0,
    }


def generate_retention_episode(plan: EpisodePlan, rng: random.Random) -> Dict[str, object]:
    templates = {
        "r0": "Memorize FACT lines, handle distractors, then answer QUERY lines exactly.",
        "r1": "Track key-value memory through a long chat and answer final recalls.",
        "r2": "Store facts despite irrelevant turns. Output only fact values when queried.",
        "r3": "Long-context retention test: keep FACT bindings stable through noise.",
        "r4": "Retain earlier facts and avoid drift under distractor-heavy interaction.",
    }

    fact_count = 4 + plan.complexity
    distractor_count = 3 + 2 * plan.complexity
    query_count = min(fact_count, 3 + plan.complexity)

    facts: Dict[str, str] = {}
    for _ in range(fact_count):
        key = f"KEY_{_random_token(rng, 3, 5).upper()}"
        value = _random_token(rng, 4, 7)
        facts[key] = value

    all_keys = list(facts.keys())
    rng.shuffle(all_keys)
    query_keys = all_keys[:query_count]

    distractors = [_make_distractor(rng) for _ in range(distractor_count)]

    return {
        "suite": plan.suite,
        "episode_id": f"{plan.suite}_{plan.split}_{plan.episode_index}",
        "template_family": plan.template_family,
        "complexity": plan.complexity,
        "intro": templates[plan.template_family],
        "facts": facts,
        "distractors": distractors,
        "query_keys": query_keys,
        "answers": [facts[k] for k in query_keys],
        "chance_level": 1.0 / max(2, fact_count),
    }


def generate_belief_update_episode(plan: EpisodePlan, rng: random.Random) -> Dict[str, object]:
    templates = {
        "b0": "Track FACT and CORRECTION updates. Answer QUERY with latest valid value.",
        "b1": "Belief update task: always prefer corrected information over earlier claims.",
        "b2": "When corrections arrive, update memory and avoid perseverating on old values.",
        "b3": "Core learning test: incorporate corrective feedback and stay consistent afterward.",
        "b4": "Update beliefs online under contradiction and delay; answer with current truth.",
    }

    key = f"BELIEF_{_random_token(rng, 3, 5).upper()}"
    initial_value = _random_token(rng, 4, 7)
    correction_1 = _random_token(rng, 4, 7)
    correction_2 = _random_token(rng, 4, 7)

    while correction_1 == initial_value:
        correction_1 = _random_token(rng, 4, 7)
    while correction_2 in (initial_value, correction_1):
        correction_2 = _random_token(rng, 4, 7)

    post_queries = 2 + plan.complexity

    timeline: List[Dict[str, str]] = [
        {"type": "fact", "key": key, "value": initial_value},
        {"type": "query", "key": key, "expected": initial_value},
    ]

    delay_turns = 1 + (1 if plan.complexity >= 3 else 0)
    for _ in range(delay_turns):
        timeline.append({"type": "distractor", "text": _make_distractor(rng)})

    timeline.append({"type": "correction", "key": key, "value": correction_1})

    for _ in range(post_queries):
        timeline.append({"type": "query", "key": key, "expected": correction_1})

    if plan.complexity >= 4:
        timeline.append({"type": "correction", "key": key, "value": correction_2})
        timeline.append({"type": "query", "key": key, "expected": correction_2})

    return {
        "suite": plan.suite,
        "episode_id": f"{plan.suite}_{plan.split}_{plan.episode_index}",
        "template_family": plan.template_family,
        "complexity": plan.complexity,
        "intro": templates[plan.template_family],
        "timeline": timeline,
        "initial_value": initial_value,
        "final_value": correction_2 if plan.complexity >= 4 else correction_1,
        "chance_level": 0.0,
    }


def generate_reward_episode(plan: EpisodePlan, rng: random.Random) -> Dict[str, object]:
    templates = {
        "s0": "Choose options trial-by-trial. Learn from REWARD feedback to maximize return.",
        "s1": "Bandit learning task: adapt choices using observed rewards.",
        "s2": "Incremental skill learning: improve action selection based on outcomes.",
        "s3": "Reward adaptation under shift: update policy when payoff structure changes.",
        "s4": "Online reinforcement test with changed reward mapping in harder conditions.",
    }

    action_count = 3 if plan.complexity <= 2 else 4
    n_trials = 12 + 3 * plan.complexity
    actions = [_random_action_token(rng) for _ in range(action_count)]

    phase1 = {a: rng.uniform(0.2, 0.8) for a in actions}
    best_phase1 = max(phase1, key=phase1.get)

    shift_trial = None
    phase2 = None
    if plan.split == "test" or plan.complexity >= 3:
        shift_trial = n_trials // 2
        phase2 = phase1.copy()
        boost_target = rng.choice([a for a in actions if a != best_phase1])
        phase2[boost_target] = min(0.95, phase1[boost_target] + 0.2)
        phase2[best_phase1] = max(0.05, phase1[best_phase1] - 0.2)

    return {
        "suite": plan.suite,
        "episode_id": f"{plan.suite}_{plan.split}_{plan.episode_index}",
        "template_family": plan.template_family,
        "complexity": plan.complexity,
        "intro": templates[plan.template_family],
        "actions": actions,
        "n_trials": n_trials,
        "phase1_means": phase1,
        "phase2_means": phase2,
        "shift_trial": shift_trial,
        "chance_level": 1.0 / len(actions),
    }


def reward_mean_for_trial(scenario: Dict[str, object], trial_idx: int, action: str) -> float:
    shift_trial = scenario.get("shift_trial")
    if shift_trial is not None and scenario.get("phase2_means") is not None and trial_idx >= shift_trial:
        return float(scenario["phase2_means"].get(action, 0.0))
    return float(scenario["phase1_means"].get(action, 0.0))


def _sample_rule(complexity: int, rng: random.Random) -> Tuple[str, Callable[[str, str], str]]:
    rules: List[Tuple[str, Callable[[str, str], str]]] = [
        ("identity", lambda s, suffix: s),
        ("suffix", lambda s, suffix: s + suffix),
        ("reverse", lambda s, suffix: s[::-1]),
        ("reverse_suffix", lambda s, suffix: s[::-1] + suffix),
        ("vowel_shift", lambda s, suffix: s.translate(str.maketrans({"a": "e", "e": "i", "i": "o", "o": "u", "u": "a"}))),
    ]

    if complexity <= 1:
        pool = rules[:2]
    elif complexity == 2:
        pool = rules[:4]
    else:
        pool = rules

    return rng.choice(pool)


def _random_word(rng: random.Random, min_len: int, max_len: int) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _random_token(rng: random.Random, min_len: int = 3, max_len: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _make_distractor(rng: random.Random) -> str:
    verbs = ["summarize", "classify", "rewrite", "tag", "count"]
    nouns = ["signal", "snippet", "note", "sentence", "token"]
    return f"Please {rng.choice(verbs)} this {rng.choice(nouns)}: {_random_word(rng, 5, 9)}"


def _random_action_token(rng: random.Random) -> str:
    return rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"))

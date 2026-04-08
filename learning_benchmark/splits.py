from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .types import EpisodePlan


SUITES = [
    "belief_update",
    "concept_rule_learning",
    "long_context_retention",
    "reward_skill_learning",
]


_SPLIT_CONFIG: Dict[str, Dict[str, Dict[str, object]]] = {
    "belief_update": {
        "train": {"template_families": ["b0", "b1"], "complexity": (1, 2)},
        "dev": {"template_families": ["b2"], "complexity": (2, 3)},
        "test": {"template_families": ["b3", "b4"], "complexity": (3, 4)},
    },
    "concept_rule_learning": {
        "train": {"template_families": ["c0", "c1"], "complexity": (1, 2)},
        "dev": {"template_families": ["c2"], "complexity": (2, 3)},
        "test": {"template_families": ["c3", "c4"], "complexity": (3, 4)},
    },
    "long_context_retention": {
        "train": {"template_families": ["r0", "r1"], "complexity": (1, 2)},
        "dev": {"template_families": ["r2"], "complexity": (2, 3)},
        "test": {"template_families": ["r3", "r4"], "complexity": (3, 4)},
    },
    "reward_skill_learning": {
        "train": {"template_families": ["s0", "s1"], "complexity": (1, 2)},
        "dev": {"template_families": ["s2"], "complexity": (2, 3)},
        "test": {"template_families": ["s3", "s4"], "complexity": (3, 4)},
    },
}


def get_split_spec(suite: str, split: str) -> Dict[str, object]:
    if suite not in _SPLIT_CONFIG:
        raise ValueError(f"Unknown suite: {suite}")
    if split not in _SPLIT_CONFIG[suite]:
        raise ValueError(f"Unknown split for {suite}: {split}")
    return _SPLIT_CONFIG[suite][split]


def build_episode_plans(suite: str, split: str, n_episodes: int, seed: int) -> List[EpisodePlan]:
    spec = get_split_spec(suite, split)
    template_families = list(spec["template_families"])
    c_min, c_max = spec["complexity"]

    rng = random.Random(_stable_mix_seed(seed=seed, suite=suite, split=split))
    plans: List[EpisodePlan] = []

    for i in range(n_episodes):
        template_family = rng.choice(template_families)
        complexity = rng.randint(c_min, c_max)
        episode_seed = seed * 100000 + i * 37 + rng.randint(1, 999999)
        leakage_key = f"{suite}|{template_family}|{complexity}"
        plans.append(
            EpisodePlan(
                suite=suite,
                split=split,
                episode_index=i,
                episode_seed=episode_seed,
                template_family=template_family,
                complexity=complexity,
                leakage_key=leakage_key,
            )
        )

    return plans


def validate_no_leakage(plans_by_split: Dict[str, List[EpisodePlan]]) -> Tuple[bool, Dict[str, int]]:
    """Validate train/dev/test disjointness by leakage_key.

    Disjoint template-family assignment should keep intersections at 0.
    """
    keys_by_split = {
        split: {p.leakage_key for p in plans}
        for split, plans in plans_by_split.items()
    }

    overlaps: Dict[str, int] = {}
    splits = list(keys_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap = len(keys_by_split[a].intersection(keys_by_split[b]))
            overlaps[f"{a}_{b}"] = overlap

    ok = all(v == 0 for v in overlaps.values())
    return ok, overlaps


def build_and_validate_split_plans(suite: str, n_per_split: int, seed: int) -> Dict[str, object]:
    plans = {
        split: build_episode_plans(suite=suite, split=split, n_episodes=n_per_split, seed=seed)
        for split in ("train", "dev", "test")
    }
    ok, overlaps = validate_no_leakage(plans)
    return {"ok": ok, "overlaps": overlaps, "plans": plans}


def _stable_mix_seed(seed: int, suite: str, split: str) -> int:
    text = f"{suite}:{split}"
    mix = 0
    for i, ch in enumerate(text):
        mix += (i + 1) * ord(ch)
    return seed * 131 + mix

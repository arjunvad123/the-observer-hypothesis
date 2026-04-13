"""
=== KAGGLE NOTEBOOK: Belief Update with Temporal Control ===

Paste this into a Kaggle Benchmarks notebook at:
  https://www.kaggle.com/benchmarks/tasks/new

This is a single self-contained file. No imports from local packages.
Everything the notebook needs is defined inline.

After running, execute in the final cell:
  %choose belief_update
"""

# ============================================================
# Cell 1: Imports and Setup
# ============================================================

import kaggle_benchmarks as kbench
import pandas as pd
import random
import re
import string
from typing import Dict, List


# ============================================================
# Cell 2: Episode Generator (condition-aware)
# ============================================================

SYSTEM_INSTRUCTIONS = (
    "You are playing a memory tracking game. "
    "I will give you FACT statements that assign values to keys. "
    "I will sometimes give you CORRECTION statements that update a key to a new value. "
    "When I say QUERY followed by a key, respond with ONLY the current value for that key — "
    "nothing else, no explanation, just the value. "
    "If a CORRECTION has been given for a key, use the corrected value, not the original. "
    "Ignore any DISTRACTOR messages — they are irrelevant to the game."
)

TEMPLATES = {
    "b0": SYSTEM_INSTRUCTIONS,
    "b1": SYSTEM_INSTRUCTIONS + " Always prefer the most recent CORRECTION over earlier values.",
    "b2": SYSTEM_INSTRUCTIONS + " When corrections arrive, update your memory immediately.",
    "b3": SYSTEM_INSTRUCTIONS + " This tests whether you can incorporate corrective feedback.",
    "b4": SYSTEM_INSTRUCTIONS + " Values may be corrected multiple times. Always use the latest.",
}


def _random_token(rng, min_len=3, max_len=6):
    alphabet = string.ascii_lowercase + string.digits
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _random_word(rng, min_len, max_len):
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _make_distractor(rng):
    verbs = ["summarize", "classify", "rewrite", "tag", "count"]
    nouns = ["signal", "snippet", "note", "sentence", "token"]
    return f"Please {rng.choice(verbs)} this {rng.choice(nouns)}: {_random_word(rng, 5, 9)}"


def _build_canonical_timeline(key, initial_value, correction_1, correction_2,
                               complexity, has_second_correction, rng):
    timeline = []

    # Phase 1: initial fact + pre-correction query
    timeline.append({
        "type": "fact", "key": key, "value": initial_value,
        "post_correction": False,
    })
    timeline.append({
        "type": "query", "key": key, "expected": initial_value,
        "stale_value": None, "post_correction": False,
    })

    # Phase 2: distractors
    delay_turns = 1 + (1 if complexity >= 3 else 0)
    for _ in range(delay_turns):
        timeline.append({"type": "distractor", "text": _make_distractor(rng)})

    # Phase 3: first correction + post-correction queries
    timeline.append({"type": "correction", "key": key, "value": correction_1})
    post_queries = 2 + complexity
    for _ in range(post_queries):
        timeline.append({
            "type": "query", "key": key, "expected": correction_1,
            "stale_value": initial_value, "post_correction": True,
        })

    # Phase 4: second correction (complexity >= 4)
    if has_second_correction:
        timeline.append({"type": "correction", "key": key, "value": correction_2})
        timeline.append({
            "type": "query", "key": key, "expected": correction_2,
            "stale_value": correction_1, "post_correction": True,
        })

    return timeline


def _shuffle_timeline(canonical, seed):
    informational = [t for t in canonical if t["type"] in ("fact", "correction", "distractor")]
    queries = [t for t in canonical if t["type"] == "query"]
    shuffle_rng = random.Random(seed + 7919)
    shuffled_info = list(informational)
    shuffle_rng.shuffle(shuffled_info)
    return shuffled_info + queries


def generate_belief_episode(episode_seed, complexity, template_family,
                             condition="canonical"):
    rng = random.Random(episode_seed)

    key = f"BELIEF_{_random_token(rng, 3, 5).upper()}"
    initial_value = _random_token(rng, 4, 7)
    correction_1 = _random_token(rng, 4, 7)
    correction_2 = _random_token(rng, 4, 7)

    while correction_1 == initial_value:
        correction_1 = _random_token(rng, 4, 7)
    while correction_2 in (initial_value, correction_1):
        correction_2 = _random_token(rng, 4, 7)

    has_second_correction = complexity >= 4
    final_value = correction_2 if has_second_correction else correction_1

    canonical = _build_canonical_timeline(
        key=key, initial_value=initial_value,
        correction_1=correction_1, correction_2=correction_2,
        complexity=complexity, has_second_correction=has_second_correction,
        rng=rng,
    )

    timeline = canonical if condition == "canonical" else _shuffle_timeline(canonical, episode_seed)

    return {
        "intro": TEMPLATES[template_family],
        "timeline": timeline,
        "initial_value": initial_value,
        "final_value": final_value,
        "condition": condition,
        "key": key,
        "complexity": complexity,
    }


# ============================================================
# Cell 3: Response Parsing
# ============================================================

def _normalize_response(text):
    text = text.strip().lower()
    match = re.search(r"([a-z0-9_\-]+)", text)
    return match.group(1) if match else ""


# ============================================================
# Cell 4: Task Definition
# ============================================================

@kbench.task(name="belief_update")
def belief_update(
    llm,
    episode_seed: int,
    complexity: int,
    template_family: str,
    condition: str = "canonical",
) -> float:
    """Belief update task with temporal control condition.

    Measures whether the model incorporates corrective feedback during
    multi-turn interaction. The `condition` parameter determines the
    episode variant:
      - "canonical": coherent temporal order (FACT -> CORRECTION -> QUERY)
      - "shuffled_control": same tokens, shuffled informational turns

    Returns BeliefRevisionRate (post-correction accuracy) as float [0, 1].
    """
    scenario = generate_belief_episode(
        episode_seed=episode_seed,
        complexity=complexity,
        template_family=template_family,
        condition=condition,
    )

    post_total = 0
    post_correct = 0
    post_stale = 0

    # Intro
    llm.prompt(scenario["intro"])

    # Walk timeline
    for item in scenario["timeline"]:
        t = item["type"]

        if t == "fact":
            llm.prompt(f"FACT {item['key']} = {item['value']}")

        elif t == "distractor":
            llm.prompt(item["text"])

        elif t == "correction":
            llm.prompt(f"CORRECTION {item['key']} = {item['value']}")

        elif t == "query":
            response = llm.prompt(f"QUERY {item['key']} — respond with only the value, nothing else.")
            parsed = _normalize_response(response)
            expected = item["expected"]
            is_post = item.get("post_correction", False)

            if is_post:
                post_total += 1
                if parsed == expected:
                    post_correct += 1
                if parsed == item.get("stale_value"):
                    post_stale += 1

    if post_total == 0:
        return 0.0

    return post_correct / post_total


# ============================================================
# Cell 5: Build Evaluation Dataset
# ============================================================

def build_dataset(n_per_condition=25, seed=42):
    rng = random.Random(seed)
    templates = list(TEMPLATES.keys())
    complexities = (1, 2, 3, 4, 5)
    rows = []

    for i in range(n_per_condition):
        ep_seed = rng.randint(0, 2**31 - 1)
        cx = complexities[i % len(complexities)]
        tf = templates[i % len(templates)]

        rows.append({"episode_seed": ep_seed, "complexity": cx,
                      "template_family": tf, "condition": "canonical"})
        rows.append({"episode_seed": ep_seed, "complexity": cx,
                      "template_family": tf, "condition": "shuffled_control"})

    return pd.DataFrame(rows)


df = build_dataset(n_per_condition=25)
print(f"Dataset: {len(df)} rows ({len(df)//2} canonical + {len(df)//2} shuffled)")
print(df.head(6))


# ============================================================
# Cell 6+7: Run Evaluation with Retries + Analysis
# ============================================================

import time

df = build_dataset(n_per_condition=25)
all_results = []
errors = 0

for idx, row in df.iterrows():
    cond = row["condition"]
    cx = row["complexity"]
    ep = row["episode_seed"]
    tf = row["template_family"]
    success = False
    for attempt in range(3):
        try:
            score = belief_update.run(kbench.llm, episode_seed=ep, complexity=cx, template_family=tf, condition=cond)
            result_val = score.result if hasattr(score, 'result') else score
            all_results.append({"episode_seed": ep, "complexity": cx, "condition": cond, "score": result_val})
            print("[" + cond.rjust(16) + "] cx=" + str(cx) + " -> " + str(result_val))
            success = True
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                all_results.append({"episode_seed": ep, "complexity": cx, "condition": cond, "score": None})
                errors += 1
                print("[" + cond.rjust(16) + "] cx=" + str(cx) + " FAILED")

results_df = pd.DataFrame(all_results)
results_df = results_df.dropna(subset=["score"])
canonical = results_df[results_df["condition"] == "canonical"]["score"]
shuffled = results_df[results_df["condition"] == "shuffled_control"]["score"]

print("")
print("=" * 60)
print("TEMPORAL SENSITIVITY ANALYSIS (n=25 per condition)")
print("=" * 60)
print("Canonical mean:            " + str(round(canonical.mean(), 4)))
print("Shuffled control mean:     " + str(round(shuffled.mean(), 4)))
print("Temporal sensitivity gap:  " + str(round(canonical.mean() - shuffled.mean(), 4)))
print("Episodes completed:        " + str(len(results_df)) + " / " + str(len(df)))
print("Episodes failed:           " + str(errors))
print("")
print("By Complexity:")
print("Complexity | Canonical | Shuffled  | Gap")
print("-" * 45)
for c in sorted(results_df["complexity"].unique()):
    c_can = results_df[(results_df["condition"] == "canonical") & (results_df["complexity"] == c)]["score"]
    c_shf = results_df[(results_df["condition"] == "shuffled_control") & (results_df["complexity"] == c)]["score"]
    gap = c_can.mean() - c_shf.mean()
    print(str(c).rjust(10) + " | " + str(round(c_can.mean(), 4)).rjust(9) + " | " + str(round(c_shf.mean(), 4)).rjust(9) + " | " + str(round(gap, 4)).rjust(9))


# ============================================================
# Cell 8: List Available Models
# ============================================================

print("Available models:")
for name in sorted(kbench.llms.keys()):
    print("  " + name)


# ============================================================
# Cell 9: Second Model Evaluation
# ============================================================

# Replace MODEL_NAME with one from Cell 8 output
# Good choices: anthropic/claude-3.5-sonnet, meta/llama-3.1-70b, google/gemini-2.5-pro
MODEL_NAME = "deepseek-ai/deepseek-r1-0528"

import time

second_llm = kbench.llms[MODEL_NAME]
df2 = build_dataset(n_per_condition=25)
results_m2 = []
errors_m2 = 0

for idx, row in df2.iterrows():
    cond = row["condition"]
    cx = row["complexity"]
    ep = row["episode_seed"]
    tf = row["template_family"]
    for attempt in range(3):
        try:
            score = belief_update.run(second_llm, episode_seed=ep, complexity=cx, template_family=tf, condition=cond)
            result_val = score.result if hasattr(score, 'result') else score
            results_m2.append({"episode_seed": ep, "complexity": cx, "condition": cond, "score": result_val})
            print("[" + cond.rjust(16) + "] cx=" + str(cx) + " -> " + str(result_val))
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                results_m2.append({"episode_seed": ep, "complexity": cx, "condition": cond, "score": None})
                errors_m2 += 1
                print("[" + cond.rjust(16) + "] cx=" + str(cx) + " FAILED")

df_m2 = pd.DataFrame(results_m2).dropna(subset=["score"])
can_m2 = df_m2[df_m2["condition"] == "canonical"]["score"]
shf_m2 = df_m2[df_m2["condition"] == "shuffled_control"]["score"]

print("")
print("=" * 60)
print("MODEL 2: " + MODEL_NAME)
print("=" * 60)
print("Canonical mean:            " + str(round(can_m2.mean(), 4)))
print("Shuffled control mean:     " + str(round(shf_m2.mean(), 4)))
print("Temporal sensitivity gap:  " + str(round(can_m2.mean() - shf_m2.mean(), 4)))
print("Episodes completed:        " + str(len(df_m2)) + " / " + str(len(df2)))
print("Episodes failed:           " + str(errors_m2))
print("")
print("By Complexity:")
print("Complexity | Canonical | Shuffled  | Gap")
print("-" * 45)
for c in sorted(df_m2["complexity"].unique()):
    c_can = df_m2[(df_m2["condition"] == "canonical") & (df_m2["complexity"] == c)]["score"]
    c_shf = df_m2[(df_m2["condition"] == "shuffled_control") & (df_m2["complexity"] == c)]["score"]
    gap = c_can.mean() - c_shf.mean()
    print(str(c).rjust(10) + " | " + str(round(c_can.mean(), 4)).rjust(9) + " | " + str(round(c_shf.mean(), 4)).rjust(9) + " | " + str(round(gap, 4)).rjust(9))


# ============================================================
# Cell 10: Cross-Model Comparison
# ============================================================

print("=" * 60)
print("CROSS-MODEL COMPARISON")
print("=" * 60)
print("")
print("Model                     | Canonical | Shuffled | Gap")
print("-" * 60)
print("Gemini 2.5 Flash".ljust(26) + "| " + str(round(canonical.mean(), 4)).rjust(9) + " | " + str(round(shuffled.mean(), 4)).rjust(8) + " | " + str(round(canonical.mean() - shuffled.mean(), 4)).rjust(8))
print(MODEL_NAME.ljust(26) + "| " + str(round(can_m2.mean(), 4)).rjust(9) + " | " + str(round(shf_m2.mean(), 4)).rjust(8) + " | " + str(round(can_m2.mean() - shf_m2.mean(), 4)).rjust(8))


# ============================================================
# Cell 11: Publish Task
# ============================================================

# Uncomment when ready to publish:
# %choose belief_update

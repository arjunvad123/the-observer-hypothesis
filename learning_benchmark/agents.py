from __future__ import annotations

import abc
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None


class AgentAdapter(abc.ABC):
    """Public adapter interface for benchmark agents."""

    agent_id: str

    @abc.abstractmethod
    def start_episode(self, episode_meta: Dict[str, str]) -> None:
        pass

    @abc.abstractmethod
    def respond(self, turn_input: str) -> str:
        pass

    @abc.abstractmethod
    def end_episode(self) -> None:
        pass


@dataclass
class DummyRandomAgent(AgentAdapter):
    """Random baseline for sanity checks."""

    seed: int = 42
    agent_id: str = "dummy_random"
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def start_episode(self, episode_meta: Dict[str, str]) -> None:
        return None

    def respond(self, turn_input: str) -> str:
        options = _extract_options(turn_input)
        if options:
            return self._rng.choice(options)
        if "QUERY" in turn_input:
            return "unknown"
        return "ack"

    def end_episode(self) -> None:
        return None


@dataclass
class RuleBasedAdaptiveAgent(AgentAdapter):
    """Simple adaptive baseline used by integration tests."""

    seed: int = 42
    epsilon: float = 0.1
    agent_id: str = "rule_based_adaptive"
    _rng: random.Random = field(init=False)
    _facts: Dict[str, str] = field(default_factory=dict, init=False)
    _belief_history: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _bandit_stats: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _last_action: Optional[str] = field(default=None, init=False)
    _concept_examples: List[tuple] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def start_episode(self, episode_meta: Dict[str, str]) -> None:
        self._facts.clear()
        self._belief_history.clear()
        self._bandit_stats.clear()
        self._last_action = None
        self._concept_examples.clear()

    def respond(self, turn_input: str) -> str:
        self._ingest_state(turn_input)

        if "TRANSFORM_QUERY" in turn_input:
            token = _extract_transform_token(turn_input)
            return self._solve_concept(token)

        query_key = _extract_query_key(turn_input)
        if query_key:
            return self._facts.get(query_key, "unknown")

        if "BANDIT_TRIAL" in turn_input:
            options = _extract_options(turn_input)
            if not options:
                return "unknown"
            if self._rng.random() < self.epsilon:
                action = self._rng.choice(options)
            else:
                scored = []
                for opt in options:
                    rewards = self._bandit_stats.get(opt, [])
                    avg = sum(rewards) / len(rewards) if rewards else 0.0
                    scored.append((avg, opt))
                scored.sort(reverse=True)
                action = scored[0][1]
            self._last_action = action
            return action

        return "ack"

    def end_episode(self) -> None:
        return None

    def _ingest_state(self, turn_input: str) -> None:
        for key, value in re.findall(r"FACT\s+([A-Z0-9_]+)\s*=\s*([A-Za-z0-9_\-]+)", turn_input):
            self._facts[key] = value
            self._belief_history.setdefault(key, []).append(value)

        for key, value in re.findall(r"CORRECTION\s+([A-Z0-9_]+)\s*=\s*([A-Za-z0-9_\-]+)", turn_input):
            self._facts[key] = value
            self._belief_history.setdefault(key, []).append(value)

        for match in re.findall(r"EXAMPLE\s+([a-z]+)\s*->\s*([a-z0-9_]+)", turn_input):
            self._concept_examples.append(match)

        reward_match = re.search(r"REWARD\s+([A-Za-z0-9_\-]+)\s*=\s*([0-9]+\.?[0-9]*)", turn_input)
        if reward_match:
            action = reward_match.group(1)
            reward = float(reward_match.group(2))
            self._bandit_stats.setdefault(action, []).append(reward)

    def _solve_concept(self, token: str) -> str:
        if not token:
            return "unknown"
        if not self._concept_examples:
            return token

        candidates = [
            _rule_identity,
            _rule_reverse,
            _rule_suffix,
            _rule_reverse_suffix,
            _rule_vowel_shift,
        ]
        for candidate in candidates:
            if _matches_all(candidate, self._concept_examples):
                return candidate(token, self._concept_examples)
        return token


@dataclass
class StubbornAgent(AgentAdapter):
    """Non-adaptive baseline; ignores corrections and reward feedback."""

    seed: int = 42
    agent_id: str = "stubborn"
    _rng: random.Random = field(init=False)
    _facts: Dict[str, str] = field(default_factory=dict, init=False)
    _fixed_action: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def start_episode(self, episode_meta: Dict[str, str]) -> None:
        self._facts.clear()
        self._fixed_action = None

    def respond(self, turn_input: str) -> str:
        for key, value in re.findall(r"FACT\s+([A-Z0-9_]+)\s*=\s*([A-Za-z0-9_\-]+)", turn_input):
            self._facts.setdefault(key, value)

        query_key = _extract_query_key(turn_input)
        if query_key:
            return self._facts.get(query_key, "unknown")

        if "TRANSFORM_QUERY" in turn_input:
            token = _extract_transform_token(turn_input)
            return token

        if "BANDIT_TRIAL" in turn_input:
            options = _extract_options(turn_input)
            if not options:
                return "unknown"
            if self._fixed_action is None:
                self._fixed_action = options[0]
            return self._fixed_action

        return "ack"

    def end_episode(self) -> None:
        return None


@dataclass
class AnthropicAgent(AgentAdapter):
    """Optional adapter for real model runs."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key: Optional[str] = None
    agent_id: str = "anthropic"
    _client: Optional["anthropic.Anthropic"] = field(default=None, init=False)
    _messages: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if anthropic is None:
            raise RuntimeError("anthropic package not available")
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for AnthropicAgent")
        self._client = anthropic.Anthropic(api_key=key)

    def start_episode(self, episode_meta: Dict[str, str]) -> None:
        self._messages = []

    def respond(self, turn_input: str) -> str:
        assert self._client is not None
        self._messages.append({"role": "user", "content": turn_input})
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self._messages,
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        text = text.strip() or ""
        self._messages.append({"role": "assistant", "content": text})
        return text

    def end_episode(self) -> None:
        self._messages = []


def create_agent(name: str, seed: int = 42) -> AgentAdapter:
    if name == "dummy":
        return DummyRandomAgent(seed=seed)
    if name == "anthropic":
        return AnthropicAgent()
    if name == "rule_based_adaptive":
        return RuleBasedAdaptiveAgent(seed=seed)
    if name == "stubborn":
        return StubbornAgent(seed=seed)
    raise ValueError(f"Unknown agent: {name}")


def _extract_options(text: str) -> List[str]:
    options_block = re.search(r"Options\s*:\s*\[([^\]]+)\]", text)
    if options_block:
        raw = options_block.group(1)
        return [item.strip() for item in raw.split(",") if item.strip()]

    token_block = re.search(r"choose one action from\s*\[([^\]]+)\]", text, flags=re.I)
    if token_block:
        return [item.strip() for item in token_block.group(1).split(",") if item.strip()]
    return []


def _extract_query_key(text: str) -> Optional[str]:
    match = re.search(r"QUERY\s+([A-Z0-9_]+)", text)
    return match.group(1) if match else None


def _extract_transform_token(text: str) -> str:
    match = re.search(r"TRANSFORM_QUERY\s+([a-z]+)", text)
    return match.group(1) if match else ""


def _matches_all(rule_fn, examples: List[tuple]) -> bool:
    for inp, out in examples:
        if rule_fn(inp, examples) != out:
            return False
    return True


def _rule_identity(token: str, examples: List[tuple]) -> str:
    return token


def _rule_reverse(token: str, examples: List[tuple]) -> str:
    return token[::-1]


def _rule_suffix(token: str, examples: List[tuple]) -> str:
    suffix = _infer_suffix(examples)
    return token + suffix


def _rule_reverse_suffix(token: str, examples: List[tuple]) -> str:
    suffix = _infer_suffix(examples)
    return token[::-1] + suffix


def _rule_vowel_shift(token: str, examples: List[tuple]) -> str:
    table = str.maketrans({"a": "e", "e": "i", "i": "o", "o": "u", "u": "a"})
    return token.translate(table)


def _infer_suffix(examples: List[tuple]) -> str:
    if not examples:
        return ""
    inp, out = examples[0]
    if out.startswith(inp):
        return out[len(inp) :]
    if out.startswith(inp[::-1]):
        return out[len(inp) :]
    return ""

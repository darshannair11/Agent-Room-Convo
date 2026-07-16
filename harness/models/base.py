"""The adapter seam that keeps the harness provider-agnostic.

The graph only ever calls generate(). Adding a new provider (GPT, Gemini) is a
new file implementing this interface — zero graph changes.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from harness.config import Question


@dataclass
class ModelResponse:
    """Parsed structured output from one model call."""

    stance: str
    reasoning: str
    self_confidence: int
    perceived_peer_confidence: Optional[int]  # None for solo turns
    raw_response: str
    malformed: bool = False
    token_usage: dict = field(default_factory=dict)


def _clamp_confidence(value: object) -> int:
    """Coerce a model-supplied confidence into a 0-100 int, defaulting to 50."""
    try:
        n = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 50
    return max(0, min(100, n))


def _extract_json_block(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response.

    Tolerates fenced code blocks and leading/trailing prose, which models
    sometimes add despite instructions.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def parse_response(
    raw: str,
    *,
    expect_peer_confidence: bool,
    normalize_choices: Optional[list[str]] = None,
    token_usage: Optional[dict] = None,
) -> ModelResponse:
    """Turn raw model text into a ModelResponse.

    On unparseable output, returns a ModelResponse flagged malformed with the
    raw text preserved rather than raising — one bad turn shouldn't kill a run.
    """
    data = _extract_json_block(raw)
    if data is None or "stance" not in data:
        return ModelResponse(
            stance="",
            reasoning="",
            self_confidence=50,
            perceived_peer_confidence=50 if expect_peer_confidence else None,
            raw_response=raw,
            malformed=True,
            token_usage=token_usage or {},
        )

    stance = str(data.get("stance", "")).strip()
    if normalize_choices:
        stance = _normalize_stance(stance, normalize_choices)

    peer = None
    if expect_peer_confidence:
        peer = _clamp_confidence(data.get("perceived_peer_confidence"))

    return ModelResponse(
        stance=stance,
        reasoning=str(data.get("reasoning", "")).strip(),
        self_confidence=_clamp_confidence(data.get("self_confidence")),
        perceived_peer_confidence=peer,
        raw_response=raw,
        malformed=False,
        token_usage=token_usage or {},
    )


def _normalize_stance(stance: str, choices: list[str]) -> str:
    """Map a free-typed stance onto one of the allowed choices when possible.

    Exact (case-insensitive) match wins; otherwise a unique choice appearing as
    a token in the stance wins; otherwise the original stance is returned so the
    mismatch is visible downstream rather than silently coerced.
    """
    s = stance.strip()
    lowered = {c.lower(): c for c in choices}
    if s.lower() in lowered:
        return lowered[s.lower()]
    hits = [c for c in choices if re.search(rf"\b{re.escape(c)}\b", s, re.IGNORECASE)]
    if len(hits) == 1:
        return hits[0]
    return s


class ModelAdapter(ABC):
    """Provider-agnostic interface. The graph depends only on this.

    Tracks cumulative token usage across all calls so the runner can total
    per-room cost without threading counters through graph state.
    """

    def __init__(self) -> None:
        self.usage_total: dict = {"input_tokens": 0, "output_tokens": 0}

    def _record_usage(self, usage: Optional[dict]) -> None:
        if not usage:
            return
        self.usage_total["input_tokens"] += usage.get("input_tokens", 0)
        self.usage_total["output_tokens"] += usage.get("output_tokens", 0)

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        transcript_context: str,
        question: Question,
        *,
        temperature: float,
        expect_peer_confidence: bool,
    ) -> ModelResponse:
        """Produce one structured response for the given context."""
        raise NotImplementedError

"""Claude (Anthropic API) adapter.

Handles: building the message, one retry on malformed structured output, and
exponential backoff on transient API errors. API key comes from the environment
(ANTHROPIC_API_KEY), never from config.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from harness.config import Question
from harness.models.base import ModelAdapter, ModelResponse, parse_response

_MAX_API_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 1.0


class ClaudeAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            # Imported lazily so the package is importable without anthropic
            # installed (e.g. for mock-only test runs).
            from anthropic import Anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set; export it or pass api_key="
                )
            self._client = Anthropic(api_key=key)

    def generate(
        self,
        system_prompt: str,
        transcript_context: str,
        question: Question,
        *,
        temperature: float,
        expect_peer_confidence: bool,
    ) -> ModelResponse:
        user_message = self._build_user_message(
            transcript_context, question, expect_peer_confidence
        )
        raw, usage = self._call_api(system_prompt, user_message, temperature)
        response = parse_response(
            raw,
            expect_peer_confidence=expect_peer_confidence,
            normalize_choices=question.answer_choices,
            token_usage=usage,
        )
        if not response.malformed:
            self._record_usage(response.token_usage)
            return response

        # One nudge retry — models occasionally wrap or omit the JSON.
        retry_message = (
            user_message
            + "\n\nYour previous reply could not be parsed. Reply with ONLY a "
            "single valid JSON object in the required shape, nothing else."
        )
        raw2, usage2 = self._call_api(system_prompt, retry_message, temperature)
        merged_usage = _merge_usage(usage, usage2)
        response2 = parse_response(
            raw2,
            expect_peer_confidence=expect_peer_confidence,
            normalize_choices=question.answer_choices,
            token_usage=merged_usage,
        )
        self._record_usage(merged_usage)
        return response2

    # ------------------------------------------------------------------ #
    def _build_user_message(
        self,
        transcript_context: str,
        question: Question,
        expect_peer_confidence: bool,
    ) -> str:
        parts = [f"Question: {question.text}"]
        if question.answer_choices:
            parts.append("Choose exactly one of: " + ", ".join(question.answer_choices))
        if transcript_context.strip():
            parts.append("Discussion so far:\n" + transcript_context)
        else:
            parts.append("No one has spoken yet; you are first.")

        fields = ['"stance"', '"reasoning"', '"self_confidence" (0-100)']
        if expect_peer_confidence:
            fields.append('"perceived_peer_confidence" (0-100)')
        parts.append(
            "Reply with ONLY a JSON object containing: " + ", ".join(fields) + "."
        )
        return "\n\n".join(parts)

    def _call_api(
        self, system_prompt: str, user_message: str, temperature: float
    ) -> tuple[str, dict]:
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_API_ATTEMPTS):
            try:
                resp = self._client.messages.create(  # type: ignore[attr-defined]
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                usage = {
                    "input_tokens": getattr(resp.usage, "input_tokens", 0),
                    "output_tokens": getattr(resp.usage, "output_tokens", 0),
                }
                return text, usage
            except Exception as err:  # noqa: BLE001 — backoff then re-raise
                last_err = err
                if attempt == _MAX_API_ATTEMPTS - 1:
                    break
                time.sleep(_BACKOFF_BASE_SECONDS * (2 ** attempt))
        raise RuntimeError(f"Claude API call failed after retries: {last_err}") from last_err


def _merge_usage(a: dict, b: dict) -> dict:
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
    }

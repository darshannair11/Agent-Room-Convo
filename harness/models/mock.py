"""A deterministic in-memory adapter for tests and dry runs.

Lets the entire room state machine be exercised with zero API calls and zero
cost. Behaviour is driven by a scripted queue of responses or a callable.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from harness.config import Question
from harness.models.base import ModelAdapter, ModelResponse, parse_response


class MockAdapter(ModelAdapter):
    """Returns canned structured responses.

    Provide either:
      - responses: a list of dicts (stance/reasoning/self_confidence/...), served
        in order and then repeated from the last one, or
      - responder: a callable (system_prompt, transcript_context, question) -> dict
    """

    def __init__(
        self,
        responses: Optional[list[dict]] = None,
        responder: Optional[Callable[[str, str, Question], dict]] = None,
    ) -> None:
        super().__init__()
        self._responses = list(responses or [])
        self._responder = responder
        self._cursor = 0
        self.calls: list[dict] = []

    def generate(
        self,
        system_prompt: str,
        transcript_context: str,
        question: Question,
        *,
        temperature: float,
        expect_peer_confidence: bool,
    ) -> ModelResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "transcript_context": transcript_context,
                "question_id": question.id,
                "temperature": temperature,
                "expect_peer_confidence": expect_peer_confidence,
            }
        )
        if self._responder is not None:
            payload = self._responder(system_prompt, transcript_context, question)
        elif self._responses:
            idx = min(self._cursor, len(self._responses) - 1)
            payload = self._responses[idx]
            self._cursor += 1
        else:
            payload = {
                "stance": (question.answer_choices or ["yes"])[0],
                "reasoning": "mock reasoning",
                "self_confidence": 70,
                "perceived_peer_confidence": 60,
            }
        raw = json.dumps(payload)
        response = parse_response(
            raw,
            expect_peer_confidence=expect_peer_confidence,
            normalize_choices=question.answer_choices,
            token_usage={"input_tokens": 0, "output_tokens": 0},
        )
        self._record_usage(response.token_usage)
        return response

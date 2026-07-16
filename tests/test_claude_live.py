"""One live smoke test against the real Claude API.

Skipped unless ANTHROPIC_API_KEY is set, so the default test run stays free and
offline. Run explicitly with:  pytest tests/test_claude_live.py -m live
"""

import os

import pytest

from harness.config import AgentSpec, Question, RoomConfig
from harness.models.claude import ClaudeAdapter
from harness.runner import run_room

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; live smoke test skipped",
)


def test_two_agent_live_room_runs():
    config = RoomConfig(
        num_rounds=1,
        seed=1,
        agents=[
            AgentSpec(agent_id="agent_1", model="claude-opus-4-8", temperature=0.7),
            AgentSpec(agent_id="agent_2", model="claude-opus-4-8", temperature=0.7),
        ],
    )
    question = Question(
        id="live_smoke", text="What is 1 + 1?", type="objective",
        answer_choices=["1", "2", "3", "4"], ground_truth="2",
    )
    transcript = run_room(
        config, question,
        adapter_factory=lambda spec: ClaudeAdapter(model=spec.model),
    )
    assert len(transcript.turns) == 2
    assert len(transcript.solo_pre) == 2
    # Real models should get 1+1 right and emit a normalized stance.
    assert all(t.stance in {"1", "2", "3", "4"} for t in transcript.turns)
    assert transcript.metadata["token_usage"]["output_tokens"] > 0

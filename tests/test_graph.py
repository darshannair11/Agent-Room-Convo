"""End-to-end room runs with the mock adapter — zero API calls, zero cost.

Verifies the full state machine: solo bookends, round/turn counts, first-mover
peer-confidence handling, and stance flips driven by the discussion.
"""

from harness.config import AgentSpec, Question, RoomConfig
from harness.models.mock import MockAdapter
from harness.runner import run_room


def _room(num_agents=4, num_rounds=3, temperature=0.8):
    return RoomConfig(
        num_rounds=num_rounds,
        seed=42,
        agents=[
            AgentSpec(agent_id=f"agent_{i}", model="mock", temperature=temperature)
            for i in range(1, num_agents + 1)
        ],
    )


def _question():
    return Question(id="q1", text="What is 1 + 1?", type="objective",
                    answer_choices=["1", "2", "3", "4"], ground_truth="2")


def test_room_produces_expected_structure():
    config, question = _room(4, 3), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())

    assert len(transcript.solo_pre) == 4
    assert len(transcript.solo_post) == 4
    assert len(transcript.turns) == 4 * 3  # agents * rounds
    assert {s.phase for s in transcript.solo_pre} == {"pre"}
    assert {s.phase for s in transcript.solo_post} == {"post"}


def test_every_agent_speaks_once_per_round():
    config, question = _room(4, 3), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())
    for r in range(3):
        speakers = [t.agent_id for t in transcript.turns if t.round_idx == r]
        assert sorted(speakers) == ["agent_1", "agent_2", "agent_3", "agent_4"]


def test_first_turn_has_no_peer_confidence():
    config, question = _room(3, 2), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())
    first = transcript.turns[0]
    assert first.perceived_peer_confidence is None
    # Every later turn has seen at least one peer.
    assert all(t.perceived_peer_confidence is not None for t in transcript.turns[1:])


def test_no_back_to_back_speaker_across_rounds():
    config, question = _room(4, 4), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())
    for r in range(3):
        last_of_round = [t for t in transcript.turns if t.round_idx == r][-1]
        first_of_next = [t for t in transcript.turns if t.round_idx == r + 1][0]
        assert last_of_round.agent_id != first_of_next.agent_id


def test_position_in_round_is_recorded():
    config, question = _room(3, 2), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())
    for r in range(2):
        positions = [t.position_in_round for t in transcript.turns if t.round_idx == r]
        assert positions == [0, 1, 2]


def test_stance_flip_is_captured():
    # Agents start on "1", then flip to "2" once they see any discussion.
    def responder(system_prompt, context, question):
        stance = "2" if context.strip() else "1"
        return {"stance": stance, "reasoning": "r",
                "self_confidence": 80, "perceived_peer_confidence": 60}

    config, question = _room(3, 2), _question()
    transcript = run_room(
        config, question,
        adapter_factory=lambda spec: MockAdapter(responder=responder),
    )
    assert transcript.turns[0].stance == "1"   # first mover, empty context
    assert transcript.turns[-1].stance == "2"  # later turns saw discussion


def test_metadata_is_populated():
    config, question = _room(3, 2), _question()
    transcript = run_room(config, question, adapter_factory=lambda spec: MockAdapter())
    md = transcript.metadata
    assert md["num_agents"] == 3
    assert md["num_rounds"] == 2
    assert md["malformed_turns"] == 0
    assert "token_usage" in md
    assert transcript.config_snapshot["seed"] == 42

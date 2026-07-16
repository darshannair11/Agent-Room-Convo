"""Round-trip guarantees the on-disk contract can't silently drift."""

from harness.config import Question
from harness.storage import load_transcript, save_transcript
from harness.transcript import SoloResponse, Transcript, Turn


def _sample_transcript() -> Transcript:
    q = Question(
        id="q1",
        text="What is 1 + 1?",
        type="objective",
        answer_choices=["1", "2"],
        ground_truth="2",
    )
    return Transcript(
        run_id="q1-abc123",
        config_snapshot={"num_rounds": 2, "seed": 42, "agents": []},
        question=q,
        agents=[{"agent_id": "agent_1", "model": "claude-opus-4-8",
                 "temperature": 0.8, "tool_access": False}],
        solo_pre=[SoloResponse("agent_1", "pre", "2", "because", 90,
                               '{"stance":"2"}', False, "2026-07-04T00:00:00Z")],
        turns=[Turn("agent_1", 0, 0, "2", "obvious", 95, None,
                    '{"stance":"2"}', False, "2026-07-04T00:00:01Z")],
        solo_post=[SoloResponse("agent_1", "post", "2", "still", 95,
                                '{"stance":"2"}', False, "2026-07-04T00:00:02Z")],
        metadata={"num_rounds": 2},
    )


def test_dict_round_trip_is_equal():
    t = _sample_transcript()
    assert Transcript.from_dict(t.to_dict()) == t


def test_file_round_trip_is_equal(tmp_path):
    t = _sample_transcript()
    path = save_transcript(t, tmp_path)
    assert path.exists()
    assert load_transcript(path) == t


def test_first_mover_peer_confidence_is_none():
    t = _sample_transcript()
    assert t.turns[0].perceived_peer_confidence is None

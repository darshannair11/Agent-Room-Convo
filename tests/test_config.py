"""Config loading: agent expansion (homo/hetero), validation, and the
temperature-0 homogeneous warning."""

import warnings

import pytest

from harness.config import (
    AgentSpec,
    Question,
    RoomConfig,
    load_question_bank,
    load_room_config,
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_shorthand_expands_to_distinct_agents(tmp_path):
    cfg = _write(tmp_path, "room.yaml", """
num_rounds: 2
seed: 1
agents:
  - model: claude-opus-4-8
    count: 3
    temperature: 0.8
""")
    room = load_room_config(cfg)
    assert room.num_agents == 3
    assert [a.agent_id for a in room.agents] == ["agent_1", "agent_2", "agent_3"]
    assert all(a.model == "claude-opus-4-8" for a in room.agents)


def test_explicit_heterogeneous_agents(tmp_path):
    cfg = _write(tmp_path, "room.yaml", """
num_rounds: 2
seed: 1
agents:
  - agent_id: alpha
    model: claude-opus-4-8
  - agent_id: beta
    model: claude-sonnet-5
""")
    room = load_room_config(cfg)
    assert [a.agent_id for a in room.agents] == ["alpha", "beta"]
    assert room.agents[0].temperature == 0.7  # default applied


def test_homogeneous_temp_zero_warns():
    with pytest.warns(UserWarning, match="clones"):
        RoomConfig(
            num_rounds=2,
            seed=1,
            agents=[
                AgentSpec(agent_id="agent_1", model="m", temperature=0.0),
                AgentSpec(agent_id="agent_2", model="m", temperature=0.0),
            ],
        )


def test_objective_question_requires_ground_truth():
    with pytest.raises(ValueError, match="ground_truth"):
        Question(id="q", text="?", type="objective")


def test_ground_truth_must_be_a_choice():
    with pytest.raises(ValueError, match="not one of"):
        Question(id="q", text="?", type="objective",
                 answer_choices=["A", "B"], ground_truth="C")


def test_load_question_bank(tmp_path):
    bank = _write(tmp_path, "q.yaml", """
questions:
  - id: q1
    text: "What is 1 + 1?"
    type: objective
    answer_choices: ["1", "2"]
    ground_truth: "2"
""")
    questions = load_question_bank(bank)
    assert len(questions) == 1
    assert questions[0].ground_truth == "2"

"""The room as a LangGraph state machine.

Flow (see the design spec):

    init_room -> solo_pre -> round_setup -> agent_turn -> (route)
                                  ^-------------------------|
                                                            v
                                          solo_post -> finalize

route after each turn:
  - more speakers this round      -> agent_turn (next speaker)
  - round done, more rounds left  -> round_setup (next round)
  - all rounds done               -> solo_post

Turns are sequential by design: each agent must see the prior turns, so they
cannot be parallelized within a room.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from harness.agent import Agent
from harness.config import Question, RoomConfig
from harness.scheduler import SpeakingScheduler
from harness.transcript import SoloResponse, Turn

_PROMPT_DIR = Path(__file__).parent / "prompts"


class RoomState(TypedDict, total=False):
    run_id: str
    question: Question
    agents: list[Agent]
    config: RoomConfig
    scheduler: SpeakingScheduler
    round_idx: int
    speaking_order: list[str]
    turn_cursor: int
    last_speaker_id: Optional[str]
    turns: list[Turn]
    solo_pre: list[SoloResponse]
    solo_post: list[SoloResponse]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text()


def _agents_by_id(agents: list[Agent]) -> dict[str, Agent]:
    return {a.agent_id: a for a in agents}


def serialize_transcript_context(turns: list[Turn]) -> str:
    """Render the discussion so far as readable text for the next speaker.

    Every agent sees this same shared view — real interaction, not a canned
    peer-summary.
    """
    lines = []
    for t in turns:
        tag = " [unparsed]" if t.malformed else ""
        lines.append(
            f"{t.agent_id} (round {t.round_idx + 1}): {t.stance} — {t.reasoning}{tag}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def _init_room(state: RoomState) -> dict:
    return {
        "round_idx": 0,
        "turn_cursor": 0,
        "last_speaker_id": None,
        "turns": [],
        "solo_pre": [],
        "solo_post": [],
    }


def _solo_node(state: RoomState, phase: str) -> list[SoloResponse]:
    system_template = _load_prompt("solo_system_prompt.txt")
    question = state["question"]
    out: list[SoloResponse] = []
    for agent in state["agents"]:
        system_prompt = system_template.format(agent_id=agent.agent_id)
        resp = agent.adapter.generate(
            system_prompt,
            "",  # solo: no discussion context
            question,
            temperature=agent.temperature,
            expect_peer_confidence=False,
        )
        out.append(
            SoloResponse(
                agent_id=agent.agent_id,
                phase=phase,
                stance=resp.stance,
                reasoning=resp.reasoning,
                self_confidence=resp.self_confidence,
                raw_response=resp.raw_response,
                malformed=resp.malformed,
                timestamp=_now(),
            )
        )
    return out


def _solo_pre(state: RoomState) -> dict:
    return {"solo_pre": _solo_node(state, "pre")}


def _solo_post(state: RoomState) -> dict:
    return {"solo_post": _solo_node(state, "post")}


def _round_setup(state: RoomState) -> dict:
    order = state["scheduler"].order_for_round(state["last_speaker_id"])
    return {"speaking_order": order, "turn_cursor": 0}


def _agent_turn(state: RoomState) -> dict:
    system_template = _load_prompt("room_system_prompt.txt")
    question = state["question"]
    turns = list(state["turns"])
    cursor = state["turn_cursor"]
    agent_id = state["speaking_order"][cursor]
    agent = _agents_by_id(state["agents"])[agent_id]

    # First mover with an empty transcript has no peers to assess yet.
    peers_visible = len(turns) > 0
    context = serialize_transcript_context(turns)
    system_prompt = system_template.format(agent_id=agent_id)

    resp = agent.adapter.generate(
        system_prompt,
        context,
        question,
        temperature=agent.temperature,
        expect_peer_confidence=peers_visible,
    )
    turns.append(
        Turn(
            agent_id=agent_id,
            round_idx=state["round_idx"],
            position_in_round=cursor,
            stance=resp.stance,
            reasoning=resp.reasoning,
            self_confidence=resp.self_confidence,
            perceived_peer_confidence=(
                resp.perceived_peer_confidence if peers_visible else None
            ),
            raw_response=resp.raw_response,
            malformed=resp.malformed,
            timestamp=_now(),
        )
    )
    return {
        "turns": turns,
        "turn_cursor": cursor + 1,
        "last_speaker_id": agent_id,
    }


def _route_after_turn(state: RoomState) -> str:
    if state["turn_cursor"] < len(state["speaking_order"]):
        return "agent_turn"
    if state["round_idx"] + 1 < state["config"].num_rounds:
        return "next_round"
    return "done"


def _advance_round(state: RoomState) -> dict:
    return {"round_idx": state["round_idx"] + 1}


def build_room_graph():
    """Compile and return the room StateGraph."""
    g = StateGraph(RoomState)
    g.add_node("init_room", _init_room)
    g.add_node("solo_pre", _solo_pre)
    g.add_node("round_setup", _round_setup)
    g.add_node("agent_turn", _agent_turn)
    g.add_node("advance_round", _advance_round)
    g.add_node("solo_post", _solo_post)

    g.set_entry_point("init_room")
    g.add_edge("init_room", "solo_pre")
    g.add_edge("solo_pre", "round_setup")
    g.add_edge("round_setup", "agent_turn")
    g.add_conditional_edges(
        "agent_turn",
        _route_after_turn,
        {"agent_turn": "agent_turn", "next_round": "advance_round", "done": "solo_post"},
    )
    g.add_edge("advance_round", "round_setup")
    g.add_edge("solo_post", END)
    return g.compile()

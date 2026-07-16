"""Transcript data model + JSON (de)serialization.

This is the contract the analysis pipeline and later phases consume, so it is
kept plain: dataclasses that map one-to-one onto the on-disk JSON. Round-trip
(object -> JSON -> object) is guaranteed equal, verified in tests, so the
on-disk format can't silently drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from harness.config import Question


@dataclass
class Turn:
    """One agent's contribution during the group discussion."""

    agent_id: str
    round_idx: int
    position_in_round: int
    stance: str
    reasoning: str
    self_confidence: int
    perceived_peer_confidence: Optional[int]  # None if no peer turn seen yet
    raw_response: str
    malformed: bool
    timestamp: str


@dataclass
class SoloResponse:
    """An agent's answer given alone, before or after the discussion."""

    agent_id: str
    phase: str  # "pre" | "post"
    stance: str
    reasoning: str
    self_confidence: int
    raw_response: str
    malformed: bool
    timestamp: str


@dataclass
class Transcript:
    """Everything produced by one room run."""

    run_id: str
    config_snapshot: dict
    question: Question
    agents: list[dict]  # id, model, temperature, tool_access
    solo_pre: list[SoloResponse] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    solo_post: list[SoloResponse] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_snapshot": self.config_snapshot,
            "question": asdict(self.question),
            "agents": self.agents,
            "solo_pre": [asdict(s) for s in self.solo_pre],
            "turns": [asdict(t) for t in self.turns],
            "solo_post": [asdict(s) for s in self.solo_post],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            run_id=data["run_id"],
            config_snapshot=data["config_snapshot"],
            question=Question(**data["question"]),
            agents=data["agents"],
            solo_pre=[SoloResponse(**s) for s in data.get("solo_pre", [])],
            turns=[Turn(**t) for t in data.get("turns", [])],
            solo_post=[SoloResponse(**s) for s in data.get("solo_post", [])],
            metadata=data.get("metadata", {}),
        )

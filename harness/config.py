"""Config and question schemas + loaders.

Two hand-authored inputs drive a run: a room config (who is in the room, how
many rounds, sampling) and a question bank (what they discuss). Both load from
YAML or JSON and are validated at startup so a bad config fails fast rather
than midway through an expensive run.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Question:
    """A single discussion prompt.

    objective questions carry a ground truth and (usually) constrained answer
    choices so metrics are exact string comparisons. subjective questions may
    leave both as None.
    """

    id: str
    text: str
    type: str  # "objective" | "subjective"
    answer_choices: Optional[list[str]] = None
    ground_truth: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in ("objective", "subjective"):
            raise ValueError(
                f"Question {self.id!r}: type must be 'objective' or 'subjective', "
                f"got {self.type!r}"
            )
        if self.type == "objective" and self.ground_truth is None:
            raise ValueError(
                f"Question {self.id!r}: objective questions require a ground_truth"
            )
        if self.answer_choices is not None and self.ground_truth is not None:
            if self.ground_truth not in self.answer_choices:
                raise ValueError(
                    f"Question {self.id!r}: ground_truth {self.ground_truth!r} "
                    f"is not one of answer_choices {self.answer_choices}"
                )


@dataclass
class AgentSpec:
    """One participant. tool_access is parsed now but unused until Phase 3."""

    agent_id: str
    model: str
    temperature: float
    tool_access: bool = False


@dataclass
class RoomConfig:
    """Full room setup. num_agents is always len(agents) — never separate."""

    num_rounds: int
    seed: int
    agents: list[AgentSpec]
    default_temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if len(self.agents) < 2:
            raise ValueError(
                f"a room needs at least 2 agents, got {len(self.agents)}"
            )
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent ids must be unique, got {ids}")
        self._warn_if_homogeneous_and_deterministic()

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def _warn_if_homogeneous_and_deterministic(self) -> None:
        """N instances of one model at temperature 0 are clones — no diversity
        to debate. Warn so the homogeneous baseline isn't silently degenerate."""
        by_model: dict[str, list[AgentSpec]] = {}
        for a in self.agents:
            by_model.setdefault(a.model, []).append(a)
        for model, specs in by_model.items():
            if len(specs) > 1 and all(s.temperature == 0 for s in specs):
                warnings.warn(
                    f"{len(specs)} instances of {model!r} all at temperature 0 "
                    "will behave as clones with no diversity to debate; set "
                    "temperature > 0 for a meaningful homogeneous room.",
                    stacklevel=2,
                )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_raw(path: str | Path) -> Any:
    """Read a YAML or JSON file into plain Python data."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if p.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported config extension {p.suffix!r} (use .yaml/.yml/.json)")


def _expand_agents(raw_agents: list[dict], default_temperature: float) -> list[AgentSpec]:
    """Expand agent entries into AgentSpecs.

    Supports two authoring styles, freely mixed:
      - shorthand: {model: X, count: N}     -> N instances (homogeneous block)
      - explicit:  {agent_id: .., model: X} -> one named agent (heterogeneous)
    Homogeneous vs heterogeneous is just how this list is written; there is no
    separate mode flag.
    """
    specs: list[AgentSpec] = []
    auto_idx = 1
    for entry in raw_agents:
        temperature = entry.get("temperature", default_temperature)
        tool_access = entry.get("tool_access", False)
        if "count" in entry:
            for _ in range(int(entry["count"])):
                specs.append(
                    AgentSpec(
                        agent_id=f"agent_{auto_idx}",
                        model=entry["model"],
                        temperature=temperature,
                        tool_access=tool_access,
                    )
                )
                auto_idx += 1
        else:
            agent_id = entry.get("agent_id") or f"agent_{auto_idx}"
            auto_idx += 1
            specs.append(
                AgentSpec(
                    agent_id=agent_id,
                    model=entry["model"],
                    temperature=temperature,
                    tool_access=tool_access,
                )
            )
    return specs


def load_room_config(path: str | Path) -> RoomConfig:
    """Load and validate a room config from YAML/JSON."""
    raw = _load_raw(path)
    if not isinstance(raw, dict):
        raise ValueError(f"room config must be a mapping, got {type(raw).__name__}")
    default_temperature = float(raw.get("default_temperature", 0.7))
    agents = _expand_agents(raw.get("agents", []), default_temperature)
    return RoomConfig(
        num_rounds=int(raw["num_rounds"]),
        seed=int(raw["seed"]),
        agents=agents,
        default_temperature=default_temperature,
    )


def load_question_bank(path: str | Path) -> list[Question]:
    """Load a list of Questions from YAML/JSON.

    Accepts either a top-level list, or a mapping with a 'questions' key.
    """
    raw = _load_raw(path)
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    if not isinstance(raw, list):
        raise ValueError("question bank must be a list (or have a 'questions' list)")
    questions = [
        Question(
            id=str(q["id"]),
            text=q["text"],
            type=q["type"],
            answer_choices=q.get("answer_choices"),
            ground_truth=q.get("ground_truth"),
        )
        for q in raw
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"question ids must be unique, got {ids}")
    return questions

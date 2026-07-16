"""Agent Room Convo — shared multi-agent room harness.

Public API:
    from harness import run_room, load_room_config, load_question_bank
"""

from harness.config import (
    AgentSpec,
    Question,
    RoomConfig,
    load_question_bank,
    load_room_config,
)
from harness.runner import run_room
from harness.transcript import SoloResponse, Transcript, Turn

__all__ = [
    "AgentSpec",
    "Question",
    "RoomConfig",
    "load_room_config",
    "load_question_bank",
    "run_room",
    "Turn",
    "SoloResponse",
    "Transcript",
]

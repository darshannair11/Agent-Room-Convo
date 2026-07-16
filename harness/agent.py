"""An Agent binds a spec to a live model adapter.

A heterogeneous room is agents with different adapters; a homogeneous room is
several agents sharing a model id with independent conversation state.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.config import AgentSpec
from harness.models.base import ModelAdapter


@dataclass
class Agent:
    spec: AgentSpec
    adapter: ModelAdapter

    @property
    def agent_id(self) -> str:
        return self.spec.agent_id

    @property
    def temperature(self) -> float:
        return self.spec.temperature

    def to_public_dict(self) -> dict:
        """Serializable projection stored in the transcript (no live adapter)."""
        return {
            "agent_id": self.spec.agent_id,
            "model": self.spec.model,
            "temperature": self.spec.temperature,
            "tool_access": self.spec.tool_access,
        }

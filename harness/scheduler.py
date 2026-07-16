"""Speaking-order scheduler.

Two constraints, both to keep the room fair and avoid positional bias:
  1. Order is randomized each round (no fixed 1,2,3 anchoring).
  2. Whoever spoke LAST in round N may not speak FIRST in round N+1 — no
     back-to-back turns for the same agent across the round boundary.

Deterministic given a seed, so runs are reproducible and testable.
"""

from __future__ import annotations

import random
from typing import Optional


class SpeakingScheduler:
    def __init__(self, agent_ids: list[str], seed: int) -> None:
        if len(agent_ids) < 2:
            raise ValueError("scheduler needs at least 2 agents")
        self._agent_ids = list(agent_ids)
        self._rng = random.Random(seed)

    def order_for_round(self, last_speaker_id: Optional[str]) -> list[str]:
        """Return a shuffled order whose first element != last_speaker_id.

        With >= 2 agents this always terminates quickly: reshuffle until the
        first speaker differs from the previous round's last speaker.
        """
        order = list(self._agent_ids)
        self._rng.shuffle(order)
        if last_speaker_id is not None:
            # Guaranteed to succeed since at least one other agent exists.
            while order[0] == last_speaker_id:
                self._rng.shuffle(order)
        return order

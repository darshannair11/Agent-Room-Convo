"""Top-level orchestration: config in, Transcript out.

This is the library entry point (run_room). It builds live agents from the
config, runs the compiled room graph, and assembles the Transcript with a full
config snapshot and metadata for reproducibility.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Optional

from harness.agent import Agent
from harness.config import AgentSpec, Question, RoomConfig
from harness.graph import build_room_graph
from harness.models.base import ModelAdapter
from harness.scheduler import SpeakingScheduler
from harness.transcript import Transcript


def _default_adapter_factory(spec: AgentSpec) -> ModelAdapter:
    """Build a live Claude adapter for a spec. Imported lazily so mock-only
    runs don't require the anthropic package or an API key."""
    from harness.models.claude import ClaudeAdapter

    return ClaudeAdapter(model=spec.model)


def _config_snapshot(config: RoomConfig) -> dict:
    return {
        "num_rounds": config.num_rounds,
        "seed": config.seed,
        "default_temperature": config.default_temperature,
        "agents": [asdict(a) for a in config.agents],
    }


def run_room(
    config: RoomConfig,
    question: Question,
    *,
    adapter_factory: Optional[Callable[[AgentSpec], ModelAdapter]] = None,
    run_id: Optional[str] = None,
) -> Transcript:
    """Run one room over one question and return the assembled Transcript.

    adapter_factory lets callers inject adapters (e.g. a MockAdapter in tests).
    Defaults to a live Claude adapter per agent.
    """
    factory = adapter_factory or _default_adapter_factory
    run_id = run_id or f"{question.id}-{uuid.uuid4().hex[:8]}"

    agents = [Agent(spec=spec, adapter=factory(spec)) for spec in config.agents]
    scheduler = SpeakingScheduler([a.agent_id for a in agents], seed=config.seed)

    graph = build_room_graph()
    started = datetime.now(timezone.utc)
    # Each agent turn is one super-step; give headroom beyond agents*rounds.
    recursion_limit = config.num_agents * config.num_rounds + 3 * config.num_rounds + 20

    final_state = graph.invoke(
        {
            "run_id": run_id,
            "question": question,
            "agents": agents,
            "config": config,
            "scheduler": scheduler,
        },
        config={"recursion_limit": recursion_limit},
    )
    finished = datetime.now(timezone.utc)

    turns = final_state["turns"]
    solo_pre = final_state["solo_pre"]
    solo_post = final_state["solo_post"]

    token_usage_total = _total_tokens(agents)
    metadata = {
        "num_rounds": config.num_rounds,
        "num_agents": config.num_agents,
        "seed": config.seed,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "malformed_turns": sum(1 for t in turns if t.malformed),
        "token_usage": token_usage_total,
    }

    return Transcript(
        run_id=run_id,
        config_snapshot=_config_snapshot(config),
        question=question,
        agents=[a.to_public_dict() for a in agents],
        solo_pre=solo_pre,
        turns=turns,
        solo_post=solo_post,
        metadata=metadata,
    )


def _total_tokens(agents: list[Agent]) -> dict:
    """Sum each adapter's cumulative usage tally across the room."""
    total = {"input_tokens": 0, "output_tokens": 0}
    for agent in agents:
        usage = getattr(agent.adapter, "usage_total", None)
        if usage:
            total["input_tokens"] += usage.get("input_tokens", 0)
            total["output_tokens"] += usage.get("output_tokens", 0)
    return total

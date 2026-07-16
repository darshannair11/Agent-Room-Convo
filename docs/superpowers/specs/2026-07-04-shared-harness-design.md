# Shared Room Harness — Design Spec

**Date:** 2026-07-04
**Owner:** Darshan
**Status:** Approved design, ready for implementation planning

See [README.md](../../../README.md) for research background and [WORKPLAN.md](../../../WORKPLAN.md) for how this fits the phase split. The shared harness is the prerequisite every phase depends on.

## Purpose

A Python harness that runs a multi-agent "chat room": several LLM agents discuss a question over a fixed number of rounds, with their stances, reasoning, and confidence captured in a structured, reproducible transcript. The harness is the shared foundation for all four research phases; it must be provider-agnostic, cheap to test, and produce a stable transcript format that downstream analysis (Mehal's pipeline) can consume without touching orchestration internals.

## Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python | LLM + data tooling ecosystem; matches Phase 4 / analysis work |
| Providers (first pass) | Claude only | Fastest to a working end-to-end harness; adapter interface makes adding others trivial |
| Orchestration | LangGraph | Room is naturally a state graph (round loop, conditional edges, solo bookends) |
| Turn order | Randomized per round, no back-to-back | Avoids positional/anchor bias; last speaker of round N can't speak first in round N+1 |
| Stopping condition | Fixed number of rounds — no convergence early-stop | Convergence stop would bias the phenomenon under study; better to observe full opinion trajectory |
| Rooms/rounds/agents | Configurable (`num_rounds`, `num_agents`) | No hard default baked in |
| Solo elicitation | Baked into harness core (pre + post discussion) | Cheap now; keeps one consistent transcript schema across phases (Phase 2 needs it) |
| Storage | JSON files, one per run | Human-readable, zero setup, easy for pandas-based analysis |
| Invocation | Both CLI and library API | Scriptable experiment runs + importable from notebooks/scripts |
| Config format | YAML/JSON config files | Reusable room configs + separate question-bank files, hand-authorable |

## Architecture & directory structure

```
harness/
  __init__.py
  config.py          # load & validate room configs + question banks (YAML/JSON)
  models/
    base.py          # ModelAdapter abstract interface + ModelResponse
    claude.py        # Claude implementation
  agent.py           # Agent dataclass: id, model adapter, system prompt, tool-access flag
  scheduler.py       # shuffled speaking-order generator (no-repeat-across-rounds constraint)
  graph.py           # LangGraph StateGraph definition — the room state machine
  solo.py            # solo elicitation node logic (pre/post discussion)
  transcript.py      # Turn / SoloResponse / Transcript data models + JSON (de)serialization
  storage.py         # persist transcripts to data/runs/<run_id>.json
  cli.py             # CLI entrypoint (python -m harness run ...)
  prompts/
    room_system_prompt.txt
    solo_system_prompt.txt
configs/
  rooms/example_room.yaml
  questions/phase1_objective.yaml
data/
  runs/              # output transcripts, gitignored
tests/
  test_scheduler.py
  test_graph.py
  test_transcript.py
```

Each module has one job. `scheduler.py` orders agents fairly; `models/` calls an LLM and returns a structured response; `graph.py` wires those into the round loop; `storage.py` reads/writes transcript JSON. Downstream phases consume `transcript.py`'s format and `models/base.py`'s interface without touching graph internals.

## Room state graph (LangGraph)

The room is a `StateGraph` with a shared `RoomState` threaded through nodes. Each node reads state, does one thing, returns a state update.

```
init_room  -> build agents, load question, round=0
   |
solo_pre   -> each agent answers alone -> baseline stance + confidence
   |
round_setup <----------------------------+   scheduler picks this round's order
   |                                      |
agent_turn -> next speaker reads          |
   |          transcript, produces turn   |
round_check -+-- more speakers -> agent_turn (next)
   |         +-- round done & round_idx<N -> round_setup
   |         +-- all rounds done -> solo_post
solo_post  -> each agent answers alone again -> post-discussion stance
   |
finalize   -> assemble Transcript, write JSON
```

`round_check` is the conditional edge. Since convergence early-stop was dropped, the only exit condition is the fixed round cap — deterministic. `solo_pre`/`solo_post` bracket the discussion so Phase 2's stance-stability comparison is `solo_post[i] vs solo_pre[i]`; Phase 1 can reuse `solo_pre` as the "single agent alone" condition.

**RoomState:**

```python
class RoomState(TypedDict):
    run_id: str
    question: Question
    agents: list[Agent]
    config: RoomConfig
    round_idx: int
    speaking_order: list[str]      # agent_ids for current round
    turn_cursor: int               # index into speaking_order
    last_speaker_id: str | None    # feeds no-back-to-back constraint
    turns: list[Turn]              # growing group transcript
    solo_pre: list[SoloResponse]
    solo_post: list[SoloResponse]
```

## Scheduler

Generates a speaking order per round with two constraints:
1. Order is randomized each round (no fixed 1,2,3 — avoids positional anchoring bias).
2. The agent who spoke last in round N may not speak first in round N+1 (no back-to-back turns across the round boundary).

A `seed` makes ordering reproducible and is stored in the transcript. This is the most logic-heavy unit and is unit-tested hardest.

## Data model & transcript JSON

Every model response is structured, not free text. Each group turn asks for stance, reasoning, and two confidence numbers (own + perceived peer, feeding the confidence-gap metric).

```python
@dataclass
class Turn:
    agent_id: str
    round_idx: int
    position_in_round: int         # 0-based slot in that round's order
    stance: str                    # short answer/position label (e.g. "A", "yes")
    reasoning: str                 # free-text justification
    self_confidence: int           # 0-100
    perceived_peer_confidence: int # 0-100
    raw_response: str              # full model output, for audit
    timestamp: str

@dataclass
class SoloResponse:                # pre & post discussion
    agent_id: str
    phase: str                     # "pre" | "post"
    stance: str
    reasoning: str
    self_confidence: int
    raw_response: str
    timestamp: str

@dataclass
class Transcript:
    run_id: str
    config_snapshot: dict          # full room config used (reproducibility)
    question: Question
    agents: list[dict]             # id, model, homo/hetero label, tool flag
    solo_pre: list[SoloResponse]
    turns: list[Turn]
    solo_post: list[SoloResponse]
    metadata: dict                 # num_rounds, seed, timings, model version, token usage
```

On-disk JSON (`data/runs/<run_id>.json`) mirrors this directly. Design choices:

- **`config_snapshot` + `seed` stored in every transcript** — full reproducibility; tells exactly which conditions (homo/hetero, tool on/off, question bank) produced a run. Critical for slicing results by condition.
- **`stance` short label, `reasoning` separate** — agreement-rate and flip-rate are trivial string comparisons on `stance`; `reasoning` feeds the LLM-judge flip classifier (reasoned update vs conformity flip).
- **`raw_response` always kept** — if structured parsing misfires, the ground truth of what the model said is preserved.

## Model adapter interface & prompting

The adapter interface is deliberately tiny — the graph only calls `generate`:

```python
class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, transcript_context: str,
                 question: Question) -> ModelResponse: ...

class ModelResponse:
    stance: str
    reasoning: str
    self_confidence: int
    perceived_peer_confidence: int | None   # None for solo turns
    raw_response: str
    token_usage: dict
```

`ClaudeAdapter` implements this against the Anthropic API. Adding GPT/Gemini later = one new file, zero graph changes. `Agent` holds a reference to its adapter, so a heterogeneous room is agents with different adapters; a homogeneous room is N agents sharing a model id with independent conversation state.

**Two neutral system prompts** (per PIMMUR anti-test-awareness guidance — neither mentions "experiment," "conformity," or "groupthink"):
- `room_system_prompt.txt` — frames the agent as a group-discussion participant working toward a good answer; specifies the required JSON output shape.
- `solo_system_prompt.txt` — same contract minus `perceived_peer_confidence`.

**Transcript context:** before each turn, the graph serializes the discussion so far as readable text (`Agent {id} (round N): {stance} — {reasoning}`). Every agent sees the same shared transcript — real multi-turn interaction, not a canned peer-summary (the other PIMMUR guardrail). Neutral agent ids avoid name/positional prestige.

## Error handling & operational concerns

- **Malformed structured output** — parse JSON; on failure, one retry with an explicit "return only valid JSON in this shape" nudge; on second failure, record the turn with `malformed: true` and preserved `raw_response`, then continue. One bad turn never kills an expensive run.
- **API failures (rate limit / timeout / 5xx)** — retry with exponential backoff; if still failing, abort the run but write a partial transcript with an `error` field in metadata.
- **Determinism** — `seed` controls scheduler shuffling and is stored. LLM calls aren't deterministic even at temperature 0, but ordering and config are fully reproducible — what matters for fair condition comparison.
- **Cost visibility** — token usage per call captured per turn and totaled in `metadata`.
- **Secrets** — `ANTHROPIC_API_KEY` from environment (`.env` via python-dotenv), never in config. `.env` and `data/runs/` gitignored.

## Testing strategy

- **`scheduler.py`** — pure logic, tested hardest: no agent speaks twice in a row across the round boundary; order varies across rounds; fixed seed reproduces orderings.
- **`graph.py`** — tested with a `MockAdapter` (canned structured responses, zero API calls) so the full state machine — round loop, solo bookends, transcript assembly — is verified fast and free.
- **`transcript.py` / `storage.py`** — round-trip test: object → JSON → object equality, so the on-disk contract can't silently drift.
- **`ClaudeAdapter`** — one lightweight live smoke test (marked/skippable) plus parsing unit tests against captured sample responses.

The whole harness is testable without spending money because the adapter interface lets a mock stand in for Claude everywhere except the one adapter smoke test.

## Out of scope (deferred to later phases)

- Web/search tooling (Phase 3, Mehal) — the `tool-access` flag exists on `Agent` but no search tool is wired in yet.
- Non-Claude adapters (added when heterogeneous multi-provider runs are needed).
- The analysis pipeline itself (Mehal) — this harness only produces the transcript JSON it consumes.
- Dynamic/agent-initiated turn-taking — fixed round-robin-with-shuffle only for now.

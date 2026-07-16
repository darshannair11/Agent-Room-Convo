# Room.md — How the Room Harness Works

This is the plain-English guide to the shared room harness. It explains what
the harness does, how to run it, and what comes out the other end — without
assuming you wrote the code. For the design reasoning behind these choices, see
the [design spec](docs/superpowers/specs/2026-07-04-shared-harness-design.md).

## What it does, in one paragraph

You put a few AI agents in a "room," give them a question, and let them discuss
it over a set number of rounds. Before and after the discussion, each agent also
answers the question **alone**. The harness records everything — what each agent
said, how confident it was, and how its answer changed — into a single JSON file
you can analyze later. That's it. The whole point is to watch how opinions move
when agents talk to each other.

## The flow of a single run

```
Everyone answers alone   ->   Group discussion   ->   Everyone answers alone again
   (the "before")             (several rounds)              (the "after")
```

1. **Before (solo):** each agent answers the question on its own, with no idea
   what the others think. This is the baseline.
2. **Discussion:** agents take turns speaking. Each agent sees everything said so
   far and then gives its own view — agreeing, pushing back, or changing its
   mind. This repeats for the configured number of rounds.
3. **After (solo):** each agent answers alone again. Comparing "before" vs
   "after" tells you whether the discussion actually changed anyone's mind.

Every time an agent speaks, it reports three things: its **answer**, its
**reasoning**, and a **confidence score** (0–100). During the discussion it also
estimates **how confident it thinks the others are** — that gap between "how sure
am I" and "how sure do the others seem" is a key signal for studying peer
pressure.

## How turn order works (and why it's shuffled)

Agents don't always speak in the same order. Each round the order is **reshuffled**,
so agent 1 isn't permanently the one who "anchors" the conversation by going
first. There's one rule on top of the shuffle: **whoever spoke last in a round
can't speak first in the next round** — that would give one agent two turns back
to back, which isn't fair. The shuffling is seeded, so the same config produces
the same ordering every time and runs are reproducible.

We deliberately **do not** stop early when agents agree. If we ended the room the
moment everyone converged, we'd bias the very thing we're trying to measure — so
every room runs its full set of rounds and we watch the whole trajectory.

## The pieces (what each file is for)

You don't need to read the code to use it, but here's the map:

- **`config.py`** — reads your setup files (who's in the room, what they discuss)
  and checks them for mistakes before anything runs.
- **`scheduler.py`** — decides the speaking order each round (the shuffle + the
  no-back-to-back rule).
- **`graph.py`** — the "director" of the room: runs the before/after solos and
  the discussion rounds in order. Built on **LangGraph**, which models the room
  as a flowchart of steps.
- **`models/`** — the part that actually talks to the AI. `claude.py` calls the
  real Claude API; `mock.py` is a fake stand-in used for testing (no API, no
  cost). Adding another provider later (GPT, Gemini) means adding one file here
  and nothing else changes.
- **`transcript.py` / `storage.py`** — the record of what happened and how it's
  saved to a JSON file.
- **`runner.py`** — ties it all together: `run_room(config, question)`.
- **`cli.py`** — the command-line way to run everything.

## How to run it

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Copy `.env.example` to `.env` and put your Anthropic key in it:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Try it without spending anything (mock mode)

This runs the whole machine using a fake model, so you can see it work before
using real API calls:

```bash
python -m harness run \
  --room configs/rooms/example_room.yaml \
  --questions configs/questions/phase1_objective.yaml \
  --mock
```

### 4. Run it for real

Drop the `--mock` flag and it uses Claude:

```bash
python -m harness run \
  --room configs/rooms/example_room.yaml \
  --questions configs/questions/phase1_objective.yaml
```

Transcripts land in `data/runs/` — one JSON file per question.

### Using it from Python instead

```python
from harness import load_room_config, load_question_bank, run_room
from harness.storage import save_transcript

config = load_room_config("configs/rooms/example_room.yaml")
questions = load_question_bank("configs/questions/phase1_objective.yaml")

transcript = run_room(config, questions[0])
save_transcript(transcript)
```

## Setting up a room

A **room config** describes who's in the room. The example is four copies of the
same model (a "homogeneous" room):

```yaml
num_rounds: 3
seed: 42
default_temperature: 0.7
agents:
  - model: claude-opus-4-8
    count: 4          # makes agent_1 .. agent_4
    temperature: 0.8
```

To make a **mixed** ("heterogeneous") room, list different agents instead of
using `count`:

```yaml
agents:
  - agent_id: agent_1
    model: claude-opus-4-8
  - agent_id: agent_2
    model: claude-sonnet-5
```

> **One thing to watch:** if you make a room of identical models, keep the
> temperature above 0. At temperature 0 they behave like identical twins and
> there's nothing to actually debate. The harness will warn you if you do this.

## Setting up questions

A **question bank** is a list of questions. For questions with a right answer,
give the choices and the correct one — this makes the analysis exact:

```yaml
questions:
  - id: arithmetic_easy
    text: "What is 1 + 1?"
    type: objective
    answer_choices: ["1", "2", "3", "4"]
    ground_truth: "2"
```

For opinion questions with no right answer, mark them `subjective` and leave
`ground_truth` empty. Giving `answer_choices` even here (e.g. Yes / No / It
depends) is recommended, because it makes "did they agree?" easy to measure.

## What you get out: the transcript

Each run writes one JSON file containing:

- **`solo_pre`** — everyone's "before" answers.
- **`turns`** — every turn of the discussion, in order, with answer, reasoning,
  and confidence.
- **`solo_post`** — everyone's "after" answers.
- **`config_snapshot`** — the exact setup used, so the run is reproducible.
- **`metadata`** — round/agent counts, timing, token usage (cost), and how many
  responses (if any) came back malformed.

Because it's plain JSON, the analysis pipeline can load it straight into pandas
and start comparing before/after, measuring agreement, and tracking who changed
their mind.

## What happens when things go wrong

- **An agent gives a messy, unparseable answer:** the harness asks it once more
  for a clean answer. If it still can't, that turn is flagged `malformed` (with
  the raw text kept) and the room continues — one bad answer never wastes a whole
  run.
- **The API hiccups (rate limits, timeouts):** it retries a few times with
  increasing waits before giving up.
- **Your config has a mistake:** it's caught upfront with a clear message, before
  any API calls are made.

## Testing

The whole harness can be tested without spending a cent, because the mock model
stands in for Claude everywhere:

```bash
pytest
```

There's also one real-API smoke test that only runs if your key is set:

```bash
pytest tests/test_claude_live.py
```

## What's intentionally left out (for now)

- **Web/search access** for agents — planned for Phase 3. The on/off switch
  already exists in the config (`tool_access`) but isn't wired to a search tool
  yet.
- **Other model providers** (GPT, Gemini) — the seam for them exists; only Claude
  is implemented today.
- **The analysis itself** — this harness produces the data; measuring and
  charting it is the separate analysis pipeline.

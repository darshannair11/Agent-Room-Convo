# Agent Room Convo

A multi-agent "chat room" harness for studying how LLM agents behave when they have to talk to each other: do they converge on correct answers, converge on *any* answer regardless of truth (groupthink), or stay stable under social pressure? Built to replicate and extend recent (2026) multi-agent debate / opinion-dynamics research, not just vibes-check it.

## The question

Put several LLM agents in a shared room and have them discuss a question over multiple turns. Then ask:

1. **Objective claims** (math, facts, logic) — does group discussion actually improve accuracy over a single agent, or just average opinions toward the majority?
2. **Groupthink / conformity** — do agents cave to a confident peer even when the peer is wrong? Does the answer depend on how peer opinions are presented?
3. **Subjective / ethical claims** (no ground truth) — do agents converge in stance anyway, and is that convergence "real" (shared reasoning) or just social mimicry?
4. **Tool/web access** — does giving agents search access change *what they conclude*, not just how accurate they are? Does it introduce new manipulation surface (agents citing planted/biased content)?
5. **Personality** — do models have stable, measurable personality traits (via validated psychometric instruments), and does that baseline predict how much a given model conforms in group discussion?

## Prior art this project builds on

- Homogeneous multi-agent debate has been shown to behave like a martingale — it provably cannot beat simple majority voting in expectation. Worth trying to replicate.
- Recent work decomposes stance changes during debate into genuine belief updates vs. social-conformity flips ("Not All Flips Are Conformity") — directly relevant to distinguishing real convergence from groupthink.
- A confidence gap (an agent's own confidence vs. its perceived confidence in peers) predicts whether it caves to group pressure; the *format* peer opinions are shown in also matters.
- "Stance homogenization": deliberation can make agents converge in opinion while *losing* factual accuracy — the core overthinking-vs-correctness tension this project is designed to measure.
- Search-enabled agents can be manipulated by content specifically engineered to get LLM endorsement — web access isn't a clean truth injection, it's its own variable with failure modes.
- TRAIT and similar benchmarks show models have measurable, somewhat persistent personality profiles (via Big Five / Short Dark Triad derived items), but personality readings can be unstable across scale, reasoning mode, and conversation history — repeated measurement is required, not a single reading.
- **PIMMUR** methodology warning: many "peer influence" studies aren't real multi-agent interaction — they're one agent reacting to a static summary of what peers supposedly said. Also, models that recognize they're being tested for conformity may adjust behavior toward what looks socially appropriate. Both pitfalls matter for this project's design (see below).

## Design principles (from the PIMMUR warning)

- Agents have real multi-turn conversations in a shared room — not a single agent reacting to a canned summary of "what others said."
- The room is framed as a normal discussion task in the system prompt. Agents are never told "this is a conformity/groupthink experiment," to avoid contaminating behavior with test-awareness.

## Planned phases

### Phase 1 — Objective claims: does debate help, or just average?
- Room of 4–5 agents, both homogeneous (same model, multiple instances) and heterogeneous (mixed model families) conditions.
- Questions: math/logic/factual, with clear ground truth, ranging from trivial to hard reasoning problems.
- Compare: single-agent accuracy vs. post-discussion group answer vs. simple majority vote of independently-obtained (non-interacting) answers.
- Falsifiable target: replicate or contradict the "debate can't beat majority vote" result.

### Phase 2 — Subjective/ethical claims: convergence vs. stability
- Same room setup, but ethics/values questions with no ground truth.
- Metrics:
  - Final-round agreement rate across agents.
  - Stance stability: does each agent give the same answer solo, before vs. after the room discussion?
  - Whether convergence tracks genuine reasoning exchange or just the most confident/dominant agent (log stated confidence and discussion "winners", not correctness, as the predictor).

### Phase 3 — Tool/web access ablation
- Repeat Phase 1 and 2 with web search enabled for all agents, some agents, or none, as a controlled variable.
- Log which sources get cited and whether agents that "won" a debate did so via better evidence or just more confident presentation.
- Watch for search-driven manipulation (agents citing content engineered to be endorsed).

### Phase 4 — Personality
- Before any chat-room run, profile each model with a validated psychometric instrument (TRAIT or a BFI subset), repeated multiple times to get a trait profile with variance — not a single point estimate.
- Test whether a model's baseline profile (e.g., high vs. low agreeableness) predicts its conformity rate in Phase 2. This connective question (personality baseline → conformity behavior) doesn't appear to be answered in existing literature.

## Metrics to track

- **Accuracy** (objective questions only): correct / incorrect vs. ground truth.
- **Agreement rate**: fraction of agents sharing the same final stance.
- **Flip rate**: how often an agent changes its stated position turn-over-turn.
- **Flip classification**: reasoned update vs. unexplained conformity flip (LLM-judge pass over transcripts, informed by the "Not All Flips Are Conformity" approach).
- **Confidence gap**: agent's own stated confidence vs. its estimate of peers' confidence, and whether that gap predicts caving.
- **Personality profile**: repeated-measures trait scores per model, with variance, from a validated instrument.

## Tech stack

- **Orchestration**: shared-state multi-turn graph (e.g. LangGraph or the Agents SDK) — a "room" is a shared transcript that each agent's turn appends to.
- **Models**: at least 2–3 model families (e.g. Claude, GPT, Gemini, or an open model like Qwen/Llama) for heterogeneous runs, plus multiple instances of a single model for homogeneous runs. Both conditions are required — homogeneous is the baseline.
- **Web tool**: a search API (e.g. Tavily, Exa, or a provider's native search), gated on/off per experimental condition.
- **Logging**: full transcripts plus per-turn confidence elicitation (each agent states a 0–100 confidence every turn), enabling confidence-gap and calibration-style analysis.
- **Analysis**: pandas for agreement/flip-rate tracking; an LLM-judge pass to classify stance changes as reasoned updates vs. conformity flips.

## Status

Early planning stage — no code yet. Next step: build the shared-room harness for Phase 1 (objective claims, homogeneous vs. heterogeneous, with/without discussion vs. majority vote).

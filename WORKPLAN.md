# Work Split & Timeline

Division of labor between Darshan and Mehal for the Agent Room Convo project. See [README.md](README.md) for full background and prior-art context.

## Owners

| Track | Owner |
|---|---|
| Shared room harness | Darshan |
| Phase 1: Objective claims | Darshan |
| Phase 4: Personality profiling | Darshan |
| Phase 2: Subjective/ethical claims | Mehal |
| Phase 3: Tool/web access ablation | Mehal |
| Analysis pipeline | Mehal |

## Shared harness (Darshan) — prerequisite for everything

- Define the "room" data structure: shared transcript, turn order, per-agent state (model, system prompt, tool access flag).
- Orchestration loop: each agent reads the transcript so far, produces a turn (stance + reasoning + confidence 0–100), appends to transcript.
- Model adapter layer: wrap 2–3 model APIs (e.g., Claude, GPT, Gemini/open model) behind one interface so swapping models is a config change.
- Homogeneous mode: same model, N independent instances, no shared memory between instances except the room transcript.
- Logging: persist full transcripts + per-turn confidence to disk/DB in a structured format (JSON) for later analysis.
- Neutral framing: system prompt that presents the room as a normal discussion task, no mention of "experiment," "conformity," or "test."

## Phase 1 — Objective claims (Darshan)

- Curate a question set with verifiable ground truth: start with easy factual/math questions, ramp to genuinely hard reasoning problems (so there's room to see debate help or hurt).
- Run three conditions per question: (a) single agent alone, (b) group discussion → final group answer, (c) N independent single-agent answers aggregated by majority vote (no interaction).
- Run each condition in both homogeneous and heterogeneous agent compositions.
- Score accuracy against ground truth for all three conditions.
- Compare: does discussion beat majority vote, tie it, or lose to it? (this is the replicate/contradict target)

## Phase 4 — Personality profiling (Darshan)

- Select a validated psychometric instrument (TRAIT benchmark or a Big Five Inventory subset) rather than ad hoc personality prompts.
- Run the instrument on each model multiple times (repeated measures) to get a trait profile with variance, not a single score — necessary because personality readings are known to be unstable across scale/reasoning mode/context.
- Produce a baseline personality profile per model (e.g., agreeableness, neuroticism, etc. with confidence intervals).
- Cross-reference: does a model's baseline trait score predict its conformity/flip rate from Phase 2? (e.g., do high-agreeableness models cave more often)

## Phase 2 — Subjective/ethical claims (Mehal)

- Curate a question set with no ground truth (ethical dilemmas, value judgments, open questions).
- Ask each agent solo, pre-discussion, to record a baseline stance + confidence.
- Run the group discussion.
- Ask each agent solo again, post-discussion, on the same or a reframed version of the question (to check stability, not just memorized repetition).
- Compute: final-round agreement rate across agents in the group; how often individual stances flipped pre- vs. post-discussion.
- Identify "discussion winners" — did convergence track the most confident/dominant agent rather than the best argument? (requires logging who spoke when and their confidence trajectory)

## Phase 3 — Tool/web access ablation (Mehal)

- Add a search tool (e.g., Tavily/Exa/native provider search) as an optional capability per agent.
- Rerun Phase 1 and Phase 2 question sets under three tool conditions: no agents have search, some agents have it, all agents have it.
- Log every query issued and every source cited per agent.
- Check whether "winning" a debate correlates with citing better evidence, or just with confident presentation regardless of source quality.
- Flag cases where an agent's stance shifted after citing a source — was the source reliable, or does it look like manipulated/biased content the agent got steered by?

## Analysis pipeline (Mehal) — cuts across all phases

- Metrics computation: accuracy, agreement rate, flip rate, confidence gap (self-confidence vs. perceived-peer-confidence).
- LLM-judge pass: classify each stance flip as "reasoned update" (agent explains new information/argument) vs. "unexplained conformity flip" (no new reasoning, just matches the group).
- Aggregation and reporting per phase, likely in pandas, with a way to slice by model, condition (homo/hetero, tool on/off), and question type.

## Timeline

| Week | Darshan | Mehal |
|---|---|---|
| 1 | Build shared harness + logging | Curate Phase 2 question set (ethical/subjective, no ground truth) |
| 2 | Phase 1 experiments (objective, homo/heterogeneous, all 3 conditions) | Run Phase 2 pre/post-discussion stance tracking, discussion-winner logging |
| 3 | Personality profiling (TRAIT/BFI, repeated measures per model) | Phase 3 — add search tool gating, rerun Phase 1 & 2 question sets under tool conditions |
| 4 | Cross-reference personality baseline vs. Phase 2 conformity rate; writeup | Build analysis pipeline (metrics + LLM-judge flip classifier); aggregate results across all phases; writeup |


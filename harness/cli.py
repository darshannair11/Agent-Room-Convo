"""Command-line entry point.

    python -m harness run --room ROOM.yaml --questions BANK.yaml [options]

Runs the given room over every question in the bank (or one via --question-id),
writing one JSON transcript per run into --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from harness.config import load_question_bank, load_room_config
from harness.models.mock import MockAdapter
from harness.runner import run_room
from harness.storage import save_transcript


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Run agent rooms.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a room over a question bank")
    run.add_argument("--room", required=True, help="room config (YAML/JSON)")
    run.add_argument("--questions", required=True, help="question bank (YAML/JSON)")
    run.add_argument("--out", default="data/runs", help="output dir for transcripts")
    run.add_argument("--question-id", help="run only this question id")
    run.add_argument(
        "--mock",
        action="store_true",
        help="use a mock model (no API calls) for a dry run",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    config = load_room_config(args.room)
    questions = load_question_bank(args.questions)
    if args.question_id:
        questions = [q for q in questions if q.id == args.question_id]
        if not questions:
            print(f"no question with id {args.question_id!r}", file=sys.stderr)
            return 1

    adapter_factory = (lambda spec: MockAdapter()) if args.mock else None

    for question in questions:
        transcript = run_room(config, question, adapter_factory=adapter_factory)
        path = save_transcript(transcript, args.out)
        tokens = transcript.metadata["token_usage"]
        malformed = transcript.metadata["malformed_turns"]
        print(
            f"[{question.id}] {len(transcript.turns)} turns, "
            f"{malformed} malformed, "
            f"{tokens['input_tokens']}+{tokens['output_tokens']} tokens -> {path}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

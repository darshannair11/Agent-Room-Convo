"""Persist and load transcripts as one JSON file per run."""

from __future__ import annotations

import json
from pathlib import Path

from harness.transcript import Transcript

DEFAULT_RUNS_DIR = Path("data/runs")


def save_transcript(transcript: Transcript, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Write a transcript to <runs_dir>/<run_id>.json and return the path."""
    d = Path(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{transcript.run_id}.json"
    path.write_text(json.dumps(transcript.to_dict(), indent=2))
    return path


def load_transcript(path: str | Path) -> Transcript:
    """Load a transcript back from JSON."""
    data = json.loads(Path(path).read_text())
    return Transcript.from_dict(data)

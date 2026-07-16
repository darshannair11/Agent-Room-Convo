"""Structured-output parsing, including the malformed fallback and stance
normalization onto answer choices."""

from harness.models.base import parse_response


def test_parses_clean_json():
    raw = '{"stance": "2", "reasoning": "obvious", "self_confidence": 90, "perceived_peer_confidence": 70}'
    r = parse_response(raw, expect_peer_confidence=True)
    assert r.stance == "2"
    assert r.self_confidence == 90
    assert r.perceived_peer_confidence == 70
    assert not r.malformed


def test_parses_fenced_json_with_prose():
    raw = 'Sure!\n```json\n{"stance": "Yes", "reasoning": "r", "self_confidence": 55}\n```\nHope that helps.'
    r = parse_response(raw, expect_peer_confidence=False)
    assert r.stance == "Yes"
    assert r.perceived_peer_confidence is None
    assert not r.malformed


def test_malformed_output_is_flagged_not_raised():
    r = parse_response("I refuse to answer in JSON.", expect_peer_confidence=True)
    assert r.malformed
    assert r.raw_response == "I refuse to answer in JSON."
    assert r.self_confidence == 50  # safe default


def test_confidence_is_clamped():
    raw = '{"stance": "A", "reasoning": "r", "self_confidence": 250}'
    r = parse_response(raw, expect_peer_confidence=False)
    assert r.self_confidence == 100


def test_stance_normalized_to_choice():
    raw = '{"stance": "the answer is canberra", "reasoning": "r", "self_confidence": 80}'
    r = parse_response(raw, expect_peer_confidence=False,
                       normalize_choices=["Sydney", "Canberra", "Perth"])
    assert r.stance == "Canberra"


def test_peer_confidence_none_when_not_expected():
    raw = '{"stance": "A", "reasoning": "r", "self_confidence": 80, "perceived_peer_confidence": 40}'
    r = parse_response(raw, expect_peer_confidence=False)
    assert r.perceived_peer_confidence is None

"""Scheduler is pure logic and encodes the fairness constraints, so it's
tested hardest: no back-to-back across rounds, order actually varies, and a
fixed seed reproduces the same orderings."""

from harness.scheduler import SpeakingScheduler

AGENTS = ["agent_1", "agent_2", "agent_3", "agent_4"]


def test_order_contains_all_agents_once():
    sched = SpeakingScheduler(AGENTS, seed=1)
    order = sched.order_for_round(last_speaker_id=None)
    assert sorted(order) == sorted(AGENTS)


def test_no_back_to_back_across_round_boundary():
    sched = SpeakingScheduler(AGENTS, seed=7)
    last = None
    for _ in range(200):
        order = sched.order_for_round(last_speaker_id=last)
        if last is not None:
            assert order[0] != last, "last speaker of prev round spoke first again"
        last = order[-1]


def test_order_varies_across_rounds():
    sched = SpeakingScheduler(AGENTS, seed=3)
    orders = {tuple(sched.order_for_round(None)) for _ in range(50)}
    assert len(orders) > 1, "expected the speaking order to vary"


def test_seed_is_reproducible():
    a = SpeakingScheduler(AGENTS, seed=99)
    b = SpeakingScheduler(AGENTS, seed=99)
    last_a = last_b = None
    for _ in range(30):
        oa = a.order_for_round(last_a)
        ob = b.order_for_round(last_b)
        assert oa == ob
        last_a, last_b = oa[-1], ob[-1]


def test_two_agents_still_satisfies_constraint():
    sched = SpeakingScheduler(["a", "b"], seed=5)
    last = None
    for _ in range(50):
        order = sched.order_for_round(last)
        if last is not None:
            assert order[0] != last
        last = order[-1]

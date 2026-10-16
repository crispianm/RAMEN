"""LOOK-01: consume_with_delay pairs every delayed prediction with its own row."""

from __future__ import annotations

import pytest

from evaluation.lookahead_eval import consume_with_delay


class FakeRunner:
    """Emits ``("pred", frame_index_of_t_minus_L)`` at step t, flushes the tail."""

    def __init__(self, L):
        self.L = L
        self.reset()

    def reset(self):
        self.buf = []

    def step(self, frame):
        self.buf.append(frame)
        if len(self.buf) - 1 < self.L:
            return None
        return ("pred", self.buf[-1 - self.L])

    def flush(self):
        n = len(self.buf)
        return [("pred", self.buf[n - 1 - d]) for d in range(min(self.L, n) - 1, -1, -1)]


@pytest.mark.parametrize("L,n", [(0, 5), (4, 10), (4, 3), (8, 1), (2, 2)])
def test_every_row_scored_once_in_order_against_its_own_frame(L, n):
    runner = FakeRunner(L)
    scored = []
    rows = [(100 + i, f"frame{i}") for i in range(n)]
    step_ms, flush_ms = consume_with_delay(runner.step, runner.flush, runner.reset, rows,
                                           lambda pos, rid, pred: scored.append((pos, rid, pred)))
    assert [s[0] for s in scored] == list(range(n))
    assert [s[1] for s in scored] == [100 + i for i in range(n)]
    assert [s[2] for s in scored] == [("pred", f"frame{i}") for i in range(n)]
    assert len(step_ms) == n and len(flush_ms) == 1


def test_timed_wrapper_separates_step_and_flush_costs():
    runner = FakeRunner(2)
    calls = []

    def timed(fn):
        calls.append(1)
        return fn(), float(len(calls))

    step_ms, flush_ms = consume_with_delay(runner.step, runner.flush, runner.reset,
                                           [(0, "a"), (1, "b"), (2, "c")], lambda *a: None, timed)
    assert step_ms == [1.0, 2.0, 3.0] and flush_ms == [4.0]


def test_runner_that_never_flushes_is_an_error():
    runner = FakeRunner(3)
    with pytest.raises(RuntimeError, match="never received"):
        consume_with_delay(runner.step, lambda: [], runner.reset, [(0, "a"), (1, "b")], lambda *a: None)

"""LOOK-01: delayed-emission bookkeeping for bounded-lookahead evaluation.

The frozen scorer (scripts/eval_streaming_npz.py) assumes one prediction per consumed
frame, returned immediately. A lookahead runner returns the prediction for frame t-L
when frame t is consumed and the last L predictions on ``flush()``. This module pairs
every prediction with the row it belongs to, independent of any model, so the pairing
rule is unit-testable (tests/test_lookahead_eval.py).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def consume_with_delay(step: Callable[[Any], Any], flush: Callable[[], list], reset: Callable[[], None],
                       rows: Iterable[tuple[int, Any]],
                       score: Callable[[int, int, Any], None],
                       timed: Callable[[Callable[[], Any]], tuple[Any, float]] | None = None,
                       ) -> tuple[list[float], list[float]]:
    """Feed one video's ``rows`` (``(row_id, frame)`` in chronological order) through a
    delayed runner and call ``score(position, row_id, prediction)`` exactly once per
    row, in order. ``timed(fn)`` wraps a call and returns ``(result, elapsed_ms)``; the
    per-step and flush timings are returned separately so the flush cost (which a
    deployment pays only at stream end) never enters the per-frame figure.
    """
    if timed is None:
        def timed(fn):
            return fn(), 0.0
    reset()
    pending: list[tuple[int, int]] = []
    step_ms: list[float] = []
    flush_ms: list[float] = []
    for position, (row_id, frame) in enumerate(rows):
        pending.append((position, row_id))
        pred, ms = timed(lambda: step(frame))
        step_ms.append(ms)
        if pred is not None:
            pos, rid = pending.pop(0)
            score(pos, rid, pred)
    preds, ms = timed(flush)
    flush_ms.append(ms)
    for pred in preds:
        if not pending:
            raise RuntimeError("runner flushed more predictions than frames were consumed")
        pos, rid = pending.pop(0)
        score(pos, rid, pred)
    if pending:
        raise RuntimeError(f"{len(pending)} frame(s) never received a prediction "
                           f"(runner.flush must return the last min(L, n) predictions)")
    return step_ms, flush_ms

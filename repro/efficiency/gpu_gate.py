"""GPU-memory gate for the N concurrent scoring workers (eff_b fix 2). Scheduling only: no numerics, no data, no timing.
A clip scored by a memory-heavy leg (manual attention, no SDPA, on KITTI: ~16 GiB peak in fp32, more than N=4 workers can share with the 2.9 GB desktop) takes the gate
EXCLUSIVELY; every other clip takes it SHARED. Writer-preference two-lock protocol so heavy clips cannot starve. Inert unless EFF_GATE_DIR is set (control / single-worker runs)."""
from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path


@contextlib.contextmanager
def gate(heavy: bool):
    d = os.environ.get("EFF_GATE_DIR")
    if not d:
        yield
        return
    Path(d).mkdir(parents=True, exist_ok=True)
    fb = open(Path(d) / "intent.lock", "a+")
    fa = open(Path(d) / "gpu.lock", "a+")
    try:
        fcntl.flock(fb, fcntl.LOCK_EX)                 # intent lock: serialises arrival; blocks new readers while a writer waits
        fcntl.flock(fa, fcntl.LOCK_EX if heavy else fcntl.LOCK_SH)
        fcntl.flock(fb, fcntl.LOCK_UN)                 # the gate itself now orders writers/readers
        yield
    finally:
        fcntl.flock(fa, fcntl.LOCK_UN)
        fcntl.flock(fb, fcntl.LOCK_UN)
        fa.close()
        fb.close()

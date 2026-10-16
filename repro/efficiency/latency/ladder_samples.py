"""Ladder harness + all per-pass samples (generate_results.py tab:latency / tab:latency_eff).

run_ladder.py (the Auditor-checked ladder, committed >= c524643) is imported UNMODIFIED. This wrapper only records what that file times but does not keep:
every timed pass of every row (`time_stages` already returns the lists; run_ladder keeps just their p50/p95). After each row it rewrites rows/<id>.json with
    "samples": {"window1_total_ms": [30 floats], "window2_total_ms": [30 floats], "protocol": ...}
(ms per WINDOW; window 1 / 32 = ms per frame, window 2 / 22 = ms per new frame, exactly the harness's own division). No timing code is touched, no number changes.
Usage is run_ladder.py's:  ladder_samples.py --stacks S0 --skip-standalone --out DIR [--lock FILE] [--warmup 5 --iters 30]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ladder as RL  # noqa: E402

_LAST: dict = {}
_orig_time_stages = RL.time_stages
_orig_run_item = RL.Session.run_item


def _time_stages(torch, stages, windows, warmup, iters):
    res, wall = _orig_time_stages(torch, stages, windows, warmup, iters)
    _LAST["res"] = res
    return res, wall


def _run_item(self, item):
    _LAST.clear()
    _orig_run_item(self, item)
    res = _LAST["res"]
    rec = self.rows[-1]                                          # the very dict run_item just wrote
    assert rec["id"] == item["id"]
    rec["samples"] = {"window1_total_ms": [float(v) for v in res[0]["total"]], "window2_total_ms": [float(v) for v in res[1]["total"]],
                      "n_samples": len(res[0]["total"]), "unit": "ms per window; window 1 / 32 = ms per frame, window 2 / 22 = ms per new frame",
                      "recorded_by": "repro/efficiency/latency/ladder_samples.py (run_ladder.py unmodified)"}
    self.common.write_json(self.out / "rows" / f"{item['id']}.json", rec)


RL.time_stages = _time_stages
RL.Session.run_item = _run_item

if __name__ == "__main__":
    sys.exit(RL.main())

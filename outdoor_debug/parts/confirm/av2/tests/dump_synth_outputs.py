"""Refusal-only proof helper: TESTS_DIR=<tests dir> DUMP=<out.json> python -m pytest -q dump_synth_outputs.py (run once at the base commit, once at the new one), then diff_synth_outputs.py before.json after.json."""
import json, os, sys
sys.path.insert(0, os.environ["TESTS_DIR"])
from test_av2_e2e import world, runs   # noqa: F401  (fixtures)
import av2_scorer as S

def test_dump(runs):
    out = {"runs": runs, "compare": S.compare(runs["H300"], runs["TS2"], stage=1)} if False else {"runs": runs}
    json.dump(out, open(os.environ["DUMP"], "w"), default=float)

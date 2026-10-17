#!/usr/bin/env python3
"""Stage-1 scoring runner. Reads the LITERAL expected scorer sha from RUNBOOK_STAGE1.md (line `EXPECT_SCORER_SHA = <64 hex>`), and runs av2_scorer.py with it as --expect-scorer-sha.
The runner NEVER hashes the scorer itself (the pin must come from outside the file); a placeholder / malformed literal is refused before anything runs; a literal that does not match the
scorer file makes the scorer itself refuse.
usage: run_score.py RUNBOOK_STAGE1.md -- <av2_scorer.py arguments>"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_expected_sha(runbook_text: str) -> str:
    m = re.findall(r"^EXPECT_SCORER_SHA\s*=\s*(\S+)\s*$", runbook_text, re.M)
    if len(m) != 1:
        raise SystemExit(f"runbook must contain exactly one 'EXPECT_SCORER_SHA = <sha>' line (found {len(m)})")
    if not re.fullmatch(r"[0-9a-f]{64}", m[0]):
        raise SystemExit(f"EXPECT_SCORER_SHA {m[0]!r} is not a 64-hex literal (placeholder not filled in?): refusing to run")
    return m[0]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[1] != "--":
        raise SystemExit(__doc__)
    sha = read_expected_sha(Path(argv[0]).read_text())
    cmd = [sys.executable, "-I", str(HERE / "av2_scorer.py"), *argv[2:], "--expect-scorer-sha", sha]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

"""check_ddad_output.py --scorer-sha: default unchanged (A3 80916c4f); an explicit sha is enforced; forced fails on synthetic meta derived from the real ours-378 row."""
import copy, json, subprocess, sys
from pathlib import Path
import pytest
LR = Path(__file__).resolve().parents[1] / "outdoor_debug" / "parts" / "LR_DEV"
REF = json.load(open(LR / "ddad_ref2" / "ddad_scores.json")); A4 = "8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503"; CK = REF["meta"]["ckpt"]["sha256"]

def run(tmp_path, d, *extra):
    p = tmp_path / "s.json"; p.write_text(json.dumps(d)); r = tmp_path / "r.json"; r.write_text(json.dumps(REF))
    out = subprocess.run([sys.executable, "-I", str(LR / "check_ddad_output.py"), str(p), "--expect-ckpt-sha", CK, "--reference", str(r), *extra], capture_output=True, text=True)
    return out.returncode, out.stdout

def test_default_accepts_a3_scorer_sha(tmp_path):
    rc, o = run(tmp_path, copy.deepcopy(REF)); assert rc == 0, o; assert "PASS scorer sha 80916c4f" in o

def test_default_rejects_a4_sha(tmp_path):
    d = copy.deepcopy(REF); d["meta"]["ddad_scorer_sha256"] = A4
    rc, o = run(tmp_path, d); assert rc == 1 and "FAIL scorer sha 80916c4f" in o

def test_explicit_a4_sha_accepted_and_enforced(tmp_path):
    d = copy.deepcopy(REF); d["meta"]["ddad_scorer_sha256"] = A4
    rc, o = run(tmp_path, d, "--scorer-sha", A4); assert rc == 0, o; assert "PASS scorer sha 8e71511b" in o
    d["meta"]["ddad_scorer_sha256"] = "0" * 64
    rc, o = run(tmp_path, d, "--scorer-sha", A4); assert rc == 1 and "FAIL scorer sha 8e71511b" in o
    rc, o = run(tmp_path, copy.deepcopy(REF), "--scorer-sha", A4); assert rc == 1       # the A3 row fails an A4 pin

"""Executor: shells out to the vendored scorer per unit, resume by (existing result file with matching limit_clips + bench_version)."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import config as C
from .plan import ENV, VENDORED, Unit
from .util import REPO, sha256_file, sha_matches

BENCH_VERSION = 3


def finished(u: Unit, out: Path, smoke: bool) -> tuple[bool, str]:
    """(done, reason). A result is only 'finished' if it carries this run's limit_clips and bench_version; a mismatch raises (never resume a smoke as the full read)."""
    p = u.out_path(out)
    if u.group and not all(q.exists() for q in u.all_paths(out)):
        return False, "grid pass incomplete"
    if u.derive_from:
        bp = Path(out) / u.rel_dir() / u.filename().replace(f"__{u.dataset}__", f"__{u.derive_from['base']}__")
        return bp.exists(), "derived from base"
    if not p.exists():
        return False, "missing"
    try:
        art = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False, "corrupt json (rerun)"
    want = 1 if smoke else None
    if u.kind in ("bench",) and (art.get("limit_clips") != want or ("bench_version" in art and art["bench_version"] != BENCH_VERSION)):
        raise RuntimeError(f"REFUSED to resume {p}: limit_clips={art.get('limit_clips')!r} bench_version={art.get('bench_version')!r}; this run wants limit_clips={want!r}, bench_version={BENCH_VERSION}. Use a clean --out.")
    return True, "resumed"


def check_inputs(cfg: dict, units: list[Unit], hash_files: bool = True) -> tuple[dict, list[str]]:
    """Checkpoint existence + sha256 against the config pins, for EVERY model any planned unit uses (finished or not; FT n47/n48 included on the host where they live).
    Returns ({model_id: {path, sha256, ok, status}}, problems). status = ok | missing | MISMATCH; a MISMATCH is never skippable (see mismatches())."""
    info, problems = {}, []
    used = {u.model_id for u in units}
    for mid in sorted(used):
        ck = (cfg["models"][mid].get("ckpt") or {})
        if not ck:
            continue
        path = C.resolve_path(cfg, ck["path"])
        rec = {"path": str(path), "pinned_sha256": ck["sha256"]}
        if not path.is_file():
            problems.append(f"{mid}: checkpoint missing at {path} (download: {ck.get('download_url') or ck.get('download') or 'not given in config'})")
            rec["ok"] = False
            rec["status"] = "missing"
        elif hash_files:
            rec["sha256"] = sha256_file(path)
            rec["ok"] = sha_matches(rec["sha256"], ck["sha256"])
            rec["status"] = "ok" if rec["ok"] else "MISMATCH"
            if not rec["ok"]:
                problems.append(f"{mid}: sha256 {rec['sha256'][:12]} != pinned {ck['sha256'][:12]} ({path})")
        info[mid] = rec
    return info, problems


def mismatches(info: dict) -> list[str]:
    """Models whose checkpoint file exists but hashes differently from the pin: the run aborts (no smoke / resume / skip exemption)."""
    return [f"{mid}: sha256 {r.get('sha256', '')[:12]} != pinned {r['pinned_sha256'][:12]} ({r['path']})" for mid, r in info.items() if r.get("status") == "MISMATCH"]


def check_result_pins(cfg: dict, u: Unit, art: dict) -> list[str]:
    d = cfg["datasets"].get(u.dataset, {})
    out = []
    for k in ("manifest_sha256", "llvar_draws_sha256"):
        if k in d and art.get(k) is not None and not sha_matches(art[k], d[k]):
            out.append(f"{u.label()}: {k} {str(art[k])[:10]} != config {d[k][:10]}")
    return out


def verify_vendored() -> list[str]:
    """The vendored wrapper scripts must be byte-identical to the Auditor-cleared originals (repro/vendored/VENDORED.json)."""
    vj = json.loads((VENDORED / "VENDORED.json").read_text())["files"]
    return [f"vendored/{n}: sha256 {sha256_file(VENDORED / n)[:12]} != {rec['sha256'][:12]}" for n, rec in vj.items() if sha256_file(VENDORED / n) != rec["sha256"]]


SCORER_PATHS = ("bench_joint.py", "metrics.py", "metrics_temporal.py", "evaluation", "scripts/dn01_headline.py", "scripts/dn01_heldout_eval.py", "scripts/dn01_bvi_fnoise.py")


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)


def ensure_scorer_clone(dest: Path, commit: str = C.DEFAULT_SCORER_COMMIT, repo: Path = REPO) -> Path:
    """A pristine checkout of the scorer commit at `dest` (HEAD == commit, clean), for the vendored wrappers and the figure renderers, which assert exactly that.
    Refuses unless this repo's scorer files are byte-identical to `commit` (nothing in the scorer is ever edited or patched; the clone is `git clone --shared`, no copy of history)."""
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise RuntimeError(f"scorer commit {commit[:8]} is not in this repository's history")
    d = _git(repo, "diff", "--stat", commit, "--", *SCORER_PATHS)
    if d.stdout.strip():
        raise RuntimeError(f"scorer files differ from {commit[:8]} (the scorer must be vendored unchanged):\n{d.stdout}")
    dest = Path(dest)
    if dest.exists():
        head, por = _git(dest, "rev-parse", "HEAD").stdout.strip(), _git(dest, "status", "--porcelain").stdout.strip()
        if head == commit and not por:
            return dest
        raise RuntimeError(f"existing scorer clone {dest} is not at {commit[:8]} or not clean; remove it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for cmd in (["clone", "--shared", "--no-checkout", str(repo), str(dest)],):
        r = subprocess.run(["git", *cmd], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
    r = _git(dest, "checkout", "--detach", commit)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return dest


def unit_env(u: Unit, out: Path) -> dict:
    env = {**os.environ, **ENV, "RAMEN_TEST_ACCESS_LOG": os.environ.get("RAMEN_TEST_ACCESS_LOG", str(Path(out) / "TEST_access_log.jsonl"))}
    if u.tag_backend == "fast":
        env["RAMEN_AGF_BACKEND"] = "gpu"                      # the scorer's own device TAG backend (bit-identical per WP24); repro/tag_fast.py is not wired into the scorer
    return env


def run_unit(u: Unit, out: Path, smoke: bool, allow_test: bool, bvi_root: str | None, log_dir: Path, scorer_clone: Path | None = None) -> str:
    if u.host == "mac":
        return "not_measured"                                  # no executor for host mac, by design: nothing is launched
    cmd = u.command(out, smoke=smoke, allow_test=allow_test, bvi_root=bvi_root, scorer_clone=scorer_clone)
    if cmd is None:
        return "derived"
    Path(out, u.rel_dir()).mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / (f"latency_ladder_{u.harness['stack']}.log" if u.harness else u.filename().replace(".json", ".log"))
    env = unit_env(u, out)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(u.cwd(scorer_clone)), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and u.out_path(out).exists()
    return f"ok({time.time() - t0:.0f}s)" if ok else f"failed(rc={rc}, log={log})"

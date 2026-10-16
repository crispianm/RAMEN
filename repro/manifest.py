"""MANIFEST.json writer + git/env capture."""
from __future__ import annotations

import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from .util import REPO, jdump, sha256_file

PACKAGES = ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "pyyaml", "scipy", "pillow", "imageio", "imageio-ffmpeg", "matplotlib", "pytest")


def git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def git_state(out_dir: Path | None = None) -> dict:
    """{commit, dirty, dirty_files}. Untracked files under out_dir (the results folder) never count."""
    porc = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    if out_dir is not None:
        try:
            rel = str(Path(out_dir).resolve().relative_to(REPO.resolve()))
            porc = [ln for ln in porc if not ln[3:].startswith(rel)]
        except ValueError:
            pass
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(porc), "dirty_files": porc[:50]}


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def gpu_name() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def hash_tree(out: Path, skip=("MANIFEST.json",)) -> dict:
    out = Path(out)
    return {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in skip}


def frozen_state(cfg_path: Path) -> dict:
    """The freeze is the commit itself: the config must be tracked and unmodified at HEAD (so MANIFEST.git.commit + config sha256 identify the frozen bytes), and its header must say
    'FROZEN (this commit)' and not 'DRAFT'. Returns {header_problems, tracked, modified, commit, ok}."""
    from . import config as C
    cfg_path = Path(cfg_path).resolve()
    hp = C.frozen_header_problems(cfg_path.read_text())
    try:
        rel = str(cfg_path.relative_to(REPO.resolve()))
    except ValueError:
        return {"header_problems": hp, "tracked": False, "modified": None, "commit": git("rev-parse", "HEAD"), "ok": False, "why": "config is outside the repository"}
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True).returncode == 0
    modified = bool(git("status", "--porcelain", "--", rel))
    return {"header_problems": hp, "tracked": tracked, "modified": modified, "commit": git("rev-parse", "HEAD"), "config": rel, "ok": not hp and tracked and not modified}


def bvi_file_shas(root: Path, registration: Path) -> dict:
    """sha256 of every BVI-RLV frame the F_noise rows read (included sequences, the registration's listed frames, low + normal): {'<scene>/<kind>_light_<lv>/<file>': sha}."""
    import json
    reg = json.loads(Path(registration).read_text())
    out = {}
    for key, rec in sorted(reg["sequences"].items()):
        if not rec.get("included"):
            continue
        scene, lv = key.rsplit("__", 1)
        for kind in ("low", "normal"):
            for fn in rec["frames"][kind]:
                rel = f"{scene}/{kind}_light_{lv}/{fn}"
                out[rel] = sha256_file(Path(root) / rel)
    return out


def write_manifest(out: Path, *, cfg_path: Path, items: list[str], units: list, unit_status: dict, checkpoints: dict, clip_lists: dict,
                   git_info: dict, started: float, smoke: bool, allow_test: bool, argv: list[str], extra: dict | None = None) -> Path:
    out = Path(out)
    man = {
        "schema": 1, "argv": argv, "smoke": smoke, "allow_test": allow_test,
        "git": git_info, "config": {"path": str(cfg_path), "sha256": sha256_file(cfg_path)},
        "python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(), "gpu": gpu_name(),
        "packages": package_versions(),
        "items": items, "units": {u.label() + f"|{u.kind}": unit_status.get(u.key, "not_run") for u in units},
        "checkpoints": checkpoints, "clip_lists": clip_lists,
        "wall_time_s": round(time.time() - started, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_sha256": hash_tree(out),
        **(extra or {}),
    }
    p = out / "MANIFEST.json"
    jdump(man, p)
    return p

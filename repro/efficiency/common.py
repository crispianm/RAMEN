"""Shared constants and helpers for the efficiency cache builder / launcher / worker. Pure stdlib + numpy (no torch import here), so tests and --dry-run run anywhere."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
SCORER_CLONE = Path(os.environ.get("EFF_SCORER_CLONE", HOME / ".cache" / "ramen_scorer" / "368b5c53"))
SCORER_COMMIT = "368b5c53b37ebd22f5d44ce0c9d514122efc88da"
VDA_REPO = Path("/mnt/DATA/DegradedDepth/competition/Video-Depth-Anything")
VDA_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
FOCUSMAMBA_VENV = Path(os.environ.get("EFF_FOCUSMAMBA_VENV", "FocusMamba/.venv"))
FOCUSMAMBA_FREEZE_SHA = "8475f8800b8e8a31f2c06eb8d522b0c2c4a61df5c30e5f360f2f90026074ff82"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/mnt/DATA/DegradedDepth/claude_outputs/scratch/paper_repro")
DEFAULT_CACHE = SCRATCH / "effic_input_cache"
DEFAULT_OUT = SCRATCH / "effic_runs"

RAMEN_CKPT = Path(os.environ.get("RAMEN_CKPT", str(REPO_ROOT / "checkpoints_release" / "ramen.pt")))
RAMEN_CKPT_SHA = "898c50da69841b5c6710a026ffd42e570fba08a349f7485ee7e35747a22f46dd"
# RAMEN_DISPLAY: False = time the depth path only (spec ours:<ramen.pt>, which needs a scorer whose ours: loader ignores the display head, i.e. the pinned
# eval clone); True = depth + display head (spec l15d:<ramen.pt>, needs this repo's bench_joint). One switch, nothing else changed.
RAMEN_DISPLAY = os.environ.get("RAMEN_DISPLAY", "0") == "1"


def ramen_spec(display: bool | None = None) -> str:
    display = RAMEN_DISPLAY if display is None else display
    return f"{'l15d' if display else 'ours'}:{RAMEN_CKPT}"

# DEV set of (b): dataset -> expected number of DEV clips (a count mismatch is a FAIL, never silently accepted)
DEV_DATASETS = {"kitti": 7, "bonn": 13, "scannetpp": 31}
CELLS = ("llvar", "st_llvar")
TAG_VARIANTS = ("none", "agf0.2t0.06", "agf0.5t0.12")        # raw + both TAG settings; one model pass, TAG applied to the raw depth (--grid semantics)

# pinned-path numerics flags (R-TF32): PyTorch defaults. Nothing may change them.
PINNED_NUMERICS = {"cudnn_allow_tf32": True, "matmul_allow_tf32": False, "float32_matmul_precision": "highest"}

TOGGLE_FIELDS = ("attn_backend", "precision", "bn_folded", "cudnn_benchmark", "channels_last", "compiled", "refine_hook", "input_staging",
                 "agf_backend", "venv", "torch", "cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def array_sha256(a) -> str:
    """sha256 over dtype, shape and the C-contiguous bytes of an array (native dtype, no cast). Accepts numpy arrays or CPU torch tensors."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def safe_key(key: str) -> str:
    """Filesystem-safe, collision-free file stem for a clip key."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key) + "-" + sha256_bytes(key.encode())[:8]


def cache_path(root: Path, ds: str, cell: str, key: str) -> Path:
    return Path(root) / ds / cell / (safe_key(key) + ".npy")


def run(cmd, cwd=None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


def git_record(path) -> dict:
    return {"path": str(path), "HEAD": run(["git", "rev-parse", "HEAD"], path).strip(), "status": run(["git", "status", "--porcelain"], path).strip()}


def freeze_sha(venv: Path = FOCUSMAMBA_VENV) -> str:
    """sha256 of `uv pip freeze` of the shared venv (read-only query; the venv is never modified)."""
    uv = os.environ.get("UV", "uv")
    out = subprocess.run([uv, "pip", "freeze", "--python", str(Path(venv) / "bin" / "python")], capture_output=True, text=True).stdout
    return sha256_bytes(out.encode())


def untouched_record(label: str) -> dict:
    """BEFORE / AFTER record: eval clone HEAD+status, VDA HEAD+status, FocusMamba uv-freeze sha (equal before/after = untouched)."""
    import datetime
    return {"label": label, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_clone": git_record(SCORER_CLONE), "vda": git_record(VDA_REPO),
            "focusmamba_freeze_sha256": freeze_sha(), "expected_freeze_sha256": FOCUSMAMBA_FREEZE_SHA}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def write_sha256sums(root: Path, name: str = "SHA256SUMS") -> Path:
    """SHA256SUMS over every file below root (relative paths), excluding the sums file itself."""
    root = Path(root)
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in sorted(root.rglob("*")) if p.is_file() and p.name != name and not p.name.endswith(".tmp")]
    out = root / name
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


def import_scorer(clone: Path = SCORER_CLONE):
    """Import the pinned scorer's bench_joint from the clone (no copy). Refuses unless the clone is at the pinned commit; never writes bytecode into the clone."""
    sys.dont_write_bytecode = True
    head = run(["git", "rev-parse", "HEAD"], clone).strip()
    if head != SCORER_COMMIT:
        raise RuntimeError(f"scorer clone {clone} is at {head[:8]}, not {SCORER_COMMIT[:8]}")
    if str(clone) not in sys.path:
        sys.path.insert(0, str(clone))
    os.chdir(clone)
    import bench_joint as bj  # noqa: PLC0415
    return bj

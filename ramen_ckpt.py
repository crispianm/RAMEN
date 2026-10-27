"""Default RAMEN checkpoint resolution: --ckpt > RAMEN_CKPT env > <repo>/checkpoints_release/ramen.pt."""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RELATIVE = Path("checkpoints_release") / "ramen.pt"


def default_ckpt() -> str:
    return os.environ.get("RAMEN_CKPT") or str(REPO_ROOT / DEFAULT_RELATIVE)


def require_ckpt(path: str) -> str:
    """Return ``path`` if the file exists; else exit with a message saying where to download it (links are TODO)."""
    if not Path(path).is_file():
        raise SystemExit(f"[ramen] checkpoint not found: {path}\n"
                         "  Download ramen.pt / ramen_l.pt and put it at <repo>/checkpoints_release/ramen.pt, or pass --ckpt PATH, or set RAMEN_CKPT=PATH.\n"
                         "  Download links: TODO (none exist yet; they are announced in README.md and CHECKPOINTS.md at release).")
    return path

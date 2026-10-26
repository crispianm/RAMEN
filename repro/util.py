"""Small helpers shared by the repro harness (no torch import: everything here runs on a CPU-only box)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def item_stem(item_id: str) -> str:
    """'tab:static' -> 'tab_static' (file stem and renderer suffix)."""
    return item_id.replace(":", "_").replace("-", "_")


def slug(spec: str) -> str:
    """bench_joint._slug, copied (importing bench_joint pulls torch/cv2). Kept byte-identical in behaviour; tests pin it."""
    return spec.replace("ours_up:", "oursup-").replace("ours:", "ours-").replace("l15d:", "l15d-").replace("/", "_").replace(".pt", "")


def sha256_file(p, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha_matches(full: str, want: str) -> bool:
    want = want.lower()
    return full.lower() == want or (len(want) >= 8 and full.lower().startswith(want))


def nanmean(xs) -> float:
    v = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def expand(path: str, env: dict | None = None) -> str:
    env = os.environ if env is None else env
    out = path
    for k, v in env.items():
        out = out.replace("${" + k + "}", v)
    return out


def jdump(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=lambda o: None if o != o else str(o)))

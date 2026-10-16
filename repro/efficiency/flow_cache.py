"""Clean-flow (RAFT) cache for tOF (Amendment 1 (ii)): ONLY used if cached flow == recomputed flow bitwise on every clip (``verify-flow``); otherwise flow is recomputed per leg.
GPU work (RAFT-large, 12 updates, the pinned TOF class): not part of the CPU-only cache verify. ``CachedFlowTOF`` is a drop-in subclass of the scorer's TOF whose
``flows`` returns the cached flow when the input tensor's content sha equals the cached clean clip's sha, else recomputes (so a wrong hit is impossible by construction)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro.efficiency import common as C


def flow_path(cache: Path, ds: str, key: str) -> Path:
    return Path(cache) / "flow" / ds / (C.safe_key(key) + ".npy")


def main_flow(a) -> int:
    import torch
    bj = C.import_scorer()
    from evaluation.temporal_metrics import TOF
    from repro.efficiency.cache_builder import dev_clips
    tof = TOF("cuda")
    recs, ok = [], True
    for ds, clips in dev_clips(bj, a.datasets).items():
        for key, frames in list(clips.items())[:a.limit_clips]:
            clean, _ = bj.load_frames(ds, frames)
            fl = tof.flows(clean).cpu().numpy()
            p = flow_path(a.cache, ds, key)
            rec = {"dataset": ds, "key": key, "clean_sha256": C.array_sha256(clean), "flow_sha256": C.array_sha256(fl)}
            if a.mode == "build-flow":
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, fl)
                rec["status"] = "built"
            else:
                same = p.exists() and C.array_sha256(np.load(p)) == rec["flow_sha256"]
                rec["status"] = "MATCH" if same else "MISMATCH"
                ok &= same
            recs.append(rec)
    C.write_json(Path(a.cache) / ("flow_manifest.json" if a.mode == "build-flow" else "flow_verify_report.json"), {"entries": recs, "PASS": ok})
    return 0 if ok else 1


def make_cached_tof(tof_cls, cache: Path, clean_by_sha: dict):
    """Subclass factory (no torch import at module import). ``clean_by_sha``: clean-clip content sha -> flow .npy path."""
    import torch

    class CachedFlowTOF(tof_cls):
        hits = 0
        misses = 0

        def flows(self, v):
            h = C.array_sha256(v.detach().cpu())
            p = clean_by_sha.get(h)
            if p is not None and Path(p).exists():
                type(self).hits += 1
                return torch.from_numpy(np.load(p)).to(self.device)
            type(self).misses += 1
            return super().flows(v)

    return CachedFlowTOF

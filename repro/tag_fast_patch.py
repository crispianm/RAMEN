"""EXPLORATORY, UNAUDITED. Runtime patch (outside the scorer; no scorer file is edited): route the scorer's agf stage to the fused Triton kernel V3 (repro/tag_fast.py, class E).

``patch()`` replaces ``evaluation.depth_probe.depth_agf`` (numpy path) and ``depth_agf_torch`` (device path) by V3 wrappers and sets ``evaluation.depth_probe.AGF_BACKEND = BACKEND_NAME``, which
bench_joint writes to every output JSON as ``agf_backend``. ``make_post`` looks both functions up by module global at call time, so every agf token takes V3, whatever the input type
(numpy in -> numpy out in the input dtype, via a device copy; CUDA tensor in -> CUDA tensor out, fp32). ``unpatch()`` restores the originals and the pinned backend string. amin == 1 stays the bit-exact identity.
NB: ``cmd_latency`` takes its device-resident branch only for AGF_BACKEND == 'gpu'; under this patch it would time the host-copy path (flag when timing; use the standalone kernel timing or set the string accordingly)."""
from __future__ import annotations

import numpy as np
import torch

from evaluation import depth_probe as dp
from repro.tag_fast import agf_v3

BACKEND_NAME = "fast_triton_v3"
_ORIG = {"depth_agf": dp.depth_agf, "depth_agf_torch": dp.depth_agf_torch, "AGF_BACKEND": dp.AGF_BACKEND}
_state = {"patched": False}


def _v3(depth, amin, tau, beta=dp.AGF_BETA):
    if isinstance(depth, np.ndarray):
        out = agf_v3(torch.from_numpy(np.ascontiguousarray(depth)).cuda(), amin, tau, beta).cpu().numpy()
        return out.astype(depth.dtype, copy=False)
    return agf_v3(depth, amin, tau, beta)


def patch() -> None:
    dp.depth_agf, dp.depth_agf_torch, dp.AGF_BACKEND = _v3, _v3, BACKEND_NAME
    _state["patched"] = True


def unpatch() -> None:
    dp.depth_agf, dp.depth_agf_torch, dp.AGF_BACKEND = _ORIG["depth_agf"], _ORIG["depth_agf_torch"], _ORIG["AGF_BACKEND"]
    _state["patched"] = False


def original(name: str):
    """The unpatched scorer function (``depth_agf`` = the float64 numpy reference)."""
    return _ORIG[name]

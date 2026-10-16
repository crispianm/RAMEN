#!/usr/bin/env python3
"""DN-01 P4: per-frame latency of plain VDA-S vs VDA-S + restoration head, one offline
32-frame window at 518x518, fp16 autocast (infer_video_depth's default), 4090 only.
Weights don't affect latency, so both models use the DA-V2 init. Run on an idle GPU.

  python scripts/dn01_latency.py --out reports/dn01_latency_4090.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.vda_net import VDANet  # noqa: E402


@torch.no_grad()
def time_model(model, x, iters: int, warmup: int) -> list[float]:
    ms = []
    for i in range(warmup + iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.float16):
            model(x, return_recon=True) if model.enh_decoder else model(x)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            ms.append(start.elapsed_time(end) / x.shape[2])   # per frame
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = torch.rand(1, 3, a.frames, a.size, a.size, device="cuda")
    rows = {}
    for name, head in (("vda_s", False), ("vda_s_restore_h0", True)):
        m = VDANet(init="da_v2", restoration_head=head).cuda().eval()
        ms = time_model(m, x, a.iters, a.warmup)
        rows[name] = {"per_frame_ms_p50": float(np.median(ms)), "per_frame_ms_p95": float(np.percentile(ms, 95))}
        del m
        torch.cuda.empty_cache()
    rows["overhead_p50_pct"] = 100 * (rows["vda_s_restore_h0"]["per_frame_ms_p50"] / rows["vda_s"]["per_frame_ms_p50"] - 1)
    out = {"gpu": torch.cuda.get_device_name(), "frames": a.frames, "size": a.size, **rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

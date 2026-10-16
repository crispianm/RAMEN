"""Protocol v3 WP24 (partial): per-frame latency of every candidate model / cascade on the local 4090: fp16 autocast, 518^2, ONE 32-frame window, CUDA events, p50 after warmup, idle GPU only (bench_joint's own ``latency``
method, one subprocess per spec so no allocator state leaks between models). DA3-S (not wired in bench_joint) is timed here by the SAME method on ``models.D3Net`` (arch d3: DA3-small backbone + DualDPT head, per frame, no
temporal module). Writes ``latency_v3.json`` = per-model p50 / p95 per frame, stage breakdown, ratio to the released VDA-S, and the list within ``--budget`` x VDA-S (default 2.0).
``RAMEN_AGF_BACKEND=gpu`` times the TAG (agf) stage on the device (evaluation/depth_probe.depth_agf_torch, no host copy); the default ``cpu`` is the numpy reference.
``python scripts/dn01_v3_latency.py --out DIR --ours-a CKPT --ours-b CKPT``  (ours checkpoints can be step-0: latency depends on the architecture, not the weights)."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TAG = "agf0.2t0.06"                       # TAG = the agf depth post-filter (PAPER_TODO_V3 goal line)
DA3_CKPT = "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth-anything-v3-small.safetensors"


def specs(a) -> list[tuple[str, str]]:
    s = [("vda_s", "vda_s"), ("metric_vda_s", "metric_vda_s"), ("metric_vda_l", "metric_vda_l"), ("dav2_s", "dav2_s"), ("daac", "daac"),
         ("fastdvdnet_blind -> metric_vda_s", "fastdvdnet_blind+metric_vda_s"), ("pocketdvdnet -> metric_vda_s", "pocketdvdnet+metric_vda_s"),
         ("metric_vda_s + TAG (stage timed alone inside)", f"metric_vda_s+{TAG}")]
    if a.ours_a:
        s += [("ours A (a1h + V2 restorer), no TAG", f"ours:{a.ours_a}"), ("ours A + TAG", f"ours:{a.ours_a}+{TAG}")]
    if a.ours_b:
        s += [("ours B (c2 coupling), no TAG", f"ours:{a.ours_b}"), ("ours B + TAG", f"ours:{a.ours_b}+{TAG}")]
    return s


def time_da3(size: int, warmup: int, iters: int) -> dict:
    import torch
    sys.path.insert(0, str(REPO))
    from models import build_model
    m = build_model({"model": {"arch": "d3", "checkpoint_path": DA3_CKPT, "config_name": "da3-small", "use_temporal": False, "per_frame_encode": True, "device": "cuda"}}).cuda().eval()
    x = torch.rand(1, 3, 32, size, size, device="cuda")
    ms = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(warmup + iters):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for s in range(0, 32, 8):             # window-1 models run 8 frames at a time (bench_joint's per-frame path)
                m(x[:, :, s:s + 8])
            e1.record()
            torch.cuda.synchronize()
            if i >= warmup:
                ms.append(e0.elapsed_time(e1) / 32)
    return {"model": "da3_s (arch d3, released DA3-small, per frame)", "per_frame_ms_p50": float(np.median(ms)), "per_frame_ms_p95": float(np.percentile(ms, 95)), "stage_ms_p50": {"da3_s": float(np.median(ms))},
            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours-a")
    ap.add_argument("--ours-b")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--skip-da3", action="store_true")
    ap.add_argument("--only", help="comma-separated substrings: time only the specs whose label contains one (e.g. TAG)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, spec in specs(a):
        if a.only and not any(k in label for k in a.only.split(",")):
            continue
        r = subprocess.run([sys.executable, str(REPO / "bench_joint.py"), "latency", "--model", spec, "--out", str(out / "raw"), "--iters", str(a.iters), "--warmup", str(a.warmup)],
                           cwd=REPO, capture_output=True, text=True)
        files = sorted((out / "raw").glob("latency__*.json"), key=lambda p: p.stat().st_mtime) if r.returncode == 0 else []
        if r.returncode != 0 or not files:
            rows.append({"model": label, "spec": spec, "error": (r.stderr or r.stdout)[-400:]})
            print("FAILED", label, (r.stderr or r.stdout)[-200:], flush=True)
            continue
        d = json.loads(files[-1].read_text())
        rows.append({"model": label, "spec": spec, "per_frame_ms_p50": d["per_frame_ms_p50"], "per_frame_ms_p95": d["per_frame_ms_p95"], "stage_ms_p50": d["stage_ms_p50"], "gpu": d["gpu"], "commit": d["ramen_commit"],
                     "idle_check": d["gpu_idle_check"]})
        print(f"{label}: p50 {d['per_frame_ms_p50']:.2f} ms/frame", flush=True)
    if not a.skip_da3:
        try:
            rows.append(time_da3(518, a.warmup, a.iters))
            print("da3_s: p50 %.2f" % rows[-1]["per_frame_ms_p50"], flush=True)
        except Exception as e:                                         # noqa: BLE001  (recorded, not hidden)
            rows.append({"model": "da3_s", "error": repr(e)[-300:]})
    ref = next((r["per_frame_ms_p50"] for r in rows if r.get("spec") == "vda_s" and "per_frame_ms_p50" in r), None)
    for r in rows:
        if ref and "per_frame_ms_p50" in r:
            r["x_vda_s"] = round(r["per_frame_ms_p50"] / ref, 3)
    res = {"protocol": "v3 WP24 (partial)", "gpu": next((r.get("gpu") for r in rows if r.get("gpu")), None), "host": socket.gethostname(), "precision": "fp16 autocast", "size": [518, 518], "window_frames": 32,
           "method": "bench_joint.py latency: one 32-frame window, CUDA events, p50 per frame after warmup, idle GPU (no other compute process >= 1 GB); DA3-S timed identically in this script", "tag_token": TAG,
           "ref_vda_s_ms_p50": ref, "budget_x_vda_s": a.budget, "models": rows,
           "within_budget": [r["model"] for r in rows if r.get("x_vda_s") is not None and r["x_vda_s"] <= a.budget],
           "over_budget": [r["model"] for r in rows if r.get("x_vda_s") is not None and r["x_vda_s"] > a.budget],
           "note": "ours A / B ckpts are step-0 (latency depends on architecture); the stitched VDA pipeline advances 22 new frames per 32-frame window, multiply by 32/22 for per-NEW-frame cost; cascades include the restorer stage"}
    (out / "latency_v3.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("ref_vda_s_ms_p50", "within_budget", "over_budget")}, indent=1))


if __name__ == "__main__":
    main()

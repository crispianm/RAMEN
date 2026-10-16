"""NEW latency launcher (bench_joint.py is NOT edited). Imports the pinned bench_joint read-only, replaces ONLY depth_post_stages (the pre-branch
torch.cuda.synchronize() at bench_joint.py:1471-1472 is dropped), then runs bench_joint.cmd_latency UNCHANGED (its end event ev[-1] is recorded after the last stage's
last op and elapsed time is read only after torch.cuda.synchronize()). Optional --eff-flags applies patches.py toggles (class-labelled) to the model before timing.
  python latency_copy.py --eff-flags S1 -- latency --model <spec> --out <dir> ...   (args after -- are bench_joint's CLI)
NO GPU TIMING before an Auditor token."""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/DATA/DegradedDepth/claude_outputs/scratch/c3_diag_20261009/eval_clone")
import torch
import bench_joint as bj


GPU_BACKENDS = ("gpu", "fast_triton_v3")      # P2 (Auditor R-latency): the device-resident filter branch applies to every GPU backend, incl. repro/tag_fast_patch.py's V3


def use_device_branch(device, backend, tokens):
    return device == "cuda" and backend in GPU_BACKENDS and all(t.startswith("agf") for t in tokens)


def precision_record():
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def depth_post_stages_nosync(p, size, device):
    if "depth_post" not in p:
        return []
    probe = torch.rand(bj.INFER_LEN, size, size, device=device)
    from evaluation import depth_probe
    on_device = use_device_branch(device, depth_probe.AGF_BACKEND, p["depth_post_tokens"])

    def stage(y, s, b):                       # NO synchronize here: the GPU branch is enqueued behind the model's work and timed by the events;
        guides = {"i": y, "r": y} if isinstance(y, torch.Tensor) else None   # the host (numpy) branch blocks on .cpu() anyway
        if on_device:
            p["depth_post"](probe.float(), guides)
            return y
        p["depth_post"](probe.float().cpu().numpy(), guides)
        return y
    return [("+".join(p["depth_post_tokens"]), stage)]


if __name__ == "__main__":
    argv = sys.argv[1:]
    flags_name = None
    if argv and argv[0] == "--eff-flags":
        flags_name = argv[1]; argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    bj.depth_post_stages = depth_post_stages_nosync
    if flags_name:
        import patches, run_variant as RV
        fl = RV.VARIANTS[flags_name]
        o_load, o_rel = bj.load_ours, bj._vda_released
        def load_ours(ckpt, device):            # C3: patch the freshly loaded SharedTemporalNet (runtime only, nothing on disk changes)
            net = o_load(ckpt, device); patches.apply(net, fl, True); return net
        def vda_released(device, metric):       # released Metric-VDA-S: identical shared toggles (E1 attention, E3 compile of encoder/head)
            bj._vda_path()
            from video_depth_anything.video_depth import VideoDepthAnything
            key = "metric_vda_s" if metric else "vda_s"
            net = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=metric)
            net.load_state_dict(torch.load(bj.PATHS[key], map_location="cpu", weights_only=False), strict=True)
            net = net.to(device).eval(); patches.apply(net, [f for f in fl if f in ("sdpa", "sdpa_motion", "cudnn", "c_enc", "c_head")], False)
            return dict(fwd=lambda x: (net(bj._norm(x)[None])[0], None), metric=metric, window=32, weights={key: bj.PATHS[key]})
        bj.load_ours, bj._vda_released = load_ours, vda_released
        bj.DEPTH_MODELS["metric_vda_s"] = lambda d: vda_released(d, True)   # explicit: do not rely on how DEPTH_MODELS captured the builder
    sys.argv = ["bench_joint.py"] + argv
    try:
        bj.main()
    finally:                                   # after cmd_latency returned (all timing done): annotate its JSON; cmd_latency itself is unchanged
        import json, glob
        if argv and argv[0] == "latency" and "--out" in argv:
            from evaluation import depth_probe as _dp
            for f in glob.glob(os.path.join(argv[argv.index("--out") + 1], "latency__*.json")):
                art = json.load(open(f)); art.update(precision_record())
                if art.get("depth_post_probe"):
                    art["depth_post_probe"]["label"] = "RAND PROBE: torch.rand noise, not model output; valid for value-independent fused kernels only; real-depth kernel timing is a separate standalone measurement"
                    art["depth_post_branch"] = "device (no D2H)" if _dp.AGF_BACKEND in GPU_BACKENDS else "host (numpy; stage time includes the D2H copy as released)"
                json.dump(art, open(f, "w"), indent=2)

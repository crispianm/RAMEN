"""One variant per process (global monkeypatches). No timing, no DEV data.
  run_variant.py --model c3|mvda --variant NAME --prec fp32|fp16 [--h 518 --w 518] [--save-base | --compare | --mem]
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLONE = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/c3_diag_20261009/eval_clone"
CK = os.environ.get("RAMEN_CKPT", "/mnt/DATA/DegradedDepth/checkpoints_release/ramen.pt")
sys.path.insert(0, CLONE)
import torch, torch.nn.functional as F
import patches

VARIANTS = {
    "base": [], "e1": ["sdpa"], "e2a": ["bnfold"], "e2b": ["cudnn"], "e2c": ["chlast"], "e2d": ["hookoff"], "e2e": ["fp16build"],
    "e2": ["bnfold", "cudnn", "chlast", "hookoff", "fp16build"],
    "e1e2": ["sdpa", "bnfold", "cudnn", "chlast", "hookoff", "fp16build"],
    "e1e2_hf": ["sdpa", "hookoff", "fp16build"],                          # the bitwise-clean subset only
    "e1m": ["sdpa_motion"], "e1all": ["sdpa", "sdpa_motion"], "e2f": ["dense_group"],
    # declared cumulative stacks (see AUDITOR_PACKAGE_DRAFT.md): S1=+E1(dinov2+motion), S2=S1+E2 (all NEAR/EXACT restorer toggles), S3=S2+E3 compile
    "X": ["hookoff", "fp16build"],
    "S1": ["hookoff", "fp16build", "sdpa", "sdpa_motion"],
    "S2": ["sdpa", "sdpa_motion", "bnfold", "cudnn", "chlast", "hookoff", "fp16build", "dense_group"],
    "S3": ["sdpa", "sdpa_motion", "bnfold", "cudnn", "chlast", "hookoff", "fp16build", "dense_group", "c_enc", "c_head", "c_rest"],
    "a3_enc": ["sdpa", "c_enc"], "a3_enc_head": ["sdpa", "c_enc", "c_head"],
    "a3_full": ["sdpa", "bnfold", "cudnn", "chlast", "hookoff", "fp16build", "c_enc", "c_head", "c_rest"],
}


def make_input(h, w, T=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = F.interpolate(torch.rand(1, 3, 6, 8, generator=g), size=(h + 64, w + 64), mode="bicubic", align_corners=False)
    mid = F.interpolate(torch.rand(1, 3, 40, 52, generator=g), size=(h + 64, w + 64), mode="bilinear", align_corners=False)
    img = (0.6 * base + 0.4 * mid).clamp(0, 1)
    fr = []
    for t in range(T):                                      # slow camera drift + sigma=25/255 AWGN per frame (llvar-like noisy input)
        c = img[:, :, 2 * t % 60:2 * t % 60 + h, t % 60:t % 60 + w]
        fr.append((c + torch.randn(c.shape, generator=g) * 25 / 255).clamp(0, 1))
    return torch.cat(fr)


def build(model, flags):
    import bench_joint as bj
    if model == "c3":
        net = bj.load_ours(CK, "cuda")
        log = patches.apply(net, flags, True)
        res = bj._ours("cuda", CK) if False else None
        def fwd(x):
            rgb = x.permute(1, 0, 2, 3)[None]
            restored = net.restore_batch(rgb)
            d, _ = net._depth_and_feats(net.consume(rgb, restored))
            return d[0, 0], restored[0].permute(1, 0, 2, 3)
        return fwd, log, net
    net = bj._vda_upstream_net("vits").to("cuda").eval()
    log = patches.apply(net, flags, False)
    return (lambda x: (net(bj._norm(x)[None])[0], None)), log, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--prec", required=True)
    ap.add_argument("--h", type=int, default=518); ap.add_argument("--w", type=int, default=518)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--save-base", action="store_true"); ap.add_argument("--mem", action="store_true")
    a = ap.parse_args()
    import subprocess
    busy = [l for l in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
            if l and int(l.rsplit(",", 1)[1]) >= 1024]
    if busy:
        raise SystemExit(f"GPU busy, refusing: {busy}")
    flags = VARIANTS[a.variant]
    out = {"toggles": patches.toggle_record(flags, a.prec), "model": a.model, "variant": a.variant, "flags": flags, "class": patches.variant_class(flags), "prec": a.prec, "hw": [a.h, a.w], "T": a.frames,
           "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
           "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    fwd, log, net = build(a.model, flags); out["patch_log"] = log
    out.update(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),   # re-read AFTER patches.apply (a toggle must not change them)
               float32_matmul_precision=torch.get_float32_matmul_precision(), cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    x = make_input(a.h, a.w, a.frames).cuda()
    ctx = torch.autocast("cuda", dtype=torch.float16, enabled=(a.prec == "fp16"))
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(), ctx:
            d, r = fwd(x)
            torch.cuda.synchronize()
            if not a.mem:                                    # second pass: compile/benchmark warm, outputs from the warm pass
                d, r = fwd(x); torch.cuda.synchronize()
        out["peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        out["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        out["oom"] = False
    except torch.OutOfMemoryError as e:
        out.update(oom=True, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30, err=str(e)[:200])
        print(json.dumps(out)); os.makedirs(f"{HERE}/out", exist_ok=True)
        json.dump(out, open(f"{HERE}/out/{'mem' if a.mem else 'cmp'}_{a.model}_{a.variant}_{a.prec}_{a.h}x{a.w}.json", "w"), indent=1); return
    d = d.float().cpu(); r = None if r is None else r.float().cpu()
    bp = f"{HERE}/out/base_{a.model}_{a.prec}.pt"
    if a.save_base:
        torch.save({"d": d, "r": r}, bp)
    if not a.mem and os.path.exists(bp):
        b = torch.load(bp)
        def cmp(x, y):
            ad = (x - y).abs(); rel = ad / y.abs().clamp_min(1e-6)
            q = lambda t, p: float(t.flatten()[: 4_000_000].quantile(p)) if t.numel() > 4_000_000 else float(t.flatten().quantile(p))
            return {"max_abs": float(ad.max()), "mean_abs": float(ad.mean()), "max_rel": float(rel.max()), "p999_rel": q(rel, .999), "median_rel": q(rel, .5),
                    "bitwise": bool(torch.equal(x, y)), "finite": bool(torch.isfinite(x).all())}
        out["depth_vs_base"] = cmp(d, b["d"])
        if r is not None:
            out["restored_vs_base"] = cmp(r, b["r"])
        out["depth_range"] = [float(b["d"].min()), float(b["d"].max())]
    suffix = f"{a.h}x{a.w}"
    json.dump(out, open(f"{HERE}/out/{'mem' if a.mem else 'cmp'}_{a.model}_{a.variant}_{a.prec}_{suffix}.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restorer-only PSNR/SSIM/tOF scorer for the FIDELITY read (arm: restorer-before-joint-training = distilled temp2).

Imports bench_joint from the PINNED scorer clone (commit 368b5c5) WITHOUT editing it and reuses, unchanged: select_clips, load_frames, llvar_draw,
degrade_llvar, llvar_noise_seed, lowlight_reference, vda_windows (32-frame windows, step 22, keyframe hand-over, restored frames = first window that
sees a frame as a real input; the depth output is replaced by a 2x2 dummy), psnr, ssim, blind_gain, TOF, llvar_pin, git_state.
Output = per_video JSONs in the c3rep schema (psnr_native / ssim_native / tof_native, *_input_native twins, display secondary).

  modes   --mode init : temp2 block of --restorer-ckpt (distilled init; default temp2_distilled.pt, sha-verified), stage 1 only (as at init: no block2)
          --mode c3r  : the FULL restorer R of a train.py C3 checkpoint (restore_batch per 32-frame window = exactly what bench_joint's fwd_c returns) -> PIPELINE CHECK
  sub-commands  (default) score | --check-p2-sample SAMPLE.json OUT.txt | --check-vs-bench BENCH.json OUT.txt   (compare an existing output with a reference)
No GPU unless --device cuda is passed (the runner does that, after the 4090-free gate). --device cpu hides CUDA before torch is imported.
"""
import argparse, hashlib, json, math, os, sys, time

TEMP2_SHA = "96d35f08ecd5c988387e502bbd0ad85875539cafd9e329448ad5c76804916c4b"
TEMP2_DEFAULT = "/mnt/DATA/DegradedDepth/RAMEN_aaai/results/dn01_st_recal/recal_full/temp2_distilled.pt"
SCORER_CLONE = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/c3_diag_20261009/eval_clone"
SCORER_COMMIT = "368b5c53b37ebd22f5d44ce0c9d514122efc88da"
DATASETS = ["kitti", "bonn", "scannetpp", "nyuv2", "sintel_rgbfix"]
SCHEMA_VERSION = "fidelity_init_score/1"
# PRE-DECLARED tolerances of the two checks (stated before the first number was seen)
TOL_P2 = {"psnr_native": 0.01, "ssim_native": 0.001}      # same code, same CPU, same 32 frames: expect ~1e-5; 0.01 dB / 0.001 is the allowed slack
TOL_BENCH = {"psnr_native": 0.05, "ssim_native": 0.002, "tof_native": 0.01}   # CPU here vs the 4090's stored row: fp32 reduction order, per clip; tof_native only compared when both are finite (the GPU gate)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


LABEL = "exposed exploratory, split all"


def log_exposure(path, rec):
    """Append-only, fsynced jsonl line: one per (dataset) BEFORE any frame of a non-dev split is loaded (Auditor C2)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def _nanmean(v):
    import math
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def _cmp(ref, new, tol, label, lines):
    ok = True
    for k, t in tol.items():
        if k == "tof_native" and not (math.isfinite(ref.get(k, float("nan"))) and math.isfinite(new.get(k, float("nan")))):
            lines.append(f"  {label} {k}: not compared (not finite in ref or new, e.g. CPU --no-tof run)"); continue
        d = abs(ref[k] - new[k]); good = d <= t; ok &= good
        lines.append(f"  {label} {k}: ref {ref[k]:.6f} new {new[k]:.6f} |diff| {d:.2e} tol {t} -> {'OK' if good else 'FAIL'}")
    return ok


def check_p2_sample(sample, out_json, out_txt):
    s, o = json.load(open(sample)), json.load(open(out_json)); lines = []; ok = True; n = 0
    for dskey, rec in o["per_video"].items():
        key = dskey if "/" in dskey else None
        if key not in s: continue
        n += 1; lines.append(key)
        ok &= _cmp(s[key]["distilled_temp2_init"], rec, TOL_P2, "init-vs-p2_cpu_sample", lines)
    ok &= n > 0
    lines.append(f"CHECK_P2_SAMPLE clips compared={n} tol={TOL_P2} -> {'PASS' if ok and n else 'FAIL'}")
    open(out_txt, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); return 0 if ok and n else 1


def check_vs_bench(bench_json, out_json, out_txt):
    b, o = json.load(open(bench_json))["per_video"], json.load(open(out_json)); lines = []; ok = True; n = 0
    for key, rec in o["per_video"].items():
        k = key.split("/", 1)[-1]
        if k not in b: lines.append(f"{key}: not in bench json"); ok = False; continue
        n += 1; lines.append(k)
        ok &= _cmp(b[k], rec, TOL_BENCH, "c3r-vs-bench-row", lines)
        lines.append(f"  input twin psnr_input_native ref {b[k]['psnr_input_native']:.6f} new {rec['psnr_input_native']:.6f}")
    lines.append(f"CHECK_VS_BENCH clips compared={n} tol={TOL_BENCH} -> {'PASS' if ok and n else 'FAIL'}")
    open(out_txt, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); return 0 if ok and n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["init", "c3r"], default="init")
    ap.add_argument("--restorer-ckpt", default=TEMP2_DEFAULT)
    ap.add_argument("--expect-sha", default=None, help="sha256 the restorer ckpt must have (init mode defaults to the pinned 96d35f08...)")
    ap.add_argument("--dataset", nargs="+", default=DATASETS)
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--clips", nargs="*", default=None, help="explicit clip keys (smoke); default every clip of the split")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="SMOKE ONLY: truncate each clean clip BEFORE degradation (reproduces the 32-frame CPU sample)")
    ap.add_argument("--allow-test", action="store_true", help="required for --split test/all (as in bench_joint); every such run is logged")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-tof", action="store_true", help="skip RAFT tOF (tof_native then NaN); smoke only")
    ap.add_argument("--bench-root", default=SCORER_CLONE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-p2-sample", nargs=3, metavar=("SAMPLE", "OUT_JSON", "OUT_TXT"))
    ap.add_argument("--check-vs-bench", nargs=3, metavar=("BENCH_JSON", "OUT_JSON", "OUT_TXT"))
    a = ap.parse_args()
    if a.check_p2_sample: sys.exit(check_p2_sample(*a.check_p2_sample))
    if a.check_vs_bench: sys.exit(check_vs_bench(*a.check_vs_bench))
    if not a.out: ap.error("--out required")
    if a.split != "dev" and not a.allow_test: sys.exit("ABORT: --split %s reads TEST clips; refused without --allow-test" % a.split)
    a.out, a.restorer_ckpt, a.bench_root = (os.path.abspath(v) for v in (a.out, a.restorer_ckpt, a.bench_root))   # before the chdir into the scorer clone
    if a.device == "cpu": os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    import numpy as np, torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.bench_root); os.chdir(a.bench_root)
    import bench_joint as bj
    from data.elvis_noise import lowlight_reference
    state = bj.git_state(bj.REPO)
    if state != SCORER_COMMIT: sys.exit(f"ABORT: scorer clone state {state} != {SCORER_COMMIT} (must be the clean pinned commit)")
    if a.split != "dev" and a.mode == "c3r": sys.exit("ABORT: c3r pipeline check is DEV-only by design")
    dev = a.device
    got = sha256_file(a.restorer_ckpt); want = a.expect_sha or (TEMP2_SHA if a.mode == "init" else None)
    if want and got != want: sys.exit(f"ABORT: restorer ckpt sha {got} != expected {want}")
    if a.mode == "init" and a.restorer_ckpt == TEMP2_DEFAULT and got != TEMP2_SHA: sys.exit("ABORT: temp2 sha")

    if a.mode == "init":
        import torch.nn as nn
        from models.shared_temporal import SharedTemporalNet, temp2_block
        class Shell(nn.Module):   # only the restorer part, driven by the REAL SharedTemporalNet.restore_clip
            def __init__(s):
                super().__init__(); s.batch = 8; s.proj = None; s.temp2 = temp2_block("full")
            restore_clip = SharedTemporalNet.restore_clip
        shell = Shell().eval()
        sd = torch.load(a.restorer_ckpt, map_location="cpu", weights_only=False)
        print("load temp2:", shell.temp2.load_state_dict(sd.get("temp2", sd), strict=True)); shell = shell.to(dev)
        def restore_window(cur):          # cur (32,3,H,W) -> (32,3,H,W)
            return shell.restore_clip(cur, None)
        weights = {"temp2_distilled": {"path": a.restorer_ckpt, "sha256": got}}; model = "restorer_init:temp2_distilled(stage1_only)"
    else:
        net = bj.load_ours(a.restorer_ckpt, dev)
        def restore_window(cur):          # == bench_joint._ours.fwd_c's restored output
            return net.restore_batch(cur.permute(1, 0, 2, 3)[None])[0].permute(1, 0, 2, 3)
        weights = {"ours": {"path": a.restorer_ckpt, "sha256": got}}; model = "restorer_R_of:" + os.path.basename(os.path.dirname(a.restorer_ckpt)) + "/" + os.path.basename(a.restorer_ckpt)

    def fwd(cur):                          # bench_joint.vda_windows contract: (depth-like (32,h,w), restored (32,3,H,W)); depth = 2x2 dummy
        return torch.zeros(cur.shape[0], 2, 2, device=cur.device), restore_window(cur)
    tofnet = None
    if not a.no_tof:
        from evaluation.temporal_metrics import TOF
        tofnet = TOF(dev)
    os.makedirs(a.out, exist_ok=True)
    for ds in a.dataset:
        proto = bj.DATASETS[ds]; clips = bj.select_clips(ds, a.split, a.limit_clips)
        if a.clips: clips = [(k, f) for k, f in clips if k in set(a.clips)]
        if not clips: sys.exit(f"ABORT: no clips for {ds} {a.split} {a.clips}")
        if a.split != "dev":      # Auditor C2: log BEFORE any TEST frame is loaded. (1) out-dir jsonl with the clips; (2) bench_joint's own log_test_access (default ~/.ramen/TEST_access_log.jsonl)
            import datetime, argparse as _ap, getpass
            logrec = dict(ts_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), pid=os.getpid(), host=__import__("socket").gethostname(), user=getpass.getuser(),
                          dataset=ds, split=a.split, cell="llvar", clips=[k for k, _ in clips], mode=a.mode, weights_sha256=got, weights_path=a.restorer_ckpt, ramen_commit=state,
                          script_sha256=sha256_file(__file__), device=dev, label=LABEL, bench_version=bj.BENCH_VERSION)
            log_exposure(os.path.join(a.out, "TEST_access_log.jsonl"), logrec)
            bj.log_test_access(_ap.Namespace(model=("restorer_init:" if a.mode == "init" else "restorer_R_of:") + a.restorer_ckpt, dataset=[ds], cells=["llvar"], split=a.split,
                                             allow_test=True, limit_clips=a.limit_clips, out=a.out))
        pv = {}
        for key, frames in clips:
            t0 = time.time()
            with torch.no_grad():
                clean, _ = bj.load_frames(ds, frames)
                if a.max_frames: clean = clean[:a.max_frames]
                draw = bj.llvar_draw(ds, key)
                x = bj.degrade_llvar(clean, "llvar", draw, bj.llvar_noise_seed(ds, key, "llvar")).to(dev)
                _, restored = bj.vda_windows(fwd, x, (2, 2), True)
                restored = restored.clamp(0, 1)
                sl = proto.get("score_slice"); clean_d = clean.to(dev)
                if sl: restored, x, clean_d = restored[sl[0]:sl[1]], x[sl[0]:sl[1]], clean_d[sl[0]:sl[1]]   # = vda5_sets.score_slice on the restoration inputs
                ref = torch.from_numpy(lowlight_reference(clean_d.cpu().numpy(), draw["ev"])).to(dev)
                r = {"frames": int(restored.shape[0]),
                     "psnr_native": float(bj.psnr(restored, ref).mean()), "psnr_input_native": float(bj.psnr(x, ref).mean()),
                     "ssim_native": float(bj.ssim(restored, ref).mean()), "ssim_input_native": float(bj.ssim(x, ref).mean())}
                if tofnet is not None: r["tof_native"], r["tof_input_native"] = tofnet(restored, ref), tofnet(x, ref)
                else: r["tof_native"] = r["tof_input_native"] = float("nan")
                g = bj.blind_gain(x); shown = (restored * g).clamp(0, 1)
                r.update(gain=g, psnr_display=float(bj.psnr(shown, clean_d).mean()), ssim_display=float(bj.ssim(shown, clean_d).mean()), llvar_draw=draw)
            pv[key if not a.max_frames else f"{ds}/{key}"] = r
            print(f"{model} {ds} {key}: psnr_native {r['psnr_native']:.4f} (input {r['psnr_input_native']:.4f}) ssim_native {r['ssim_native']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        keys = [k for k in next(iter(pv.values())) if k not in ("frames", "llvar_draw")]
        art = {"bench": "fidelity_init_score", "schema": SCHEMA_VERSION, "bench_version": bj.BENCH_VERSION, "model": model, "dataset": ds, "cell": "llvar", "split": a.split, "label": LABEL,
               "limit_clips": a.limit_clips, "max_frames_smoke": a.max_frames, "short_side": proto["short_side"], "num_clips": len(pv),
               "llvar_draws_sha256": bj.llvar_pin(ds), "manifest_sha256": bj._sha(bj.PATHS[f"{ds}_manifest"]), "ramen_commit": state,
               "scorer_script_sha256": sha256_file(__file__), "weights": weights, "host": __import__("socket").gethostname(),
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
               "window_path": "bench_joint.vda_windows(32, step 22, keyframe hand-over); restored = first window seeing a frame as real input",
               "video_macro": {k: _nanmean([v[k] for v in pv.values()]) for k in keys}, "per_video": pv}
        path = os.path.join(a.out, f"restorer_{'temp2_init' if a.mode == 'init' else 'c3R_check'}__{ds}__llvar__{a.split}__v3.json")
        if os.path.exists(path): sys.exit(f"ABORT: {path} exists (unique-output rule)")
        open(path, "w").write(json.dumps(art, indent=2)); assert bj.git_state(bj.REPO) == SCORER_COMMIT, "scorer clone became dirty"; print("wrote", path, {k: round(art['video_macro'][k], 4) for k in ("psnr_native", "ssim_native")}, flush=True)


if __name__ == "__main__":
    main()

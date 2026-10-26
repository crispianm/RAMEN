"""Single-session latency ladder runner (AUD-EFF-D / AUD-FASTTAG-C, C3_LATENCY_4090_20261012 protocol). PREPARED, NOT RUN: needs the Auditor's go and an idle 4090.

ONE process / session / env. Protocol per row: 518x518, fp16 autocast, CUDA events, 5 warm-up passes + median of 30, end event after the last op, ms/frame = window/32.
Each pass times window 1 (32 new frames, /32) and window 2 (22 new frames, /22) of a 54-frame input; no caching is used, so window 2 is the same work (reported anyway).
Ladder (cumulative, declared order): S0, X, S1, S2, S3, S4 (= S3 + fused TAG V3). Per stack: RAMEN raw; RAMEN+TAG V3 x2 settings (every stack);
RAMEN+TAG numpy x2 settings (S0 only); Metric-VDA-S bracket of six runs interleaved among the stack's rows, with the shared toggles of that stack (T3,T4,T6,T9).
Never mixes stacks in a ratio: ratios are row / upper-median of the SAME stack's six bracket runs.
Usage:  run_ladder.py --dry-run            (no GPU, no torch)
        run_ladder.py [--out DIR]           (real session; refuses unless the card is idle and GPU_LOCK is free)
Nothing here edits the scorer, the VDA repo, the venv or paper_cvpr2027."""
from __future__ import annotations

import argparse
import atexit
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EFF = HERE.parent
REPO_ROOT = EFF.parents[1]
sys.path.insert(0, str(EFF))

SCRATCH = Path("/mnt/DATA/DegradedDepth/claude_outputs/scratch/paper_repro")
DEFAULT_OUT = SCRATCH / "latency_ladder"
GPU_LOCK = Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/GPU_LOCK")
DEPTH_DIR = SCRATCH / "fasttag_b/run/depth"
MIN_HARNESS_COMMIT = "c524643"

SIZE, INFER_LEN, OVERLAP, WARMUP, ITERS = 518, 32, 10, 5, 30
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
SETTINGS = {"agf0.2t0.06": (0.2, 0.06), "agf0.5t0.12": (0.5, 0.12)}
RELEASED = {"metric_vda_s": 5.73, "ramen_raw": 9.51, "ramen_tag0.2": 13.67}     # C3_LATENCY_4090_20261012, ms/frame
S0_GATE, HARNESS_GATE, DRIFT_FLAG = 0.10, 0.03, 0.03

# cumulative stacks (identical to run_variant.VARIANTS; a test asserts it)
_E1 = ["sdpa", "sdpa_motion"]
STACK_FLAGS = {
    "S0": [], "X": ["hookoff", "fp16build"], "S1": ["hookoff", "fp16build"] + _E1,
    "S2": _E1 + ["bnfold", "cudnn", "chlast", "hookoff", "fp16build", "dense_group"],
    "S3": _E1 + ["bnfold", "cudnn", "chlast", "hookoff", "fp16build", "dense_group", "c_enc", "c_head", "c_rest"],
}
STACK_FLAGS["S4"] = list(STACK_FLAGS["S3"])
# L (Addendum 16, user Add.33): C = S1+T5 (bnfold), AZ = S1+T5+T6+T7+T9+T10; toggles identical to X' legs.py (2ce36e8d). Appended AFTER the original six; they are
# only timed when named in --stacks (LADDER_DEFAULT = the original six is the default plan, so every pre-L invocation is unchanged).
STACK_FLAGS["C"] = ["hookoff", "fp16build"] + _E1 + ["bnfold"]
STACK_FLAGS["AZ"] = ["hookoff", "fp16build"] + _E1 + ["bnfold", "cudnn", "chlast", "c_enc", "c_head", "c_rest"]
LADDER_DEFAULT = ["S0", "X", "S1", "S2", "S3", "S4"]
STACK_ORDER = LADDER_DEFAULT + ["C", "AZ"]
BUILD_FROM = {"S4": "S3"}                                                       # S4 times the S3 nets with the V3 TAG rows
MVDA_SHARED = ("sdpa", "sdpa_motion", "cudnn", "c_enc", "c_head")               # toggles that apply to Metric-VDA-S (T3, T4, T6, T9)
TOGGLE_FIELDS = ("attn_backend", "precision", "bn_folded", "cudnn_benchmark", "channels_last", "compiled", "refine_hook", "input_staging",
                 "agf_backend", "venv", "torch", "cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision")


def mvda_flags(stack): return [f for f in STACK_FLAGS[stack] if f in MVDA_SHARED]


# ---------------------------------------------------------------------------------------------------------------- plan (pure, no torch)
def build_plan(stacks=None) -> list[dict]:
    """Ordered list of timed items. Per stack: the RAMEN rows with the six Metric-VDA-S bracket runs interleaved (before / between / after)."""
    plan = []
    for st in (stacks or LADDER_DEFAULT):
        rows = [dict(kind="row", stack=st, model="ramen", post=None, setting=None, id=f"{st}__ramen_raw")]
        if st == "S0":
            rows += [dict(kind="row", stack=st, model="ramen", post="numpy", setting=s, id=f"{st}__ramen_tag_numpy_{s}") for s in SETTINGS]
        rows += [dict(kind="row", stack=st, model="ramen", post="v3", setting=s, id=f"{st}__ramen_tag_v3_{s}") for s in SETTINGS]
        n, slots = len(rows), {}
        for k in range(6):
            slots.setdefault(round(k * n / 5), []).append(k)
        for i in range(n + 1):
            for k in slots.get(i, []):
                plan.append(dict(kind="bracket", stack=st, model="mvda", post=None, setting=None, id=f"{st}__mvda_bracket{k}", k=k))
            if i < n:
                plan.append(rows[i])
    return plan


PASS_S = {"ramen": 0.30, "mvda": 0.19}      # rough seconds per 32-frame window (dry-run estimate only)
TAG_NP_S, TAG_V3_S = 0.15, 0.003


def estimate_minutes(plan, iters=ITERS, warmup=WARMUP) -> dict:
    tot, per = 0.0, {}
    for it in plan:
        w = PASS_S[it["model"]] + (TAG_NP_S if it["post"] == "numpy" else TAG_V3_S if it["post"] == "v3" else 0.0)
        t = (warmup + iters) * 2 * w + 1.0
        per[it["id"]] = t / 60
        tot += t
    compile_min = 2 * 3.0 + 0.5 * 4          # two compiled nets in S3 (guess 3 min each) + net builds
    extra = 2.0 + 3.0 + 4.0                   # harness cross-check (2 subprocess runs) + FASTTAG-C standalone + untouched/freeze records
    return {"per_item_min": per, "timing_min": tot / 60, "compile_and_build_min": compile_min, "other_min": extra, "total_min": tot / 60 + compile_min + extra}


def dry_run(plan) -> str:
    est = estimate_minutes(plan)
    lines = ["| # | stack | item | model | TAG | setting | est min |", "|---|---|---|---|---|---|---|"]
    for i, it in enumerate(plan):
        lines.append(f"| {i} | {it['stack']} | {it['id']} | {'RAMEN C3' if it['model']=='ramen' else 'Metric-VDA-S'} | {it['post'] or '-'} | {it['setting'] or '-'} | {est['per_item_min'][it['id']]:.1f} |")
    lines += ["", f"timed items {len(plan)} ({sum(i['kind']=='row' for i in plan)} RAMEN rows + {sum(i['kind']=='bracket' for i in plan)} bracket runs); each = {WARMUP} warm-up + {ITERS} timed passes x 2 windows",
              f"estimated: timing {est['timing_min']:.0f} min + compile/build {est['compile_and_build_min']:.0f} + harness check/standalone V3/records {est['other_min']:.0f} = ~{est['total_min']:.0f} min (estimate)",
              "phases: 0 lock+idle+untouched BEFORE; 1 original-harness cross-check (2 subprocess runs, before CUDA init here); 2 ladder S0..S4 (S0 gates: 10% vs 5.73/9.51/13.67);",
              "        3 FASTTAG-C standalone V3 (+V3s streaming) on saved real depth 518^2 / 374x1242 / 480x640; 4 untouched AFTER, summary, SHA256SUMS, lock release"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------- summaries (pure)
def upper_median(v): s = sorted(v); return s[len(s) // 2]


def bracket_stats(vals) -> dict:
    if not vals: return {"n": 0}
    m = statistics.median(vals); drift = (max(vals) - min(vals)) / m
    return {"n": len(vals), "min": min(vals), "max": max(vals), "drift": drift, "flag": "OK" if drift <= DRIFT_FLAG else "FLAG", "reference_upper_median": upper_median(vals), "runs": list(vals)}


def rel(a, b): return (a - b) / b


def summarize(rows: list[dict], harness: dict | None = None) -> dict:
    """rows = per-row result dicts (kind row|bracket). Builds per-stack tables (ratio vs THAT stack's bracket) and the S0 / harness gates."""
    out = {"stacks": {}, "gates": {}}
    for st in STACK_ORDER:
        br = [r["window1"]["ms_per_frame_p50"] for r in rows if r["stack"] == st and r["kind"] == "bracket"]
        bs = bracket_stats(br)
        tab = []
        for r in rows:
            if r["stack"] == st and r["kind"] == "row":
                v = r["window1"]["ms_per_frame_p50"]
                tab.append({"id": r["id"], "post": r["post"], "setting": r["setting"], "ms_per_frame_w1": v, "ms_per_new_frame_w2": r["window2"]["ms_per_new_frame_p50"],
                            "peak_alloc_GiB": r["mem"]["peak_alloc_GiB"], "peak_reserved_GiB": r["mem"]["peak_reserved_GiB"],
                            "ratio_vs_same_stack_bracket": (v / bs["reference_upper_median"]) if bs.get("n") else None})
        if bs.get("n") or tab: out["stacks"][st] = {"bracket": bs, "rows": tab}
    g = {}
    s0 = {r["id"]: r for r in rows if r["stack"] == "S0"}
    chk = [("metric_vda_s", out["stacks"].get("S0", {}).get("bracket", {}).get("reference_upper_median")),
           ("ramen_raw", s0.get("S0__ramen_raw", {}).get("window1", {}).get("ms_per_frame_p50")),
           ("ramen_tag0.2", s0.get("S0__ramen_tag_numpy_agf0.2t0.06", {}).get("window1", {}).get("ms_per_frame_p50"))]
    for k, v in chk:
        if v is not None: g[f"S0_vs_released_{k}"] = {"now": v, "released": RELEASED[k], "rel_diff": rel(v, RELEASED[k]), "flag": "OK" if abs(rel(v, RELEASED[k])) <= S0_GATE else "FLAG (do not adjust)"}
    if harness:
        for k, orig in harness.items():
            ours = {"ramen_raw": chk[1][1], "metric_vda_s": statistics.median([r["window1"]["ms_per_frame_p50"] for r in rows if r["stack"] == "S0" and r["kind"] == "bracket"] or [float("nan")])}.get(k)
            if ours is not None and ours == ours:
                g[f"harness_check_{k}"] = {"copy_ladder": ours, "original_harness": orig, "rel_diff": rel(ours, orig), "flag": "OK" if abs(rel(ours, orig)) <= HARNESS_GATE else "FLAG"}
    out["gates"] = g
    return out


def markdown_tables(summ: dict) -> str:
    L = ["# Latency ladder tables (descriptive, one session, one 4090; ratios only within a stack, vs that stack's six-run Metric-VDA-S bracket upper median)", ""]
    for st, d in summ["stacks"].items():
        b = d["bracket"]
        L += [f"## {st}", f"bracket: {json.dumps({k: v for k, v in b.items() if k != 'runs'})}", "",
              "| row | ms/frame (w1 /32) | ms/new frame (w2 /22) | peak alloc GiB | peak reserved GiB | ratio vs bracket |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            ra = r["ratio_vs_same_stack_bracket"]
            L.append(f"| {r['id']} | {r['ms_per_frame_w1']:.3f} | {r['ms_per_new_frame_w2']:.3f} | {r['peak_alloc_GiB']:.2f} | {r['peak_reserved_GiB']:.2f} | {'' if ra is None else format(ra, '.3f')} |")
        L.append("")
    L += ["## Gates", "```", json.dumps(summ["gates"], indent=1), "```"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- environment / idle / lock (no torch)
def smi(q, fmt="csv,noheader,nounits"):
    return subprocess.run(["nvidia-smi", f"--query-{q[0]}={q[1]}", f"--format={fmt}"], capture_output=True, text=True).stdout.strip()


def gpu_snapshot() -> dict:
    return {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "compute_apps": smi(("compute-apps", "pid,process_name,used_memory")).splitlines(),
            "clocks": smi(("gpu", "clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu"))}


def other_heavy(apps, own_pid=None, thresh_mib=1024):
    own = str(own_pid if own_pid is not None else os.getpid()); bad = []
    for ln in apps:
        try:
            pid, name, mem = [x.strip() for x in ln.rsplit(",", 2)] if ln.count(",") >= 2 else (None, ln, "0")
            if pid != own and int(mem) >= thresh_mib: bad.append(ln)
        except ValueError:
            bad.append(ln)
    return bad


def require_idle(snap, own_pid=None):
    bad = other_heavy(snap["compute_apps"], own_pid)
    if bad: raise SystemExit(f"GPU not idle (other compute process >= 1 GB): {bad}")


def lock_acquire(path: Path, label: str):
    path = Path(path)
    if path.exists() and path.read_text().strip(): raise SystemExit(f"GPU_LOCK held: {path.read_text().strip()[:200]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"run_ladder pid={os.getpid()} {label} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")


def lock_release(path: Path):
    try: Path(path).write_text("")
    except OSError: pass


def sha(p):
    from common import sha256_file
    return sha256_file(p)


def code_record() -> dict:
    g = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)
    head = g("rev-parse", "HEAD").stdout.strip()
    anc = g("merge-base", "--is-ancestor", MIN_HARNESS_COMMIT, "HEAD").returncode == 0
    files = {n: EFF / n for n in ("latency_copy.py", "patches.py", "run_variant.py")} | {"tag_fast_patch.py": REPO_ROOT / "repro/tag_fast_patch.py", "tag_fast.py": REPO_ROOT / "repro/tag_fast.py", "run_ladder.py": HERE / "run_ladder.py"}
    return {"worktree": str(REPO_ROOT), "HEAD": head, "min_commit": MIN_HARNESS_COMMIT, "min_commit_is_ancestor": anc, "dirty_status": g("status", "--porcelain", "--", "repro").stdout.strip(),
            "sha256": {k: sha(v) for k, v in files.items()}}


# ---------------------------------------------------------------------------------------------------------------- timing core (torch imported lazily)
def make_post_stage(bj, token, size, device="cuda"):
    """Timed post-filter stage, as latency_copy.depth_post_stages_nosync: no pre-stage synchronize; device branch for GPU backends (tensor in, tensor out, no D2H);
    numpy reference row hands the filter a host array (the .cpu() blocks, the numpy time + D2H are inside the timer). Filter input = torch.rand probe (labelled)."""
    import torch
    from evaluation import depth_probe
    fn = bj.depth_post(token)
    probe = torch.rand(INFER_LEN, size, size, device=device)
    on_device = depth_probe.AGF_BACKEND in ("gpu", "fast_triton_v3")

    def stage(y, s, b):
        guides = {"i": y, "r": y} if isinstance(y, torch.Tensor) else None
        if on_device: fn(probe.float(), guides)
        else: fn(probe.float().cpu().numpy(), guides)
        return y
    return (token, stage), ("device (no D2H)" if on_device else "host (numpy; stage includes the D2H copy)")


def time_stages(torch, stages, windows, warmup, iters):
    """Replicates bench_joint.cmd_latency's loop for each window of the 54-frame input. Returns per-window lists of (total ms, per-stage ms) and the warm-up wall times."""
    res = [{"total": [], "stage": {n: [] for n, _ in stages}} for _ in windows]
    wall = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        for wi, x in enumerate(windows):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(stages) + 1)]
            y = x; ev[0].record()
            with torch.autocast("cuda", dtype=torch.float16):
                for k, (_, fn) in enumerate(stages):
                    y = fn(y, 25.0, 3.0); ev[k + 1].record()
            torch.cuda.synchronize()
            if i >= warmup:
                for k, (n, _) in enumerate(stages): res[wi]["stage"][n].append(ev[k].elapsed_time(ev[k + 1]))
                res[wi]["total"].append(ev[0].elapsed_time(ev[-1]))
        if i < warmup: wall.append(time.perf_counter() - t0)
    return res, wall


def pct(v, q):
    import numpy as np
    return float(np.percentile(v, q))


def row_stats(res, wall, nnew):
    """ms values are per WINDOW; window 1 -> /32, window 2 -> /22 per new frame."""
    w1, w2 = res
    return ({"ms_per_frame_p50": statistics.median(w1["total"]) / INFER_LEN, "ms_per_frame_p95": pct(w1["total"], 95) / INFER_LEN, "window_ms_p50": statistics.median(w1["total"]),
             "stage_ms_per_frame_p50": {k: statistics.median(v) / INFER_LEN for k, v in w1["stage"].items()}},
            {"ms_per_new_frame_p50": statistics.median(w2["total"]) / nnew, "ms_per_new_frame_p95": pct(w2["total"], 95) / nnew, "window_ms_p50": statistics.median(w2["total"]),
             "new_frames": nnew, "stage_ms_per_new_frame_p50": {k: statistics.median(v) / nnew for k, v in w2["stage"].items()}},
            {"first_call_s": wall[0], "warmup_wall_s": wall, "steady_warmup_s": wall[-1]})


class Session:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.rows, self.nets, self.group = [], {}, None

    # -- model construction ------------------------------------------------------------------------------------------------
    def setup_torch(self):
        import common, torch
        self.torch, self.common = torch, common
        self.bj = common.import_scorer()
        import patches
        self.patches = patches
        self.bj._vda_path()
        from video_depth_anything.dinov2_layers import attention as A
        from video_depth_anything.motion_module import motion_module as M
        self._orig = dict(attn=A.Attention.forward, memeff=A.MemEffAttention.forward, mot=M.TemporalAttention._attention, bench=torch.backends.cudnn.benchmark)
        self._A, self._M = A, M
        sys.path.insert(0, str(REPO_ROOT))
        import repro.tag_fast_patch as tfp
        self.tfp = tfp
        assert common.sha256_file(common.RAMEN_CKPT) == common.RAMEN_CKPT_SHA, "RAMEN checkpoint sha mismatch"

    def reset_globals(self):
        self._A.Attention.forward, self._A.MemEffAttention.forward = self._orig["attn"], self._orig["memeff"]
        self._M.TemporalAttention._attention = self._orig["mot"]
        self.torch.backends.cudnn.benchmark = self._orig["bench"]

    def build_stack(self, stack):
        torch, bj, patches = self.torch, self.bj, self.patches
        for k in list(self.nets): self.nets.pop(k)
        import gc; gc.collect(); torch.cuda.empty_cache()
        self.reset_globals()
        fl, mf = STACK_FLAGS[stack], mvda_flags(stack)
        t0 = time.perf_counter(); logs = {}
        o_load, o_dm = bj.load_ours, bj.DEPTH_MODELS["metric_vda_s"]

        def load_ours(ckpt, device):
            net = o_load(ckpt, device); logs["ramen"] = patches.apply(net, fl, True); return net

        def vda_released(device, metric):          # == latency_copy.vda_released == bench_joint._vda_released + patches.apply (Auditor-checked)
            bj._vda_path()
            from video_depth_anything.video_depth import VideoDepthAnything
            key = "metric_vda_s" if metric else "vda_s"
            net = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=metric)
            net.load_state_dict(torch.load(bj.PATHS[key], map_location="cpu", weights_only=False), strict=True)
            net = net.to(device).eval(); logs["mvda"] = patches.apply(net, mf, False)
            return dict(fwd=lambda x: (net(bj._norm(x)[None])[0], None), metric=metric, window=32, weights={key: bj.PATHS[key]})
        try:
            bj.load_ours = load_ours; bj.DEPTH_MODELS["metric_vda_s"] = lambda d: vda_released(d, True)
            ramen = bj.build_pipeline(self.common.ramen_spec(), "cuda")
            mvda = bj.build_pipeline("metric_vda_s", "cuda")
        finally:
            bj.load_ours, bj.DEPTH_MODELS["metric_vda_s"] = o_load, o_dm
        self.nets = {"ramen": ramen, "mvda": mvda, "stack": stack, "build_s": time.perf_counter() - t0, "patch_logs": logs}
        # the three flags that no toggle may change (R-TF32)
        assert self.precision()["cudnn_allow_tf32"] is True and self.precision()["matmul_allow_tf32"] is False and self.precision()["float32_matmul_precision"] == "highest", self.precision()

    def precision(self):
        t = self.torch
        return {"cudnn_allow_tf32": bool(t.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(t.backends.cuda.matmul.allow_tf32),
                "float32_matmul_precision": t.get_float32_matmul_precision(), "cudnn_benchmark": bool(t.backends.cudnn.benchmark)}

    # -- one timed item -----------------------------------------------------------------------------------------------------
    def inputs(self):
        t = self.torch
        g = t.Generator(device="cuda").manual_seed(0)
        x54 = t.rand(INFER_LEN + (INFER_LEN - OVERLAP), 3, SIZE, SIZE, device="cuda", generator=g)
        w1 = x54[:INFER_LEN].clone(); w2 = x54[INFER_LEN - OVERLAP:INFER_LEN - OVERLAP + INFER_LEN].clone(); w2[:OVERLAP] = w1[KEYFRAMES]    # vda_windows: keyframes carried
        return [w1, w2]

    def toggles(self, item):
        pa = self.patches
        fl = STACK_FLAGS[item["stack"]] if item["model"] == "ramen" else mvda_flags(item["stack"])
        rec = pa.toggle_record(fl, "fp16 autocast")
        if item["model"] == "mvda": rec.update(bn_folded=False, channels_last=False, refine_hook="n/a", input_staging="n/a (no restorer)", dense_group_conv=False)
        from evaluation import depth_probe
        rec["agf_backend"] = depth_probe.AGF_BACKEND if item["post"] else None
        rec.update(self.precision()); rec["venv"] = sys.prefix + " (shared FocusMamba venv; runtime patches only)"
        rec["flags"] = fl
        return rec

    def run_item(self, item):
        torch, bj = self.torch, self.bj
        p = self.nets[item["model"]]
        depth = p["fwd"]

        def depth_stage(y, s, b):
            out = depth(y)
            if "st_restore" in p: return p["st_restore"](y, out[1])
            return out[1] if isinstance(out, tuple) and out[1] is not None else y
        stages = p.get("stages", []) + [(item["model"], depth_stage)] + p.get("post_stages", [])
        branch = "n/a (no post filter; depth stays on the GPU)"
        if item["post"] == "numpy": self.tfp.unpatch()
        elif item["post"] == "v3": self.tfp.patch()
        if item["post"]:
            ps, branch = make_post_stage(bj, item["setting"], SIZE)
            if item["post"] == "v3":
                from evaluation import depth_probe
                assert depth_probe.AGF_BACKEND == "fast_triton_v3" and branch.startswith("device"), (depth_probe.AGF_BACKEND, branch)
            stages = stages + [ps]
        windows = self.inputs()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated() / 2**30
        with torch.no_grad():
            res, wall = time_stages(torch, stages, windows, self.a.warmup, self.a.iters)
        w1, w2, wu = row_stats(res, wall, INFER_LEN - OVERLAP)
        self.tfp.unpatch()
        rec = {"id": item["id"], "kind": item["kind"], "stack": item["stack"], "model": item["model"], "post": item["post"], "setting": item["setting"],
               "toggles": self.toggles(item) | {"agf_backend": ("fast_triton_v3" if item["post"] == "v3" else "cpu" if item["post"] == "numpy" else None)},
               "protocol": {"size": SIZE, "infer_len": INFER_LEN, "precision": "fp16 autocast", "warmup": self.a.warmup, "iters": self.a.iters, "window2_new_frames": INFER_LEN - OVERLAP,
                            "input_54_frames": "window 1 = frames 0..31; window 2 = frames 22..53 with slots 0..9 = window-1 keyframes; no caching, so window 2 is the same work",
                            "ms_per_frame": "window ms / 32 (window 1); window 2 ms / 22 new frames"},
               "window1": w1, "window2": w2, "warmup": wu,
               "compile_and_build": {"stack_build_s": self.nets["build_s"], "compiled": self.toggles(item)["compiled"], "note": "first_call_s (warm-up pass 0 wall, both windows) includes compile/benchmark/Triton JIT; excluded from the medians"},
               "mem": {"peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30, "resident_before_GiB": resident,
                       "note": "peak includes both the RAMEN and Metric-VDA-S nets resident in this stack" + ("; cudnn.benchmark ON: peak includes algorithm-search workspace" if "cudnn" in STACK_FLAGS[item["stack"]] else "")},
               "output_location": "GPU (end event after the last op)" if item["post"] != "numpy" else "host (numpy TAG; D2H copy inside the timed stage, as released)",
               "post_branch": branch, "post_filter_input": "torch.rand probe (value-independent kernel only; not model depth)" if item["post"] else None,
               "gpu_group": self.group, "code": self.code}
        self.common.write_json(self.out / "rows" / f"{item['id']}.json", rec)
        self.rows.append(rec)
        print(f"{item['id']}: w1 {w1['ms_per_frame_p50']:.3f} ms/frame  w2 {w2['ms_per_new_frame_p50']:.3f} ms/new frame  peak {rec['mem']['peak_alloc_GiB']:.2f}/{rec['mem']['peak_reserved_GiB']:.2f} GiB  first-call {wu['first_call_s']:.1f}s", flush=True)

    # -- phases -------------------------------------------------------------------------------------------------------------
    def harness_check(self):
        """Original bench_joint.cmd_latency in a subprocess BEFORE this process touches CUDA (its own idle check would otherwise see us)."""
        outs = {}
        for key, spec in (("ramen_raw", self.common.ramen_spec()), ("metric_vda_s", "metric_vda_s")):
            d = self.out / "harness_check" / key
            cmd = [sys.executable, "-I", str(HERE / "harness_orig.py"), "latency", "--model", spec, "--out", str(d), "--size", str(SIZE), "--warmup", str(self.a.warmup), "--iters", str(self.a.iters)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            f = next(d.glob("latency__*.json")); outs[key] = json.loads(f.read_text())["per_frame_ms_p50"]
        return outs

    def standalone_v3(self):
        """FASTTAG-C: kernel timing on saved real depth (fasttag_b/run/depth) at 518^2 (real depth resized), 374x1242 (KITTI), 480x640 (Bonn), both settings; V3s streaming per-frame cost."""
        import numpy as np, torch
        import torch.nn.functional as F
        from repro.tag_fast import agf_v3, Agf3Stream
        com, res = self.common, []
        clips = {ds: sorted((DEPTH_DIR / ds / "llvar").glob("*.npy"))[0] for ds in ("kitti", "bonn", "scannetpp")}
        cases = [("518x518", "kitti", (518, 518)), ("374x1242", "kitti", None), ("480x640", "bonn", None)]
        for label, ds, resize in cases:
            arr = np.load(clips[ds])[:INFER_LEN]
            h5 = com.array_sha256(arr)
            x = torch.from_numpy(np.ascontiguousarray(arr)).cuda().float()
            if resize: x = F.interpolate(x[:, None], size=resize, mode="bilinear", align_corners=True)[:, 0].contiguous()
            for src, xin in (("real_depth", x), ("torch.rand", torch.rand(x.shape, device="cuda") * 80)):
                for tok, (amin, tau) in SETTINGS.items():
                    def timeit(fn, n=INFER_LEN):
                        for _ in range(WARMUP): fn()
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); v = []
                        for _ in range(ITERS):
                            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            a.record(); fn(); b.record(); b.synchronize(); v.append(a.elapsed_time(b))
                        return statistics.median(v), pct(v, 95)
                    w, w95 = timeit(lambda: agf_v3(xin, amin, tau))
                    st = Agf3Stream(xin[0], amin, tau)
                    def stream():
                        for t in range(1, INFER_LEN): st.step(xin[t])
                    s, s95 = timeit(stream)
                    res.append({"resolution": label, "clip": str(clips[ds].name), "source": src, "real_array_sha256_32f": h5 if src == "real_depth" else None, "resized_from_native": bool(resize) and src == "real_depth",
                                "setting": tok, "v3_window_ms_p50": w, "v3_window_ms_p95": w95, "v3_ms_per_frame_amortised_in_window": w / INFER_LEN,
                                "v3s_stream_31steps_ms_p50": s, "v3s_ms_per_frame_streaming": s / (INFER_LEN - 1),
                                "label": "descriptive, one session, one GPU; the window kernel loops over 32 frames inside one launch (amortised), per-frame streaming cost = V3s (one launch per frame); input on GPU, no H2D/D2H",
                                "peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30})
        com.write_json(self.out / "fasttag_c_standalone_v3.json", {"results": res, "agf_backend": "fast_triton_v3", "code": self.code})
        return res

    def run(self):
        a, com = self.a, None
        import common as com
        self.common = com
        self.out.mkdir(parents=True, exist_ok=True)
        s0 = gpu_snapshot(); require_idle(s0, own_pid=-1)
        lock_acquire(GPU_LOCK if not a.lock else Path(a.lock), "latency ladder"); atexit.register(lock_release, GPU_LOCK if not a.lock else Path(a.lock))
        self.code = code_record()
        if not self.code["min_commit_is_ancestor"]: raise SystemExit(f"harness commit {MIN_HARNESS_COMMIT} is not an ancestor of HEAD {self.code['HEAD']}")
        before = com.untouched_record("BEFORE (ladder session)")
        com.write_json(self.out / "UNTOUCHED_BEFORE.json", before)
        harness = {} if a.skip_harness_check else self.harness_check()
        self.setup_torch()
        plan = build_plan([s for s in STACK_ORDER if s in a.stacks])
        com.write_json(self.out / "session.json", {"plan": plan, "code": self.code, "gpu_before": s0, "torch": self.torch.__version__, "venv": sys.prefix, "precision": self.precision() if hasattr(self, "torch") else None,
                                                   "harness_original_ms_per_frame": harness, "args": vars(a), "device": self.torch.cuda.get_device_name()})
        cur_build, last_stack = None, None
        for it in plan:
            if it["stack"] != last_stack:
                if last_stack is not None:
                    self.group["after"] = gpu_snapshot()
                bf = BUILD_FROM.get(it["stack"], it["stack"])
                if bf != cur_build: self.build_stack(bf); cur_build = bf
                self.nets["stack"] = it["stack"]
                self.group = {"stack": it["stack"], "before": gpu_snapshot(), "after": None}
                last_stack = it["stack"]
            self.run_item(it)
            self.group["after"] = gpu_snapshot()
        standalone = [] if a.skip_standalone else self.standalone_v3()
        after = com.untouched_record("AFTER (ladder session)"); com.write_json(self.out / "UNTOUCHED_AFTER.json", after)
        summ = summarize(self.rows, harness)
        summ["untouched_equal"] = all(before[k]["HEAD"] == after[k]["HEAD"] and before[k]["status"] == after[k]["status"] for k in ("eval_clone", "vda")) and before["focusmamba_freeze_sha256"] == after["focusmamba_freeze_sha256"]
        com.write_json(self.out / "LADDER_SUMMARY.json", summ); (self.out / "LADDER_TABLES.md").write_text(markdown_tables(summ))
        s1 = gpu_snapshot(); com.write_json(self.out / "gpu_after.json", s1)
        com.write_sha256sums(self.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stacks", nargs="*", default=LADDER_DEFAULT); ap.add_argument("--warmup", type=int, default=WARMUP); ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--lock", default=None, help="override GPU_LOCK path (tests)"); ap.add_argument("--skip-harness-check", action="store_true"); ap.add_argument("--skip-standalone", action="store_true")
    a = ap.parse_args(argv)
    plan = build_plan([s for s in STACK_ORDER if s in a.stacks])
    if a.dry_run:
        print(dry_run(plan)); return 0
    Session(a).run(); return 0


if __name__ == "__main__":
    sys.exit(main())

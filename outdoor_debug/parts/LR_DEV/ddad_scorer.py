#!/usr/bin/env python3
"""DDAD val LR_DEV scorer. EXPLORATORY DEV, REPORT-ONLY (REGISTRATION.md A1/A2, Auditor O11). Never enters a decision.

Metrics (O11, exactly): GT valid = projected LIDAR depth in (0, 200] m. Bins by GT distance [0,20) [20,50) [50,80) [80,120) [120,200] m.
Per bin: unaligned PIXEL-POOLED AbsRel = sum|p-g|/g / n over all valid px of all scored frames; median pred:gt = median over the pooled px of p/g;
px count; number of frames and scenes contributing. Overall AbsRel over (0,200]. No alignment of any kind, no clamping of pred, GT never resized.
Resolution rule A1: pred -> bilinear (align_corners=False) to native 1936x1216, sampled at the projected LIDAR px (ddad_lr_dataset.sample_prediction).
Model input A2: bench_joint Resize lower_bound, ensure_multiple_of 14, INTER_CUBIC, short_side 378, fp16 autocast.

Usage:  python -I ddad_scorer.py --ckpt <RAMEN resumable .pt> --out result.json [--repo <repo root>] [--device cuda] [--limit-scenes N]
        python -I ddad_scorer.py --selftest        (synthetic frames only, no DDAD data, no model)
"""
from __future__ import annotations

import argparse, hashlib, json, socket, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scorer"))
import ddad_lr_dataset as D          # noqa: E402  (loader; imported, not modified)
import scorer as S                   # noqa: E402  (BJ_SHA, sha256_file; imported, not modified)

FRAMES_SHA = "b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098"
PAIRS_SHA = "962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245"
BJ_SHA = "5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef"
SHORT_SIDE = 378
BINS_M = [(0.0, 20.0), (20.0, 50.0), (50.0, 80.0), (80.0, 120.0), (120.0, 200.0)]   # [lo,hi) except the last, closed [120,200]
MAX_M = 200.0
LABEL = "exploratory DEV, report-only (DDAD val LR_DEV; not a held-out test, never enters the decision)"
# REGISTRATION.md A3 (Auditor O29): release-spec branch. Same frames / bins / pins / 378 input / fp16 / out_hw trick / shape assert as ours:.
RELEASE_LABEL = "released Metric-VDA-S at the registered 378 input (not its default 518), fp16, exploratory DEV, report-only"
RELEASE = {"metric_vda_s": dict(   # bench_joint DEPTH_MODELS key -> _vda_released(dev, metric=True): vits, metric=True head, fwd returns METRES (no scale/shift in vda_windows)
    isambard_path="/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth",
    sha256="3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33", size=116444063, label=RELEASE_LABEL)}
assert BINS_M == D.BINS_M or [tuple(map(float, b)) for b in D.BINS_M] == BINS_M


def bin_labels() -> list[str]:
    return [f"[{lo:g},{hi:g}{']' if i == len(BINS_M) - 1 else ')'}" for i, (lo, hi) in enumerate(BINS_M)]


def valid_mask(gt: np.ndarray) -> np.ndarray:
    """0 < gt <= 200, finite (NaN / 0 / negative / >200 excluded)."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(gt) & (gt > 0) & (gt <= MAX_M)


def bin_masks(g: np.ndarray) -> list[np.ndarray]:
    """Masks over a 1-D array of valid GT depths. Half-open [lo,hi), last bin closed at 200."""
    out = []
    for i, (lo, hi) in enumerate(BINS_M):
        out.append((g >= lo) & ((g <= hi) if i == len(BINS_M) - 1 else (g < hi)))
    return out


class Accumulator:
    """Pools (pred, gt) samples over frames. Stores per-bin float64 |p-g|/g sums and the p/g ratios (for the pooled median)."""

    def __init__(self):
        n = len(BINS_M)
        self.n = [0] * n
        self.s = [0.0] * n
        self.ratios = [[] for _ in range(n)]
        self.frames = [0] * n
        self.scenes = [set() for _ in range(n)]
        self.scored_frames = 0
        self.scored_scenes = set()
        self.n_nonfinite_pred = 0

    def add(self, p: np.ndarray, g: np.ndarray, scene: str):
        """p, g: 1-D arrays at the valid GT px of ONE frame (pred sampled at GT px; no alignment, no clamp)."""
        p = np.asarray(p, np.float64).ravel(); g = np.asarray(g, np.float64).ravel()
        assert p.shape == g.shape
        v = valid_mask(g)
        p, g = p[v], g[v]
        self.n_nonfinite_pred += int((~np.isfinite(p)).sum())
        self.scored_frames += 1
        self.scored_scenes.add(scene)
        for i, m in enumerate(bin_masks(g)):
            if not m.any():
                continue
            pm, gm = p[m], g[m]
            self.n[i] += int(pm.size)
            self.s[i] += float((np.abs(pm - gm) / gm).sum())
            self.ratios[i].append((pm / gm).astype(np.float64))
            self.frames[i] += 1
            self.scenes[i].add(scene)

    def result(self) -> dict:
        bins = {}
        for i, lab in enumerate(bin_labels()):
            r = np.concatenate(self.ratios[i]) if self.ratios[i] else np.zeros(0)
            bins[lab] = dict(n_px=self.n[i], absrel_unaligned_pooled=(self.s[i] / self.n[i]) if self.n[i] else float("nan"),
                             median_pred_over_gt=float(np.median(r)) if r.size else float("nan"),
                             n_frames=self.frames[i], n_scenes=len(self.scenes[i]))
        tn = sum(self.n)
        return dict(label=LABEL, bins=bins, overall_absrel_0_200=(sum(self.s) / tn) if tn else float("nan"), overall_n_px=tn,
                    n_scored_frames=self.scored_frames, n_scenes=len(self.scored_scenes), n_nonfinite_pred_px=self.n_nonfinite_pred)


def score_frame(acc: Accumulator, pred: np.ndarray, gt: np.ndarray, scene: str):
    """pred: (h,w) metres at ANY resolution; gt: native (1216,1936) sparse map. GT is never resized (asserted)."""
    if tuple(gt.shape) != tuple(D.NATIVE_HW):
        raise ValueError(f"GT shape {gt.shape} != native {D.NATIVE_HW}: the registered rule forbids resizing GT")
    p, g = D.sample_prediction(pred, gt)       # A1: bilinear pred -> native, sampled at valid projected-LIDAR px
    acc.add(p, g, scene)


def assert_pins(frames_tsv=None, pairs_tsv=None) -> dict:
    ft = Path(frames_tsv or D.FRAMES_TSV); pt = Path(pairs_tsv or D.PAIRS_TSV)
    fs, ps = S.sha256_file(ft), S.sha256_file(pt)
    assert fs == FRAMES_SHA, f"frame list sha {fs} != registered {FRAMES_SHA}"
    assert ps == PAIRS_SHA, f"pairs sha {ps} != registered {PAIRS_SHA}"
    return dict(frames_tsv=str(ft), frames_sha256=fs, pairs_tsv=str(pt), pairs_sha256=ps)


# ----------------------------------------------------------------------------- model path
def model_input(paths, hw_cache=None):
    """bench_joint.load_frames equivalent for DDAD (A2): per-frame RGB -> /255 -> Resize(lower_bound, mult 14, cubic, short side 378) -> (N,3,h,w)."""
    import cv2, torch
    import bench_joint as bj
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=SHORT_SIDE, height=SHORT_SIDE, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    ims = [rs({"image": D.load_rgb(p).astype(np.float32) / 255.0})["image"] for p in paths]
    return torch.from_numpy(np.clip(np.stack(ims), 0, 1)).permute(0, 3, 1, 2).contiguous()


def score_clip(acc, scene, frames, predict, root=None, loader=None):
    """frames: list of frame dicts (ddad_lr_dataset.load_clips), windowed to max_len. predict(frames)->(N,h,w) metres at MODEL resolution
    (the A1 bilinear upsample is done here by sample_prediction, so the model output is NOT pre-resized). Scored = frames with GT."""
    frames = frames[:D.DDAD_LR_SPEC["max_len"]]
    depth = predict(frames)
    assert len(depth) == len(frames)
    n = 0
    for i, f in enumerate(frames):
        if not f["scored"]:
            continue
        gt = (loader or (lambda ff: D.load_gt([ff], root)[0]))(f)
        score_frame(acc, depth[i], gt, scene)
        n += 1
    return n


def release_meta(key: str, paths: dict) -> dict:
    """A3: weights provenance of a released model. Unknown key / missing file / sha != registered -> raises. `paths` = bench_joint.PATHS."""
    if key not in RELEASE:
        raise ValueError(f"unknown release spec {key!r}; registered: {sorted(RELEASE)}")
    r = RELEASE[key]; w = Path(paths[key])
    assert w.is_file(), f"release weights missing: {w}"
    sha = S.sha256_file(w)
    assert sha == r["sha256"], f"release weights sha {sha} != registered {r['sha256']} ({w})"
    assert r["label"] == RELEASE_LABEL, "release label altered"
    return dict(spec=key, release_weights=dict(key=key, path=str(w), sha256=sha, size=w.stat().st_size, registered_isambard_path=r["isambard_path"]))


def model_meta(ckpt, spec, paths: dict) -> tuple[str, dict]:
    """-> (bench_joint spec string, meta fragment). ours: path unchanged (trainable_ema assert + ckpt meta); release path via release_meta."""
    if spec:
        assert not ckpt, "release --spec must not be combined with --ckpt (an 'ours' checkpoint)"
        return spec, release_meta(spec, paths)
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "trainable_ema" in ck, "checkpoint has no trainable_ema (bench_joint.load_ours would silently use 'trainable'); refused"
    ck_meta = dict(path=str(ckpt), sha256=S.sha256_file(ckpt), weights_key="trainable_ema", step=ck.get("step"),
                   arch=(ck.get("cfg", {}).get("model", {}) or {}).get("arch"))
    del ck
    return None, dict(ckpt=ck_meta)


def assert_same_hw(d, x):
    assert tuple(d.shape[-2:]) == tuple(x.shape[-2:]), f"pipeline depth {tuple(d.shape[-2:])} != model input {tuple(x.shape[-2:])}: an extra resample happened (A1 allows only the sample_prediction upsample)"   # Auditor O26 (1)


def selftest(spec=None) -> dict:
    """Synthetic frames only (no DDAD data, no model): smooth depth ramp D at native res, sparse GT = D at random px, stub model = 0.9*D at model res."""
    import torch, torch.nn.functional as F
    rng = np.random.default_rng(0)
    H, W = D.NATIVE_HW
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    acc = Accumulator()
    for scene in ("sA", "sB"):
        for k in range(3):
            field = (5.0 + 190.0 * (xx / W) * (0.5 + 0.5 * yy / H) + 3 * k).astype(np.float32)
            gt = np.full((H, W), np.nan, np.float32)
            idx = rng.choice(H * W, 20000, replace=False)
            gt.flat[idx] = field.flat[idx]
            pred_small = F.interpolate(torch.from_numpy(0.9 * field)[None, None], size=(378, 602), mode="bilinear", align_corners=False)[0, 0].numpy()
            score_frame(acc, pred_small, gt, scene)
    r = acc.result()
    if spec:
        assert spec in RELEASE; r["label"] = RELEASE_LABEL; r["spec"] = spec
    r["selftest_note"] = "pred=0.9*smooth ramp at 378x602, bilinear back to native: ratios ~0.9 except at image borders (edge clamp)"
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--out")
    ap.add_argument("--spec", default=None, help="A3: released model = a bench_joint DEPTH_MODELS key registered in RELEASE (metric_vda_s); mutually exclusive with --ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp16", choices=["fp16"], help="fp16 autocast only (registered)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--spec-prefix", default="ours:")
    ap.add_argument("--limit-scenes", type=int, default=None, help="smoke only; recorded in meta, pins_asserted stays True for the lists but status=PARTIAL")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(a.spec), indent=1)); return
    assert a.out and (a.ckpt or a.spec), "--out and one of --ckpt / --spec required"
    if a.spec: assert not a.ckpt, "release --spec must not be combined with --ckpt (an 'ours' checkpoint)"
    out_path = Path(a.out); assert not out_path.exists(), f"{out_path} exists (never overwrite)"
    pins = assert_pins()
    D.read_frame_list(); D.read_pairs()     # the loader's own sha checks too
    assert D.DDAD_LR_SPEC["short_side"] == SHORT_SIDE == 378
    import torch
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import bench_joint as bj
    bj_sha = S.sha256_file(bj.__file__)
    assert bj_sha == BJ_SHA, f"bench_joint.py sha {bj_sha} != pinned {BJ_SHA}"
    bj_spec, mm = model_meta(a.ckpt, a.spec, bj.PATHS)
    dev = a.device
    ac = dict(device_type="cuda", dtype=torch.float16, enabled=dev.startswith("cuda"))
    assert ac["enabled"], "fp16 autocast requires cuda (registered precision)"
    pipe = bj.build_pipeline(bj_spec or (a.spec_prefix + a.ckpt), dev)
    assert pipe["metric"], "scorer needs a metric model"
    root = D.SCENES_DIR

    def predict(frames):
        x = model_input([root / f["image"] for f in frames]).to(dev)
        with torch.autocast(**ac):
            d, _ = bj.run_pipeline(pipe, x, 0.0, 0.0, tuple(x.shape[-2:]))   # out_hw = model resolution: the A1 bilinear upsample happens in sample_prediction
        assert_same_hw(d, x)
        if not getattr(predict, "_shape_logged", False):
            print(f"[ddad_scorer] first clip: model input {tuple(x.shape)} -> depth {tuple(d.shape)} (same HxW, identity resample)", flush=True); predict._shape_logged = True
        return d

    clips = D.load_clips(a.limit_scenes)
    acc = Accumulator(); t0 = time.time(); per_scene = {}
    for sc, frames in clips:
        per_scene[sc] = score_clip(acc, sc, frames, predict, root)
        print(f"{sc} scored={per_scene[sc]} {time.time() - t0:.0f}s", flush=True)
    res = acc.result()
    if a.spec: res["label"] = RELEASE_LABEL
    full = a.limit_scenes is None
    if full:
        assert res["n_scored_frames"] == 395 and res["n_scenes"] == 50, (res["n_scored_frames"], res["n_scenes"])
    meta = dict(label=RELEASE_LABEL if a.spec else LABEL, status="FULL" if full else "PARTIAL (--limit-scenes)", limit_scenes=a.limit_scenes, pins_asserted=full,
                pins=pins, bench_joint_sha256=bj_sha, bench_joint_pinned=BJ_SHA, ddad_scorer_sha256=S.sha256_file(__file__),
                dataset_module_sha256=S.sha256_file(D.__file__), scorer_module_sha256=S.sha256_file(S.__file__), **mm,
                ramen_commit=bj.git_state(bj.REPO), precision="fp16", short_side=SHORT_SIDE, max_len=D.DDAD_LR_SPEC["max_len"],
                resolution_rule="A1: bilinear(align_corners=False) pred -> 1936x1216, sampled at projected LIDAR px; GT never resized",
                alignment="none", host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                seconds=time.time() - t0, bins_m=bin_labels(), per_scene_scored=per_scene)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(meta=meta, **res), indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

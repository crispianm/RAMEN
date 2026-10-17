#!/usr/bin/env python3
"""AV2 Sensor VAL scorer for the STRICT confirmatory run TS2+H300 vs TS2 (AV2_REGISTRATION.md). DRAFT: the model/real-data path has NEVER run.

Per checkpoint (one JSON):
  PRIMARY statistic input (prediction only, no GT): for each scored frame, the median PREDICTED depth over CERTAIN-SKY pixels (frozen mask, nearest-resized to the model grid,
    frames with < 2000 certain-sky px at the model grid are dropped and counted); P = median over the kept frames (frame-balanced). Also: pooled median (description), frac of sky px > 200 / 300 m.
  GUARDS: unaligned pixel-pooled AbsRel and pooled median pred:gt in the bins [0,20) [20,50) [50,80) [80,120) [120,200] m over LiDAR-projected px with the ANY-SKY mask excluded
    (A1: prediction bilinearly resized to native 2048x1550 and sampled at the projected LiDAR px; GT never resized; no alignment, no clamp). Overall AbsRel 0-200.
Model input mirrors ddad_scorer (A2): bench_joint Resize lower_bound, multiple of 14, INTER_CUBIC, short side 378, fp16 autocast; portrait 2048x1550 -> 504x378 (Resize rounding).
Decision (Stage 1, Add. 48): per pair d_i = ln(P_H300 / P_TS2) for pairs s50/s51/s52; CONFIRMED iff all three d_i AND their mean >= ln 1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0 (Auditor O39); unevaluable guard never passes.
Inputs (Auditor O45): the model clip and the scored frames come ONLY from the pinned av2_clips.tsv + av2_frames.tsv (cross-checked against objects.tsv); the data dir is never globbed.
The scorer's OWN sha256 is pinned OUTSIDE the file: --expect-scorer-sha <literal from RUNBOOK_STAGE1.md> is REQUIRED and compared with the sha of the running file.
Usage:  python -I av2_scorer.py --ckpt <ours .pt> --pair s50|s51|s52|stage2 --arm TS2|H300 | --spec metric_vda_s  --out r.json --root <av2_val dir> --mask <sky_masks.npz>
            --frames <av2_frames.tsv> --clips <av2_clips.tsv> --objects <objects.tsv> --logs <logs.txt> --expect-scorer-sha <sha256>
        python -I av2_scorer.py --compare H300.json TS2.json [--out verdict.json]
        python -I av2_scorer.py --selftest
"""
from __future__ import annotations

import argparse, json, math, socket, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "LR_DEV"))
sys.path.insert(0, str(HERE.parent.parent / "scorer"))
import av2_dataset as A            # noqa: E402

UNSET = "UNSET"
# ---- pins: filled in the registration BEFORE the first prediction. UNSET makes assert_pins fail (a scorer with unset pins cannot run on real data).
# The scorer's OWN sha is NOT in this dict (it cannot contain its own hash): it is passed from outside as --expect-scorer-sha (check_scorer_sha).
PINS = dict(logs_sha="f7f508f958e8e9d173772677d239c7a53165a0fb67520a68a4bfcf134592e370", **A.REGISTERED_SHAS,        # frames_sha, clips_sha, objects_sha
            mask_sha=UNSET, meta_sha=UNSET, dataset_module_sha=UNSET, ddad_scorer_sha=UNSET, sums_sha=UNSET,
            segmenter_a_sha="a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df",
            segmenter_b_sha="876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c",
            bench_joint_sha="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef")
EXPECT_STEP = 14000                  # registration 'Pins': checkpoint step 14000 (read from the checkpoint file)
HOST_GPU = "GH200"                   # registration 'Pins': host type GH200 (substring of the CUDA device name)
IOU_MIN = 0.85                       # R1 M2: the mask meta's pooled segmenter IoU must be >= this
BINS_M = A.BINS_M
MAX_M = A.MAX_DEPTH_M
MIN_SKY_PX = 2000                    # certain-sky px at the model grid for a frame to count in P
LN_MARGIN = math.log(1.5)
GUARD_TOL = 0.05                     # H300 AbsRel <= (1 + 0.05) * same-seed TS2 AbsRel, per bin
GUARD_MIN_PX = 20000                 # a bin with fewer pooled px in either arm is reported "UNEVALUABLE" (not silently passed): the verdict says so
PAIRS = ("s50", "s51", "s52")           # Stage 1 pairs: c3_n46/seed 50, c3_n47/seed 51, c3_n48/seed 52 (Add. 48)
PAIR_LABELS = PAIRS + ("stage2",)        # 'stage2' = the single from-scratch seed-50 pair of Stage 2
SEED_OF = dict(s50=50, s51=51, s52=52, stage2=50)
ARMS = ("H300", "TS2")
# Registered disclosure strings, one per stage (AV2_REGISTRATION.md 'Stage structure' and 'Metrics'); compare()/compare_pairs() print the matching one.
STAGE1_NOTE = ("Stage 1: THREE paired 2000-step continuations (n46/s50, n47/s51, n48/s52); 3 pairs size the pair-to-pair spread only roughly; the A0/A0b screen pair showed 26% spread on a guard. "
               "P measures what the H300 hinge enforces by design (a PASS shows transfer of the hinge's effect, not accuracy; the guards carry the information). "
               "TS2+H300 was chosen after the screen on a descriptive sky secondary and was NOT BETTER on its primary (the screen primary).")
STAGE2_NOTE = ("SINGLE SEED (50): one paired difference cannot size seed noise; the A0/A0b screen pair showed 26% spread on a guard. Sign and magnitude only. "
               "Framing (O35): P measures what the hinge enforces by design; a pass is near built-in and is NOT an accuracy claim; the guards carry the information.")
LABEL_STAGE1 = "confirmatory STRICT, Stage 1 (AV2 sensor val, held out); three paired starts s50/s51/s52; primary = predicted sky depth by design, NOT accuracy"
LABEL_STAGE2 = "confirmatory STRICT, Stage 2 (AV2 sensor val, held out); single seed 50; primary = predicted sky depth by design, NOT accuracy"
REF_LABEL = "REFERENCE row, description only: released Metric-VDA-S at the registered 378 input, fp16; never enters the decision"


def bin_labels() -> list[str]:
    return [f"[{lo:g},{hi:g}{']' if i == len(BINS_M) - 1 else ')'}" for i, (lo, hi) in enumerate(BINS_M)]


def bin_masks(g: np.ndarray) -> list[np.ndarray]:
    return [(g >= lo) & ((g <= hi) if i == len(BINS_M) - 1 else (g < hi)) for i, (lo, hi) in enumerate(BINS_M)]


def assert_pins(actual: dict, registered: dict | None = None) -> dict:
    reg = PINS if registered is None else registered
    for k, want in reg.items():
        if want == UNSET:
            raise AssertionError(f"pin {k} is UNSET in the scorer: the registration has not been completed")
        got = actual.get(k)
        if got != want:
            raise AssertionError(f"pin {k}: {got} != registered {want}")
    return dict(actual)


def check_scorer_sha(expected: str | None, path=None) -> str:
    """D1: the running scorer file must hash to the sha passed from OUTSIDE (a literal committed in RUNBOOK_STAGE1.md). Absent / mismatch -> raises."""
    if not expected:
        raise SystemExit("--expect-scorer-sha is required (the literal sha256 of the final committed av2_scorer.py, from RUNBOOK_STAGE1.md)")
    got = A.sha256_file(path or __file__)
    if got != expected:
        raise AssertionError(f"scorer sha {got} != expected {expected}: refusing to run")
    return got


def assert_ckpt_facts(ck_meta: dict, pair: str) -> dict:
    """Registered checkpoint facts: step == 14000 (read from the checkpoint file by ddad_scorer.model_meta); the seed is carried by the pair label (s50/s51/s52 -> 50/51/52; stage2 -> 50)."""
    if ck_meta.get("step") != EXPECT_STEP:
        raise AssertionError(f"checkpoint step {ck_meta.get('step')!r} != registered {EXPECT_STEP}")
    if pair not in SEED_OF:
        raise AssertionError(f"pair label {pair!r} not in {sorted(SEED_OF)}")
    return dict(ckpt_step=ck_meta["step"], expected_seed=SEED_OF[pair])


def assert_host(gpu_name: str) -> str:
    if HOST_GPU not in str(gpu_name):
        raise AssertionError(f"host GPU {gpu_name!r} does not contain {HOST_GPU!r} (registered host type)")
    return gpu_name


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name()


def autocast_ctx(dev: str):
    """Registered precision: fp16 autocast on cuda ONLY. A non-cuda device raises (tests monkeypatch this function, not the check)."""
    import torch
    if not str(dev).startswith("cuda"):
        raise AssertionError("fp16 autocast requires cuda (registered precision)")
    return torch.autocast(device_type="cuda", dtype=torch.float16)


# ----------------------------------------------------------------------------- guards (LiDAR px, sky excluded)
class Accumulator:
    def __init__(self):
        n = len(BINS_M)
        self.n = [0] * n; self.s = [0.0] * n
        self.ratios = [[] for _ in range(n)]
        self.frames = [0] * n; self.logs = [set() for _ in range(n)]
        self.scored_frames = 0; self.logs_seen = set()
        self.n_sky_excluded = 0; self.n_lidar_valid_before_sky = 0; self.n_nonfinite_pred = 0

    def add(self, p, g, log, sky_at_px=None):
        """p, g: 1-D at the valid LiDAR px of one frame. sky_at_px: bool 1-D (any-sky) or None (variant that includes sky: for tests / description only)."""
        p = np.asarray(p, np.float64).ravel(); g = np.asarray(g, np.float64).ravel()
        assert p.shape == g.shape
        with np.errstate(invalid="ignore"):
            v = np.isfinite(g) & (g > 0) & (g <= MAX_M)
        self.n_lidar_valid_before_sky += int(v.sum())
        if sky_at_px is not None:
            sk = np.asarray(sky_at_px, bool).ravel(); assert sk.shape == g.shape
            self.n_sky_excluded += int((v & sk).sum())
            v = v & ~sk
        p, g = p[v], g[v]
        self.n_nonfinite_pred += int((~np.isfinite(p)).sum())
        self.scored_frames += 1; self.logs_seen.add(log)
        for i, m in enumerate(bin_masks(g)):
            if not m.any():
                continue
            pm, gm = p[m], g[m]
            self.n[i] += int(pm.size); self.s[i] += float((np.abs(pm - gm) / gm).sum())
            self.ratios[i].append(pm / gm); self.frames[i] += 1; self.logs[i].add(log)

    def result(self) -> dict:
        bins = {}
        for i, lab in enumerate(bin_labels()):
            r = np.concatenate(self.ratios[i]) if self.ratios[i] else np.zeros(0)
            bins[lab] = dict(n_px=self.n[i], absrel_unaligned_pooled=(self.s[i] / self.n[i]) if self.n[i] else float("nan"),
                             median_pred_over_gt=float(np.median(r)) if r.size else float("nan"), n_frames=self.frames[i], n_logs=len(self.logs[i]))
        tn = sum(self.n)
        return dict(bins=bins, overall_absrel_0_200=(sum(self.s) / tn) if tn else float("nan"), overall_n_px=tn, n_scored_frames=self.scored_frames,
                    n_logs=len(self.logs_seen), n_lidar_px_sky_excluded=self.n_sky_excluded, n_lidar_px_valid_before_sky=self.n_lidar_valid_before_sky,
                    n_nonfinite_pred_px=self.n_nonfinite_pred)


def sample_prediction(pred: np.ndarray, gt: np.ndarray):
    """A1: bilinear (align_corners=False) pred (h,w) metres -> native 2048x1550; return (pred_at_valid, gt_at_valid, valid_mask). GT never resized (shape asserted by the caller)."""
    import torch, torch.nn.functional as F
    m = np.isfinite(gt) & (gt > 0)
    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    if tuple(p.shape[-2:]) != tuple(A.NATIVE_HW):
        p = F.interpolate(p, size=A.NATIVE_HW, mode="bilinear", align_corners=False)
    return p[0, 0].numpy()[m], gt[m], m


def score_frame_guard(acc: Accumulator, pred, gt, log, any_sky=None):
    if tuple(gt.shape) != tuple(A.NATIVE_HW):
        raise ValueError(f"GT shape {gt.shape} != native {A.NATIVE_HW}: the registered rule forbids resizing GT")
    p, g, m = sample_prediction(pred, gt)
    sk = None
    if any_sky is not None:
        if tuple(any_sky.shape) != tuple(A.NATIVE_HW):
            raise ValueError("any-sky mask must be at native resolution")
        sk = any_sky[m]
    acc.add(p, g, log, sk)


# ----------------------------------------------------------------------------- primary (prediction only)
def mask_to_grid(mask_native: np.ndarray, hw) -> np.ndarray:
    import cv2
    return cv2.resize(mask_native.astype(np.uint8), (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_NEAREST).astype(bool)


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v, kind="stable"); v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


class SkyAccumulator:
    def __init__(self):
        self.frame_medians = []; self.frame_n = []; self.dropped = 0; self.seen = 0
        self.kept_ids = []; self.dropped_ids = []; self.mask_px = []
        self.n_gt200 = 0; self.n_gt300 = 0; self.n_px = 0; self._samples = []; self._weights = []

    def add(self, pred: np.ndarray, certain_native: np.ndarray, fid=None):
        fid = fid if fid is not None else f"#{self.seen}"
        self.seen += 1
        m = mask_to_grid(certain_native, pred.shape[-2:])
        self.mask_px.append(int(m.sum()))
        v = np.asarray(pred, np.float64)[m]
        v = v[np.isfinite(v)]
        if v.size < MIN_SKY_PX:
            self.dropped += 1; self.dropped_ids.append(fid); return
        self.kept_ids.append(fid)
        self.frame_medians.append(float(np.median(v))); self.frame_n.append(int(v.size))
        self.n_px += v.size; self.n_gt200 += int((v > 200).sum()); self.n_gt300 += int((v > 300).sum())
        rng = np.random.default_rng(0)
        smp = v if v.size <= 20000 else rng.choice(v, 20000, replace=False)
        self._samples.append(smp); self._weights.append(np.full(smp.size, v.size / smp.size))     # weights restore the true pixel pooling

    def result(self) -> dict:
        ids = dict(kept_frame_ids=list(self.kept_ids), dropped_frame_ids=list(self.dropped_ids), frame_mask_px=list(self.mask_px))
        if not self.frame_medians:
            return dict(P_framebalanced_median=float("nan"), n_frames_kept=0, n_frames_dropped=self.dropped, n_frames_seen=self.seen, **ids)
        return dict(P_framebalanced_median=float(np.median(self.frame_medians)), n_frames_kept=len(self.frame_medians), n_frames_dropped=self.dropped,
                    n_frames_seen=self.seen, pooled_median_description=weighted_median(np.concatenate(self._samples), np.concatenate(self._weights)),
                    frac_sky_px_gt200=self.n_gt200 / self.n_px, frac_sky_px_gt300=self.n_gt300 / self.n_px, n_sky_px=int(self.n_px),
                    frame_medians=self.frame_medians, **ids)


def route_frame(acc: Accumulator, sky: SkyAccumulator, fid: str, pred, certain, any_sky, gt, log: str, mask_dropped) -> bool:
    """Per scored frame (R1 M4): a mask-dropped frame (< 1% certain-sky px, listed in the mask meta) is removed from the PRIMARY P ONLY (the sky accumulator never sees it);
    the guard accumulator ALWAYS sees every scored frame. Returns True iff the frame was mask-dropped."""
    dropped = is_mask_dropped(fid, mask_dropped)
    if not dropped:
        sky.add(pred, certain, fid)
    score_frame_guard(acc, pred, gt, log, any_sky)
    return dropped


# ----------------------------------------------------------------------------- decision (Auditor O36, single seed)
def decide(d: float, guards_pass: bool, unevaluable=()) -> str:
    if not np.isfinite(d):
        return "INVALID (non-finite d)"
    if d <= 0:
        return "REFUTED"
    if unevaluable:             # Auditor O37: an unevaluable guard NEVER passes by default (the far guard is the informative leg)
        return "NOT CONFIRMED (" + ", ".join(sorted(unevaluable)) + " guard unevaluable)"
    if d >= LN_MARGIN and guards_pass:
        return "CONFIRMED"
    return "NOT CONFIRMED"      # 0 < d < ln 1.5, or d >= ln 1.5 with a failed guard


def guard_check(h_bins: dict, t_bins: dict, tol: float = GUARD_TOL, min_px: int = GUARD_MIN_PX) -> dict:
    """Per bin: H300 AbsRel <= (1+tol) * TS2 AbsRel (same seed). Bins with < min_px in either arm -> status UNEVALUABLE (reported; the verdict line says so)."""
    out = {}
    for lab in h_bins:
        h, t = h_bins[lab], t_bins[lab]
        if h["n_px"] < min_px or t["n_px"] < min_px:
            out[lab] = dict(status="UNEVALUABLE", n_px_H300=h["n_px"], n_px_TS2=t["n_px"]); continue
        ratio = h["absrel_unaligned_pooled"] / t["absrel_unaligned_pooled"]
        out[lab] = dict(status="PASS" if ratio <= 1.0 + tol else "FAIL", absrel_ratio_H300_over_TS2=ratio, n_px_H300=h["n_px"], n_px_TS2=t["n_px"])
    return out


def decide_pairs(ds, guards_pass_all: bool, unevaluable=()) -> str:
    """Stage 1 (Add. 48): CONFIRMED iff ALL pair d_i >= ln 1.5 AND mean(d) >= ln 1.5 AND all guards pass in every pair.
    REFUTED iff ANY d_i <= 0 (Auditor O39). An unevaluable guard in any pair never passes (Auditor O37); else NOT CONFIRMED. min/max d_i = pair spread."""
    ds = list(ds)
    if len(ds) != len(PAIRS) or not all(np.isfinite(ds)):
        return "INVALID (need 3 finite pair d values)"
    mean = float(np.mean(ds))
    if min(ds) <= 0:                       # Auditor O39: any d_i <= 0 = REFUTED
        return "REFUTED"
    if unevaluable:
        return "NOT CONFIRMED (" + ", ".join(sorted(set(unevaluable))) + " guard unevaluable)"
    if min(ds) >= LN_MARGIN and mean >= LN_MARGIN and guards_pass_all:
        return "CONFIRMED"
    return "NOT CONFIRMED"


def check_pair_runs(h: dict, t: dict) -> None:
    """D11: argument order (first = H300, second = TS2), different checkpoints, one pair label / stage, and IDENTICAL kept/dropped frame ids, certain-sky px counts per frame,
    scored-frame and LiDAR px counts in both arms (the drop sets and the pixel sets must be the same: same mask file, same frames)."""
    for tag, r in (("first", h), ("second", t)):
        if "meta" not in r:
            raise AssertionError(f"{tag} run has no meta")
    if h["meta"].get("arm") != "H300" or t["meta"].get("arm") != "TS2":
        raise AssertionError(f"argument order: first must be arm H300, second arm TS2 (got {h['meta'].get('arm')!r}, {t['meta'].get('arm')!r})")
    if not h["meta"].get("ckpt_sha") or h["meta"]["ckpt_sha"] == t["meta"].get("ckpt_sha"):
        raise AssertionError("the two arms must use different checkpoints (ckpt sha equal or missing)")
    if h["meta"].get("seed_label") != t["meta"].get("seed_label") or h["meta"].get("stage") != t["meta"].get("stage"):
        raise AssertionError("a pair shares its pair label and stage")
    for k in ("kept_frame_ids", "dropped_frame_ids", "frame_mask_px"):
        if h["sky"].get(k) != t["sky"].get(k):
            raise AssertionError(f"arms differ in sky.{k}: the frame/mask sets must be identical across the two arms")
    for k in ("n_scored_frames", "n_lidar_px_valid_before_sky", "n_lidar_px_sky_excluded"):
        if h.get(k) != t.get(k):
            raise AssertionError(f"arms differ in {k}: {h.get(k)} vs {t.get(k)}")


def compare(h: dict, t: dict, stage: int) -> dict:
    """Per-pair comparison. `stage` selects the registered disclosure string (1 or 2)."""
    assert stage in (1, 2)
    check_pair_runs(h, t)
    d = math.log(h["sky"]["P_framebalanced_median"] / t["sky"]["P_framebalanced_median"])
    g = guard_check(h["bins"], t["bins"])
    ok = all(v["status"] == "PASS" for v in g.values())      # UNEVALUABLE is not a pass
    unev = [k for k, v in g.items() if v["status"] == "UNEVALUABLE"]
    return dict(label=LABEL_STAGE1 if stage == 1 else LABEL_STAGE2, stage=stage, disclosure=STAGE1_NOTE if stage == 1 else STAGE2_NOTE, d_ln_PH300_over_PTS2=d,
                ln_margin=LN_MARGIN, P_H300=h["sky"]["P_framebalanced_median"], P_TS2=t["sky"]["P_framebalanced_median"], guards=g, guards_pass=ok,
                guards_unevaluable=unev, verdict=decide(d, ok, unev),
                rule="CONFIRMED iff d>=ln1.5 AND all guards pass; 0<d<ln1.5 NOT CONFIRMED; d<=0 REFUTED")


def compare_pairs(pairs: dict) -> dict:
    """pairs: {'s50': (h_json, t_json), 's51': ..., 's52': ...}; per-pair compare() plus the Stage 1 rule over the three."""
    assert tuple(sorted(pairs)) == PAIRS, "need exactly the pairs s50, s51, s52"
    for k in PAIRS:
        for r in pairs[k]:
            assert r["meta"].get("stage") == 1 and r["meta"].get("seed_label") == k, f"pair {k}: runs must carry stage 1 and the pair label"
    per = {k: compare(*pairs[k], stage=1) for k in PAIRS}
    ds = [per[k]["d_ln_PH300_over_PTS2"] for k in PAIRS]
    unev = [f"{k}:{b}" for k in PAIRS for b in per[k]["guards_unevaluable"]]
    gp = all(per[k]["guards_pass"] for k in PAIRS)
    return dict(label=LABEL_STAGE1, stage=1, disclosure=STAGE1_NOTE, d_per_pair=dict(zip(PAIRS, ds)), d_mean=float(np.mean(ds)), d_min=float(np.min(ds)), ln_margin=LN_MARGIN,
                guards_pass_all=gp, guards_unevaluable=unev, per_pair=per, verdict=decide_pairs(ds, gp, unev),
                rule="CONFIRMED iff all 3 pair d >= ln1.5 AND mean d >= ln1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0; unevaluable guard never passes; else NOT CONFIRMED; min/max d_i = pair spread")


# ----------------------------------------------------------------------------- model path (NEVER run) [UNVERIFIED until data exists]
def model_input(paths):
    """Same as ddad_scorer.model_input (A2) with the AV2 jpg reader."""
    import cv2, torch
    import bench_joint as bj
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=A.SHORT_SIDE, height=A.SHORT_SIDE, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    ims = [rs({"image": A.load_rgb(p).astype(np.float32) / 255.0})["image"] for p in paths]
    return torch.from_numpy(np.clip(np.stack(ims), 0, 1)).permute(0, 3, 1, 2).contiguous()


def check_mask_meta(m: dict) -> None:
    """The mask store's sidecar must say OK and carry a pooled segmenter IoU >= 0.85 (R1 M2); anything else refuses to score.
    bool, NaN, inf, non-numbers and values outside [0, 1] are rejected explicitly."""
    if m.get("status") != "OK":
        raise RuntimeError(f"mask meta status {m.get('status')!r} != 'OK'")
    iou = m.get("pooled_iou")
    if isinstance(iou, bool) or not isinstance(iou, (int, float)):
        raise RuntimeError(f"mask meta pooled_iou {iou!r} is not a number: mask set rejected")
    if math.isnan(iou):
        raise RuntimeError("mask meta pooled_iou is NaN: mask set rejected")
    if iou < 0.0 or iou > 1.0:                       # IoU lives in [0, 1] (also rejects +-inf)
        raise RuntimeError(f"mask meta pooled_iou {iou!r} outside [0, 1]: mask set rejected")
    if iou < IOU_MIN:
        raise RuntimeError(f"mask meta pooled_iou {iou!r} < {IOU_MIN}: mask set rejected")
    if not isinstance(m.get("dropped"), list):
        raise RuntimeError("mask meta has no dropped list")


def is_mask_dropped(fid: str, dropped: set) -> bool:
    return fid in dropped


def load_mask_meta(mask_path, frames_path) -> tuple[dict, str]:
    """Reads the sidecar next to the mask store, runs check_mask_meta, and asserts meta.frames_sha == sha256(frames tsv) (the mask was built for THESE frames).
    Returns (meta, sha256 of the sidecar file: the meta_sha pin)."""
    mp = Path(mask_path).with_suffix(".meta.json")
    mmeta = json.load(open(mp))
    check_mask_meta(mmeta)
    fs = A.sha256_file(frames_path)
    if mmeta.get("frames_sha") != fs:
        raise AssertionError(f"mask meta frames_sha {mmeta.get('frames_sha')} != frames tsv sha {fs}")
    return mmeta, A.sha256_file(mp)


def to_numpy(x) -> np.ndarray:
    """Model output (torch tensor, possibly cuda/half) -> float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float32)


def selftest() -> dict:
    """Synthetic frames only: pred = 0.9 * smooth field -> 0.9 / 0.1 in every bin; sky region predicted at 450 m -> P = 450."""
    H, W = A.NATIVE_HW
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = (1.0 + 199.0 * (xx / (W - 1))).astype(np.float32)
    rng = np.random.default_rng(0)
    acc = Accumulator(); sk = SkyAccumulator()
    sky = np.zeros((H, W), bool); sky[: H // 4] = True
    for log in ("a", "b"):
        gt = np.full((H, W), np.nan, np.float32); idx = rng.choice(H * W, 40000, replace=False); gt.flat[idx] = field.flat[idx]
        pred = 0.9 * field; pred[sky] = 450.0
        score_frame_guard(acc, pred, np.where(sky, np.nan, gt), log, sky)
        sk.add(pred[::4, ::4], sky)
    return dict(label="selftest (synthetic)", **acc.result(), sky=sk.result())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--spec", default=None); ap.add_argument("--out")
    ap.add_argument("--root"); ap.add_argument("--mask"); ap.add_argument("--frames"); ap.add_argument("--clips"); ap.add_argument("--objects"); ap.add_argument("--logs")
    ap.add_argument("--expect-scorer-sha", help="REQUIRED for a scoring run: the literal sha256 of the final committed av2_scorer.py (from RUNBOOK_STAGE1.md)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]))
    ap.add_argument("--spec-prefix", default="ours:")
    ap.add_argument("--compare", nargs=2, metavar=("H300_JSON", "TS2_JSON"))
    ap.add_argument("--compare-pairs", nargs=6, metavar=("H300_s50", "TS2_s50", "H300_s51", "TS2_s51", "H300_s52", "TS2_s52"))
    ap.add_argument("--pair", choices=PAIR_LABELS, help="pair label of this run (required with --ckpt): s50 = c3_n46 start/seed 50, s51 = n47/51, s52 = n48/52, stage2 = Stage 2 seed 50")
    ap.add_argument("--arm", choices=ARMS, help="arm of this run (required with --ckpt): H300 = TS2+H300 (treatment), TS2 = control")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest(), indent=1, default=float)); return
    if a.compare_pairs:
        js = [json.load(open(p)) for p in a.compare_pairs]
        pairs = {}
        for i, k in enumerate(PAIRS):
            h, t = js[2 * i], js[2 * i + 1]
            for r in (h, t):
                assert r["meta"]["pins_asserted"] and r["meta"]["status"] == "FULL" and r["meta"]["seed_label"] == k, f"pair {k}: FULL pinned runs with the pair label"
            pairs[k] = (h, t)
        res = compare_pairs(pairs)
        print(json.dumps(res, indent=1, default=float))
        if a.out:
            assert not Path(a.out).exists(); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    if a.compare:
        h, t = (json.load(open(p)) for p in a.compare)
        for r in (h, t):
            assert r["meta"]["pins_asserted"] and r["meta"]["status"] == "FULL", "compare needs FULL runs with pins asserted"
            assert r["meta"]["seed_label"] in PAIR_LABELS, "pair label"
        assert h["meta"]["seed_label"] == t["meta"]["seed_label"], "a pair shares its start and seed"
        res = compare(h, t, stage=2 if h["meta"]["seed_label"] == "stage2" else 1)
        print(json.dumps(res, indent=1, default=float))
        if a.out:
            assert not Path(a.out).exists(); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    assert a.out and (a.ckpt or a.spec) and a.root and a.mask and a.frames and a.clips and a.objects and a.logs
    assert not (a.ckpt and a.spec)
    scorer_sha = check_scorer_sha(a.expect_scorer_sha)                       # D1: pinned OUTSIDE the file
    if a.ckpt:
        assert a.pair and a.arm, "--pair and --arm are required with --ckpt"
    else:
        assert not a.pair and not a.arm, "the reference row has no pair/arm"
    out_path = Path(a.out); assert not out_path.exists(), f"{out_path} exists (never overwrite)"
    import torch
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import bench_joint as bj
    import ddad_scorer as DS
    mmeta, meta_sha = load_mask_meta(a.mask, a.frames)                       # check_mask_meta + frames_sha == sha(frames tsv)
    actual = dict(logs_sha=A.sha256_file(a.logs), frames_sha=A.sha256_file(a.frames), clips_sha=A.sha256_file(a.clips), objects_sha=A.sha256_file(a.objects),
                  mask_sha=A.sha256_file(a.mask), meta_sha=meta_sha, dataset_module_sha=A.sha256_file(A.__file__), ddad_scorer_sha=A.sha256_file(DS.__file__),
                  bench_joint_sha=A.sha256_file(bj.__file__), segmenter_a_sha=mmeta["segmenter_a_sha"], segmenter_b_sha=mmeta["segmenter_b_sha"],
                  sums_sha=A.sha256_file(Path(a.root).parent / "SHA256SUMS"))
    pins = assert_pins(actual)
    reg = A.load_registered_inputs(a.frames, a.clips, a.objects, a.root, registered={k: PINS[k] for k in ("frames_sha", "clips_sha", "objects_sha")})
    bj_spec, mm = DS.model_meta(a.ckpt, a.spec, bj.PATHS)
    facts = dict(assert_ckpt_facts(mm["ckpt"], a.pair), arm=a.arm, ckpt_sha=mm["ckpt"]["sha256"]) if a.ckpt else dict(arm="reference", ckpt_sha=None)
    gpu = assert_host(gpu_name())
    dev = a.device
    ctx = lambda: autocast_ctx(dev)                                            # raises on a non-cuda device (registered fp16)
    ctx()
    pipe = bj.build_pipeline(bj_spec or (a.spec_prefix + a.ckpt), dev); assert pipe["metric"]
    masks = A.MaskStore(a.mask)
    mask_dropped = set(mmeta["dropped"]); n_mask_dropped = 0
    acc, sky = Accumulator(), SkyAccumulator(); t0 = time.time(); dtypes = set()
    for log, d in reg.items():
        L = A.Log(a.root, log)
        x = model_input([L.image_path(c) for c in d["clip"]]).to(dev)
        with ctx():
            dep, _ = bj.run_pipeline(pipe, x, 0.0, 0.0, tuple(x.shape[-2:]))
        DS.assert_same_hw(dep, x)
        for (_, c, l, dt, i) in d["rows"]:
            fid = A.frame_id(log, c)
            cert, anys = masks.get(fid)
            n_mask_dropped += int(route_frame(acc, sky, fid, to_numpy(dep[i]), cert, anys, L.sparse_gt(c, l), log, mask_dropped))
        dtypes |= L.xyz_dtypes
        print(f"{log} {len(d['rows'])} frames {time.time() - t0:.0f}s", flush=True)
    res = acc.result()
    stage = 2 if a.pair == "stage2" else 1
    meta = dict(label=REF_LABEL if a.spec else (LABEL_STAGE1 if stage == 1 else LABEL_STAGE2), status="FULL", pins_asserted=True, pins=pins, scorer_sha=scorer_sha,
                seed_label=(a.pair if a.ckpt else "reference"), stage=(stage if a.ckpt else None), host_gpu=gpu, **facts, **mm,
                precision="fp16", short_side=A.SHORT_SIDE, resolution_rule="A1 bilinear pred->2048x1550 sampled at projected LiDAR px; GT never resized; sky = any-sky mask excluded",
                n_mask_dropped_frames=n_mask_dropped, mask_pooled_iou=mmeta["pooled_iou"], mask_n_candidates=mmeta.get("n_candidates"), mask_segmenters=dict(a=mmeta.get("segmenter_a"), b=mmeta.get("segmenter_b")),
                lidar_xyz_dtypes=sorted(dtypes | set(mmeta.get("lidar_xyz_dtypes", []))), host=socket.gethostname(), seconds=time.time() - t0, bins_m=bin_labels(),
                disclosure=(STAGE1_NOTE if stage == 1 else STAGE2_NOTE) if a.ckpt else REF_LABEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(meta=meta, sky=sky.result(), **res), indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

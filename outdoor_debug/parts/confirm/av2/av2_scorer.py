#!/usr/bin/env python3
"""AV2 Sensor VAL scorer for the STRICT confirmatory run TS2+H300 vs TS2 (AV2_REGISTRATION.md). DRAFT: the model/real-data path has NEVER run.

Per checkpoint (one JSON):
  PRIMARY statistic input (prediction only, no GT): for each scored frame, the median PREDICTED depth over CERTAIN-SKY pixels (frozen mask, nearest-resized to the model grid,
    frames with < 2000 certain-sky px at the model grid are dropped and counted); P = median over the kept frames (frame-balanced). Also: pooled median (description), frac of sky px > 200 / 300 m.
  GUARDS: unaligned pixel-pooled AbsRel and pooled median pred:gt in the bins [0,20) [20,50) [50,80) [80,120) [120,200] m over LiDAR-projected px with the ANY-SKY mask excluded
    (A1: prediction bilinearly resized to native 2048x1550 and sampled at the projected LiDAR px; GT never resized; no alignment, no clamp). Overall AbsRel 0-200.
Model input mirrors ddad_scorer (A2): bench_joint Resize lower_bound, multiple of 14, INTER_CUBIC, short side 378, fp16 autocast; portrait 2048x1550 -> 504x378 (Resize rounding).
Decision (Stage 1, Add. 48): per pair d_i = ln(P_H300 / P_TS2) for pairs s50/s51/s52; CONFIRMED iff all three d_i AND their mean >= ln 1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0 (Auditor O39); unevaluable guard never passes.
Usage:  python -I av2_scorer.py --ckpt <ours .pt> | --spec metric_vda_s  --out r.json --root <av2_val dir> --mask <sky_masks.npz> --frames <av2_frames.tsv>
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
PINS = dict(logs_sha=UNSET, frames_sha=UNSET, mask_sha=UNSET, segmenter_a_sha=UNSET, segmenter_b_sha=UNSET, scorer_sha=UNSET,
            dataset_module_sha=UNSET, ddad_scorer_sha=UNSET, sums_sha=UNSET,
            bench_joint_sha="5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef")
BINS_M = A.BINS_M
MAX_M = A.MAX_DEPTH_M
MIN_SKY_PX = 2000                    # certain-sky px at the model grid for a frame to count in P
LN_MARGIN = math.log(1.5)
GUARD_TOL = 0.05                     # H300 AbsRel <= (1 + 0.05) * same-seed TS2 AbsRel, per bin
GUARD_MIN_PX = 20000                 # a bin with fewer pooled px in either arm is reported "UNEVALUABLE" (not silently passed): the verdict says so
SINGLE_SEED_NOTE = "Stage 1: THREE paired starts (n46/s50, n47/s51, n48/s52), 2000-step continuations; 3 pairs size the pair-to-pair spread only roughly; the A0/A0b screen pair showed 26% spread on a guard."
PAIRS = ("s50", "s51", "s52")           # Stage 1 pairs: c3_n46/seed 50, c3_n47/seed 51, c3_n48/seed 52 (Add. 48)
LABEL = "confirmatory STRICT (AV2 sensor val, held out); single seed 50; primary = predicted sky depth by design, NOT accuracy"
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
        self.n_gt200 = 0; self.n_gt300 = 0; self.n_px = 0; self._samples = []; self._weights = []

    def add(self, pred: np.ndarray, certain_native: np.ndarray):
        self.seen += 1
        m = mask_to_grid(certain_native, pred.shape[-2:])
        v = np.asarray(pred, np.float64)[m]
        v = v[np.isfinite(v)]
        if v.size < MIN_SKY_PX:
            self.dropped += 1; return
        self.frame_medians.append(float(np.median(v))); self.frame_n.append(int(v.size))
        self.n_px += v.size; self.n_gt200 += int((v > 200).sum()); self.n_gt300 += int((v > 300).sum())
        rng = np.random.default_rng(0)
        smp = v if v.size <= 20000 else rng.choice(v, 20000, replace=False)
        self._samples.append(smp); self._weights.append(np.full(smp.size, v.size / smp.size))     # weights restore the true pixel pooling

    def result(self) -> dict:
        if not self.frame_medians:
            return dict(P_framebalanced_median=float("nan"), n_frames_kept=0, n_frames_dropped=self.dropped, n_frames_seen=self.seen)
        return dict(P_framebalanced_median=float(np.median(self.frame_medians)), n_frames_kept=len(self.frame_medians), n_frames_dropped=self.dropped,
                    n_frames_seen=self.seen, pooled_median_description=weighted_median(np.concatenate(self._samples), np.concatenate(self._weights)),
                    frac_sky_px_gt200=self.n_gt200 / self.n_px, frac_sky_px_gt300=self.n_gt300 / self.n_px, n_sky_px=int(self.n_px),
                    frame_medians=self.frame_medians)


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


def compare_pairs(pairs: dict) -> dict:
    """pairs: {'s50': (h_json, t_json), 's51': ..., 's52': ...}; per-pair compare() plus the Stage 1 rule over the three."""
    assert tuple(sorted(pairs)) == PAIRS, "need exactly the pairs s50, s51, s52"
    per = {k: compare(*pairs[k]) for k in PAIRS}
    ds = [per[k]["d_ln_PH300_over_PTS2"] for k in PAIRS]
    unev = [f"{k}:{b}" for k in PAIRS for b in per[k]["guards_unevaluable"]]
    gp = all(per[k]["guards_pass"] for k in PAIRS)
    return dict(label=LABEL, note=SINGLE_SEED_NOTE, d_per_pair=dict(zip(PAIRS, ds)), d_mean=float(np.mean(ds)), d_min=float(np.min(ds)), ln_margin=LN_MARGIN,
                guards_pass_all=gp, guards_unevaluable=unev, per_pair=per, verdict=decide_pairs(ds, gp, unev),
                rule="CONFIRMED iff all 3 pair d >= ln1.5 AND mean d >= ln1.5 AND all guards pass in every pair; REFUTED iff any d_i <= 0; unevaluable guard never passes; else NOT CONFIRMED; min/max d_i = pair spread")


def compare(h: dict, t: dict) -> dict:
    d = math.log(h["sky"]["P_framebalanced_median"] / t["sky"]["P_framebalanced_median"])
    g = guard_check(h["bins"], t["bins"])
    ok = all(v["status"] == "PASS" for v in g.values())      # UNEVALUABLE is not a pass
    return dict(label=LABEL, single_seed_note=SINGLE_SEED_NOTE, d_ln_PH300_over_PTS2=d, ln_margin=LN_MARGIN, P_H300=h["sky"]["P_framebalanced_median"],
                P_TS2=t["sky"]["P_framebalanced_median"], guards=g, guards_pass=ok,
                guards_unevaluable=[k for k, v in g.items() if v["status"] == "UNEVALUABLE"], verdict=decide(d, ok, [k for k, v in g.items() if v["status"] == "UNEVALUABLE"]),
                rule="CONFIRMED iff d>=ln1.5 AND all guards pass; 0<d<ln1.5 NOT CONFIRMED; d<=0 REFUTED")


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


def read_frames_tsv(path) -> list[tuple[str, int, int, int, int]]:
    out = []
    for ln in open(path):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 5:
            out.append((p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])))        # log, cam_ts, lidar_ts, dt_ns, clip_idx
    return out


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
    ap.add_argument("--root"); ap.add_argument("--mask"); ap.add_argument("--frames"); ap.add_argument("--logs")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[4]))
    ap.add_argument("--spec-prefix", default="ours:")
    ap.add_argument("--compare", nargs=2, metavar=("H300_JSON", "TS2_JSON"))
    ap.add_argument("--compare-pairs", nargs=6, metavar=("H300_s50", "TS2_s50", "H300_s51", "TS2_s51", "H300_s52", "TS2_s52"))
    ap.add_argument("--pair", choices=PAIRS, help="pair label of this run (required with --ckpt): s50 = c3_n46 start/seed 50, s51 = n47/51, s52 = n48/52")
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
            assert r["meta"]["seed_label"] in PAIRS, "pair label"
        assert h["meta"]["seed_label"] == t["meta"]["seed_label"], "a pair shares its start and seed"
        res = compare(h, t)
        print(json.dumps(res, indent=1, default=float))
        if a.out:
            assert not Path(a.out).exists(); Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return
    assert a.out and (a.ckpt or a.spec) and a.root and a.mask and a.frames and a.logs
    assert not (a.ckpt and a.spec)
    out_path = Path(a.out); assert not out_path.exists(), f"{out_path} exists (never overwrite)"
    import torch, importlib
    sys.path.insert(0, str(Path(a.repo).resolve()))
    import bench_joint as bj
    import ddad_scorer as DS
    S = DS.S
    actual = dict(logs_sha=S.sha256_file(a.logs), frames_sha=S.sha256_file(a.frames), mask_sha=S.sha256_file(a.mask), scorer_sha=S.sha256_file(__file__),
                  dataset_module_sha=S.sha256_file(A.__file__), ddad_scorer_sha=S.sha256_file(DS.__file__), bench_joint_sha=S.sha256_file(bj.__file__),
                  segmenter_a_sha=json.load(open(Path(a.mask).with_suffix(".meta.json")))["segmenter_a_sha"],
                  segmenter_b_sha=json.load(open(Path(a.mask).with_suffix(".meta.json")))["segmenter_b_sha"],
                  sums_sha=S.sha256_file(Path(a.root).parent / "SHA256SUMS"))
    pins = assert_pins(actual)
    bj_spec, mm = DS.model_meta(a.ckpt, a.spec, bj.PATHS)
    dev = a.device
    ac = dict(device_type="cuda", dtype=torch.float16, enabled=dev.startswith("cuda")); assert ac["enabled"]
    pipe = bj.build_pipeline(bj_spec or (a.spec_prefix + a.ckpt), dev); assert pipe["metric"]
    frames = read_frames_tsv(a.frames); masks = A.MaskStore(a.mask)
    by_log: dict[str, list] = {}
    for fr in frames:
        by_log.setdefault(fr[0], []).append(fr)
    acc, sky = Accumulator(), SkyAccumulator(); t0 = time.time()
    for log, frs in by_log.items():
        L = A.Log(a.root, log)
        sel = A.select_frames(L.cam_timestamps(), L.lidar_timestamps())
        assert [(i, c, l, d) for i, c, l, d in sel["scored"]] == [(f[4], f[1], f[2], f[3]) for f in frs], f"{log}: scored list differs from the registered frames tsv"
        x = model_input([L.image_path(c) for c in sel["clip"]]).to(dev)
        with torch.autocast(**ac):
            d, _ = bj.run_pipeline(pipe, x, 0.0, 0.0, tuple(x.shape[-2:]))
        DS.assert_same_hw(d, x)
        for (i, c, l, dt) in sel["scored"]:
            cert, anys = masks.get(A.frame_id(log, c))
            pred = np.asarray(d[i], np.float32)
            sky.add(pred, cert)
            score_frame_guard(acc, pred, L.sparse_gt(c, l), log, anys)
        print(f"{log} {len(sel['scored'])} frames {time.time() - t0:.0f}s", flush=True)
    res = acc.result()
    full = True
    meta = dict(label=REF_LABEL if a.spec else LABEL, status="FULL", pins_asserted=full, pins=pins, seed_label=(a.pair if a.ckpt else "reference"), **mm,
                precision="fp16", short_side=A.SHORT_SIDE, resolution_rule="A1 bilinear pred->2048x1550 sampled at projected LiDAR px; GT never resized; sky = any-sky mask excluded",
                host=socket.gethostname(), seconds=time.time() - t0, bins_m=bin_labels(), single_seed_note=SINGLE_SEED_NOTE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(meta=meta, sky=sky.result(), **res), indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

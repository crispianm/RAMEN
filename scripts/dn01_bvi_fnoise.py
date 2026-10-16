"""DN-01 K49 real-data flicker cell (Auditor (j), (j1)-(j5)): NOISE-INDUCED metric-depth flicker F_noise on the
registered BVI-RLV low / normal-light PAIRS of the display split's TEST scenes (provenance/dn01_display/
bvi_rlv_split.json, sha256 9a509cf4; 13 scenes x levels 10 / 20). BVI-RLV's camera MOVES (a repeatable trajectory), so
raw low-light flicker = motion + noise; subtracting the SAME model's frame-to-frame change on the normal-light twin
cancels the motion.

FRAMES: the first N_FRAMES = 64 frames of each sequence (sorted file names); pairs / frames t >= T0 = 8 (warm-up).

(j1) REGISTRATION PRE-CHECK (no model; run and its output committed BEFORE any model score). Per sequence (scene,
level), frames 0..63, luma at half resolution, each frame divided by its mean luma and Gaussian-blurred (sigma 2 px):
  - counts: the low and normal sequences have the same number of frames (>= N_FRAMES);
  - registration: cv2.phaseCorrelate shift between normal_t and low_t (Hanning window), in NATIVE pixels;
    PASS iff median over t <= SHIFT_MEDIAN_MAX = 1.0 px AND the 90th percentile <= SHIFT_P90_MAX = 2.0 px (one bad
    phase-correlation estimate on a dark / textureless frame does not exclude a sequence; responses reported);
  - frame sync: R(k) = mean over t of mean |normal_t - low_{t+k}| (normalised, blurred), k in [-3, 3]; SYNC-TESTABLE
    iff max_k R(k) - min_k R(k) >= SYNC_MIN_CONTRAST x R(0) (1%): then PASS iff argmin_k R(k) == 0 (the camera moves,
    so a misaligned lag raises the residual). A flat R(k) = a near-still camera over these frames, where a lag cannot
    create a motion mismatch: labelled "sync untestable" and KEPT;
  - reported: R(0) (the gain-normalised residual), the per-frame shifts + responses, R(k).
  A sequence failing any check is EXCLUDED; a scene with both levels excluded leaves the macro.
  The pre-check records each sequence's 64 low + 64 normal file names and a sha256 over their bytes; `run` recomputes
  them on its host and REFUSES to score a sequence whose frames differ (the committed exclusions describe these frames).
  python scripts/dn01_bvi_fnoise.py precheck --root <BVI-Lowlight-videos> --out provenance/dn01_fnoise/registration.json

(j5) F_NOISE (per model; METRIC models only; UNALIGNED metric depth at model resolution: VDA Resize, short side 518;
bench_joint's pipeline and windowing, blind restorers). D^low / D^norm = the SAME model on the low / normal clip,
clamped to [0.1, 80] m. VALID pixel at pair (t-1, t): D^norm_{t-1} and D^norm_t both inside [0.1, 80] BEFORE the clamp.
  F_noise = mean over pairs t = T0..63 and valid pixels of |(D^low_t - D^low_{t-1}) - (D^norm_t - D^norm_{t-1})| / D^norm_t
  (j2) level = mean over frames t = T0..63 and pixels with D^norm_t inside [0.1, 80] of |D^low_t - D^norm_t| / D^norm_t
  descriptive: flicker_low / flicker_norm = mean over the same pairs of |dD^low| / D^norm_t, |dD^norm| / D^norm_t.
  Per sequence -> per scene (mean over its included levels) -> MACRO over scenes (the reported number).
  python scripts/dn01_bvi_fnoise.py run --model <bench_joint spec> --root <...> --registration <json> --out DIR
ADDITIVE like-for-like keys (audit/SKY_fnoise_all_20261001.md): f_noise_all / flicker_low_all / flicker_norm_all = the same formulas over ALL pixels (no model-defined mask), depth clamped to [0.1, 80] m, the same pairs; metric path only.
DIAGNOSTICS (descriptive fields, formula unchanged; the gate rule that reads them is in docs/DN01_JOB_LEDGER.md):
f_noise_offset (D^norm's step shifted one frame: the no-cancellation reference) and corr (Pearson of dD^low vs dD^norm).
RELATIVE mode (--relative; DESCRIPTIVE ONLY, never gates): fnoise_relative, one affine fit per sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_joint as bj  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "provenance/dn01_display/bvi_rlv_split.json"
SPLIT_SHA256 = "9a509cf4965a02ce497af8171efbdc29f0d8d6f1ca8762957abf5945979556af"
N_FRAMES, T0, LAGS = 64, 8, range(-3, 4)
SHIFT_MEDIAN_MAX, SHIFT_P90_MAX, SYNC_MIN_CONTRAST = 1.0, 2.0, 0.01
DEPTH_RANGE, SHORT_SIDE = (0.1, 80.0), 518


def sequences() -> list[tuple[str, int]]:
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA256, "display split changed"
    s = json.loads(SPLIT.read_text())
    return [(scene, lv) for scene in sorted(s["test"]) for lv in s["levels"]]


def frame_files(root: Path, scene: str, lv: int, kind: str) -> list[Path]:
    return sorted(p for p in (root / scene / f"{kind}_light_{lv}").iterdir() if p.suffix == ".png")


def frames_digest(low: list[Path], normal: list[Path]) -> dict:
    """The exact frames a sequence is checked and scored on: names + one sha256 over their bytes, in order."""
    h = hashlib.sha256()
    for f in [*low, *normal]:
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return {"low": [f.name for f in low], "normal": [f.name for f in normal], "sha256": h.hexdigest()}


def _prep(p: Path) -> np.ndarray:
    """Half-resolution luma, / its mean, Gaussian sigma 2 px (float32)."""
    y = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(y / max(float(y.mean()), 1e-6), (0, 0), 2.0)


def precheck(root: Path) -> dict:
    out = {"rule": {"n_frames": N_FRAMES, "shift_median_max_px": SHIFT_MEDIAN_MAX, "shift_p90_max_px": SHIFT_P90_MAX,
                    "sync_min_contrast": SYNC_MIN_CONTRAST, "lags": list(LAGS), "split_sha256": SPLIT_SHA256},
           "sequences": {}}
    for scene, lv in sequences():
        lo, no = frame_files(root, scene, lv, "low"), frame_files(root, scene, lv, "normal")
        rec = {"n_low": len(lo), "n_normal": len(no)}
        ok = len(lo) == len(no) and len(lo) >= N_FRAMES
        if ok:
            rec["frames"] = frames_digest(lo[:N_FRAMES], no[:N_FRAMES])
            L = [_prep(p) for p in lo[:N_FRAMES]]
            Nn = [_prep(p) for p in no[:N_FRAMES]]
            win = cv2.createHanningWindow(L[0].shape[::-1], cv2.CV_32F)
            pc = [cv2.phaseCorrelate(n, lw, win) for n, lw in zip(Nn, L)]
            shifts = [2.0 * float(np.hypot(*c[0])) for c in pc]
            R = {k: float(np.mean([np.abs(Nn[t] - L[t + k]).mean() for t in range(3, N_FRAMES - 3)])) for k in LAGS}
            kstar = min(R, key=R.get)
            testable = max(R.values()) - min(R.values()) >= SYNC_MIN_CONTRAST * R[0]
            reg_ok = float(np.median(shifts)) <= SHIFT_MEDIAN_MAX and float(np.percentile(shifts, 90)) <= SHIFT_P90_MAX
            sync_ok = kstar == 0 if testable else True
            rec.update(shift_median_px=float(np.median(shifts)), shift_p90_px=float(np.percentile(shifts, 90)),
                       shift_max_px=max(shifts), shifts_px=shifts, responses=[float(c[1]) for c in pc],
                       residual_lag={str(k): v for k, v in R.items()}, residual0=R[0], best_lag=kstar,
                       sync_testable=bool(testable), registration_ok=bool(reg_ok), sync_ok=bool(sync_ok))
            ok = reg_ok and sync_ok
        rec["included"] = bool(ok)
        out["sequences"][f"{scene}__{lv}"] = rec
        print(f"{scene}__{lv}: n {len(lo)}/{len(no)} shift med {rec.get('shift_median_px', float('nan')):.2f} "
              f"p90 {rec.get('shift_p90_px', float('nan')):.2f} lag* {rec.get('best_lag')} "
              f"(testable {rec.get('sync_testable')}) R0 {rec.get('residual0', float('nan')):.4f} -> "
              f"{'IN' if ok else 'EXCLUDED'}", flush=True)
    out["excluded"] = sorted(k for k, v in out["sequences"].items() if not v["included"])
    return out


def _metrics(dl: np.ndarray, dn: np.ndarray, inr: np.ndarray, t0: int) -> dict:
    """The (j5) fields + the DIAGNOSTICS (Auditor 2026-09-28, descriptive; the gate rule on them is in the ledger):
      f_noise_offset = the F_noise formula with D^norm's step shifted by ONE frame, |dD^low_t - dD^norm_{t+1}| / D^norm_t,
                       over pairs t = t0..N-2 valid at (t-1, t) and (t, t+1): the no-cancellation reference;
      corr           = Pearson correlation of dD^low_t and dD^norm_t over the F_noise pairs / valid pixels."""
    pair = inr[t0:] & inr[t0 - 1:-1]
    step_l, step_n = dl[t0:] - dl[t0 - 1:-1], dn[t0:] - dn[t0 - 1:-1]
    den = dn[t0:]
    off = pair[:-1] & pair[1:]
    a, b = step_l[pair], step_n[pair]
    return {"f_noise": float((np.abs(step_l - step_n) / den)[pair].mean()),
            "level": float((np.abs(dl[t0:] - dn[t0:]) / den)[inr[t0:]].mean()),
            "flicker_low": float((np.abs(step_l) / den)[pair].mean()),
            "flicker_norm": float((np.abs(step_n) / den)[pair].mean()),
            "f_noise_offset": float((np.abs(step_l[:-1] - step_n[1:]) / den[:-1])[off].mean()),
            "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"),
            "valid_frac": float(pair.mean())}


def fnoise(d_low: np.ndarray, d_norm: np.ndarray, t0: int = T0) -> dict:
    """(j5) on one sequence. ``d_low`` / ``d_norm`` (N,H,W) metric depth (metres), UNCLAMPED as the model gave it."""
    lo_, hi_ = DEPTH_RANGE
    inr = (d_norm >= lo_) & (d_norm <= hi_)
    dl, dn = np.clip(d_low, lo_, hi_).astype(np.float64), np.clip(d_norm, lo_, hi_).astype(np.float64)
    # E4 far-field guard fields (additive): from the UNCLAMPED depth the model gave, so depth pushed past the cap is visible
    far = {"far_frac80_low": float((d_low >= 80.0).mean()), "median_depth_low": float(np.median(d_low[:, ::4, ::4])), "median_depth_norm": float(np.median(d_norm[:, ::4, ::4]))}
    return {**_metrics(dl, dn, inr, t0), **far, **_all_pixels(dl, dn, t0)}


def _all_pixels(dl: np.ndarray, dn: np.ndarray, t0: int) -> dict:
    """LIKE-FOR-LIKE fields (audit/SKY_fnoise_all_20261001.md, ADDITIVE; every existing key is unchanged): the F_noise / flicker formulas over ALL pixels (NO model-defined validity mask) of the pairs t = t0..N-1, on the
    depth CLAMPED to [0.1, 80] m for both D^low and D^norm (``dl`` / ``dn`` are already clamped), so every model is scored on the SAME pixel set: a sky beyond 80 m contributes ~0 change at the clamp, a sky pulled inside 80 m
    contributes its real change. f_noise_all = mean |(D^low_t - D^low_{t-1}) - (D^norm_t - D^norm_{t-1})| / D^norm_t; flicker_low_all / flicker_norm_all = mean |dD^low| / D^norm_t, |dD^norm| / D^norm_t."""
    step_l, step_n = dl[t0:] - dl[t0 - 1:-1], dn[t0:] - dn[t0 - 1:-1]
    den = dn[t0:]
    return {"f_noise_all": float((np.abs(step_l - step_n) / den).mean()), "flicker_low_all": float((np.abs(step_l) / den).mean()), "flicker_norm_all": float((np.abs(step_n) / den).mean())}


def fnoise_relative(q_low: np.ndarray, q_norm: np.ndarray, t0: int = T0) -> dict:
    """RELATIVE mode (Auditor (b): DESCRIPTIVE ONLY, never gates; for relative models such as DA-AC, with MVDA / ST-U
    side rows). ``q_*`` (N,H,W) DISPARITY-like outputs (a metric model's 1 / depth). ONE least-squares (scale, shift)
    per SEQUENCE maps q_low onto q_norm over frames t >= t0 and valid pixels (q_norm > 0, finite; never per frame);
    depth-like D = 1 / q, both divided by median(D^norm); valid = D^norm / median AND D^low / median in [0.01, 100]
    (an aligned q_low <= 0 has no depth: invalid, never floored); then the same fields (no metric clamp), plus
    ``invalid_low_frac`` = the share of frames-t >= t0 pixels valid for D^norm but dropped because D^low is not.
    Its level is NOT comparable to the metric rows'.
    AMENDMENT 2026-09-28 (descriptive path only; after the evalx run @fd5f37b showed DA-AC's S20_books3__10 at
    F_noise 5318): the first version floored the aligned q_low at 1e-6 (depth 1e6 x) and kept those pixels valid."""
    q_low, q_norm = q_low.astype(np.float64), q_norm.astype(np.float64)
    ok = np.isfinite(q_norm) & (q_norm > 0) & np.isfinite(q_low)
    x, y = q_low[t0:][ok[t0:]], q_norm[t0:][ok[t0:]]
    s, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    ql = s * np.where(ok, q_low, 0.0) + b
    dn = 1.0 / np.where(ok, q_norm, 1.0)
    dl = 1.0 / np.where(ql > 0, ql, 1.0)
    med = float(np.median(dn[t0:][ok[t0:]]))
    dn, dl = dn / med, dl / med
    norm_ok = ok & (dn >= 0.01) & (dn <= 100.0)
    inr = norm_ok & (ql > 0) & (dl >= 0.01) & (dl <= 100.0)
    lost = float((norm_ok[t0:] & ~inr[t0:]).sum() / max(norm_ok[t0:].sum(), 1))
    return {**_metrics(dl, dn, inr, t0), "fit_scale": float(s), "fit_shift": float(b), "invalid_low_frac": lost}


def load(files: list[Path]) -> torch.Tensor:
    """(N,3,h,w) in [0,1] at model resolution (bench_joint's VDA Resize, short side 518)."""
    bj._vda_path()
    from video_depth_anything.util.transform import Resize
    rs = Resize(width=SHORT_SIDE, height=SHORT_SIDE, resize_target=False, keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC)
    imgs = [rs({"image": cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0})["image"]
            for p in files]
    return torch.from_numpy(np.clip(np.stack(imgs), 0, 1)).permute(0, 3, 1, 2).contiguous()


def checked_files(root: Path, key: str, rec: dict) -> dict:
    """The sequence's scored frames on THIS host; refuses unless they equal the committed pre-check's."""
    scene, lv = key.rsplit("__", 1)
    files = {k: frame_files(root, scene, int(lv), k)[:N_FRAMES] for k in ("low", "normal")}
    if frames_digest(files["low"], files["normal"]) != rec["frames"]:
        raise SystemExit(f"{key}: this host's frames differ from the committed registration pre-check")
    return files


def run(a) -> None:
    reg_p = Path(a.registration)
    reg = json.loads(reg_p.read_text())
    bj.log_test_access(a)                        # Auditor WP18 B4: F_noise reads the BVI-RLV test sequences; logged before any model or frame is loaded
    p = bj.build_pipeline(a.model, a.device)
    if not p["metric"] and not a.relative:
        raise SystemExit("F_noise is defined on METRIC depth; this model is relative (descriptive: --relative)")
    vs = bj._grid_variants(a)                    # probe package: [None] (unchanged path) or the --grid variants, ONE model pass
    spec = lambda v: a.model if v in (None, "none") else f"{a.model}+{v}"        # noqa: E731
    pers = {v: {} for v in vs}
    limit = getattr(a, "limit_clips", None)      # F3 smoke only (default None = every included sequence, unchanged)
    for key, rec in reg["sequences"].items():
        if not rec["included"]:
            continue
        if limit is not None and len(pers[vs[0]]) >= limit:
            break
        files = checked_files(Path(a.root), key, rec)
        raw, extra = {}, {}
        for kind in ("low", "normal"):
            x = load(files[kind]).to(a.device)
            depth, _ = bj.run_pipeline(p, x, 0.0, 0.0, tuple(x.shape[-2:]))
            raw[kind] = {v: depth if v is None else bj.apply_variant(v, depth, p["last_guides"]) for v in vs}
            extra[kind] = {v: bj.variant_extras(v) for v in vs}          # F3: the DVP record of this clip ({} for every other variant)
        for v in vs:
            d = {kind: raw[kind][v] for kind in raw}
            if a.relative:                       # disparity-like: a metric model's 1 / depth, a relative model's own
                q = {k: (1.0 / np.maximum(z, 1e-6) if p["metric"] else z) for k, z in d.items()}
                pers[v][key] = fnoise_relative(q["low"], q["normal"])
            else:
                pers[v][key] = fnoise(d["low"], d["normal"])
            pers[v][key].update({f"{k}_{kind}": z for kind in extra for k, z in extra[kind][v].items()})
            print(f"{spec(v)} {key}: F_noise {pers[v][key]['f_noise']:.4f} level {pers[v][key]['level']:.4f}", flush=True)
    for v in vs:
        per = pers[v]
        out = Path(a.out) / f"{bj._slug(spec(v))}__bvi_rlv_test__fnoise{'_rel' if a.relative else ''}.json"
        if vs != [None] and out.exists():
            continue
        scenes = sorted({k.rsplit("__", 1)[0] for k in per})
        by_scene = {s: {m: float(np.mean([z[m] for k, z in per.items() if k.rsplit("__", 1)[0] == s]))
                        for m in next(iter(per.values()))} for s in scenes}
        art = {"bench": "dn01_bvi_fnoise", "model": spec(v), "rule": __doc__, "n_frames": N_FRAMES, "t0": T0,
               "agf_backend": getattr(sys.modules.get("evaluation.depth_probe"), "AGF_BACKEND", "cpu"),   # protocol v3 WP24
               "mode": "relative (DESCRIPTIVE ONLY; never gates)" if a.relative else "metric",
               "registration": {"path": str(reg_p), "sha256": bj.sha256_file(reg_p), "excluded": reg["excluded"]},
               "weights": {k: {"path": z, "sha256": bj.sha256_file(z)} for k, z in p["weights"].items()},
               "ramen_commit": bj.git_state(REPO), "host": socket.gethostname(), "torch": torch.__version__,
               "macro": {m: float(np.mean([z[m] for z in by_scene.values()])) for m in next(iter(by_scene.values()))},
               "per_scene": by_scene, "per_sequence": per, **bj.ckpt_info(a.model), **({"limit_clips": limit} if limit is not None else {})}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(art, indent=1))
        print(f"wrote {out}: F_noise {art['macro']['f_noise']:.4f} level {art['macro']['level']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("precheck")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("run")
    s.add_argument("--model", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--registration", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--relative", action="store_true", help="descriptive relative mode (fnoise_relative)")
    s.add_argument("--limit-clips", type=int, default=None, help="smoke use only: score the first N included sequences (low + normal clip each); recorded in the JSON")
    s.add_argument("--grid", nargs="+", default=None,
                   help="probe package: score these depth post-filter variants (group names or literal variants) from ONE model pass; "
                        "one result file per variant, named as a single-token run")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "precheck":
        res = precheck(Path(a.root))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"excluded: {res['excluded']}")
    else:
        run(a)


if __name__ == "__main__":
    main()

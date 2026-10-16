"""DN-01 display path fit + evaluation (Auditor K44(b)(c)). Run ONCE, after the split was committed
(provenance/dn01_display/bvi_rlv_split.json, sha256 9a509cf4...).
FIT (FIT scenes only, both light levels, 8 evenly spaced frames per clip, INTER_AREA to a 518 px short side):
  - GAIN_REF: log2 GAIN_REF = median over fit clips of log2(g*_clip) + log2(stat_clip), where g* = the least-squares
    scalar low -> normal (RGB) and stat = the median over frames of evaluation.display.frame_stat (the blind statistic).
  - LUT: 256 bins of z = clip(GAIN_REF / stat_clip x low) (all channels pooled); bin value = the mean normal-light
    value; empty bins interpolated; made monotone by weighted pool-adjacent-violators; then FROZEN to display_fit.json.
  EMA_ALPHA (0.2) is declared in evaluation/display.py, not fitted.
EVALUATE (after freezing; TEST scenes, the first 32 consecutive frames per clip, the causal per-frame gain as deployed):
  PSNR of the display of the RAW low-light frames vs the normal-light frames, for: the v2 rule (bench_joint.blind_gain,
  per clip), the refit causal gain without / with the curve, the oracle scalar without / with the curve; the per-clip
  gain error in stops; the 3 figure scenes listed too. Plus bench_joint LL cells (dev, first 32 frames): the display of a
  PERFECT native restoration (lowlight_reference) vs clean, without / with the curve (descriptive: synthetic darkening is
  a scalar, so the right curve there is the identity). Plus the display function's 4090 cost (ms/frame).
  python scripts/dn01_display_fit.py --out provenance/dn01_display
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_joint as bj  # noqa: E402
from data.elvis_noise import lowlight_reference  # noqa: E402
from evaluation import display as D  # noqa: E402

ROOT = Path("/mnt/DATA/DegradedDepth/data/test/BVI-RLV/BVI-Lowlight-videos")
SPLIT = Path(__file__).resolve().parents[1] / "provenance/dn01_display/bvi_rlv_split.json"
SPLIT_SHA = "9a509cf4965a02ce497af8171efbdc29f0d8d6f1ca8762957abf5945979556af"


def load(scene: str, kind: str, level: int, idx: list[int]) -> torch.Tensor:
    out = []
    for i in idx:
        im = cv2.imread(str(ROOT / scene / f"{kind}_light_{level}" / f"{i:05d}.png"), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        s = 518 / min(h, w)
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def nframes(scene: str, level: int) -> int:
    return len(list((ROOT / scene / f"low_light_{level}").glob("*.png")))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(bj.psnr(a, b).mean())


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression, pool-adjacent-violators."""
    blocks = []
    for v, wt in zip(y, w):
        blocks.append([v * wt, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, ww, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += ww
            blocks[-1][2] += n
    return np.concatenate([np.full(n, s / ww) for s, ww, n in blocks])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="code check only: FIT scenes only (2 fit, 1 as 'test'), 1 LL "
                    "clip; never the real fit")
    a = ap.parse_args()
    out = Path(a.out)
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == SPLIT_SHA, "split file changed"
    split = json.loads(SPLIT.read_text())
    if a.smoke:
        split.update(fit=split["fit"][:2], test=split["fit"][2:3], figure_scenes=split["fit"][2:3])
    fit_path = out / "display_fit.json"
    assert not fit_path.exists(), "display_fit.json exists: the fit is run ONCE"

    # ---- FIT (fit scenes only) ----
    rows, zs = [], []
    for scene in split["fit"]:
        for lv in split["levels"]:
            idx = [int(v) for v in np.linspace(1, nframes(scene, lv), 8)]
            lo, no = load(scene, "low", lv, idx), load(scene, "normal", lv, idx)
            g_star = float((lo * no).sum() / (lo * lo).sum())
            stat = float(D.frame_stat(lo).median())
            rows.append({"scene": scene, "level": lv, "g_star": g_star, "stat": stat})
            zs.append((lo, no, stat))
    gain_ref = float(2.0 ** np.median([np.log2(r["g_star"]) + np.log2(r["stat"]) for r in rows]))
    s, n = np.zeros(256), np.zeros(256)
    for lo, no, stat in zs:
        z = (lo * min(max(gain_ref / stat, 1.0), D.GAIN_MAX)).clamp(0, 1)
        b = (z * 255).round().long().flatten().numpy()
        s += np.bincount(b, weights=no.flatten().numpy(), minlength=256)
        n += np.bincount(b, minlength=256)
    have = n > 0
    raw = np.interp(np.arange(256), np.flatnonzero(have), s[have] / n[have])
    lut = np.clip(pava(raw, np.maximum(n, 1e-3)), 0.0, 1.0)
    fit = {"gain_ref": gain_ref, "ema_alpha": D.EMA_ALPHA, "q": D.Q, "down": D.DOWN, "gain_max": D.GAIN_MAX,
           "lut": [float(v) for v in lut], "fit_scenes": split["fit"], "split_sha256": SPLIT_SHA,
           "script_sha256": bj.sha256_file(__file__), "ramen_commit": bj.git_state(bj.REPO),
           "fit_clips": rows, "bins_empty": int((~have).sum())}
    out.mkdir(parents=True, exist_ok=True)
    fit_path.write_text(json.dumps(fit, indent=1))
    fit = D.load_fit(fit_path)                                        # FROZEN from here on
    print(f"FIT: gain_ref {gain_ref:.4f}, {int((~have).sum())} empty bins; frozen to {fit_path}", flush=True)

    # ---- EVALUATE (test scenes, causal gain as deployed) ----
    ev = []
    for scene in split["test"]:
        for lv in split["levels"]:
            idx = list(range(1, 33))
            lo, no = load(scene, "low", lv, idx), load(scene, "normal", lv, idx)
            g_star = float((lo * no).sum() / (lo * lo).sum())
            g = D.causal_gain(lo, fit["gain_ref"], fit["ema_alpha"])
            g2 = bj.GAIN_REF / bj.gain_stat(lo, bj.GAIN_STAT)
            orc = (lo * g_star).clamp(0, 1)
            ev.append({"scene": scene, "level": lv, "figure": scene in split["figure_scenes"],
                       "psnr_v2_rule": psnr((lo * float(np.clip(g2, 1, bj.GAIN_MAX))).clamp(0, 1), no),
                       "psnr_gain": psnr(D.display(lo, lo, fit, curve=False), no),
                       "psnr_gain_curve": psnr(D.display(lo, lo, fit), no),
                       "psnr_oracle": psnr(orc, no), "psnr_oracle_curve": psnr(D.apply_lut(orc, fit["lut"]), no),
                       "gain_err_stops": float(np.log2(g.mean().item()) - np.log2(g_star)),
                       "v2_rule_err_stops": float(np.log2(np.clip(g2, 1, bj.GAIN_MAX)) - np.log2(g_star))})
    keys = [k for k in ev[0] if k.startswith(("psnr", "gain", "v2"))]
    summ = {k: float(np.mean([r[k] for r in ev])) for k in keys}
    summ["abs_gain_err_stops"] = float(np.mean([abs(r["gain_err_stops"]) for r in ev]))
    summ["abs_v2_rule_err_stops"] = float(np.mean([abs(r["v2_rule_err_stops"]) for r in ev]))
    summ_fig = {k: float(np.mean([r[k] for r in ev if r["figure"]])) for k in keys}

    # LL cells: the display of a PERFECT native restoration (descriptive).
    ll = {}
    for ds in ("bonn", "kitti"):
        for cell in ("ll", "llb3", "llhard"):
            p0, p1 = [], []
            for key, frames in bj.load_clips(ds, "dev", 1 if a.smoke else None):
                clean, _ = bj.load_frames(ds, frames[:32])
                blur, _ = bj.parse_cell(cell)
                x = bj.degrade(clean, blur, 0.0, bj.noise_seed(ds, key, blur))
                ref = torch.from_numpy(lowlight_reference(clean.numpy(), bj.LL_CELLS[cell]["ev"]))
                p0.append(psnr(D.display(ref, x, fit, curve=False), clean))
                p1.append(psnr(D.display(ref, x, fit), clean))
            ll[f"{ds}__{cell}"] = {"psnr_gain": float(np.mean(p0)), "psnr_gain_curve": float(np.mean(p1))}

    # Cost on the 4090 (the same function for every pipeline: its cost does not depend on the pipeline).
    x = torch.rand(32, 3, 518, 686, device=a.device)
    for _ in range(3):
        D.display(x, x, fit)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        D.display(x, x, fit)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 / 32 * 1000
    res = {"test_split": summ, "figure_scenes": summ_fig, "test_clips": ev, "ll_cells_perfect_native": ll,
           "cost_ms_per_frame_4090_518x686": ms, "fit_file_sha256": bj.sha256_file(fit_path),
           "ramen_commit": bj.git_state(bj.REPO)}
    (out / "display_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "test_clips"}, indent=1))


if __name__ == "__main__":
    main()

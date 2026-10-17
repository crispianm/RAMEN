"""Forced-fail control (Manager rule 1): on the REAL VKITTI DEV GT (the 10 pinned clips), the scorer's own far_mask must (a) exclude the sentinel and (b) make the primary statistic change when the sentinel mask is switched off (sentinel bound -> 1e9). Synthetic pred: 0.842*gt on valid px, 300 m on sentinel/sky px. CPU only; no model."""
import sys, numpy as np
sys.path.insert(0, ".")
import bench_joint as bj
sys.path.insert(0, "outdoor_debug/parts/scorer")
import scorer as S
num = {"on": [], "off": []}; n_sent = n_far = 0
for key, frames in bj.select_clips("vkitti", "dev"):
    gt = bj.load_gt("vkitti", frames[:160]).astype(np.float64)
    pred = np.where(gt >= S.SENTINEL_M, 300.0, 0.842 * gt)
    for name, sent in (("on", S.SENTINEL_M), ("off", 1e9)):
        m = S.far_mask(gt, sentinel=sent) & np.isfinite(gt) & (gt > 0)
        num[name].append((pred[m] / gt[m]).astype(np.float32)[::7])
    n_sent += int(np.isfinite(gt[gt >= S.SENTINEL_M]).sum()); n_far += int(S.far_mask(gt).sum())
on, off = (float(np.median(np.concatenate(num[k]))) for k in ("on", "off"))
print(f"sentinel px (gt>=655, every 1st): {n_sent}; far px (80<=gt<655): {n_far}")
print(f"pooled median pred:gt  sentinel EXCLUDED (scorer) = {on:.5f}   sentinel INCLUDED (mask off) = {off:.5f}   changed = {abs(on-off) > 1e-4}")
assert abs(on - 0.842) < 1e-3, on
assert abs(on - off) > 1e-4, "forced-fail control did NOT change the statistic: the sentinel mask is not doing anything"
print("FORCED-FAIL CONTROL OK: switching the sentinel mask off changes the statistic; the scorer's mask is active")

import csv, json, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
H = Path(__file__).parent; BIG = Path("/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/T_trajectory")
rows = list(csv.DictReader(open(H / "trajectory.csv")))
def get(model, step, ds, q):
    for r in rows:
        if r["model"] == model and int(r["step"]) == step and r["dataset"] == ds and r["quantity"] == q: return float(r["value"])
    return np.nan
S = [0, 6000, 8000, 12000]; col = {"ft_n46": "#d95f02", "c3_n46": "#1b9e77"}; lab = {"ft_n46": "FT n46", "c3_n46": "RAMEN (C3) n46"}
def line(ax, ds, q, title, ylab, log=False):
    ref = get("released", 0, ds, q)
    ax.axhline(ref, color="k", ls="--", lw=1.2, label="released (step 0)")
    for m in col: ax.plot(S, [ref if s == 0 else get(m, s, ds, q) for s in S], "o-", color=col[m], label=lab[m])
    ax.set_title(title, fontsize=9); ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_xticks(S); ax.grid(alpha=.3)
    if log: ax.set_yscale("log")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
line(ax[0], "kitti_dev", "sky_median_of_frame_medians_m", "KITTI DEV sky (median of per-frame medians)", "m")
line(ax[1], "vkitti_dev", "sky_median_m", "VKITTI DEV sky (GT label 655 m)", "m")
line(ax[2], "horse", "sky_top15_median_m", "horse real clip, sky band", "m")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(BIG / "sky_vs_step.png", dpi=150); plt.savefig(H / "sky_vs_step.png", dpi=80)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
line(ax[0], "vkitti_dev", "pred_over_gt_median_far80-655", "VKITTI far pred:gt (GT 80-655 m)", "median pred/gt")
line(ax[1], "vkitti_dev", "pred_over_gt_median_near<30", "VKITTI near pred:gt (GT<30 m)", "median pred/gt")
line(ax[2], "kitti_dev", "absrel_raw", "KITTI DEV AbsRel raw", "AbsRel")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(BIG / "far_ratio_vs_step.png", dpi=150); plt.savefig(H / "far_ratio_vs_step.png", dpi=80)
# gradients
try:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (q, t) in zip(ax, [("vda_l2_output_conv", "VDA head gradient norm (output_conv)"), ("silog_l2_output_conv", "SiLog head gradient norm (output_conv)"), ("silog_over_vda_norm_output_conv", "SiLog / VDA norm ratio (output_conv)")]):
        line(a, "train_batches", q, t, "")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(BIG / "grad_norm_vs_step.png", dpi=150); plt.savefig(H / "grad_norm_vs_step.png", dpi=80)
except Exception as e: print("grad plot skipped", e)
# panels
P = np.load(BIG / "panels.npz")
keys = [("released", 0)] + [(m, s) for m in ("ft_n46", "c3_n46") for s in (6000, 8000, 12000)]
def panel(pref, rgbk, out, skyk=None, crop=None):
    arrs = [P[f"{pref}|{m}|{s}"] for m, s in keys]
    vmin, vmax = max(min(a.min() for a in arrs), 0.5), max(a.max() for a in arrs)
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    axs = axs.ravel(); axs[0].imshow(P[rgbk]); axs[0].set_title("RGB (input)", fontsize=9)
    for a, (m, s), d in zip(axs[1:], keys, arrs):
        im = a.imshow(d, norm=LogNorm(vmin, vmax), cmap="turbo"); sm = np.median(d[P[skyk]]) if skyk else np.median(d[:int(.15 * d.shape[0])])
        a.set_title(f"{'released' if s == 0 else lab[m] + ' ' + str(s)}  sky med {sm:.0f} m", fontsize=9)
    for a in axs: a.axis("off")
    fig.colorbar(im, ax=axs, shrink=.6, label="depth (m), shared log scale"); plt.savefig(BIG / out, dpi=110, bbox_inches="tight"); plt.savefig(H / out.replace(".png", "_small.png"), dpi=45, bbox_inches="tight")
panel("horse_f25", "horse_rgb25", "panel_horse_f25.png")
for c in ("b0s0", "llb3"): panel(f"kitti0023_f109_{c}", f"kitti0023_f109_{c}|rgb", f"panel_kitti0023_f109_{c}.png", skyk="kitti0023_f109_sky")

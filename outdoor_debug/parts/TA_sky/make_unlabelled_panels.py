"""Visual decision panels for TartanAir envs without a usable sky label. TRAIN data only, no model output.
args: <depth_root> <out_dir>.  Frame rule (fixed): FIRST trajectory (sorted, with depth dir), frames at q = 0.2 / 0.5 / 0.8 of its sorted index.
Panel per env, one row per frame: RGB | saturated-pixel mask (~isfinite(d) | d >= 1e4, red on RGB) | log10 depth. Also writes frames.csv."""
import sys, os, csv
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT, OUT = sys.argv[1], sys.argv[2]
ENVS = ["CoalMine", "FactoryWeather", "HongKong", "JapaneseAlley", "Ocean", "OldBrickHouseNight", "OldTownFall", "OldTownNight", "OldTownSummer",
        "OldTownWinter", "SeasideTown", "SeasonalForestWinterNight", "Sewerage", "Supermarket", "ArchVizTinyHouseNight"]
os.makedirs(OUT, exist_ok=True)
rows = []
for env in ENVS:
    base = f"{ROOT}/{env}/Data_easy"
    trajs = sorted(t for t in os.listdir(base) if t.startswith("P") and os.path.isdir(f"{base}/{t}/depth_lcam_front"))
    t = trajs[0]; fs = sorted(os.listdir(f"{base}/{t}/depth_lcam_front"))
    fig, ax = plt.subplots(3, 3, figsize=(12, 7.2))
    for r, q in enumerate((0.2, 0.5, 0.8)):
        f = fs[int(q * (len(fs) - 1))]
        a = cv2.imdecode(np.fromfile(f"{base}/{t}/depth_lcam_front/{f}", np.uint8), cv2.IMREAD_UNCHANGED)
        d = np.squeeze(np.ascontiguousarray(a).view("<f4"), -1)
        rgb = cv2.cvtColor(cv2.imread(f"{base}/{t}/image_lcam_front/" + f.replace("_depth", "")), cv2.COLOR_BGR2RGB)
        sat = ~np.isfinite(d) | (d >= 1e4)
        o = rgb.astype(np.float32); o[sat] = 0.4 * o[sat] + 0.6 * np.array([255, 0, 0], np.float32)
        fin = d[np.isfinite(d)]
        ld = np.log10(np.clip(np.where(np.isfinite(d), d, 65504), 0.1, 65504))
        ax[r, 0].imshow(rgb); ax[r, 1].imshow(o.astype(np.uint8)); im = ax[r, 2].imshow(ld, cmap="viridis", vmin=-1, vmax=4.82)
        ax[r, 0].set_ylabel(f"{t}/{f[:6]} q={q}", fontsize=8)
        ax[r, 1].set_title(f"saturated {int(sat.sum())} px ({100*sat.mean():.2f}%)", fontsize=8)
        ax[r, 2].set_title(f"log10 depth; finite max {fin.max():.0f} m", fontsize=8)
        rows.append(dict(env=env, traj=t, frame=f, q=q, n_sat_px=int(sat.sum()), sat_pct=round(100 * sat.mean(), 4), n_nonfinite=int((~np.isfinite(d)).sum()),
                         n_65504=int((d == 65504).sum()), finite_max_m=float(fin.max())))
    for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
    ax[0, 0].set_title(f"{env}: RGB", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/{env}.png", dpi=75); plt.close(fig)
    print(env, [r_["n_sat_px"] for r_ in rows[-3:]], flush=True)
with open(f"{OUT}/frames.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

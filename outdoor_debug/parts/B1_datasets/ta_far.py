import sys; sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/B1_datasets")
from common import *
from data.video_clip_dataset import _decode_depth
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = Path(f"{T}/tartanair"); rng = random.Random(11)
envs = ["AbandonedFactory","BrushifyMoon","DesertGasStation","ForestEnv","NordicHarbor","PolarSciFi","SeasonalForestWinter","Slaughter","ArchVizTinyHouseDay","JapaneseCity","WesternDesertTown","ConstructionSite"]
hist = np.zeros(60); edges = np.linspace(-1, 5, 61); tiles = []; cat_rows = []
for e in envs:
    trajs = [t for t in sorted((root / e / "Data_easy").iterdir()) if (t / "depth_lcam_front").is_dir() and any((t / "depth_lcam_front").glob("*.png"))]; tr = rng.choice(trajs)
    fl = sorted((tr / "depth_lcam_front").glob("*.png")); f = rng.choice(fl)
    d = _decode_depth(str(f)); im = np.asarray(Image.open(tr / "image_lcam_front" / f.name.replace("_depth", "")).convert("RGB"))
    assert im.shape[:2] == d.shape
    h, _ = np.histogram(np.log10(np.clip(d[np.isfinite(d) & (d > 0)], 0.1, 1e5)), bins=edges); hist += h
    cat = np.zeros(d.shape, np.uint8)  # 0:<80 1:80-655 2:655-1e4 3:>1e4/nonfinite
    cat[(d >= 80)] = 1; cat[d >= 655] = 2; cat[(d >= 1e4) | ~np.isfinite(d)] = 3
    pal = np.array([[40, 160, 60], [240, 220, 40], [240, 120, 0], [220, 0, 0]], np.uint8)
    ov = (0.5 * im + 0.5 * pal[cat]).astype(np.uint8)
    tiles.append(np.concatenate([im, ov, color_depth(np.nan_to_num(d, posinf=1e5), np.ones(d.shape, bool), 1, 1e5)], 1))
    cat_rows.append((e, tr.name, f.name, *[round(float((cat == k).mean()), 4) for k in range(4)], float(np.nanmax(d))))
sheet = np.concatenate(tiles, 0); Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(f"{OUT}/tartanair_far_categories.png")
for r in cat_rows: print(r)
plt.figure(figsize=(7, 3.5)); plt.bar(10 ** edges[:-1], hist, width=np.diff(10 ** edges), align="edge"); plt.xscale("log"); plt.yscale("log")
for x, c in ((80, "g"), (655, "orange"), (1e4, "r")): plt.axvline(x, color=c)
plt.xlabel("TartanAir depth (m), 12 outdoor frames"); plt.ylabel("pixels"); plt.tight_layout(); plt.savefig(f"{OUT}/tartanair_depth_hist_raw.png", dpi=110)
print("hist>1e4 counts by decade", {f"1e{int(edges[i])}": int(hist[i]) for i in range(0, 60, 10)})

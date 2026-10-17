"""Fixed-frame visual panels, TartanAir TRAIN only, no model output.  args: <task> <depth_root> <seg_root> <sky_ids_74.json> <out_dir> <env,env,...>
Frame rule (fixed): FIRST trajectory (sorted, with depth dir); scan q in 0.2,0.5,0.8,0.1,0.3,0.6,0.9,0.4,0.7 (index int(q*(n-1)));
a frame is informative if the TESTED set has >= 500 px; take informative frames in scan order until 3 or the sequence runs out (uninformative ones are
replaced by the next q). tasks: seg = tested set is seg-sky; satnon = saturated & ~seg-sky (uncertain under TS v2); sat = all saturated px (unlabelled envs, ignores labels);
Panel per env, one row per used frame: RGB | seg-sky overlay (green) | tested set (red) | log10 depth.  Writes <task>_frames.csv with every scanned frame."""
import sys, os, csv, json, zipfile
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TASK, ROOT, SEGROOT, IDS, OUT, ENVS = sys.argv[1:7]; ENVS = ENVS.split(",")
QS = [0.2, 0.5, 0.8, 0.1, 0.3, 0.6, 0.9, 0.4, 0.7]; MINPX = 500
ids_all = json.load(open(IDS))["envs"]; os.makedirs(OUT, exist_ok=True); rows = []
for env in ENVS:
    base = f"{ROOT}/{env}/Data_easy"; sky_ids = set(ids_all[env]["sky_ids"])
    trajs = sorted(t for t in os.listdir(base) if t.startswith("P") and os.path.isdir(f"{base}/{t}/depth_lcam_front")); t = trajs[0]
    fs = sorted(os.listdir(f"{base}/{t}/depth_lcam_front")); z = zipfile.ZipFile(f"{SEGROOT}/{env}/Data_easy/seg_lcam_front.zip")
    used = []
    for q in QS:
        f = fs[int(q * (len(fs) - 1))]
        d = np.squeeze(np.ascontiguousarray(cv2.imdecode(np.fromfile(f"{base}/{t}/depth_lcam_front/{f}", np.uint8), cv2.IMREAD_UNCHANGED)).view("<f4"), -1)
        seg = cv2.imdecode(np.frombuffer(z.read(f"{env}/Data_easy/{t}/seg_lcam_front/" + f.replace("_depth", "_seg")), np.uint8), cv2.IMREAD_UNCHANGED)
        sat = ~np.isfinite(d) | (d >= 1e4)
        sk = np.isin(seg, list(sky_ids - {0})) if sky_ids else np.zeros(d.shape, bool)
        if 0 in sky_ids: sk |= (seg == 0) & sat
        tested = {"seg": sk, "satnon": sat & ~sk, "sat": sat}[TASK]
        inf = int(tested.sum()) >= MINPX
        rows.append(dict(task=TASK, env=env, traj=t, frame=f, q=q, n_seg_sky=int(sk.sum()), n_sat=int(sat.sum()), n_tested=int(tested.sum()),
                         informative=inf, used=False, finite_max=float(d[np.isfinite(d)].max())))
        if inf and sum(r["used"] for r in rows if r["env"] == env) < 3:
            rows[-1]["used"] = True; used.append((q, f, d, sk, tested, sat))
        if len(used) == 3: break
    shown = used if used else None
    if shown is None:  # nothing informative: show first three scanned frames for the record
        shown = []
        for q in QS[:3]:
            f = fs[int(q * (len(fs) - 1))]
            d = np.squeeze(np.ascontiguousarray(cv2.imdecode(np.fromfile(f"{base}/{t}/depth_lcam_front/{f}", np.uint8), cv2.IMREAD_UNCHANGED)).view("<f4"), -1)
            seg = cv2.imdecode(np.frombuffer(z.read(f"{env}/Data_easy/{t}/seg_lcam_front/" + f.replace("_depth", "_seg")), np.uint8), cv2.IMREAD_UNCHANGED)
            sat = ~np.isfinite(d) | (d >= 1e4); sk = np.isin(seg, list(sky_ids - {0})) if sky_ids else np.zeros(d.shape, bool)
            shown.append((q, f, d, sk, {"seg": sk, "satnon": sat & ~sk, "sat": sat}[TASK], sat))
    fig, ax = plt.subplots(len(shown), 4, figsize=(15, 3.1 * len(shown)), squeeze=False)
    for r, (q, f, d, sk, tested, sat) in enumerate(shown):
        rgb = cv2.cvtColor(cv2.imread(f"{base}/{t}/image_lcam_front/" + f.replace("_depth", "")), cv2.COLOR_BGR2RGB)
        o1 = rgb.astype(np.float32); o1[sk] = 0.45 * o1[sk] + 0.55 * np.array([0, 255, 0], np.float32)
        o2 = rgb.astype(np.float32); o2[tested] = 0.4 * o2[tested] + 0.6 * np.array([255, 0, 0], np.float32)
        ld = np.log10(np.clip(np.where(np.isfinite(d), d, 65504), 0.1, 65504))
        for c, im in enumerate((rgb, o1.astype(np.uint8), o2.astype(np.uint8))): ax[r, c].imshow(im)
        ax[r, 3].imshow(ld, cmap="viridis", vmin=-1, vmax=4.82)
        ax[r, 0].set_ylabel(f"{t}/{f[:6]} q={q}", fontsize=8)
        ax[r, 1].set_title(f"seg-sky green: {int(sk.sum())} px ({100*sk.mean():.1f}%)", fontsize=8)
        ax[r, 2].set_title(f"tested [{TASK}] red: {int(tested.sum())} px ({100*tested.mean():.1f}%) sat {int(sat.sum())}", fontsize=8)
        ax[r, 3].set_title("log10 depth (vmax 65504)", fontsize=8)
    for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
    ax[0, 0].set_title(f"{env} [{TASK}] used q={[u[0] for u in used]}", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/{env}.png", dpi=62); plt.close(fig)
    print(env, [(r["q"], r["n_tested"], r["informative"]) for r in rows if r["env"] == env], flush=True)
with open(f"{OUT}/{TASK}_frames.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

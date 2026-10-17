"""Visual check of seg-sky vs released sky rule on TartanAir TRAIN envs. args: [out_dir]
Frame rule: first trajectory (sorted, with local depth) having seg-sky in a scan of every 5th frame; frames at the
10/40/70/90 percent index of that trajectory.  Hospital (0.01% sky): the 4 scanned frames (all trajs, every 5th)
with the most seg-sky pixels instead.  Columns: RGB | seg-sky (red) | released rule d>1e4 or non-finite (blue) | log-depth."""
import sys, os, json, zipfile, collections
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/mnt/DATA/DegradedDepth/data/TRAIN/tartanair"
SEG = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_seg"
LABELS = "/home/wg19671/.claude/jobs/01e6735a/tmp/tartanairpy_src/segfiles_x"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_visual"
ENVS = ["Downtown", "AmusementPark", "Prison", "ShoreCaves", "Hospital", "GreatMarsh", "BrushifyMoon"]
os.makedirs(OUT, exist_ok=True)

def depth(p):
    a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    return np.squeeze(np.ascontiguousarray(a).view("<f4"), -1)

def overlay(rgb, m, col, a=0.6):
    o = rgb.astype(np.float32).copy(); o[m] = (1 - a) * o[m] + a * np.array(col, np.float32); return o.astype(np.uint8)

lines = []
for env in ENVS:
    nm = json.load(open(f"{LABELS}/{env}/seg_label_map.json"))["name_map"]
    ids = [v for k, v in nm.items() if k.lower() in ("sky", "skysphere", "sky0", "cloud")]
    z = zipfile.ZipFile(f"{SEG}/{env}/Data_easy/seg_lcam_front.zip")
    bytraj = collections.defaultdict(list)
    for n in z.namelist():
        if n.endswith(".png"): bytraj[n.split("/")[-3]].append(n)
    trajs = [t for t in sorted(bytraj) if os.path.isdir(f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front")]
    segof = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
    for t in trajs: bytraj[t].sort()
    if env == "Hospital":
        cands = []
        for t in trajs:
            for n in bytraj[t][::5]: cands.append((int(np.isin(segof(n), ids).sum()), n, t))
        cands.sort(reverse=True); picks = [(t, n) for _, n, t in cands[:4]]
        rule = f"4 frames with most seg-sky px (scan every 5th frame, all trajs): {[c[0] for c in cands[:4]]} px"
    else:
        tsel = next(t for t in trajs if any(np.isin(segof(n), ids).any() for n in bytraj[t][::5]))
        L = bytraj[tsel]; picks = [(tsel, L[int(q * (len(L) - 1))]) for q in (0.1, 0.4, 0.7, 0.9)]
        rule = f"traj {tsel}, frames at 10/40/70/90% index"
    fig, ax = plt.subplots(len(picks), 4, figsize=(16, 2.6 * len(picks)))
    for r, (t, n) in enumerate(picks):
        fn = os.path.basename(n).replace("_seg", "")
        idx = fn.split("_")[0]
        rgb = cv2.cvtColor(cv2.imread(f"{ROOT}/{env}/Data_easy/{t}/image_lcam_front/{fn}"), cv2.COLOR_BGR2RGB)
        d = depth(f"{ROOT}/{env}/Data_easy/{t}/depth_lcam_front/{idx}_lcam_front_depth.png")
        sk = np.isin(segof(n), ids); rl = ~np.isfinite(d) | (d > 1e4)
        ld = np.log10(np.clip(np.where(np.isfinite(d), d, 65504), 0.1, 65504))
        ax[r, 0].imshow(rgb); ax[r, 1].imshow(overlay(rgb, sk, (255, 0, 0))); ax[r, 2].imshow(overlay(rgb, rl, (0, 80, 255))); ax[r, 3].imshow(ld, cmap="viridis")
        ax[r, 0].set_ylabel(f"{t}/{idx}", fontsize=8)
        ax[r, 1].set_title(f"seg-sky {100*sk.mean():.2f}%", fontsize=8); ax[r, 2].set_title(f"rule d>1e4 {100*rl.mean():.2f}% (of seg-sky caught {100*(sk&rl).sum()/max(sk.sum(),1):.0f}%)", fontsize=8)
        ax[r, 3].set_title("log10 depth m", fontsize=8)
        lines.append((env, t, idx))
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_title(f"{env}: RGB", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{env}.png", dpi=70); plt.close(fig)
    print(env, rule, [f"{t}/{n.split('/')[-1][:6]}" for t, n in picks], flush=True)

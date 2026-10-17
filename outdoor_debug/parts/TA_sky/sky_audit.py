"""Per-env TartanAir sky audit. args: <seg_root> <out_dir> [N_frames_per_env]
Reads depth PNGs from the local TRAIN tree (read only), seg PNGs from the zips under seg_root. Train data only."""
import sys, os, json, zipfile, csv, glob, collections
import numpy as np, cv2
DEPTH_ROOT = "/mnt/DATA/DegradedDepth/data/TRAIN/tartanair"
LABELS = "/home/wg19671/.claude/jobs/01e6735a/tmp/tartanairpy_src/segfiles_x"
GLASS = ["NordicHarbor","AbandonedCable","AbandonedFactory2","GothicIsland","Slaughter","VictorianStreet","Prison","WesternDesertTown","Office","OldBrickHouseDay","OldBrickHouseNight"]
TRAIN_LABEL = {"Downtown","Hospital","ForestEnv","OldBrickHouseNight"}
TREEWORDS = ("tree","foliage","leaf","leaves","branch","bush","plant","grass")
seg_root, out_dir = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(out_dir, exist_ok=True)

def decode_depth(b):
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return np.squeeze(np.ascontiguousarray(a).view("<f4"), axis=-1)

rows, detail = [], {}
ONLY = os.environ.get("ONLY"); 
for env in sorted(os.listdir(DEPTH_ROOT)):
    if ONLY and env not in ONLY.split(","): continue
    zp = f"{seg_root}/{env}/Data_easy/seg_lcam_front.zip"
    if not os.path.exists(zp):
        print("no seg zip", env); continue
    nm = json.load(open(f"{LABELS}/{env}/seg_label_map.json"))["name_map"]
    id2name = {v: k for k, v in nm.items()}
    sky_ids = {v for k, v in nm.items() if k.lower() in ("sky", "skysphere", "sky0", "cloud")}
    sky_main = nm.get("sky", nm.get("skysphere"))
    z = zipfile.ZipFile(zp)
    names = sorted(n for n in z.namelist() if n.endswith(".png"))
    bytraj = collections.defaultdict(list)
    for n in names:
        bytraj[n.split("/")[-3] if n.count("/") >= 2 else n.split("/")[0]].append(n)
    # depth frames per traj available locally
    trajs = [t for t in sorted(bytraj) if os.path.isdir(f"{DEPTH_ROOT}/{env}/Data_easy/{t}/depth_lcam_front")]
    picks = []
    for k in range(N):
        t = trajs[k % len(trajs)]
        lst = bytraj[t]; j = (k // len(trajs)); per = -(-N // len(trajs))
        picks.append(lst[int((j + 0.5) * len(lst) / per) % len(lst)])
    acc = collections.Counter(); skyvals = collections.Counter()
    nonsky_names = collections.Counter(); finite_sky_lt655 = 0; fin_vals_sky=[]
    tot = dict(px=0, sky=0, sky_caught=0, sky_fin_lt655=0, sky_fin_ge655=0, sky_65504=0, nonsky_sat=0, nonsky_65504=0, sat_any=0, sky_inf=0, sky_gt1e4=0)
    n_used = 0
    for p in picks:
        t = p.split("/")[-3] if p.count("/") >= 2 else p.split("/")[0]
        fn = os.path.basename(p).replace("_seg", "_depth")
        dp = f"{DEPTH_ROOT}/{env}/Data_easy/{t}/depth_lcam_front/{fn}"
        if not os.path.exists(dp): continue
        seg = cv2.imdecode(np.frombuffer(z.read(p), np.uint8), cv2.IMREAD_UNCHANGED)
        d = decode_depth(open(dp, "rb").read())
        if seg.shape != d.shape: print("shape mismatch", env, p); continue
        n_used += 1
        rule = ~np.isfinite(d) | (d > 1e4)
        sk = np.isin(seg, list(sky_ids)) if sky_ids else np.zeros(seg.shape, bool)
        tot["px"] += seg.size; tot["sky"] += int(sk.sum()); tot["sky_caught"] += int((sk & rule).sum())
        fin = np.isfinite(d)
        tot["sky_fin_lt655"] += int((sk & fin & (d < 655)).sum())
        tot["sky_fin_ge655"] += int((sk & fin & (d >= 655) & ~rule).sum())
        tot["sky_65504"] += int((sk & (d == 65504)).sum())
        tot["sky_inf"] += int((sk & ~fin).sum()); tot["sky_gt1e4"] += int((sk & fin & (d > 1e4)).sum())
        sat = ~fin | (d >= 65000)
        tot["sat_any"] += int(sat.sum())
        ns = ~sk
        tot["nonsky_sat"] += int((rule & ns).sum()); tot["nonsky_65504"] += int((ns & (d == 65504)).sum())
        if sk.any():
            v = d[sk]; u, c = np.unique(v[np.isfinite(v)], return_counts=True)
            for a, b in zip(u.tolist(), c.tolist()): skyvals[a] += b
            fin_vals_sky.append(int((~np.isfinite(v)).sum()))
        m = rule & ns
        if m.any():
            u, c = np.unique(seg[m], return_counts=True)
            for a, b in zip(u.tolist(), c.tolist()): nonsky_names[id2name.get(a, f"id{a}")] += b
    # sky depth stats
    if skyvals:
        ks = np.array(sorted(skyvals)); cs = np.array([skyvals[k] for k in ks])
        smin, smax = float(ks[0]), float(ks[-1]); smode = float(ks[cs.argmax()])
        cum = np.cumsum(cs) / cs.sum(); p01 = float(ks[np.searchsorted(cum, 0.01)])
    else:
        smin = smax = smode = p01 = float("nan")
    sky_tot = tot["sky"]
    top_ns = nonsky_names.most_common(3); nsr = tot["nonsky_sat"]
    tree_like = sum(c for n_, c in nonsky_names.items() if any(w in n_.lower() for w in TREEWORDS))
    row = dict(env=env, status="training environment, not held out" if env in TRAIN_LABEL else "local TRAIN env",
        glass_problem=env in GLASS, sky_label=("none" if not sky_ids else ",".join(f"{id2name[i]}={i}" for i in sorted(sky_ids))),
        frames=n_used, sky_pct_of_px=round(100 * sky_tot / max(tot["px"], 1), 3),
        sky_depth_min=smin, sky_depth_p1=p01, sky_depth_mode=smode, sky_depth_max=smax,
        sky_nonfinite_pct=round(100 * tot["sky_inf"] / max(sky_tot, 1), 3) if sky_tot else "",
        sky_65504_pct=round(100 * tot["sky_65504"] / max(sky_tot, 1), 3) if sky_tot else "",
        rule_catches_pct_of_seg_sky=round(100 * tot["sky_caught"] / max(sky_tot, 1), 3) if sky_tot else "",
        seg_sky_finite_lt655_pct=round(100 * tot["sky_fin_lt655"] / max(sky_tot, 1), 3) if sky_tot else "",
        seg_sky_655_to_1e4_pct=round(100 * tot["sky_fin_ge655"] / max(sky_tot, 1), 3) if sky_tot else "",
        nonsky_rule_pct_of_px=round(100 * nsr / max(tot["px"], 1), 3),
        nonsky_rule_pct_of_rule=round(100 * nsr / max(nsr + tot["sky_caught"], 1), 3),
        nonsky_rule_top_classes="; ".join(f"{n_}:{100*c/max(nsr,1):.0f}%" for n_, c in top_ns),
        nonsky_rule_treelike_pct=round(100 * tree_like / max(nsr, 1), 1) if nsr else 0.0,
        sat_any_pct_of_px=round(100 * tot["sat_any"] / max(tot["px"], 1), 3))
    rows.append(row); print(env, row["rule_catches_pct_of_seg_sky"], row["nonsky_rule_pct_of_rule"], flush=True)
with open(f"{out_dir}/sky_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

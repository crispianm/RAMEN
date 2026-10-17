#!/usr/bin/env python3
"""far_diag (1): per-depth-bin accounting of GT / mask / SiLog-eligible / sky pixels through the REAL loaders (train._build_multisource, C3 data block, degrade off).
Stratified: N clips per source (seeded), 8 frames, 518 VDA resize+crop as in training. Source weights = sampling_mass share (the C3 sampler draws a clip with P = mass_s/sum(mass), uniform within a source).
CPU only. Run (repo root): CUDA_VISIBLE_DEVICES= ~/.venvs/ramen/bin/python -I outdoor_debug/parts/far_diag/perbin_accounting.py --out <dir> [--clips 40]
"""
import argparse, copy, json, pathlib, sys, bisect, random
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
import train
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--clips", type=int, default=40); ap.add_argument("--frames", type=int, default=8)
args = ap.parse_args(); OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
T_ROOT = pathlib.Path("/mnt/DATA/DegradedDepth/data/TRAIN")
E2 = pathlib.Path("/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/E2_by_source")
cfg = train.load_config(REPO / "configs/experiments/dn01_c3rep_20261009_c3_n46.yaml")   # C3 n46 (same data block as C3 s45 / FT n46)
d = copy.deepcopy(cfg["data"]); d["seed"] = 45
srcs, names = [], []
nyu_dir = OUT / "nyu_local"; nyu_dir.mkdir(exist_ok=True)
for p in sorted((T_ROOT / "nyuv2/data").glob("train-*")):
    if not (nyu_dir / p.name).exists(): (nyu_dir / p.name).symlink_to(p)
for s in d["sources"]:
    s = dict(s); t = s["type"]
    if t == "nyu_train": nm = t; s["root"] = str(nyu_dir)
    elif t == "scannetpp_train":
        nm = t; s["root"] = str(T_ROOT / "scannetpp_train_full"); s["extra_root"] = s["root"]; s["allow_missing"] = True
    elif t == "pointodyssey":
        nm = "pointodyssey_" + ("train" if s["root"].endswith("/train") else "val")
        s["root"] = str(T_ROOT / "point_odyssey" / ("train" if nm.endswith("train") else "val"))
        kl = json.load(open(REPO / s["po_keep_list"])); kl["kept"] = [k for k in kl["kept"] if (pathlib.Path(s["root"]) / k).is_dir()]
        p = OUT / f"po_keep_local_{nm}.json"; json.dump(kl, open(p, "w")); s["po_keep_list"] = str(p); s.pop("po_keep_sha256", None)
    else:
        nm = t; s["root"] = str(T_ROOT / t)
        if "gt_root" in s: s["gt_root"] = str(T_ROOT / "kitti_raw_gt")
    srcs.append(s); names.append(nm)
d["sources"] = srcs; d["num_frames"] = args.frames; d["num_workers"] = 0; d["clean_prob"] = 1.0   # degrade stays on (loader requirement for metric_supervision); clean_prob 1 = RGB untouched, GT is never degraded
loader, _ = train._build_multisource({"data": d, "training": {"batch_size": 1}}, "train", 1, 0)
ds = loader.dataset; offs = ds.cumulative_sizes
train._set_severity_max(ds, 1.0)
for k, v in (("clean_prob", 1.0), ("seeded_noise", bool(d["seeded_noise"])), ("blur_only_prob", float(d["blur_only_prob"])), ("static_prob", float(d["static_prob"])),
             ("static_noise_mult", tuple(float(x) for x in d["static_noise_mult"])), ("static_ev_jitter", float(d["static_ev_jitter"])), ("lowlight", d["lowlight"])):
    train._set_dataset_attr(ds, k, v)
mass = {n: float(s["sampling_mass"]) for n, s in zip(names, srcs)}
sizes = {}; prev = 0
for n, c in zip(names, offs): sizes[n] = int(c - prev); prev = c
EDGES = [0, 20, 50, 80, 120, 200, 655, 1e9]; BL = ["0-20", "20-50", "50-80", "80-120", "120-200", "200-655", ">=655"]
res = {}; rng = random.Random(45); lo = 0
for n, c in zip(names, offs):
    idx = rng.sample(range(lo, c), min(args.clips, c - lo)); lo = c
    a = {k: np.zeros(len(BL)) for k in ("valid", "silog")}; sky = tot = 0.0; skybin = 0.0
    sup = None; ceil = None; sparse = None
    for i in idx:
        it = ds[i]
        dep = torch.as_tensor(it["depth"]).float(); m = (torch.as_tensor(it["mask"]) * (dep > 0)) > 0.5
        sk = (torch.as_tensor(it["sky"]) > 0.5) & ~m if "sky" in it else torch.zeros_like(m)
        sup = float(torch.as_tensor(it.get("metric_sup", 1.0)).reshape(-1)[0]) > 0.5
        ceil = float(torch.as_tensor(it["silog_ceil"]).reshape(-1)[0]) if "silog_ceil" in it else float("inf")
        sparse = bool(float(torch.as_tensor(it.get("sparse", 0.0)).reshape(-1)[0]) > 0.5)
        ms = m & sup & (dep < ceil)
        tot += dep.numel(); sky += float(sk.sum())
        for j in range(len(BL)):
            b = (dep >= EDGES[j]) & (dep < EDGES[j + 1])
            a["valid"][j] += float((m & b).sum()); a["silog"][j] += float((ms & b).sum())
    res[n] = dict(clips=len(idx), mass=mass[n], n_clips_in_source=sizes[n], total_px=tot, sky_px=sky, metric_sup=sup, silog_ceil=ceil, sparse=sparse,
                  valid_px=a["valid"].tolist(), silog_px=a["silog"].tolist())
    print(n, len(idx), "valid frac %.3f" % (sum(a["valid"]) / tot), "sky %.3f" % (sky / tot), "far>=80 %.4f" % (sum(a["valid"][3:]) / tot), flush=True)
json.dump(dict(bins=BL, edges=EDGES, sources=res, scheme="stratified N clips/source seeded random.Random(45); weight = mass/sum(mass)"), open(OUT / "perbin_raw.json", "w"), indent=1)

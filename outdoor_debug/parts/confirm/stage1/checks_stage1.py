"""Stage 1 real-path checks on INSTANTIATED objects (CPU only, local data), for all six run configs, with forced fails. Run from repo root:
  ~/.venvs/ramen/bin/python outdoor_debug/parts/confirm/stage1/checks_stage1.py
(a) class per source via the REAL train._build_multisource -> train._make_source recorder (as arm_checks/classes.py): TartanAir class has a TS2Labeller iff the run is TS2/H300 (all six are), vkitti bound 655,
    TS2 TartanAir differs from the non-TS2 (screen A0) dataset on the same clip (masks differ). (b) real train._compute_loss_vda on a real TS2 TartanAir batch: hinge term nonzero (pred 10 m) ONLY in H300 configs, absent in controls."""
import sys, copy, json, random, math
from pathlib import Path
import numpy as np, torch, yaml
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "outdoor_debug/parts/arm_checks"))
import train
from classes import remap
VDA_REPO = "/home/wg19671/repos/Video-Depth-Anything"
NAMES = [f"S1_n{n}_{a}" for n in ("46", "47", "48") for a in ("TS2", "H300")]
CFG = {n: yaml.safe_load(open(HERE / f"configs/arm_{n}.yaml")) for n in NAMES}
res = {}
def P(*a): print(*a, flush=True)

def classes_of(cfg):
    real = train._make_source; rec = []
    def w(spec, split, common):
        ds = real(spec, split, common)
        if split == "train": rec.append(dict(type=spec["type"], cls=type(ds).__name__, ts2=type(getattr(ds, "ts2", None)).__name__, vkitti_max_depth=getattr(ds, "vkitti_max_depth", None), n=len(ds)))
        return ds
    train._make_source = w
    try: train._build_multisource(remap(cfg), "train", 1, 0)
    except FileNotFoundError as e: rec.append(dict(type="<local build stopped>", err=str(e)[:90]))    # local tree lacks ScanNet++ split list (as for the screen); tartanair/vkitti/PO built before it
    finally: train._make_source = real
    return {r["type"]: r for r in rec}

def eval_classes(name, cfg):
    r = classes_of(cfg); ts2_want = True   # all six runs are TS2
    t, v = r.get("tartanair", {}), r.get("vkitti", {})
    L = [(t.get("cls") == "VDASemanticsDegradedTartanAirDataset", f"tartanair class {t.get('cls')}"),
         ((t.get("ts2") == "TS2Labeller") == ts2_want, f"tartanair.ts2 = {t.get('ts2')} (want TS2Labeller)"),
         (v.get("cls") == "VDASemanticsDegradedVKITTIDataset" and v.get("vkitti_max_depth") == 655, f"vkitti {v.get('cls')} max_depth {v.get('vkitti_max_depth')} (want 655)"),
         (r.get("vkitti", {}).get("ts2") == "NoneType", "vkitti has no ts2")]
    return L, r

def mk_ta(cfg):
    spec = copy.deepcopy([s for s in remap(cfg)["data"]["sources"] if s["type"] == "tartanair"][0]); spec["envs"] = ["Downtown"]
    d = cfg["data"]; common = dict(num_frames=int(d["num_frames"]), image_size=tuple(d["image_size"]), clip_stride=int(d["clip_stride"]), frame_stride=int(d["frame_stride"]), max_depth=float(d["max_depth"]),
        seed=int(d["seed"]), degrade=True, degrade_mode=d["degrade_mode"], degrade_severity=d.get("degrade_severity"), max_clips_per_video=d.get("max_clips_per_video"), resize_mode=d.get("resize_mode"))
    train._with_split_seed(common, d)
    ds = train._make_source(spec, "train", common)
    for k in ("clean_prob", "blur_only_prob", "blur_disk_prob", "lowlight", "seeded_noise", "static_prob", "static_noise_mult", "static_ev_jitter"):
        if k in d and hasattr(ds, k): setattr(ds, k, d[k])
    return ds
def sample(ds, idx, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); return ds[idx]
def batchify(s): return {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in s.items()}

def hinge_of(cfg, batch, val=10.0):
    cfg = copy.deepcopy(cfg); cfg["model"]["vda_repo"] = VDA_REPO; train.check_sky_loss(cfg)
    train.term_flush(); pred = torch.full(batch["depth"].shape, val).requires_grad_(True)
    L = train._compute_loss_vda(pred, batch, cfg, False); L.backward(); return train.term_flush().get("sky_hinge_weighted")

def expect_hinge(name, h): return (h is not None and h > 0.0) if name.endswith("H300") else (h is None)

ok_all = True
P("== (a) classes through the real _build_multisource, six configs")
for n in NAMES:
    L, r = eval_classes(n, CFG[n]); res[f"classes_{n}"] = r
    for g, m in L: P(f"  [{'OK' if g else 'DEFECT'}] {n}: {m}"); ok_all &= g
P("== (a2) TS2 TartanAir vs the screen A0 (no TS2) dataset, same clip, same seed: masks differ, depth/frames equal")
a0 = yaml.safe_load(open(REPO / "outdoor_debug/parts/arm_checks/configs/arm_A0.yaml"))
dA, dT = mk_ta(a0), mk_ta(CFG["S1_n47_TS2"]); res["a2"] = {}
best = max(range(0, len(dA), max(1, len(dA) // 40)), key=lambda i: float((~np.isfinite(__import__('data.video_clip_dataset', fromlist=['x'])._decode_depth(str(dA.clips[i][1][dA.clips[i][3]]))) | (__import__('data.video_clip_dataset', fromlist=['x'])._decode_depth(str(dA.clips[i][1][dA.clips[i][3]])) > 1e4)).mean()))
sa, st = sample(dA, best, 0), sample(dT, best, 0)
differ = bool((sa["mask"] != st["mask"]).any() or (sa["sky"] != st["sky"]).any()); same = bool(torch.equal(sa["depth"], st["depth"]) and torch.equal(sa["frames"], st["frames"]))
P(f"  clip {best}: classes {type(dA).__name__}/{type(dT).__name__}, ts2 {type(dA.ts2).__name__} vs {type(dT.ts2).__name__}; masks_differ={differ} depth+frames_equal={same}")
ok_all &= differ and same and dA.ts2 is None and type(dT.ts2).__name__ == "TS2Labeller"; res["a2"] = dict(clip=best, masks_differ=differ, depth_frames_equal=same)
P("== (b) hinge on a real TS2 TartanAir batch (pred=10 m): nonzero only in H300")
batch = batchify(sample(dT, best, 0)); res["b"] = {}
for n in NAMES:
    h = hinge_of(CFG[n], batch); g = expect_hinge(n, h); ok_all &= g; res["b"][n] = h
    P(f"  [{'OK' if g else 'DEFECT'}] {n}: sky_hinge_weighted(pred=10m) = {h}")
P("== forced fails (each must be detected)")
ff = True
def ffcheck(label, detected): 
    global ff; ff &= detected; P(f"  forced-fail [{label}] -> {'detected' if detected else 'NOT DETECTED'}")
def detected_cls(m):
    try: Lm, _ = eval_classes("mut", m); return not all(g for g, _ in Lm)
    except ValueError as e: P("    (real loader raised:", str(e)[:80], ")"); return True
m = copy.deepcopy(CFG["S1_n47_TS2"]); [s.__setitem__("ts2_sky", False) for s in m["data"]["sources"] if s["type"] == "tartanair"]
ffcheck("control with ts2_sky false (seg_root kept) -> loader/class check", detected_cls(m))
m = copy.deepcopy(CFG["S1_n48_TS2"]); [(s.pop("ts2_sky"), s.pop("ts2_seg_root")) for s in m["data"]["sources"] if s["type"] == "tartanair"]
ffcheck("control with both ts2 keys removed -> class check", detected_cls(m))
m = copy.deepcopy(CFG["S1_n48_H300"]); m["loss"]["sky_hinge_weight"] = 0.0
ffcheck("H300 with hinge weight 0 -> hinge check", not expect_hinge("S1_n48_H300", hinge_of(m, batch)))
m = copy.deepcopy(CFG["S1_n46_TS2"]); m["loss"]["sky_hinge_weight"] = 0.1; m["loss"]["sky_hinge_min_depth"] = 300.0
ffcheck("control with the hinge keys -> hinge check", not expect_hinge("S1_n46_TS2", hinge_of(m, batch)))
m = copy.deepcopy(CFG["S1_n47_H300"]); m["loss"]["sky_hinge_min_depth"] = 1.0
ffcheck("H300 floor 1 m (hinge zero at pred 10 m) -> hinge check", not expect_hinge("S1_n47_H300", hinge_of(m, batch)))
m = copy.deepcopy(CFG["S1_n47_TS2"]); [s.__setitem__("vkitti_max_depth", 656) for s in m["data"]["sources"] if s["type"] == "vkitti"]
ffcheck("vkitti bound 656 -> class check", detected_cls(m))
P(f"\nSTAGE1 REAL-PATH CHECKS: positives {'OK' if ok_all else 'DEFECT'}; forced-fail suite {'all detected' if ff else 'SOME NOT DETECTED'}")
json.dump(res, open(HERE / "checks_stage1_results.json", "w"), indent=1, default=str)
sys.exit(0 if (ok_all and ff) else 3)

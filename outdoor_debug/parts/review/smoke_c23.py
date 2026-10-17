"""Checks 2-5 (real local data, CPU). Run from repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/arm_checks/smoke.py
Sources are built with the REAL train._make_source (common dict mirrored from train._build_multisource lines 427-444 + _with_split_seed)."""
import sys, copy, json, random, math, importlib.util
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path('/home/wg19671/.claude/jobs/a0d391cb/tmp/od'); sys.path.insert(0, str(REPO)); sys.path.insert(0, '/tmp/rv_smoke')
import train
import data.vda_semantics_dataset as vsd
from data.video_clip_dataset import _decode_depth
from data.vda_resize import vda_resize_depth_like
from data.tartanair_ts2 import table_sha256, load_env_table
from arms import build
from classes import remap
HERE = Path('/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/arm_checks')
spec_ = importlib.util.spec_from_file_location("ts_v2_draft", REPO / "outdoor_debug/parts/TA_sky/ts_v2_draft.py"); draft = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(draft)
VDA_REPO = "/home/wg19671/repos/Video-Depth-Anything"
import os
FF=os.environ.get('FF','none')
out = {}
def P(*a):
    print(*a, flush=True)

def common_for(cfg, degrade):
    d = cfg["data"]; nf = int(d["num_frames"])
    c = dict(num_frames=nf, image_size=tuple(d["image_size"]), clip_stride=int(d["clip_stride"]), frame_stride=int(d["frame_stride"]), max_depth=float(d["max_depth"]),
             seed=int(d["seed"]), degrade=degrade, degrade_mode=d["degrade_mode"], degrade_severity=d.get("degrade_severity"), max_clips_per_video=d.get("max_clips_per_video"), resize_mode=d.get("resize_mode"))
    train._with_split_seed(c, d)
    return c

def mk(cfg, typ, degrade, envs=None, scenes=None):
    """real train._make_source for the (single) source of type typ; plus the mixin attributes train.py sets in _build_multisource / the dataset-level C3 knobs."""
    spec = copy.deepcopy([s for s in cfg["data"]["sources"] if s["type"] == typ][0])
    if envs: spec["envs"] = envs
    if not degrade:   # the clean (non-degraded) VDA classes refuse the mixin-only keys; they do not affect mask/sky/depth
        for k in ("metric_supervision", "sparse_gt", "silog_max_depth"): spec.pop(k, None)
    ds = train._make_source(spec, "train", common_for(cfg, degrade))
    d = cfg["data"]
    for k in ("clean_prob", "blur_only_prob", "blur_disk_prob", "lowlight", "seeded_noise", "static_prob", "static_noise_mult", "static_ev_jitter"):
        if k in d and hasattr(ds, k): setattr(ds, k, d[k])   # mirrors the train.py data-level knob copy (not exercised by the checks; masks are GT-side)
    if degrade and typ == "vkitti" and getattr(ds, "silog_max_depth", None) is None:
        ds.silog_max_depth = float("inf")                       # train.py:477-480 (any part has a ceiling -> the others get +inf)
    return ds

class Cap:  # capture crop/resize chosen by the dataset
    def __enter__(self):
        self.orig = vsd._vda_frame_and_crop; self.rec = []
        def w(*a, **k):
            r = self.orig(*a, **k); self.rec.append(r); return r
        vsd._vda_frame_and_crop = w; return self
    def __exit__(self, *a): vsd._vda_frame_and_crop = self.orig

def get(ds, idx, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with Cap() as c:
        s = ds[idx]
    return s, c.rec[0]

def pct(t, thr=0.5): return 100.0 * float((t > thr).float().mean())

base, arms = build()
tcfg = {n: remap(c) for n, c in arms.items()}
SEGL = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_seg"

# ---------- check 2: TS2 vs A0 on a real Downtown clip --------------------------------------------------------------
P("== check 2: TartanAir TS2 real Downtown clip")
out["c2"] = {}
for degrade in (False, True):
    a0 = mk(tcfg["A0"], "tartanair", degrade, envs=["Downtown"]); ts = mk(tcfg["TS2"], "tartanair", degrade, envs=["Downtown"])
    _lab = ts.ts2
    if FF=="ts2off": ts.ts2=None
    # pick the clip (same index in both) with the largest released-rule sky share in its first frame, scanning every 20th clip
    best = None
    for i in range(0, len(a0), max(1, len(a0)//40)):
        img_f, dep_f, vid, st = a0.clips[i]
        d = _decode_depth(str(dep_f[st])); sk = float((~np.isfinite(d) | (d > 1e4)).mean())
        if best is None or sk > best[0]: best = (sk, i)
    idx = best[1]
    res = dict(degrade=degrade, clip_idx=idx, video=str(a0.clips[idx][2]), start=int(a0.clips[idx][3]), n_clips=len(a0), cls_A0=type(a0).__name__, cls_TS2=type(ts).__name__, ts2_attr=type(ts.ts2).__name__, a0_ts2=str(a0.ts2))
    for seed in (0, 1):
        sa, ca = get(a0, idx, seed); st_, ct = get(ts, idx, seed)
        assert ca[1:] == ct[1:], "crop/resize differs between arms (RNG not matched)"
        T = sa["mask"].shape[1]
        r = dict(A0_sky_pct=pct(sa["sky"]), A0_mask_pct=pct(sa["mask"]), TS2_sky_pct=pct(st_["sky"]), TS2_mask_pct=pct(st_["mask"]),
                 TS2_uncertain_pct=100.0 - pct(st_["sky"]) - pct(st_["mask"]), A0_uncertain_pct=100.0 - pct(sa["sky"]) - pct(sa["mask"]),
                 masks_differ=bool((sa["mask"] != st_["mask"]).any() or (sa["sky"] != st_["sky"]).any()),
                 depth_equal=bool(torch.equal(sa["depth"], st_["depth"])), frames_equal=bool(torch.equal(sa["frames"], st_["frames"])),
                 static=float(st_["static"]) if "static" in st_ else None)
        # reference: ts_v2_guarded on raw depth+seg, then the released resize/crop
        (x0, y0), (rw, rh) = ct[1], ct[2]
        lab = _lab; env = "Downtown"; fl = load_env_table()[env]
        img_f, dep_f, vid, start = ts.clips[idx]; H = ts.image_size[0]
        refs, refk = [], []
        for t in range(T):
            f = dep_f[start + t * ts.frame_stride]
            dr = _decode_depth(str(f)); seg = lab.read_seg(f)
            sky_, mask_, unc_ = draft.ts_v2_guarded(dr, seg, fl["sky_ids"], fl["far_dome"], fl["seg_sky_ok"], fl["sat_nonsky_is_sky"], ts.max_depth, fl["fallback_sat_is_sky"])
            m = vda_resize_depth_like(mask_.astype(np.float32), rw, rh)[y0:y0+H, x0:x0+H]; k = vda_resize_depth_like(sky_.astype(np.float32), rw, rh)[y0:y0+H, x0:x0+H]
            refs.append(m); refk.append(k)
        refm, refk = torch.from_numpy(np.stack(refs))[None, None], torch.from_numpy(np.stack(refk))[None, None]   # (1,1,T,H,W)
        if r["static"]: refm, refk = [x[:, :, T//2:T//2+1].expand(-1, -1, T, -1, -1) for x in (refm, refk)]
        r["max_abs_diff_mask_vs_ts_v2_guarded"] = float((st_["mask"].reshape(refm.shape) - refm).abs().max())
        r["max_abs_diff_sky_vs_ts_v2_guarded"] = float((st_["sky"].reshape(refk.shape) - refk).abs().max())
        res[f"seed{seed}"] = r
        P(json.dumps({"degrade": degrade, "seed": seed, **r}))
    out["c2"][f"degrade_{degrade}"] = res; P(json.dumps({k: v for k, v in res.items() if not k.startswith("seed")}))
    if degrade: ts_batch_ds, ts_idx = ts, idx

# ---------- check 3: VKITTI TS2(cap655) vs VK655 -------------------------------------------------------------------------
P("== check 3: VKITTI real frames")
out["c3"] = {}
def vstats(s):
    d, m, k = s["depth"], s["mask"] > 0.5, s["sky"] > 0.5
    valid = m
    in_band = (m & (d >= 80) & (d <= 655.4)).float().sum() / m.float().sum().clamp_min(1)
    return dict(sentinel_px_pct=pct((d > 655.2).float()), mask_pct=pct(s["mask"]), sky_pct=pct(s["sky"]), depth_max=float(d.max()), valid_px_in_80_655_pct=100*float(in_band),
                valid_depth_max=float(d[m].max()) if m.any() else None, n_valid_px=int(m.sum()))
for degrade in (False, True):
    a = mk(tcfg["TS2"], "vkitti", degrade); b = mk(tcfg["TS2_VK655"], "vkitti", degrade)
    if FF=="flagoff_recipe" and not getattr(vsd,'_patched',0):
        vsd._patched=1      # break the flag in the class the recipe actually runs (VDASemanticsVKITTIDataset.__getitem__)
        _orig = vsd.VDASemanticsVKITTIDataset.__getitem__
        def _ig(self, idx):
            v = self.__dict__.get("sky_as_valid_depth"); self.__dict__["sky_as_valid_depth"]=False
            try: return _orig(self, idx)
            finally: self.__dict__["sky_as_valid_depth"]=v
        vsd.VDASemanticsVKITTIDataset.__getitem__ = _ig
    if FF=="flagoff_base_only" and not globals().get('_p2'):
        _p2=1   # break ONLY the base-class copy of the feature (data/vkitti_dataset.py), which the recipe class overrides
        import data.vkitti_dataset as vk
        _o2 = vk.VKITTIDataset.__getitem__
        def _ig2(self, idx):
            v = self.__dict__.get("sky_as_valid_depth"); self.__dict__["sky_as_valid_depth"]=False
            try: return _o2(self, idx)
            finally: self.__dict__["sky_as_valid_depth"]=v
        vk.VKITTIDataset.__getitem__ = _ig2
    from data.vkitti_dataset import _decode_vkitti_depth
    best = None
    for i in range(0, len(a), max(1, len(a)//60)):
        dep_f = a.clips[i][1]; st = a.clips[i][3]
        sh = float((_decode_vkitti_depth(str(dep_f[st])) >= 655).mean())
        if best is None or sh > best[0]: best = (sh, i)
    idx = best[1]
    sa, _ = get(a, idx, 0); sb, _ = get(b, idx, 0)
    res = dict(degrade=degrade, clip_idx=idx, clip=a.clips[idx][2], raw_first_frame_sentinel_pct=100*best[0], cls_TS2=type(a).__name__, cls_VK655=type(b).__name__,
               TS2_state=dict(vkitti_max_depth=a.vkitti_max_depth, sky_as_valid_depth=getattr(a, "sky_as_valid_depth", None)),
               VK655_state=dict(vkitti_max_depth=b.vkitti_max_depth, sky_as_valid_depth=getattr(b, "sky_as_valid_depth", None)),
               effective_depth_bound=dict(TS2=train.effective_depth_bound(a), VK655=train.effective_depth_bound(b)),
               TS2=vstats(sa), VK655=vstats(sb), frames_equal=bool(torch.equal(sa["frames"], sb["frames"])))
    out["c3"][f"degrade_{degrade}"] = res; P(json.dumps(res))
    if degrade: vk_ds = (a, b, idx)


json.dump(out, open("/tmp/rv_smoke/out_%s.json"%FF, "w"), indent=1)

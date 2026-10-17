"""Checks 2-5 (real local data, CPU). Run from repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/arm_checks/smoke.py
Sources are built with the REAL train._make_source (common dict mirrored from train._build_multisource lines 427-444 + _with_split_seed)."""
import sys, copy, json, random, math, importlib.util
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
import train
import data.vda_semantics_dataset as vsd
from data.video_clip_dataset import _decode_depth
from data.vda_resize import vda_resize_depth_like
from data.tartanair_ts2 import table_sha256, load_env_table
from arms import build
from classes import remap
HERE = Path(__file__).parent
spec_ = importlib.util.spec_from_file_location("ts_v2_draft", REPO / "outdoor_debug/parts/TA_sky/ts_v2_draft.py"); draft = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(draft)
VDA_REPO = "/home/wg19671/repos/Video-Depth-Anything"
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
        lab = ts.ts2; env = "Downtown"; fl = load_env_table()[env]
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

# ---------- check 4: loss on VK655 smoke batch ----------------------------------------------------------------------------
P("== check 4: SiLog on VKITTI batch (TS2 vs VK655), real train._compute_loss_vda")
def cfg_for(name):
    c = copy.deepcopy(arms[name]); c["model"]["vda_repo"] = VDA_REPO; return c
def batchify(s): return {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in s.items()}
def run_loss(cfg, batch, pred):
    train.term_flush(); p = pred.clone().requires_grad_(True)
    L = train._compute_loss_vda(p, batch, cfg, False); L.backward()
    return float(L), train.term_flush(), p.grad
def silog_mask_count(batch, pred_shape):
    B, _, T, H, W = pred_shape
    m = (batch["mask"].reshape(B, T, H, W) * (batch["depth"].reshape(B, T, H, W) > 0)).bool()
    m_si = train.silog_mask(m, batch)
    if m_si is not None and batch.get("silog_ceil") is not None:
        m_si = m_si & (batch["depth"].reshape(B, T, H, W) < batch["silog_ceil"].reshape(-1, 1, 1, 1))
    return m_si
out["c4"] = {}
a, b, idx = vk_ds
for name, ds in (("TS2", a), ("TS2_VK655", b)):
    s, _ = get(ds, idx, 0); batch = batchify(s); cfg = cfg_for(name); D = batch["depth"]
    m_si = silog_mask_count(batch, (1, 1) + tuple(D.shape[2:]))
    sent = (D.reshape(m_si.shape) > 655.2)
    r = dict(keys=sorted(k for k in batch if k in ("metric_sup", "silog_ceil", "sparse", "sky", "mask", "depth")), silog_ceil=float(batch["silog_ceil"]), metric_sup=float(batch["metric_sup"]),
             silog_mask_px=int(m_si.sum()), sentinel_px_in_batch=int(sent.sum()), sentinel_px_in_silog_mask=int((m_si & sent).sum()))
    for lab, f in (("0.5x", lambda d: 0.5 * d), ("1x", lambda d: d), ("2x", lambda d: 2 * d), ("min(GT,80)", lambda d: d.clamp(max=80.0))):
        pred = f(D).clone()
        L, terms, g = run_loss(cfg, batch, pred)
        gs = g.reshape(m_si.shape)
        r[lab] = dict(total_loss=L, silog=terms.get("silog"), silog_mean_log_err=terms.get("silog_mean_log_err"), grad_abs_sum_on_sentinel_px=float(gs[sent].abs().sum()) if sent.any() else 0.0)
    out["c4"][name] = r; P(name, json.dumps(r))

# ---------- check 5: hinge on a TS2 TartanAir batch with sky -------------------------------------------------------------
P("== check 5: hinge")
out["c5"] = {}
s, _ = get(ts_batch_ds, ts_idx, 0); batch = batchify(s)
sky_px = ((batch["sky"] > 0.5) & ~(batch["mask"] > 0.5)).sum()
P("sky px in TS2 TartanAir batch:", int(sky_px))
out["c5"]["sky_px"] = int(sky_px); out["c5"]["tartanair_batch_static"] = float(s["static"]) if "static" in s else None
for name in arms:
    cpath = HERE / "configs" / f"arm_{name}.yaml"
    cfg = train.load_config(cpath)                    # real loader path on the written arm yaml
    train.check_sky_loss(cfg)
    floor = cfg["loss"].get("sky_hinge_min_depth", 80.0); w = cfg["loss"]["sky_hinge_weight"]
    cfg["model"]["vda_repo"] = VDA_REPO
    r = dict(cfg_sky_hinge_weight=w, cfg_sky_hinge_min_depth_key_present="sky_hinge_min_depth" in cfg["loss"], floor_used=floor)
    for lab, val in (("pred=10m", 10.0), ("pred=100m", 100.0), ("pred=1000m", 1000.0)):
        pred = torch.full(batch["depth"].shape, val)
        L, terms, g = run_loss(cfg, batch, pred)
        r[lab] = dict(sky_hinge_weighted=terms.get("sky_hinge_weighted"), total_loss=L)
    r["expected_pred10"] = w * (math.log(floor) - math.log(10.0)) if w > 0 else 0.0
    out["c5"][name] = r; P(name, json.dumps(r))
json.dump(out, open(HERE / "smoke_results.json", "w"), indent=1)

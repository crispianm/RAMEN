#!/usr/bin/env python3
"""E2_by_source: per-source gradient decomposition on the TRUE nine-source FT-n46 mix (NYU + ScanNet++ now local).

Derived from E_gradients/loss_diag.py (per-source decomposition, group 'source') and T_trajectory/grad_traj.py (local NYU/ScanNet++ roots).
Real code paths only: train.load_config, train._build_multisource (mass sampler), train._compute_loss_vda (every term is a re-call with one switch),
models.build_model + trainable_ema, bf16 autocast forward in model.train(), activation-checkpointing runtime patch for RAMEN. No optimizer step, no repo edit.

Per clip i (the pred-map gradient is separable per clip): parameter-gradient L2 norm (output_conv, all) of
  comp in {full, vda, silog, sky_term}  [whole clip]
  comp in {vda, silog} x class in {near<10, mid 10-80, far 80-655, far>=655 valid, sky}
plus projection of the clip gradient onto the batch total, pixel counts per class, L1 pred-grad mass per class.
Run (repo root):  PYTORCH_ALLOC_CONF=expandable_segments:True ~/.venvs/ramen/bin/python -I outdoor_debug/parts/E2_by_source/grad_by_source.py --out <dir>
"""
import argparse, copy, csv, json, pathlib, sys, time, bisect
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import train
from models import build_model, trainable_modules

D = "/mnt/DATA/DegradedDepth/"
T_ROOT = pathlib.Path(D + "data/TRAIN")
VDA_REPO = D + "competition/Video-Depth-Anything"
C = D + "checkpoints_v3/"
CK = {
    "released": None,
    "FT_n46": C + "eval_20261012_ft_n46/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt",
    "RAMEN_n46": C + "eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt",
}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--models", default="released,FT_n46,RAMEN_n46")
args = ap.parse_args()
OUT = pathlib.Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"

cfg_ft = train.load_config(REPO / "configs/experiments/dn01_c3rep_20261009_ft_n46.yaml")
LOSS = copy.deepcopy(cfg_ft["loss"]); LOSS["log_terms"] = False
assert LOSS["sky_vda_zero_disparity"] is True and LOSS["sky_hinge_weight"] == 0.0 and LOSS["metric_silog_weight"] == 1.0


def local_model_cfg(cfg):
    cfg = copy.deepcopy(cfg); m = cfg["model"]
    m["vda_repo"] = VDA_REPO
    m["vda_checkpoint_path"] = D + "RAMEN/checkpoints/metric_video_depth_anything_vits.pth"
    m["da_v2_checkpoint_path"] = D + "RAMEN/checkpoints/depth_anything_v2_vits.pth"
    m["st_temp2_init"] = None
    return cfg


def load_model(name):
    if name == "released":
        net = build_model(local_model_cfg(cfg_ft))
    else:
        ck = torch.load(CK[name], map_location="cpu", weights_only=False)
        assert int(ck["step"]) == 12000, ck["step"]
        net = build_model(local_model_cfg(ck["cfg"]))
        torch.nn.ModuleDict(trainable_modules(net)).load_state_dict(ck.get("trainable_ema", ck.get("trainable")), strict=True)
        del ck
    net = net.to(dev)
    from torch.utils.checkpoint import checkpoint
    for mod in net.modules():
        if hasattr(mod, "attn") and hasattr(mod, "mlp") and hasattr(mod, "norm1"):
            f = mod.forward
            mod.forward = (lambda *a, _f=f, **k: checkpoint(_f, *a, use_reentrant=False, **k))
    if hasattr(net, "restore_two_stage"):
        net.restore_two_stage = (lambda clip, _f=net.restore_two_stage: checkpoint(_f, clip, use_reentrant=False))
    net.train()
    return net


def group_of(n):
    if "motion_modules" in n: return "motion"
    if "output_conv" in n: return "output_conv"
    if any(s in n for s in ("refinenet1.", "refinenet2.", "layer1_rn", "layer2_rn")): return "last_dpt"
    if any(s in n for s in ("scratch.", "projects.", "resize_layers.")): return "dpt_other"
    return "other"


def build_loader():
    d = copy.deepcopy(cfg_ft["data"]); d["seed"] = 45
    roots = {"tartanair": T_ROOT / "tartanair", "vkitti": T_ROOT / "vkitti", "irs": T_ROOT / "irs", "spring": T_ROOT / "spring", "kitti_raw": T_ROOT / "kitti_raw"}
    srcs, names = [], []
    nyu_dir = OUT / "nyu_local"                      # symlinks to the TRAIN shards only (train-*), never val-*
    nyu_dir.mkdir(exist_ok=True)
    for p in sorted((T_ROOT / "nyuv2/data").glob("train-*")):
        if not (nyu_dir / p.name).exists(): (nyu_dir / p.name).symlink_to(p)
    for s in d["sources"]:
        s = dict(s); t = s["type"]
        if t == "nyu_train":
            nm = t; s["root"] = str(nyu_dir)
        elif t == "scannetpp_train":
            nm = t; s["root"] = str(T_ROOT / "scannetpp_train_full"); s["extra_root"] = s["root"]; s["allow_missing"] = True   # config: splits [sem_test] under scannetpp_train (absent; same scenes live in _full) + subset140 under _full
        elif t == "pointodyssey":
            nm = "pointodyssey_" + ("train" if s["root"].endswith("/train") else "val")
            s["root"] = str(T_ROOT / "point_odyssey" / ("train" if nm.endswith("train") else "val"))
            kl = json.load(open(REPO / s["po_keep_list"])); kl["kept"] = [k for k in kl["kept"] if (pathlib.Path(s["root"]) / k).is_dir()]
            p = OUT / f"po_keep_local_{nm}.json"; json.dump(kl, open(p, "w")); s["po_keep_list"] = str(p); s.pop("po_keep_sha256", None)
        else:
            nm = t; s["root"] = str(roots[t])
            if "gt_root" in s: s["gt_root"] = str(T_ROOT / "kitti_raw_gt")
        srcs.append(s); names.append(nm)
    d["sources"] = srcs; d["num_frames"] = args.frames; d["num_workers"] = 2
    cfg = {"data": d, "training": {"batch_size": args.bs}}
    loader, _ = train._build_multisource(cfg, "train", 1, 0)
    ds = loader.dataset
    offs = ds.cumulative_sizes
    weights = list(loader.sampler.weights)
    train._set_severity_max(ds, 1.0)
    train._set_dataset_attr(ds, "clean_prob", float(d["clean_prob"])); train._set_dataset_attr(ds, "seeded_noise", bool(d["seeded_noise"]))
    train._set_dataset_attr(ds, "blur_only_prob", float(d["blur_only_prob"]))
    train._set_dataset_attr(ds, "static_prob", float(d["static_prob"]))
    train._set_dataset_attr(ds, "static_noise_mult", tuple(float(v) for v in d["static_noise_mult"]))
    train._set_dataset_attr(ds, "static_ev_jitter", float(d["static_ev_jitter"]))
    train._set_dataset_attr(ds, "lowlight", d["lowlight"])

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, i):
            it = ds[i]; it["src"] = torch.tensor(bisect.bisect_right(offs, i)); return it
    gen = torch.Generator().manual_seed(45)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
    dl = torch.utils.data.DataLoader(Wrap(), batch_size=args.bs, sampler=sampler, num_workers=2, drop_last=True)
    mass = {n: float(s["sampling_mass"]) for n, s in zip(names, srcs)}
    sizes = {}
    prev = 0
    for n, c in zip(names, offs):
        sizes[n] = int(c - prev); prev = c
    extra = {}
    for sub in ds.datasets if hasattr(ds, "datasets") else []:
        pass
    return dl, names, mass, sizes, ds


def lossv(pred, batch, **over):
    cfg = {"model": {"vda_repo": VDA_REPO, "metric": True}, "loss": {**LOSS, **over}}
    return train._compute_loss_vda(pred, batch, cfg, pred_is_disparity=False)


VARIANTS = {"full": {}, "vda": dict(metric_silog_weight=0.0), "vda_skyoff": dict(metric_silog_weight=0.0, sky_vda_zero_disparity=False)}
DERIVED = {"full": {"full": 1}, "vda": {"vda": 1}, "silog": {"full": 1, "vda": -1}, "sky_term": {"vda": 1, "vda_skyoff": -1}}
CLS = ["near_lt10", "mid_10_80", "far_80_655", "far_ge655_valid", "sky"]


def class_masks(batch):
    depth = batch["depth"].to(dev); m = (batch["mask"].to(dev) * (depth > 0)) > 0.5
    sky = (batch["sky"].to(dev) > 0.5) & ~m
    c = {"near_lt10": m & (depth < 10), "mid_10_80": m & (depth >= 10) & (depth < 80), "far_80_655": m & (depth >= 80) & (depth < 655),
         "far_ge655_valid": m & (depth >= 655), "sky": sky}
    return c, m, sky, depth


rows = []


def emit(ck, b, clip, src, group, comp, metric, val):
    rows.append((ck, b, clip, src, group, comp, metric, float(val)))


def flat(vjps, names, grp):
    return torch.cat([g.detach().float().flatten() for n, g in zip(names, vjps) if g is not None and (grp == "all" or group_of(n) == grp)])


def process(ck, bi, net, pnames, plist, batch, src_names):
    t0 = time.time()
    rgb = batch.get("degraded_frames", batch.get("frames")).to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = net(rgb)
    cls, m, sky, depth = class_masks(batch)
    B = pred.shape[0]
    srcs = batch["src"].tolist()
    gv, lv = {}, {}
    for v, over in VARIANTS.items():
        pp = pred.detach().clone().requires_grad_(True)
        L = lossv(pp, batch, **over)
        gv[v] = torch.autograd.grad(L, pp)[0].float(); lv[v] = float(L.detach())
    gp = {c: sum(coef * gv[v] for v, coef in lin.items()) for c, lin in DERIVED.items()}

    def vjp(gmap):
        return torch.autograd.grad(pred, plist, grad_outputs=gmap.to(pred.dtype), retain_graph=True, allow_unused=True)

    GRPS = ("output_conv", "all")
    tot = {}
    for c in ("full", "vda", "silog", "sky_term"):
        V = vjp(gp[c]); tot[c] = {g: flat(V, pnames, g) for g in GRPS}
        for g in GRPS: emit(ck, bi, -1, "batch", f"total|{g}", c, "l2_norm", tot[c][g].norm())
        del V
    sparse = batch.get("sparse", torch.zeros(B)).float().tolist()
    for i in range(B):
        sname = src_names[srcs[i]]
        # ---- pixel statistics
        v_i = m[i]
        sup = float(batch["metric_sup"][i]) > 0.5
        ceil = float(batch["silog_ceil"][i]) if "silog_ceil" in batch else float("inf")
        emit(ck, bi, i, sname, "pix", "all", "total_px", v_i.numel())
        emit(ck, bi, i, sname, "pix", "all", "valid_px", int(v_i.sum()))
        emit(ck, bi, i, sname, "pix", "all", "silog_px", int((v_i & (depth[i] < ceil)).sum()) if sup else 0)
        emit(ck, bi, i, sname, "pix", "all", "sparse_flag", sparse[i])
        for k in CLS:
            emit(ck, bi, i, sname, "pix", k, "px", int(cls[k][i].sum()))
        # ---- whole-clip gradient
        for c in ("full", "vda", "silog", "sky_term"):
            gi = torch.zeros_like(gp[c]); gi[i] = gp[c][i]
            V = vjp(gi)
            for g in GRPS:
                f = flat(V, pnames, g)
                emit(ck, bi, i, sname, f"clip|{g}", c, "l2_norm", f.norm())
                emit(ck, bi, i, sname, f"clip|{g}", c, "dot_with_total", f @ tot[c][g])
            emit(ck, bi, i, sname, "clip_l1", c, "pred_l1", gp[c][i].abs().sum())
            del V
        # ---- per pixel class (vda, silog)
        for k in CLS:
            if int(cls[k][i].sum()) == 0: continue
            for c in ("vda", "silog"):
                gi = torch.zeros_like(gp[c]); gi[i] = gp[c][i] * cls[k][i].to(gp[c].dtype)
                emit(ck, bi, i, sname, f"cls_l1|{k}", c, "pred_l1", gi.abs().sum())
                if float(gi.abs().sum()) == 0: continue
                V = vjp(gi)
                for g in GRPS:
                    emit(ck, bi, i, sname, f"cls|{k}|{g}", c, "l2_norm", flat(V, pnames, g).norm())
                del V
    print(f"  [{ck}] batch {bi} {time.time()-t0:.1f}s src={[src_names[s] for s in srcs]} full={lv['full']:.4f}", flush=True)
    del tot, gv, gp, pred
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(45); np.random.seed(45)
    lock = pathlib.Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/GPU_LOCK")
    cache = OUT / "batches_seed45.pt"
    dl, src_names, mass, sizes, ds = build_loader()
    print("sources", src_names, "clips per source", sizes, flush=True)
    for s in ds.datasets:
        for att in ("missing", "excluded", "scenes"):
            if hasattr(s, att):
                v = getattr(s, att)
                print(type(s).__name__, att, len(v), flush=True)
                if att in ("missing", "excluded", "scenes"):
                    json.dump(sorted(v), open(OUT / f"{type(s).__name__}_{att}.json", "w"))
    nb = 1 if args.smoke else args.batches
    if cache.exists() and not args.smoke:
        batches = torch.load(cache, weights_only=False)
    else:
        it = iter(dl); batches = [next(it) for _ in range(nb)]
        if not args.smoke: torch.save(batches, cache)
    print("batches", len(batches), "sources seen", [[src_names[i] for i in b["src"].tolist()] for b in batches], flush=True)
    json.dump({"sources": src_names, "sampling_mass": mass, "clips_per_source": sizes, "batches": nb, "bs": args.bs, "frames": args.frames, "seed": 45,
               "src_per_batch": [[src_names[i] for i in b["src"].tolist()] for b in batches]}, open(OUT / "batch_meta.json", "w"), indent=1)
    rows_path = OUT / ("smoke_" if args.smoke else "") / "x" if False else OUT / ("smoke_diag.csv" if args.smoke else "grad_by_source_rows.csv")
    # GPU lock
    t0 = time.time()
    while True:
        cur = lock.read_text().strip() if lock.exists() else ""
        if cur == "" or cur == "E2_by_source": break
        if time.time() - t0 > 2400: raise SystemExit("GPU_LOCK busy 40 min: " + cur)
        time.sleep(60)
    lock.write_text("E2_by_source")
    try:
        for nm in args.models.split(","):
            net = load_model(nm)
            params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            pn = [n for n, _ in params]; pl = [p for _, p in params]
            print("loaded", nm, "trainable", len(pl), flush=True)
            for bi, batch in enumerate(batches[:nb]):
                net.train()
                process(nm, bi, net, pn, pl, batch, src_names)
                with open(rows_path, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["ckpt", "batch", "clip", "source", "group", "comp", "metric", "value"]); w.writerows(rows)
            del net; torch.cuda.empty_cache()
    finally:
        lock.write_text("")
    print("wrote", rows_path, len(rows))


if __name__ == "__main__":
    main()

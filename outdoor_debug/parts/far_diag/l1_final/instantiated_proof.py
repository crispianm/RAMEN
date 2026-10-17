"""L1: 'the feature is active' evidence from INSTANTIATED objects and MEASURED effects (not config keys). CPU, real local data and real local model weights, no checkpoint inference, no GPU.
Run from the repo root:  ~/.venvs/ramen/bin/python outdoor_debug/parts/far_diag/l1_final/instantiated_proof.py   (writes instantiated_proof_output.txt via the caller's tee)
Sections (each ends in PASS/FAIL lines; CONTROL lines are forced-fail controls that must show the opposite):
  0 arm yaml -> train.load_config round trip (the trainer reads exactly the committed file)
  1 SOURCES (S1,S2,S3): real train._build_multisource sampler: per-source sampler-weight sums + an empirical draw; S1: the instantiated dataset classes
  2 S4: measured SiLog value of the real train.compute_loss for a pure log bias, from the arm's own loss block
  3 S5: real model (local weights) + real train.build_optimizer: param groups and initial_lr per group
  4 S6: measured effect through the real compute_loss from the arm's own loss block, equals the float64 reference; B0 equals the original expression
  5 SEEDS (B0 vs B0b): sampler indices / clip ids and a noise checksum of the REAL DataLoader's first batches DIFFER; B0 re-run is identical (positive control)
Local limits: scannetpp_train cannot be built locally (no split list), so section 1/5 drop it for ALL arms (ratios between the remaining sources are unaffected); the model's temp2 init and weights are the local copies."""
import copy, hashlib, json, math, random, sys
from collections import Counter
from itertools import islice
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "outdoor_debug/parts/arm_checks")); sys.path.insert(0, str(Path(__file__).parent))
import torch, yaml
import train
from classes import remap
import arms_l1
HERE = Path(__file__).parent
FAILS = []
def rep(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if detail else ""));
    if not ok: FAILS.append(name)
def ctrl(name, ok, detail=""):   # a forced-fail control: ok means 'the check correctly saw the difference'
    print(("CONTROL-OK " if ok else "CONTROL-FAIL ") + name + (f"   [{detail}]" if detail else ""))
    if not ok: FAILS.append("CONTROL " + name)
def arm(a, k=2000): return yaml.safe_load(open(HERE / f"configs/k{k}/arm_{a}.yaml"))
def local(cfg):
    c = remap(cfg); c["data"]["sources"] = [s for s in c["data"]["sources"] if s["type"] != "scannetpp_train"]; c["data"]["val_sources"] = [s for s in c["data"].get("val_sources", []) if s["type"] != "scannetpp_train"]
    c["data"]["num_workers"] = 0; return c

print("== 0 load_config round trip")
for k in (2000, 4000):
    for a in arms_l1.ARMS:
        p = HERE / f"configs/k{k}/arm_{a}.yaml"; rep(f"{a}_k{k}: train.load_config(yaml) == yaml", train.load_config(p) == yaml.safe_load(open(p)))

print("\n== 1 sources: sampler weights (real train._build_multisource, world_size 4, rank 0)")
def source_shares(cfg):
    c = local(cfg); torch.manual_seed(0)
    loader, sampler = train._build_multisource(c, "train", 4, 0); ds = loader.dataset
    names = [type(d).__name__ for d in ds.datasets]; types = [s["type"] for s in c["data"]["sources"]]
    assert len(names) == len(types), (names, types)
    w = sampler.weights.double(); cum = ds.cumulative_sizes; sums = {}; lo = 0
    for t, hi, n in zip(types, cum, names):
        key = t; i = 2 if key in sums else 1
        while key in sums: key = f"{t}#{i}"; i += 1
        sums[key] = float(w[lo:hi].sum()); lo = hi
    idx = torch.tensor(list(islice(iter(sampler), 40000)))
    cnt = Counter();
    for i in idx.tolist():
        k = 0
        while i >= cum[k]: k += 1
        cnt[types[k]] += 1
    return types, names, sums, cnt
res = {a: source_shares(arm(a)) for a in ("B0", "S1", "S2", "S3")}
for a in res: print(a, "instantiated sources:", list(zip(res[a][0], res[a][1])))
b0 = res["B0"][2]; tot = lambda s: sum(s.values())
print("B0 sampler-weight sums:", {k: round(v, 1) for k, v in b0.items()})
rep("S1 instantiated sources have NO nyu_train / scannetpp_train, all 7 others present, B0's order", res["S1"][0] == [t for t in res["B0"][0] if t not in arms_l1.DROPPED] and "nyu_train" not in res["S1"][0],
    str(res["S1"][0]))
ctrl("B0 DOES instantiate nyu_train (the S1 check is not vacuous)", "nyu_train" in res["B0"][0])
rep("S1 empirical sampler draw contains no nyu_train", res["S1"][3].get("nyu_train", 0) == 0 and sum(res["S1"][3].values()) >= 1000, f"{sum(res['S1'][3].values())} draws")
ctrl("B0 empirical draw does contain nyu_train", res["B0"][3].get("nyu_train", 0) > 1000, str(res["B0"][3].get("nyu_train")))
tt = lambda s, a_: s[a_]
r = lambda s, a_, b_: tt(s, a_) / tt(s, b_)
want_k = 299397 / 310000; rep("B0 weight-sum ratio kitti_raw/tartanair == 299397/310000 (sampler built from sampling_mass)", abs(r(b0, "kitti_raw", "tartanair") - want_k) < 1e-6, f"{r(b0, 'kitti_raw', 'tartanair'):.6f} vs {want_k:.6f}")
s2 = res["S2"][2]; rep("S2 kitti_raw/tartanair weight-sum ratio == 29940/310000 (kitti /10)", abs(r(s2, "kitti_raw", "tartanair") - 29940 / 310000) < 1e-6, f"{r(s2, 'kitti_raw', 'tartanair'):.6f}")
rep("S2: all other source weight sums unchanged vs B0", all(abs(s2[k] - b0[k]) < 1e-6 * max(1, b0[k]) for k in b0 if k != "kitti_raw"))
s3 = res["S3"][2]; rep("S3 vkitti/tartanair weight-sum ratio == 400000/310000 (vkitti x10)", abs(r(s3, "vkitti", "tartanair") - 400000 / 310000) < 1e-6, f"{r(s3, 'vkitti', 'tartanair'):.6f}")
rep("S3: all other source weight sums unchanged vs B0", all(abs(s3[k] - b0[k]) < 1e-6 * max(1, b0[k]) for k in b0 if k != "vkitti"))
ctrl("B0 vkitti/tartanair is 40000/310000, NOT S3's ratio", abs(r(b0, "vkitti", "tartanair") - 40000 / 310000) < 1e-6 and abs(r(b0, "vkitti", "tartanair") - r(s3, "vkitti", "tartanair")) > 0.5)
print("empirical shares of the sampler draws (local sources, scannetpp dropped):")
for a in res: n = sum(res[a][3].values()); print(f"  {a}: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in sorted(res[a][3].items(), key=lambda kv: -kv[1])))
sh = lambda a, t: res[a][3][t] / sum(res[a][3].values())
T0 = sum(b0.values()); exp_s2 = (29940 / (T0 - 299397 + 29940)) / (299397 / T0); exp_s3 = (400000 / (T0 - 40000 + 400000)) / (40000 / T0)
rep("S2 empirical kitti_raw share ratio S2/B0 matches the mass arithmetic (within 12% relative)", abs(sh("S2", "kitti_raw") / sh("B0", "kitti_raw") / exp_s2 - 1) < 0.12, f"measured {sh('S2', 'kitti_raw') / sh('B0', 'kitti_raw'):.3f} expected {exp_s2:.3f}")
rep("S3 empirical vkitti share ratio S3/B0 matches the mass arithmetic (within 12% relative)", abs(sh("S3", "vkitti") / sh("B0", "vkitti") / exp_s3 - 1) < 0.12, f"measured {sh('S3', 'vkitti') / sh('B0', 'vkitti'):.2f} expected {exp_s3:.2f}")
ctrl("a WRONG expectation (kitti share /2 instead of /10) is rejected by the same tolerance", abs(sh("S2", "kitti_raw") / sh("B0", "kitti_raw") / 0.5 - 1) >= 0.12)

print("\n== 2 S4 lambda: measured SiLog of the real compute_loss, pure log bias e (diff_log == e everywhere): silog = sqrt(e^2 (1 - lambda))")
class Z(torch.nn.Module):
    def __init__(self, **_): super().__init__()
    def forward(self, prediction, target, mask): return {"total_loss": prediction.sum() * 0.0}
train._get_vda_video_depth_loss = lambda repo, lc: Z()
def silog_of(cfg, e=0.3):
    depth = torch.rand(1, 1, 2, 8, 8, generator=torch.Generator().manual_seed(3)) * 30 + 1; batch = {"depth": depth, "mask": torch.ones_like(depth), "sky": torch.zeros_like(depth)}
    return float(train.compute_loss(depth * math.exp(-e), batch, cfg))
for a, lam in (("B0", 0.5), ("S4", 0.15)):
    c = arm(a); got = silog_of(c); want = math.sqrt(0.09 * (1 - lam))
    rep(f"{a}: measured silog {got:.5f} == sqrt(e^2 (1-{lam})) {want:.5f} from the arm's own loss block", abs(got - want) < 1e-4)
ctrl("S4 measured silog differs from B0's by > 0.05", abs(silog_of(arm('S4')) - silog_of(arm('B0'))) > 0.05)

print("\n== 3 S5 head lr: real model (local weights) + real train.build_optimizer")
def model_cfg(cfg):
    c = copy.deepcopy(cfg); m = c["model"]
    m["checkpoint_path"] = "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth-anything-v3-small.safetensors"; m["da_v2_checkpoint_path"] = "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/depth_anything_v2_vits.pth"
    m["vda_checkpoint_path"] = "/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth"; m["vda_repo"] = "/home/wg19671/repos/Video-Depth-Anything"
    m["st_temp2_init"] = "/mnt/DATA/DegradedDepth/RAMEN_aaai/results/dn01_st_recal/recal_full/temp2_distilled.pt"; m["device"] = "cpu"; return c
groups = {}
for a in ("B0", "S5", "S4"):
    c = model_cfg(arm(a)); core = train.build_model(c); opt = train.build_optimizer(core, c, verbose=False)
    hp = {id(p) for p in core.depth_head_parameters()}
    groups[a] = [dict(n=len(g["params"]), initial_lr=g["initial_lr"], lr=g["lr"], n_head=sum(id(p) in hp for p in g["params"])) for g in opt.param_groups]
    print(a, groups[a], "len(depth_head_parameters) =", len(hp))
hg = lambda a: [g for g in groups[a] if g["n_head"] > 0]
rep("B0: exactly one group holds the depth head; its initial_lr == 1e-4; group sizes (188, 64, 2) as in the c3_n46 checkpoint", len(hg("B0")) == 1 and hg("B0")[0]["initial_lr"] == 1e-4 and [g["n"] for g in groups["B0"]] == [188, 64, 2], str(groups["B0"]))
rep("S5: that group's initial_lr == 1e-5 (0.1x), the head group holds ALL depth-head params", len(hg("S5")) == 1 and abs(hg("S5")[0]["initial_lr"] - 1e-5) < 1e-12 and hg("S5")[0]["n_head"] == hg("S5")[0]["n"])
rep("S5: every NON-head group has the same initial_lr as B0 (1e-4, 1e-3 gates)", [g["initial_lr"] for g in groups["S5"] if g["n_head"] == 0] == [g["initial_lr"] for g in groups["B0"] if g["n_head"] == 0] and sorted(set(g["initial_lr"] for g in groups["S5"])) == [1e-5, 1e-4, 1e-3])
rep("S4: optimizer groups identical to B0 (S4 must not touch the optimizer)", groups["S4"] == groups["B0"])
ctrl("S5 head group differs from B0's head group", hg("S5")[0]["initial_lr"] != hg("B0")[0]["initial_lr"])
# the run loop sets g['lr'] = g['initial_lr'] * lr_scale(step) for EVERY group (train.py l.~2893): apply it to the real S5 optimizer for a few steps
sc = train.make_lr_scale(arm("S5")["training"], 500, 2000)
c = model_cfg(arm("S5")); core = train.build_model(c); opt5 = train.build_optimizer(core, c, verbose=False)
ratios = []
for st in (0, 499, 1000, 1999):
    for g in opt5.param_groups: g["lr"] = g["initial_lr"] * sc(st)
    lrs = [g["lr"] for g in opt5.param_groups]; ratios.append(lrs[1] / lrs[0])
    print("  S5 step", st, "scale", format(sc(st), ".4e"), "group lrs", [format(x, ".3e") for x in lrs])
rep("S5: head-group lr / group-0 lr == 0.1 at every probed step (the schedule multiplies all groups)", all(abs(x - 0.1) < 1e-9 for x in ratios))

print("\n== 4 S6 far-weighted SiLog: measured effect through the real compute_loss from the arm's own loss block")
def batch_far(seed=5):
    g = torch.Generator().manual_seed(seed)
    depth = torch.cat([torch.rand(1, 1, 2, 8, 6, generator=g) * 40 + 1, torch.rand(1, 1, 2, 8, 6, generator=g) * 300 + 60], dim=-1); mask = torch.ones_like(depth); mask[..., 0, 0] = 0.0
    return depth * torch.exp(torch.randn(depth.shape, generator=g) * 0.3 + 0.05), {"depth": depth, "mask": mask, "sky": torch.zeros_like(depth)}
def lg(cfg):
    p, b = batch_far(); p = p.clone().requires_grad_(True); l = train.compute_loss(p, b, cfg); l.backward(); return l.detach(), p.grad.clone(), p.detach(), b
l_b0, g_b0, p, b = lg(arm("B0")); l_s6, g_s6, _, _ = lg(arm("S6"))
m = b["mask"] > 0.5; dl = torch.log(b["depth"][m].float()) - torch.log(p[m].float().clamp_min(1e-3))
orig = torch.sqrt((dl.pow(2).mean() - 0.5 * dl.mean().pow(2)).clamp_min(1e-12))
w = 1 + 3 * (b["depth"][m] >= 50).double(); ref = math.sqrt(max(float((w * dl.double() ** 2).sum() / w.sum() - 0.5 * ((w * dl.double()).sum() / w.sum()) ** 2), 1e-12))
rep("B0 (arm yaml, no S6 keys): loss == the ORIGINAL SiLog expression, bit-for-bit", torch.equal(l_b0, orig), f"{float(l_b0):.6f}")
rep("S6 (arm yaml): loss == float64 reference with w=4 for gt>=50 m", abs(float(l_s6) - ref) < 1e-5, f"{float(l_s6):.6f} vs {ref:.6f}")
far = (b["depth"] >= 50) & m; shr = lambda g: float(g[far].abs().sum() / g.abs().sum())
rep("S6: far share of |grad| larger than B0's", shr(g_s6) > shr(g_b0), f"{shr(g_s6):.3f} vs {shr(g_b0):.3f}")
ctrl("S6 loss differs from B0's loss on the same batch", abs(float(l_s6) - float(l_b0)) > 1e-4, f"{float(l_s6):.5f} vs {float(l_b0):.5f}")

print("\n== 5 seeds (B0 vs B0b): REAL loader, world_size 4, first batches")
def first_batches(cfg, rank=0, nb=2, nidx=200):
    c = local(cfg); d = c["data"]; t = c["training"]; c["data"]["num_workers"] = 2
    train.seed_everything(int(d["seed"]), rank, str(t.get("seed_scheme", "add")))     # train.py main(): before the loaders are built (no resume in L1)
    loader, sampler = train._build_multisource(c, "train", 4, rank)
    for k in ("clean_prob", "seeded_noise"): train._set_dataset_attr(loader.dataset, k, d.get(k, False if k == "seeded_noise" else 0.0))
    for k in ("lowlight", "static_prob", "blur_only_prob", "blur_disk_prob", "static_exempt_shares", "static_lowlight_only"):
        if k in d: train._set_dataset_attr(loader.dataset, k, d[k])
    ds = loader.dataset
    def cid(i):
        k = 0
        while i >= ds.cumulative_sizes[k]: k += 1
        j = i - (ds.cumulative_sizes[k - 1] if k else 0); cl = ds.datasets[k].clips[j]; return (k, str(cl[0][0]), cl[3])
    idx = list(islice(iter(sampler), nidx)); ids = [cid(i) for i in idx]
    it = iter(loader); sums = []; bat_ids = []
    for _ in range(nb):
        b = next(it); sums.append(hashlib.sha256(b["frames"].contiguous().numpy().tobytes()).hexdigest()[:16])
    return idx, ids, sums
r_b0 = first_batches(arm("B0")); r_b0_again = first_batches(arm("B0")); r_b0b = first_batches(arm("B0b"))
same = lambda x, y: sum(a == b for a, b in zip(x, y))
print(f"first 200 sampler idx: B0 vs B0b identical {same(r_b0[0], r_b0b[0])}/200; clip ids identical {same(r_b0[1], r_b0b[1])}/200; B0 vs B0(re-run) identical idx {same(r_b0[0], r_b0_again[0])}/200")
print("first-2-batch frame checksums (sha256[:16] of the real loader's frame tensors, noise included): B0", r_b0[2], "| B0 re-run", r_b0_again[2], "| B0b", r_b0b[2])
rep("B0 vs B0b: first 200 sampler idx differ (<= 5/200 coincide) and clip ids differ", same(r_b0[0], r_b0b[0]) <= 5 and same(r_b0[1], r_b0b[1]) <= 5)
rep("B0 vs B0b: first-batch frame checksums DIFFER (clips and/or noise)", all(a != b for a, b in zip(r_b0[2], r_b0b[2])))
ctrl("POSITIVE control: B0 re-run is identical (idx 200/200, checksums equal) -> the comparison can see equality", r_b0[0] == r_b0_again[0] and r_b0[2] == r_b0_again[2])
# noise-only difference: SAME clip, B0 vs B0b global-RNG streams (what a fixed clip would see)
def noise_of_fixed_clip(cfg, i=0):
    c = local(cfg); d = c["data"]; c["data"]["num_workers"] = 0
    train.seed_everything(int(d["seed"]), 0, str(c["training"].get("seed_scheme", "add")))
    loader, sampler = train._build_multisource(c, "train", 4, 0)
    for k in ("clean_prob", "seeded_noise"): train._set_dataset_attr(loader.dataset, k, d.get(k, False if k == "seeded_noise" else 0.0))
    for k in ("lowlight", "static_prob", "blur_only_prob", "blur_disk_prob", "static_exempt_shares", "static_lowlight_only"):
        if k in d: train._set_dataset_attr(loader.dataset, k, d[k])
    return hashlib.sha256(loader.dataset[i]["frames"].contiguous().numpy().tobytes()).hexdigest()[:16]
na, na2, nb_ = noise_of_fixed_clip(arm("B0")), noise_of_fixed_clip(arm("B0")), noise_of_fixed_clip(arm("B0b"))
print("same clip (idx 0) frames checksum: B0", na, "| B0 re-run", na2, "| B0b", nb_)
rep("same clip, B0 vs B0b: noise checksum DIFFERS", na != nb_)
ctrl("POSITIVE control: same clip, B0 vs B0 re-run identical", na == na2)
print("\nINSTANTIATED PROOF:", "ALL PASS" if not FAILS else f"FAILURES: {FAILS}"); sys.exit(1 if FAILS else 0)

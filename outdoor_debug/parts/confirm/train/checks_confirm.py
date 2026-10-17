"""Real-path CPU checks for the confirmatory arms (Add.47). Run from the repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/confirm/train/checks_confirm.py
1 classes per source via real train._build_multisource / train._make_source (recorder), 2 TS2 vs no-TS2 masks on a real Downtown clip through train._make_source,
3 hinge through the real train._compute_loss_vda with cfg from train.load_config(arm yaml), 4 LR schedule from the real make_lr_scale (and vs the REAL lr= prints of job 7226119).
Every check has a forced-fail (negative control) that must FAIL. Exit 1 on any unexpected result."""
import sys, copy, json, math, re, random
from pathlib import Path
import numpy as np, torch, yaml
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import train
from arms_confirm import build
LOCAL = "/mnt/DATA/DegradedDepth/data/TRAIN"; SEGL = "/mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261016/TA_seg"
VDA_REPO = "/home/wg19671/repos/Video-Depth-Anything"
RES = {}; BAD = []
def P(*a): print(*a, flush=True)
def expect(name, cond, detail=""):
    P(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond: BAD.append(name)
def neg(name, cond, detail=""):   # a negative control: 'cond' is True when the check CORRECTLY failed on the bad input
    P(f"  [{'ok' if cond else 'FAIL'}] NEGATIVE CONTROL {name} {detail}")
    if not cond: BAD.append("negctl:" + name)

def remap(cfg):
    c = copy.deepcopy(cfg)
    for s in c["data"]["sources"] + c["data"].get("val_sources", []):
        for k in ("root", "gt_root", "extra_root"):
            if k in s: s[k] = s[k].replace("/projects/b5dh/data/train", LOCAL)
        if "ts2_seg_root" in s: s["ts2_seg_root"] = SEGL
    return c

base, arms = build()
EXPECT_CLS = {"tartanair": "VDASemanticsDegradedTartanAirDataset", "vkitti": "VDASemanticsDegradedVKITTIDataset", "pointodyssey": "VDASemanticsDegradedPointOdysseyDataset",
              "irs": "VDASemanticsDegradedIRSDataset", "spring": "DegradedSpringDataset", "kitti_raw": "DegradedKITTIRawTrainDataset", "nyu_train": "DegradedNYUTrainDataset"}

# ---------------- 1 classes ----------------
P("== 1 classes per source (real train._build_multisource -> _make_source, recorder)")
def classes_for(cfg):
    real = train._make_source; rec = []
    def wrapped(spec, split, common):
        ds = real(spec, split, common)
        rec.append(dict(split=split, type=spec["type"], cls=type(ds).__name__, ts2=type(getattr(ds, "ts2", None)).__name__, vk=getattr(ds, "vkitti_max_depth", None))); return ds
    train._make_source = wrapped
    try: train._build_multisource(remap(cfg), "train", 1, 0); err = None
    except Exception as e: err = f"{type(e).__name__}: {str(e)[:120]}"
    finally: train._make_source = real
    return rec, err
def class_verdict(rec, want_ts2):
    bad = []
    for r in rec:
        if r["split"] != "train": continue
        if r["cls"] != EXPECT_CLS.get(r["type"]): bad.append((r["type"], r["cls"]))
        if r["type"] == "tartanair" and (r["ts2"] == "TS2Labeller") != want_ts2: bad.append(("tartanair.ts2", r["ts2"]))
        if r["type"] != "tartanair" and r["ts2"] != "NoneType": bad.append((r["type"], "ts2 attr on non-TA", r["ts2"]))
        if r["type"] == "vkitti" and r["vk"] != 655: bad.append(("vkitti_max_depth", r["vk"]))
    return bad
RES["classes"] = {}
for n, c in arms.items():
    rec, err = classes_for(c); RES["classes"][n] = dict(rec=rec, build_error=err)
    P(f" {n}: {[(r['type'], r['cls'], r['ts2']) for r in rec if r['split']=='train']} build_error={err}")
    expect(f"{n} classes+ts2+bound", not class_verdict(rec, True), str(class_verdict(rec, True)))
    expect(f"{n} all 8 buildable train sources recorded (scannetpp local split list missing is the known gap)", len([r for r in rec if r["split"] == "train"]) >= 8, f"n={len([r for r in rec if r['split']=='train'])}")
bad_cfg = copy.deepcopy(arms["C_TS2"]); [(s.pop("ts2_sky"), s.pop("ts2_seg_root")) for s in bad_cfg["data"]["sources"] if s["type"] == "tartanair"]
rec, _ = classes_for(bad_cfg); neg("a config without ts2 keys must be flagged by the class check", bool(class_verdict(rec, True)), str(class_verdict(rec, True)))

# ---------------- 2 TS2 mask vs no-TS2 mask on a real Downtown clip ----------------
P("== 2 TS2 vs released-rule (no TS2) masks, real Downtown clip, built by train._make_source from the arm config")
def common_for(cfg):
    d = cfg["data"]
    c = dict(num_frames=int(d["num_frames"]), image_size=tuple(d["image_size"]), clip_stride=int(d["clip_stride"]), frame_stride=int(d["frame_stride"]), max_depth=float(d["max_depth"]), seed=int(d["seed"]),
             degrade=True, degrade_mode=d["degrade_mode"], degrade_severity=d.get("degrade_severity"), max_clips_per_video=d.get("max_clips_per_video"), resize_mode=d.get("resize_mode"))
    train._with_split_seed(c, d); return c
def mk_ta(cfg):
    cfg = remap(cfg); spec = copy.deepcopy([s for s in cfg["data"]["sources"] if s["type"] == "tartanair"][0]); spec["envs"] = ["Downtown"]
    ds = train._make_source(spec, "train", common_for(cfg))
    for k in ("clean_prob", "blur_only_prob", "lowlight", "seeded_noise", "static_prob", "static_noise_mult", "static_ev_jitter"):
        if k in cfg["data"] and hasattr(ds, k): setattr(ds, k, cfg["data"][k])
    return ds
def get(ds, idx, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); return ds[idx]
noTS2 = copy.deepcopy(arms["C_TS2"]); [(s.pop("ts2_sky"), s.pop("ts2_seg_root")) for s in noTS2["data"]["sources"] if s["type"] == "tartanair"]
ds_ts, ds_no, ds_ts_h = mk_ta(arms["C_TS2"]), mk_ta(noTS2), mk_ta(arms["C_TS2_H300"])
expect("TS2 arm has Labeller, no-TS2 arm does not", type(ds_ts.ts2).__name__ == "TS2Labeller" and ds_no.ts2 is None)
from data.video_clip_dataset import _decode_depth
best = None
for i in range(0, len(ds_no), max(1, len(ds_no) // 40)):
    img_f, dep_f, vid, st = ds_no.clips[i]; d = _decode_depth(str(dep_f[st])); sk = float((~np.isfinite(d) | (d > 1e4)).mean())
    if best is None or sk > best[0]: best = (sk, i)
idx = best[1]; P(f" clip idx {idx} of {len(ds_no)} ({ds_no.clips[idx][2]} start {ds_no.clips[idx][3]})")
def differs(a, b): return bool((a["mask"] != b["mask"]).any() or (a["sky"] != b["sky"]).any())
def pct(t): return 100.0 * float((t > 0.5).float().mean())
differ_any = False
for seed in (0, 1):
    sn, st_, sh = get(ds_no, idx, seed), get(ds_ts, idx, seed), get(ds_ts_h, idx, seed)
    d = differs(sn, st_); differ_any |= d
    P(f" seed{seed}: noTS2 sky%={pct(sn['sky']):.3f} mask%={pct(sn['mask']):.3f} | TS2 sky%={pct(st_['sky']):.3f} mask%={pct(st_['mask']):.3f} | masks differ={d}")
    expect(f"seed{seed} frames+depth identical TS2 vs noTS2", torch.equal(sn["frames"], st_["frames"]) and torch.equal(sn["depth"], st_["depth"]))
    expect(f"seed{seed} C_TS2 and C_TS2_H300 data identical (hinge is loss-only)", torch.equal(st_["mask"], sh["mask"]) and torch.equal(st_["sky"], sh["sky"]) and torch.equal(st_["frames"], sh["frames"]))
expect("TS2 masks differ from no-TS2 masks on the real clip (at least one seed; static seed may be tiny)", differ_any)
sn0a, sn0b = get(ds_no, idx, 1), get(ds_no, idx, 1)
neg("differs() on two identical no-TS2 samples must say False", not differs(sn0a, sn0b))
RES["ts2_clip"] = dict(idx=idx, video=str(ds_no.clips[idx][2]))

# ---------------- 3 hinge ----------------
P("== 3 hinge through real train._compute_loss_vda (cfg = train.load_config(arm yaml))")
sb = get(ds_ts, idx, 1); batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sb.items()}
sky_px = int(((batch["sky"] > 0.5) & ~(batch["mask"] > 0.5)).sum()); P(" sky px outside mask in batch:", sky_px)
expect("batch has sky px (else hinge cannot be exercised)", sky_px > 0)
def run(cfgpath, val):
    cfg = train.load_config(cfgpath); train.check_sky_loss(cfg); cfg["model"]["vda_repo"] = VDA_REPO
    train.term_flush(); p = torch.full(batch["depth"].shape, val).requires_grad_(True)
    L = train._compute_loss_vda(p, batch, cfg, False); return float(L), train.term_flush()
H = {}
for n in arms:
    path = HERE / "configs" / f"arm_{n}.yaml"; H[n] = {}
    for v in (10.0, 100.0, 1000.0): L, terms = run(path, v); H[n][v] = (L, terms.get("sky_hinge_weighted"))
    P(f" {n}: " + ", ".join(f"pred={v:g}m total={H[n][v][0]:.5f} hinge={H[n][v][1]}" for v in H[n]))
w, fl = 0.1, 300.0
expect("C_TS2 (control): hinge key absent at every pred", all(H["C_TS2"][v][1] is None for v in H["C_TS2"]))
expect("H300 pred=10: hinge == 0.1*ln(300/10)", abs(H["C_TS2_H300"][10.0][1] - w * math.log(fl / 10)) < 2e-4, f"{H['C_TS2_H300'][10.0][1]} vs {w*math.log(fl/10):.5f}")
expect("H300 pred=100: hinge == 0.1*ln(3)", abs(H["C_TS2_H300"][100.0][1] - w * math.log(3)) < 2e-4, f"{H['C_TS2_H300'][100.0][1]} vs {w*math.log(3):.5f}")
expect("H300 pred=1000: hinge == 0 (one-sided)", H["C_TS2_H300"][1000.0][1] == 0.0)
expect("total loss differs between arms only by the hinge at 10m", abs((H["C_TS2_H300"][10.0][0] - H["C_TS2"][10.0][0]) - H["C_TS2_H300"][10.0][1]) < 1e-4, f"delta={H['C_TS2_H300'][10.0][0]-H['C_TS2'][10.0][0]:.5f}")
expect("total loss equal at 1000m (hinge inactive)", abs(H["C_TS2_H300"][1000.0][0] - H["C_TS2"][1000.0][0]) < 1e-6)
neg("wrong-floor expectation (80 m) must NOT match the H300 value", abs(H["C_TS2_H300"][10.0][1] - w * math.log(80 / 10)) >= 2e-4)
neg("control must NOT match the H300 value", H["C_TS2"][10.0][1] is None or abs(H["C_TS2"][10.0][1] - w * math.log(fl / 10)) >= 2e-4)
RES["hinge"] = {n: {str(v): H[n][v] for v in H[n]} for n in H}

# ---------------- 4 LR schedule ----------------
P("== 4 LR schedule from the real make_lr_scale (arm cfg), lr_group0 = 1e-4*scale; printed lr at step G = 1e-4*scale(G-1)")
LRS = {}
for n in arms:
    cfg = train.load_config(HERE / "configs" / f"arm_{n}.yaml"); t = cfg["training"]
    f = train.make_lr_scale(t, int(t["warmup_steps"]), int(t["max_steps"]))
    LRS[n] = {s: f(s) for s in (0, 1, 19, 20, 499, 500, 501, 6000, 11999)}
    P(f" {n}: " + ", ".join(f"scale({s})={v:.6g}" for s, v in LRS[n].items()))
lr0 = LRS["C_TS2"]
expect("scale(0)=1/500 (warmup k+1)/W", abs(lr0[0] - 1 / 500) < 1e-12); expect("scale(499)=1.0", abs(lr0[499] - 1.0) < 1e-12); expect("scale(500)=1.0 (cosine start)", abs(lr0[500] - 1.0) < 1e-12)
expect("scale(6000)=0.5*(1+cos(pi*5500/11500))", abs(lr0[6000] - 0.5 * (1 + math.cos(math.pi * 5500 / 11500))) < 1e-12)
expect("scale(11999)~1.9e-12/1e-4 (cosine to ~0, NOT the restart schedule)", 0 < lr0[11999] < 1e-7, f"{lr0[11999]:.3e}")
expect("both arms same schedule", LRS["C_TS2"] == LRS["C_TS2_H300"])
# vs the REAL log of job 7226119 (same recipe, same schedule): every printed lr within 0.6% (print rounds to 3 digits)
logp = Path("/home/wg19671/.claude/jobs/a0d391cb/tmp/scr_logs/dn01_c3rep_20261009_c3_n46_7226119.out")
if logp.exists():
    t = train.load_config(HERE / "configs" / "arm_C_TS2.yaml")["training"]; f = train.make_lr_scale(t, 500, 12000)
    pr = [(int(m.group(1)), float(m.group(2))) for l in logp.read_text(errors="replace").splitlines() if (m := re.search(r"step(\d+)/12000 loss=\S+ lr=(\S+)", l))]
    worst = max(abs(lr / (1e-4 * f(s - 1)) - 1) for s, lr in pr if f(s - 1) > 1e-5)
    expect(f"all {len(pr)} lr= prints of job 7226119 reproduce 1e-4*scale(G-1) (lr>1e-9 abs part)", worst < 0.006, f"worst rel err {worst:.4f}")
    bad = f2 = train.make_lr_scale(dict(t, lr_decay_steps=14000), 500, 12000)
    worst2 = max(abs(lr / (1e-4 * f2(s - 1)) - 1) for s, lr in pr if f2(s - 1) > 1e-5)
    neg("a wrong schedule (lr_decay 14000) must NOT reproduce the real prints", worst2 > 0.006, f"worst rel err {worst2:.3f}")
    RES["lr_prints_checked"] = len(pr)
neg("a restart-key schedule must differ from the plain one at step 6000", train.make_lr_scale(dict(t, lr_decay_steps=14000, lr_restart_step=6000, lr_restart_warmup=100, lr_restart_peak=1.0), 500, 12000)(6000) != lr0[6000])
json.dump(dict(RES, lr={n: {str(k): v for k, v in d.items()} for n, d in LRS.items()}), open(HERE / "checks_confirm_results.json", "w"), indent=1, default=str)
P("\nRESULT:", "ALL CHECKS PASS (incl. negative controls)" if not BAD else f"FAILURES: {BAD}")
sys.exit(1 if BAD else 0)

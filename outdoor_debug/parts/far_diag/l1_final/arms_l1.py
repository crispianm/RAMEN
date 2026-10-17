"""L1 final (Track 2 light screen; user Add.49/50, Auditor O41): arm generator + resolved-config diff proofs + declared-VALUE asserts + forced-fail mutations.
8 arms {B0, B0b, S1..S6} x 2 stages (K = 2000 steps; K = 4000 = the automatic baseline retry / the arms' stage if the baselines needed the retry). Every run starts FROM THE RELEASE (resume null, vda_init released).
BASE = configs/experiments/dn01_c3rep_20261009_c3_n46.yaml resolved (== confirm/stage1/base/effective_ramen_n46.yaml, sha asserted), seed 46, warmup 500, severity warmup 500 (UNCHANGED recipe).
Run from the repo root:  ~/.venvs/ramen/bin/python outdoor_debug/parts/far_diag/l1_final/arms_l1.py [--forced-fail]"""
import copy, hashlib, json, sys
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "outdoor_debug/parts/arm_checks"))
from arms import flat, diff, apply            # the screen's own helpers (same flattening as config_diffs.md)
BASE_PATH = REPO / "outdoor_debug/parts/confirm/stage1/base/effective_ramen_n46.yaml"
BASE_SHA = "d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1"
STAGES = (2000, 4000)
ARMS = ["B0", "B0b", "S1", "S2", "S3", "S4", "S5", "S6"]
# declared overrides per arm (single change vs B0). 'src:<type>.<key>' targets data.sources[type==<type>].
OV = {
    "B0": [],
    "B0b": [("data.seed", 49)],                                                    # seed spread (49 = unused by c3rep 46-48 and the screen's A0b); data.seed drives sampler + (no resume) the global RNG via seed_everything
    "S1": "DROP_INDOOR",                                                           # remove nyu_train and scannetpp_train from data.sources (special-cased below)
    "S2": [("src:kitti_raw.sampling_mass", 29940)],                                # from 299397
    "S3": [("src:vkitti.sampling_mass", 400000)],                                  # from 40000
    "S4": [("loss.metric_silog_lambda", 0.15)],                                    # from 0.5
    "S5": [("training.head_lr_mult", 0.1)],                                        # from 1.0 (base already has the head in its own group, freeze_depth_head false)
    "S6": [("loss.metric_silog_far_weight", 4.0), ("loss.metric_silog_far_from", 50.0)],   # NEW opt-in keys (S6_diff.patch)
}
DROPPED = ("nyu_train", "scannetpp_train")
def common(k, arm):   # identical in every arm of a stage (the recipe is C3 otherwise): 2000/4000-step cosine from the release, warmup 500 (recipe), one permanent snapshot at the end, val effectively only at the snapshot
    return [("training.max_steps", k), ("training.lr_decay_steps", k), ("training.checkpoint_at_steps", [k]), ("training.val_every_steps", 100000),
            ("training.checkpoint_dir", f"runs/l1_{arm}_k{k}")]
def name(arm, k): return f"{arm}_k{k}"

def build():
    raw = BASE_PATH.read_bytes(); assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "base sha"
    base = yaml.safe_load(raw)
    runs = {}
    for k in STAGES:
        for a in ARMS:
            c = copy.deepcopy(base)
            if OV[a] == "DROP_INDOOR":
                assert [s["type"] for s in c["data"]["sources"] if s["type"] in DROPPED] == list(DROPPED), "indoor sources not found in the base"
                c["data"]["sources"] = [s for s in c["data"]["sources"] if s["type"] not in DROPPED]
            else:
                c = apply(c, OV[a])
            runs[name(a, k)] = apply(c, common(k, a))
    return base, runs

def keys(a, b): return {kk for kk, _, _ in diff(a, b)}
def src_keys(cfg, types):   # all flat keys of the given source types in cfg
    return {kk for kk, _ in flat(cfg) if any(kk.startswith(f"data.sources[{t}]") for t in types)}
CK = "training.checkpoint_dir"
def expect(runs):
    """(armX, armY) -> exact set of differing flat keys. B0 is the reference; S1's set is the full flat key set of the two removed sources."""
    E = {}
    for k in STAGES:
        b0 = runs[name("B0", k)]
        E[(name("B0", k), name("B0b", k))] = {"data.seed", CK}
        E[(name("B0", k), name("S1", k))] = src_keys(b0, DROPPED) | {CK}
        E[(name("B0", k), name("S2", k))] = {"data.sources[kitti_raw].sampling_mass", CK}
        E[(name("B0", k), name("S3", k))] = {"data.sources[vkitti].sampling_mass", CK}
        E[(name("B0", k), name("S4", k))] = {"loss.metric_silog_lambda", CK}
        E[(name("B0", k), name("S5", k))] = {"training.head_lr_mult", CK}
        E[(name("B0", k), name("S6", k))] = {"loss.metric_silog_far_weight", "loss.metric_silog_far_from", CK}
    for a in ARMS:   # stage change: exactly the schedule keys + the checkpoint dir
        E[(name(a, 2000), name(a, 4000))] = {"training.max_steps", "training.lr_decay_steps", "training.checkpoint_at_steps", CK}
    return E

def values():
    V = {}
    for k in STAGES:
        for a in ARMS:
            v = {"training.max_steps": k, "training.lr_decay_steps": k, "training.checkpoint_at_steps": [k], "training.val_every_steps": 100000, "training.warmup_steps": 500,
                 "training.severity_warmup_steps": 500, "training.learning_rate": 0.0001, "training.resume_from": None, "training.seed_scheme": "x1000", "training.checkpoint_dir": f"runs/l1_{a}_k{k}",
                 "training.grad_accum": 4, "training.batch_size": 1, "training.gate_lr_scale": 10, "training.precision": "bf16",
                 "model.vda_init": "released", "model.freeze_depth_head": False, "model.init_from": None, "model.metric": True,
                 "data.seed": 49 if a == "B0b" else 46, "data.split_seed": 43, "data.sampling": "mass", "data.degrade": True, "data.seeded_noise": True,
                 "loss.metric_silog_weight": 1.0, "loss.metric_silog_lambda": 0.15 if a == "S4" else 0.5,
                 "training.head_lr_mult": 0.1 if a == "S5" else 1.0,
                 "data.sources[kitti_raw].sampling_mass": 29940 if a == "S2" else 299397,
                 "data.sources[vkitti].sampling_mass": 400000 if a == "S3" else 40000,
                 "data.sources[tartanair].sampling_mass": 310000, "data.sources[vkitti].vkitti_max_depth": 655}
            if a == "S6": v.update({"loss.metric_silog_far_weight": 4.0, "loss.metric_silog_far_from": 50.0})
            V[name(a, k)] = v
    return V

def source_types(cfg): return [s["type"] for s in cfg["data"]["sources"]]

def verify(runs, base):
    ok, L = True, []
    E = expect(runs)
    for (x, y), exp in E.items():
        kk = keys(runs[x], runs[y]); g = kk == exp; ok &= g
        L.append(f"## {x} -> {y} [{'OK' if g else 'DEFECT'}] ({len(kk)} keys)" + (" [full key set of the 2 removed sources + checkpoint_dir]" if y.startswith("S1") and x.startswith("B0") else ""))
        if not y.startswith("S1") or x[:2] != "B0":
            for kk2, a, b in diff(runs[x], runs[y]): L.append(f"- {kk2}: {a} -> {b}")
        else:
            for kk2, a, b in diff(runs[x], runs[y]):
                if kk2 == CK: L.append(f"- {kk2}: {a} -> {b}")
            L.append(f"- (+{len(kk) - 1} keys data.sources[nyu_train].* / data.sources[scannetpp_train].*: all present in B0, all <absent> in S1)")
        if not g: L.append(f"  unexpected={sorted(kk - exp)} missing={sorted(exp - kk)}")
    for n, kv in values().items():
        fl = dict(flat(runs[n]))
        for kk, v in kv.items():
            g = fl.get(kk, "<absent>") == json.dumps(v); ok &= g
            L.append(f"declared value [{'OK' if g else 'DEFECT'}] {n}: {kk} = {fl.get(kk, '<absent>')} (want {json.dumps(v)})")
        a = n.rsplit("_k", 1)[0]
        for absent in ("loss.metric_silog_far_weight", "loss.metric_silog_far_from"):   # S6 keys exist ONLY in S6
            g = (absent in fl) == (a == "S6"); ok &= g
            if not g: L.append(f"DEFECT {n}: {absent} presence wrong")
        for kk in ("training.lr_restart_step", "training.lr_restart_warmup", "training.lr_restart_peak", "training.resume_reseed", "training.resume_from_trainable"):
            if kk in fl and kk != "training.resume_reseed" and fl[kk] != "null": ok = False; L.append(f"DEFECT {n}: {kk} must be absent (a from-release run)")
            if kk == "training.resume_reseed" and kk in fl: ok = False; L.append(f"DEFECT {n}: resume_reseed must be absent")
    # source lists: S1 = B0's list minus the two indoor sources, same order; every other arm has B0's list exactly
    for k in STAGES:
        b0 = source_types(runs[name("B0", k)])
        for a in ARMS:
            want = [t for t in b0 if t not in DROPPED] if a == "S1" else b0
            g = source_types(runs[name(a, k)]) == want; ok &= g
            L.append(f"source list [{'OK' if g else 'DEFECT'}] {name(a, k)}: {source_types(runs[name(a, k)])}")
        assert b0 == ["tartanair", "vkitti", "pointodyssey", "pointodyssey", "irs", "spring", "kitti_raw", "nyu_train", "scannetpp_train"], b0
    # all run dirs unique
    dirs = [r["training"]["checkpoint_dir"] for r in runs.values()]; g = len(set(dirs)) == len(dirs); ok &= g; L.append(f"checkpoint_dir unique across the {len(dirs)} runs [{'OK' if g else 'DEFECT'}]")
    # all configs also equal the base on every key that is not declared (implied by the diffs above); the base itself must be the C3 n46 config
    g = base["data"]["seed"] == 46 and base["training"]["resume_from"] is None and base["model"]["vda_init"] == "released"; ok &= g; L.append(f"base is the c3_n46 recipe from the release [{'OK' if g else 'DEFECT'}]")
    return ok, L

def forced_fail(runs, base):
    f = lambda a, b, val: (lambda c: c[a].__setitem__(b, val))
    src = lambda t, k, v: (lambda c: [s.__setitem__(k, v) for s in c["data"]["sources"] if s["type"] == t])
    M = {
     "S2 mass /100 instead of /10": ("S2_k2000", src("kitti_raw", "sampling_mass", 2994)),
     "S3 mass x5 instead of x10": ("S3_k4000", src("vkitti", "sampling_mass", 200000)),
     "S4 lambda 0.5 (arm not changed)": ("S4_k2000", f("loss", "metric_silog_lambda", 0.5)),
     "S5 head lr mult 1.0 (arm not changed)": ("S5_k2000", f("training", "head_lr_mult", 1.0)),
     "S5 head lr mult 0.01": ("S5_k4000", f("training", "head_lr_mult", 0.01)),
     "S6 far weight 2": ("S6_k2000", f("loss", "metric_silog_far_weight", 2.0)),
     "S6 far_from 80": ("S6_k4000", f("loss", "metric_silog_far_from", 80.0)),
     "S6 keys missing": ("S6_k2000", lambda c: [c["loss"].pop("metric_silog_far_weight"), c["loss"].pop("metric_silog_far_from")]),
     "S6 keys leak into S4": ("S4_k2000", f("loss", "metric_silog_far_weight", 4.0)),
     "S1 keeps nyu_train": ("S1_k2000", lambda c: c["data"]["sources"].append(copy.deepcopy(base["data"]["sources"][7]))),
     "S1 also drops spring": ("S1_k4000", lambda c: c["data"].__setitem__("sources", [s for s in c["data"]["sources"] if s["type"] != "spring"])),
     "B0b seed 46 (not changed)": ("B0b_k2000", f("data", "seed", 46)),
     "B0b seed 50": ("B0b_k4000", f("data", "seed", 50)),
     "B0b also resume_reseed": ("B0b_k2000", f("training", "resume_reseed", 49)),
     "B0 gets a restart LR key": ("B0_k2000", f("training", "lr_restart_step", 12000)),
     "warmup changed in one arm": ("S3_k2000", f("training", "warmup_steps", 100)),
     "lr_decay_steps != max_steps (2000 vs 4000)": ("S2_k4000", f("training", "lr_decay_steps", 2000)),
     "resume_from set": ("S4_k4000", f("training", "resume_from", "/projects/b5dh/x.pt")),
     "checkpoint_dir reused": ("S2_k2000", f("training", "checkpoint_dir", "runs/l1_B0_k2000")),
     "extra data change in B0 (num_frames)": ("B0_k4000", f("data", "num_frames", 16)),
     "S3 loses vkitti max depth": ("S3_k2000", src("vkitti", "vkitti_max_depth", 80)),
     "val_every_steps default": ("S5_k2000", f("training", "val_every_steps", 2000)),
    }
    allgood = True
    for t, (r, fn) in M.items():
        r2 = copy.deepcopy(runs); fn(r2[r]); ok, L = verify(r2, base); bad = [l for l in L if "DEFECT" in l]
        print(f"forced-fail [{t}] -> verify ok={ok} (must be False); first defect: {bad[0][:170] if bad else 'NONE'}"); allgood &= not ok
    print(f"FORCED-FAIL SUITE ({len(M)} mutations):", "all mutations detected" if allgood else "SOME MUTATION NOT DETECTED"); return allgood

if __name__ == "__main__":
    base, runs = build()
    for k in STAGES:
        d = HERE / "configs" / f"k{k}"; d.mkdir(parents=True, exist_ok=True)
        for a in ARMS: yaml.safe_dump(runs[name(a, k)], open(d / f"arm_{a}.yaml", "w"), sort_keys=False)
    ok, L = verify(runs, base)
    txt = ["# L1 resolved-config diffs (flat keys). BASE = c3_n46 resolved config (confirm/stage1/base/effective_ramen_n46.yaml sha d286b62c). yaml in configs/k<K>/arm_<ARM>.yaml"] + L + [f"\nALL EXPECTED DIFFS/VALUES EXACT: {ok}"]
    (HERE / "config_diffs_l1.md").write_text("\n".join(txt) + "\n"); print("\n".join(txt))
    if "--forced-fail" in sys.argv: print("\n=== FORCED FAIL ==="); sys.exit(0 if forced_fail(runs, base) and ok else 3)
    sys.exit(0 if ok else 2)

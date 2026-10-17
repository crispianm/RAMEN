"""Confirmatory (Add.47 / Auditor O36) FROM-INIT 12k arms: C_TS2 (control) and C_TS2_H300 (treatment), seed 50. Generator + resolved-config diff + declared-VALUE asserts + forced-fail.
Run from the repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/confirm/train/arms_confirm.py [--forced-fail]
BASE = the resolved config of the c3rep n46 12k from-init job 7226119 (sha256 d286b62c.., identical to provenance/dn01_c3rep_20261009/..c3_n46.resolved.yaml)."""
import copy, json, sys, hashlib
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = REPO / "outdoor_debug/parts/C2_config_train/effective_ramen_n46.yaml"
BASE_SHA = "d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1"
SEG_CLUSTER = "/projects/b5dh/data/train/tartanair_seg/"
SEED = 50
TS2 = [("src:tartanair.ts2_sky", True), ("src:tartanair.ts2_seg_root", SEG_CLUSTER)]
# COMMON (both arms): the ONLY differences from the released recipe besides the arm keys: seed, run dir, flush cadence of the already-on loss-term log
COMMON = [("data.seed", SEED), ("loss.log_terms_every", 10)]
OVERRIDES = {
    "C_TS2": TS2,
    "C_TS2_H300": TS2 + [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 300.0)],
}
# keys that must NOT exist / must keep the recipe value in every arm (Auditor O36: no resume, no restart, plain 500-warmup cosine over 12k)
ABSENT = ["training.resume_reseed", "training.lr_restart_step", "training.lr_restart_warmup", "training.lr_restart_peak", "loss.sky_hinge_min_depth_in_control"]
FIXED = {"training.resume_from": "null", "training.max_steps": "12000", "training.lr_decay_steps": "12000", "training.warmup_steps": "500", "training.learning_rate": "0.0001",
         "training.checkpoint_at_steps": "[6000, 8000, 12000]", "training.seed_scheme": '"x1000"', "data.split_seed": "43", "model.vda_init": '"released"', "model.init_from": "null",
         "model.vda_checkpoint_path": '"/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth"'}

def apply(cfg, ov):
    cfg = copy.deepcopy(cfg)
    for path, val in ov:
        if path.startswith("src:"):
            t, k = path[4:].split(".", 1)
            hits = [s for s in cfg["data"]["sources"] if s["type"] == t]
            assert len(hits) == 1, (t, len(hits)); hits[0][k] = val
        else:
            a, b = path.split("."); cfg[a][b] = val
    return cfg

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list) and o and all(isinstance(x, dict) and "type" in x for x in o):
        for x in o:
            t = x["type"]; tag = t if t != "pointodyssey" else f"{t}#{x['root'].rsplit('/',1)[-1]}"
            yield from flat(x, f"{p}[{tag}]")
    else:
        yield p, json.dumps(o)

def diff(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    return [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]

def build():
    raw = BASE.read_bytes(); assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "base config is not the c3rep n46 resolved config"
    base = yaml.safe_load(raw)
    return base, {n: apply(base, COMMON + ov + [("training.checkpoint_dir", f"runs/confirm_{n}")]) for n, ov in OVERRIDES.items()}

EXPECT = {  # exact set of differing flat keys
    ("BASE", "C_TS2"): {"data.seed", "loss.log_terms_every", "training.checkpoint_dir", "data.sources[tartanair].ts2_sky", "data.sources[tartanair].ts2_seg_root"},
    ("C_TS2", "C_TS2_H300"): {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth", "training.checkpoint_dir"},
    ("BASE", "C_TS2_H300"): {"data.seed", "loss.log_terms_every", "training.checkpoint_dir", "data.sources[tartanair].ts2_sky", "data.sources[tartanair].ts2_seg_root", "loss.sky_hinge_weight", "loss.sky_hinge_min_depth"},
}
VALUES = {"C_TS2": {"data.seed": SEED, "data.sources[tartanair].ts2_sky": True, "data.sources[tartanair].ts2_seg_root": SEG_CLUSTER, "loss.sky_hinge_weight": 0.0, "loss.log_terms": True, "loss.log_terms_every": 10,
                    "training.checkpoint_dir": "runs/confirm_C_TS2"},
          "C_TS2_H300": {"data.seed": SEED, "data.sources[tartanair].ts2_sky": True, "data.sources[tartanair].ts2_seg_root": SEG_CLUSTER, "loss.sky_hinge_weight": 0.1, "loss.sky_hinge_min_depth": 300.0,
                         "loss.log_terms": True, "loss.log_terms_every": 10, "training.checkpoint_dir": "runs/confirm_C_TS2_H300"}}

def verify(base, arms):
    """returns (ok, lines). Checks: exact diff key sets, declared VALUES, absent keys, fixed recipe values, LR/length keys identical across arms."""
    ok, L = True, []
    allarms = {"BASE": base, **arms}
    for (x, y), exp in EXPECT.items():
        keys = {k for k, _, _ in diff(allarms[x], allarms[y])}; good = keys == exp; ok &= good
        L.append(f"## {x} -> {y} [{'OK' if good else 'DEFECT'}] ({len(keys)} keys)")
        for k, a, b in diff(allarms[x], allarms[y]): L.append(f"- {k}: {a} -> {b}")
        if not good: L.append(f"  unexpected={sorted(keys - exp)} missing={sorted(exp - keys)}")
    for n, kv in VALUES.items():
        fl = dict(flat(arms[n]))
        for k, v in kv.items():
            g = fl.get(k, "<absent>") == json.dumps(v); ok &= g; L.append(f"declared value [{'OK' if g else 'DEFECT'}] {n}: {k} = {fl.get(k, '<absent>')} (want {json.dumps(v)})")
    if "sky_hinge_min_depth" in arms["C_TS2"]["loss"]: ok = False; L.append("DEFECT control carries sky_hinge_min_depth")
    for n, c in arms.items():
        fl = dict(flat(c))
        for k in ABSENT:
            if k in fl: ok = False; L.append(f"DEFECT {n}: forbidden key {k} = {fl[k]}")
        for k, v in FIXED.items():
            g = fl.get(k, "<absent>") == v; ok &= g; L.append(f"fixed recipe value [{'OK' if g else 'DEFECT'}] {n}: {k} = {fl.get(k, '<absent>')} (want {v})")
    return ok, L

def forced_fail(base, arms):
    muts = {
        "H300 floor 30 instead of 300": ("C_TS2_H300", lambda c: c["loss"].__setitem__("sky_hinge_min_depth", 30.0)),
        "H300 gets an extra learning_rate change": ("C_TS2_H300", lambda c: c["training"].__setitem__("learning_rate", 2e-4)),
        "control gets the hinge weight": ("C_TS2", lambda c: c["loss"].__setitem__("sky_hinge_weight", 0.1)),
        "control has seed 49": ("C_TS2", lambda c: c["data"].__setitem__("seed", 49)),
        "resume_from set": ("C_TS2_H300", lambda c: c["training"].__setitem__("resume_from", "/x/step12000.pt")),
        "lr_restart key added": ("C_TS2", lambda c: c["training"].__setitem__("lr_restart_step", 12000)),
        "control lost ts2_sky": ("C_TS2", lambda c: [s.__setitem__("ts2_sky", False) for s in c["data"]["sources"] if s["type"] == "tartanair"]),
        "hinge arm trains with another source mass": ("C_TS2_H300", lambda c: c["data"]["sources"][0].__setitem__("sampling_mass", 1)),
    }
    allgood = True
    for name, (arm, f) in muts.items():
        a2 = copy.deepcopy(arms); f(a2[arm]); ok, L = verify(base, a2)
        bad = [l for l in L if "DEFECT" in l]
        print(f"forced-fail [{name}] -> verify ok={ok} (must be False); first defect line: {bad[0] if bad else 'NONE'}")
        allgood &= (not ok)
    print("FORCED-FAIL SUITE:", "all mutations detected" if allgood else "SOME MUTATION NOT DETECTED")
    return allgood

if __name__ == "__main__":
    base, arms = build()
    (OUT / "configs").mkdir(exist_ok=True)
    for n, c in arms.items(): yaml.safe_dump(c, open(OUT / "configs" / f"arm_{n}.yaml", "w"), sort_keys=False)
    ok, L = verify(base, arms)
    txt = ["# Confirmatory from-init arms: resolved-config diffs (BASE = c3rep n46 job 7226119 resolved config, sha " + BASE_SHA[:8] + ")"] + L + [f"\nALL EXPECTED DIFFS/VALUES EXACT: {ok}"]
    (OUT / "config_diffs_confirm.md").write_text("\n".join(txt) + "\n"); print("\n".join(txt))
    if "--forced-fail" in sys.argv:
        print("\n=== FORCED FAIL ==="); sys.exit(0 if forced_fail(base, arms) else 3)
    sys.exit(0 if ok else 2)

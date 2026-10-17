"""Arm override dicts + config builder + resolved-config diff + instantiated-class listing (check 1).
Run from the repo root: ~/.venvs/ramen/bin/python outdoor_debug/parts/arm_checks/arms.py"""
import copy, json, sys, os
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent
BASE = REPO / "outdoor_debug/parts/C2_config_train/effective_c3_seed45.yaml"
SEG_CLUSTER = "/projects/b5dh/data/train/tartanair_seg/"

# declared overrides; each = list of (path, value); a path starting 'src:<type>.' targets data.sources[type==<type>]
TS2 = [("src:tartanair.ts2_sky", True), ("src:tartanair.ts2_seg_root", SEG_CLUSTER)]
OVERRIDES = {
    "A0": [],
    "A0b": [("data.seed", 49), ("training.resume_reseed", 49)],   # see README "R-c": data.seed = sampler order (DDP) + dataset ctor seed; resume_reseed = global RNG (worker noise seeds) AFTER the checkpoint rng restore; split_seed 43 / seed_scheme untouched
    "TS2": TS2,
    "TS2_H80": TS2 + [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 80.0)],
    "TS2_H300": TS2 + [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 300.0)],
    "TS2_VK655": TS2 + [("src:vkitti.vkitti_sky_as_valid_depth", True), ("src:vkitti.vkitti_max_depth", 656)],
}

# screen continuation (all arms identical): resume c3_n46 step12000 (Isambard path), 2000 more steps, S3 LR (Addendum 45: restart at 12000, warmup 100, peak 1.0, cosine to 0 at 14000)
CKPT_N46 = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt"
LR_S1 = [("training.lr_decay_steps", 14000)]
LR_S2 = [("training.lr_decay_steps", 1000000000)]
LR_S3 = [("training.lr_decay_steps", 14000), ("training.lr_restart_step", 12000), ("training.lr_restart_warmup", 100), ("training.lr_restart_peak", 1.0)]
COMMON = [("training.resume_from", CKPT_N46), ("training.max_steps", 14000), ("training.checkpoint_at_steps", [12500, 13000, 13500, 14000]),
          ("loss.log_terms_every", 10)] + LR_S3          # log_terms is already true in the base; log_terms_every 10 only changes the flush cadence (no loss change)
LR_KEYS = {"training.lr_decay_steps", "training.lr_restart_step", "training.lr_restart_warmup", "training.lr_restart_peak", "training.learning_rate", "training.warmup_steps", "training.max_steps"}

def apply(cfg, ov):
    cfg = copy.deepcopy(cfg)
    for path, val in ov:
        if path.startswith("src:"):
            t, k = path[4:].split(".", 1)
            hits = [s for s in cfg["data"]["sources"] if s["type"] == t]
            assert len(hits) == 1, (t, len(hits))
            hits[0][k] = val
        else:
            a, b = path.split(".")
            cfg[a][b] = val
    return cfg

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list) and o and all(isinstance(x, dict) and "type" in x for x in o):
        seen = {}
        for x in o:
            t = x["type"]; seen[t] = seen.get(t, 0) + 1
            tag = t if (t not in ("pointodyssey",)) else f"{t}#{x['root'].rsplit('/',1)[-1]}"
            yield from flat(x, f"{p}[{tag}]")
    else:
        yield p, json.dumps(o)

def diff(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    return [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]

def build():
    base = yaml.safe_load(open(BASE))
    arms = {n: apply(base, COMMON + ov + [("training.checkpoint_dir", f"runs/outdoor_screen_{n}")]) for n, ov in OVERRIDES.items()}
    (OUT / "configs").mkdir(exist_ok=True)
    for tag, ov in (("S1", LR_S1), ("S2", LR_S2), ("S3", LR_S3)):   # LR overlays (training: keys to merge over an arm yaml; S3 is what the arms bake in)
        yaml.safe_dump({"training": {k.split(".", 1)[1]: v for k, v in ov}}, open(OUT / "configs" / f"lr_{tag}.yaml", "w"), sort_keys=False)
    for n, c in arms.items():
        yaml.safe_dump(c, open(OUT / "configs" / f"arm_{n}.yaml", "w"), sort_keys=False)
    return base, arms

EXPECT = {  # (armX, armY): exact set of differing flat keys
    ("A0", "A0b"): {"data.seed", "training.resume_reseed", "training.checkpoint_dir"},
    ("A0", "TS2"): {"data.sources[tartanair].ts2_sky", "data.sources[tartanair].ts2_seg_root", "training.checkpoint_dir"},
    ("TS2", "TS2_H80"): {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth", "training.checkpoint_dir"},
    ("TS2", "TS2_H300"): {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth", "training.checkpoint_dir"},
    ("TS2", "TS2_VK655"): {"data.sources[vkitti].vkitti_sky_as_valid_depth", "data.sources[vkitti].vkitti_max_depth", "training.checkpoint_dir"},
    ("A0", "TS2_H80"): None, ("A0", "TS2_H300"): None, ("A0", "TS2_VK655"): None,
}

if __name__ == "__main__":
    base, arms = build()
    lines = ["# Resolved-config diffs (flat keys; yaml written to configs/arm_<name>.yaml)\n"]
    ok = True
    # A0 must equal the base file exactly
    dbase = diff(base, arms["A0"])
    lines.append("A0 vs effective_c3_seed45.yaml (base): differs ONLY by the common continuation fields (resume_from, max_steps, checkpoint_at_steps, checkpoint_dir, LR keys, log_terms_every): "
                 + str({k for k, _, _ in dbase} <= {"training.resume_from", "training.max_steps", "training.checkpoint_at_steps", "training.checkpoint_dir", "loss.log_terms_every"} | {"training.lr_decay_steps", "training.lr_restart_step", "training.lr_restart_warmup", "training.lr_restart_peak"}) + "\n")
    for k, a, b in dbase: lines.append(f"- {k}: {a} -> {b}")
    # L5: LR + run-length keys identical across all 6 arms; log_terms on in all
    for k in sorted(LR_KEYS):
        vals = {n: dict(flat(c)).get(k, "<absent>") for n, c in arms.items()}
        same = len(set(vals.values())) == 1; ok &= same
        lines.append(f"LR-key identical across {len(arms)} arms [{'OK' if same else 'DEFECT'}] {k} = {set(vals.values())}")
    lt = {n: (c['loss'].get('log_terms'), c['loss'].get('log_terms_every')) for n, c in arms.items()}
    ok &= all(v == (True, 10) for v in lt.values()); lines.append(f"log_terms/log_terms_every per arm: {lt}")
    lines.append("")
    for (x, y), exp in EXPECT.items():
        d = diff(arms[x], arms[y]); keys = {k for k, _, _ in d}
        if exp is None:
            status = "info"
        else:
            status = "OK" if keys == exp else "DEFECT"; ok &= (keys == exp)
        lines.append(f"\n## {x} -> {y}  [{status}]  ({len(d)} differing keys)")
        for k, a, b in d: lines.append(f"- {k}: {a} -> {b}")
    # declared VALUES (reviewer finding 2): the diff above only checks WHICH keys differ; assert the values too
    VALUES = {"A0": {"data.seed": 45},
              "A0b": {"data.seed": 49, "training.resume_reseed": 49},
              "TS2": {"data.sources[tartanair].ts2_sky": True},
              "TS2_H80": {"data.sources[tartanair].ts2_sky": True, "loss.sky_hinge_weight": 0.1, "loss.sky_hinge_min_depth": 80.0},
              "TS2_H300": {"data.sources[tartanair].ts2_sky": True, "loss.sky_hinge_weight": 0.1, "loss.sky_hinge_min_depth": 300.0},
              "TS2_VK655": {"data.sources[tartanair].ts2_sky": True, "data.sources[vkitti].vkitti_sky_as_valid_depth": True, "data.sources[vkitti].vkitti_max_depth": 656}}
    for n, kv in VALUES.items():
        fl = dict(flat(arms[n]))
        for k, v in kv.items():
            good = fl.get(k, "<absent>") == json.dumps(v); ok &= good
            lines.append(f"declared value [{'OK' if good else 'DEFECT'}] {n}: {k} = {fl.get(k, '<absent>')!r} (want {v!r})")
    lines.append(f"\nALL EXPECTED DIFFS EXACT: {ok}")
    # also vs A0 for every arm (the Manager asked for 'every arm vs A0')
    (OUT / "config_diffs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

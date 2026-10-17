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
    "A0b": [("data.seed", 46)],                       # only the seed differs (data.seed drives sampler/degradation RNG; split_seed 43 and training.seed_scheme untouched)
    "TS2": TS2,
    "TS2_H80": TS2 + [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 80.0)],
    "TS2_H300": TS2 + [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 300.0)],
    "TS2_VK655": TS2 + [("src:vkitti.vkitti_sky_as_valid_depth", True), ("src:vkitti.vkitti_max_depth", 656)],
}

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
    arms = {n: apply(base, ov) for n, ov in OVERRIDES.items()}
    (OUT / "configs").mkdir(exist_ok=True)
    for n, c in arms.items():
        yaml.safe_dump(c, open(OUT / "configs" / f"arm_{n}.yaml", "w"), sort_keys=False)
    return base, arms

EXPECT = {  # (armX, armY): exact set of differing flat keys
    ("A0", "A0b"): {"data.seed"},
    ("A0", "TS2"): {"data.sources[tartanair].ts2_sky", "data.sources[tartanair].ts2_seg_root"},
    ("TS2", "TS2_H80"): {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth"},
    ("TS2", "TS2_H300"): {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth"},
    ("TS2", "TS2_VK655"): {"data.sources[vkitti].vkitti_sky_as_valid_depth", "data.sources[vkitti].vkitti_max_depth"},
    ("A0", "TS2_H80"): None, ("A0", "TS2_H300"): None, ("A0", "TS2_VK655"): None,
}

if __name__ == "__main__":
    base, arms = build()
    lines = ["# Resolved-config diffs (flat keys; yaml written to configs/arm_<name>.yaml)\n"]
    ok = True
    # A0 must equal the base file exactly
    lines.append(f"A0 == effective_c3_seed45.yaml (no override): {arms['A0'] == base}\n")
    for (x, y), exp in EXPECT.items():
        d = diff(arms[x], arms[y]); keys = {k for k, _, _ in d}
        if exp is None:
            status = "info"
        else:
            status = "OK" if keys == exp else "DEFECT"; ok &= (keys == exp)
        lines.append(f"\n## {x} -> {y}  [{status}]  ({len(d)} differing keys)")
        for k, a, b in d: lines.append(f"- {k}: {a} -> {b}")
    lines.append(f"\nALL EXPECTED DIFFS EXACT: {ok}")
    # also vs A0 for every arm (the Manager asked for 'every arm vs A0')
    (OUT / "config_diffs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

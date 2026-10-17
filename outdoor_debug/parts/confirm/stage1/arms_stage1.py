"""Stage 1 (Add.48 + Auditor O39): THREE PAIRED 2000-step CONTINUATIONS, arms TS2 (control) vs TS2+H300 (treatment), starts c3_n46/n47/n48 step12000, seeds 50/51/52.
Generator + resolved-config diff proofs + declared-VALUE asserts + forced-fail mutations.   Run from the repo root:
  ~/.venvs/ramen/bin/python outdoor_debug/parts/confirm/stage1/arms_stage1.py [--forced-fail]
BASE per pair = the resolved config of that start's OWN c3rep job (base/effective_ramen_nXX.yaml == provenance/dn01_c3rep_20261009/..c3_nXX.resolved.yaml, sha asserted).
The screen used effective_c3_seed45 as base for a c3_n46 start; that file differs from the n46 job config only in data.seed/checkpoint_dir/checkpoint_at_steps/resume_from
(all overridden), so the screen arm yaml and the n46 Stage 1 yaml differ ONLY by seed, resume_reseed, checkpoint_dir (asserted below)."""
import copy, json, sys, hashlib
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "outdoor_debug/parts/arm_checks"))
from arms import flat, diff, apply          # the screen's own helpers (same flattening as config_diffs.md)
SEG = "/projects/b5dh/data/train/tartanair_seg/"
RUNS = "/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n{n}/step12000.pt"
BASE_SHA = {"46": "d286b62cef5f5e603ca909675b7daf22c4e120b0b99dfe3d5ec69bcbebd911a1", "47": "a3092bb1c486a00f3986be8e4ee9e3bb43fbb0ca3a1d8ed3bfae464143a4f37d", "48": "fbe2dda70a553c8c62a3d5a3dcf78607abfea9877a57f42f6af3fe7e315d8480"}
START_SHA = {"46": "f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af", "47": "2ee98b7ee25bd6d5f044ea1f6bb7854d1a0c978488be52d4c87d305a5df6a54c", "48": "bf053bd38fdb97d55228fceb737fc2d6353cca26d9ab13338c6d9dd3df95da12"}
SEED = {"46": 50, "47": 51, "48": 52}
TS2 = [("src:tartanair.ts2_sky", True), ("src:tartanair.ts2_seg_root", SEG)]
HING = [("loss.sky_hinge_weight", 0.1), ("loss.sky_hinge_min_depth", 300.0)]
LR_S3 = [("training.lr_decay_steps", 14000), ("training.lr_restart_step", 12000), ("training.lr_restart_warmup", 100), ("training.lr_restart_peak", 1.0)]
SCREEN = {"TS2": HERE.parent.parent / "arm_checks/configs/arm_TS2.yaml", "H300": HERE.parent.parent / "arm_checks/configs/arm_TS2_H300.yaml"}
NAMES = [f"S1_n{n}_{a}" for n in ("46", "47", "48") for a in ("TS2", "H300")]

def name(n, a): return f"S1_n{n}_{a}"
def build():
    bases, runs = {}, {}
    for n in SEED:
        raw = (HERE / f"base/effective_ramen_n{n}.yaml").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == BASE_SHA[n], f"base n{n} sha"
        bases[n] = yaml.safe_load(raw)
        common = [("training.resume_from", RUNS.format(n=n)), ("training.max_steps", 14000), ("training.checkpoint_at_steps", [12500, 13000, 13500, 14000]),
                  ("loss.log_terms_every", 10), ("data.seed", SEED[n]), ("training.resume_reseed", SEED[n])] + LR_S3
        for a, ov in (("TS2", TS2), ("H300", TS2 + HING)):
            runs[name(n, a)] = apply(bases[n], common + ov + [("training.checkpoint_dir", f"runs/confirm1_{name(n, a)}")])
    return bases, runs

def keys(a, b): return {k for k, _, _ in diff(a, b)}
CK = lambda x: f"training.checkpoint_dir"
def expect():
    E = {}
    for n in SEED:
        E[(name(n, "TS2"), name(n, "H300"))] = {"loss.sky_hinge_weight", "loss.sky_hinge_min_depth", "training.checkpoint_dir"}               # within a pair
    for a in ("TS2", "H300"):
        for n1, n2 in (("46", "47"), ("46", "48"), ("47", "48")):
            E[(name(n1, a), name(n2, a))] = {"data.seed", "training.resume_reseed", "training.resume_from", "training.checkpoint_dir"}          # across pairs
    return E
VALUES = {}
for n in SEED:
    for a in ("TS2", "H300"):
        v = {"data.seed": SEED[n], "training.resume_reseed": SEED[n], "training.resume_from": RUNS.format(n=n), "training.max_steps": 14000, "training.lr_decay_steps": 14000,
             "training.lr_restart_step": 12000, "training.lr_restart_warmup": 100, "training.lr_restart_peak": 1.0, "training.learning_rate": 0.0001, "training.warmup_steps": 500,
             "training.checkpoint_at_steps": [12500, 13000, 13500, 14000], "training.checkpoint_dir": f"runs/confirm1_{name(n, a)}", "loss.log_terms": True, "loss.log_terms_every": 10,
             "data.sources[tartanair].ts2_sky": True, "data.sources[tartanair].ts2_seg_root": SEG, "data.sources[vkitti].vkitti_max_depth": 655, "data.split_seed": 43, "training.seed_scheme": "x1000"}
        v.update({"loss.sky_hinge_weight": 0.1, "loss.sky_hinge_min_depth": 300.0} if a == "H300" else {"loss.sky_hinge_weight": 0.0})
        VALUES[name(n, a)] = v

def verify(runs):
    ok, L = True, []
    for (x, y), exp in expect().items():
        k = keys(runs[x], runs[y]); g = k == exp; ok &= g
        L.append(f"## {x} -> {y} [{'OK' if g else 'DEFECT'}] ({len(k)} keys)")
        for kk, a, b in diff(runs[x], runs[y]): L.append(f"- {kk}: {a} -> {b}")
        if not g: L.append(f"  unexpected={sorted(k - exp)} missing={sorted(exp - k)}")
    for n, kv in VALUES.items():
        fl = dict(flat(runs[n]))
        for k, v in kv.items():
            g = fl.get(k, "<absent>") == json.dumps(v); ok &= g
            L.append(f"declared value [{'OK' if g else 'DEFECT'}] {n}: {k} = {fl.get(k, '<absent>')} (want {json.dumps(v)})")
        if n.endswith("TS2") and "sky_hinge_min_depth" in runs[n]["loss"]: ok = False; L.append(f"DEFECT {n}: control carries sky_hinge_min_depth")
    # vs the screen arm yaml (same arm): n46 pair differs ONLY by seed/reseed/checkpoint_dir; n47/n48 additionally resume_from
    for n in SEED:
        for a in ("TS2", "H300"):
            scr = yaml.safe_load(open(SCREEN[a])); k = keys(scr, runs[name(n, a)])
            exp = {"data.seed", "training.resume_reseed", "training.checkpoint_dir"} | ({"training.resume_from"} if n != "46" else set())
            g = k == exp; ok &= g
            L.append(f"## screen arm_{'TS2' if a == 'TS2' else 'TS2_H300'} -> {name(n, a)} [{'OK' if g else 'DEFECT'}]: {sorted(k)}" + ("" if g else f" (want {sorted(exp)})"))
    return ok, L

def forced_fail(runs):
    f = lambda a, b, val: (lambda c: c[a].__setitem__(b, val))
    tart = lambda k, v: (lambda c: [s.__setitem__(k, v) for s in c["data"]["sources"] if s["type"] == "tartanair"])
    M = {
     "H300 floor 30 instead of 300": ("S1_n47_H300", f("loss", "sky_hinge_min_depth", 30.0)),
     "H300 weight 0.2": ("S1_n46_H300", f("loss", "sky_hinge_weight", 0.2)),
     "control gets the hinge weight": ("S1_n48_TS2", f("loss", "sky_hinge_weight", 0.1)),
     "H300 extra LR change": ("S1_n46_H300", f("training", "learning_rate", 2e-4)),
     "H300 differs in max_steps": ("S1_n47_H300", f("training", "max_steps", 13000)),
     "lr_restart_warmup differs in one run": ("S1_n48_TS2", f("training", "lr_restart_warmup", 50)),
     "n47 control has seed 50 (pair seed wrong)": ("S1_n47_TS2", f("data", "seed", 50)),
     "n47 H300 lacks resume_reseed match (51 vs 50)": ("S1_n47_H300", f("training", "resume_reseed", 50)),
     "n48 control resume_reseed removed": ("S1_n48_TS2", lambda c: c["training"].pop("resume_reseed")),
     "n46 pair member seeds differ (50 vs 49)": ("S1_n46_H300", f("data", "seed", 49)),
     "n47 control resumes from n46 ckpt": ("S1_n47_TS2", f("training", "resume_from", RUNS.format(n="46"))),
     "n48 control lost ts2_sky": ("S1_n48_TS2", tart("ts2_sky", False)),
     "H300 other source mass": ("S1_n47_H300", lambda c: c["data"]["sources"][0].__setitem__("sampling_mass", 1)),
     "checkpoint_dir reused across runs": ("S1_n47_TS2", f("training", "checkpoint_dir", "runs/confirm1_S1_n46_TS2")),
    }
    allgood = True
    for t, (r, fn) in M.items():
        r2 = copy.deepcopy(runs); fn(r2[r]); ok, L = verify(r2); bad = [l for l in L if "DEFECT" in l]
        print(f"forced-fail [{t}] -> verify ok={ok} (must be False); first defect: {bad[0][:200] if bad else 'NONE'}"); allgood &= not ok
    print("FORCED-FAIL SUITE:", "all mutations detected" if allgood else "SOME MUTATION NOT DETECTED"); return allgood

if __name__ == "__main__":
    bases, runs = build()
    (HERE / "configs").mkdir(exist_ok=True)
    for n, c in runs.items(): yaml.safe_dump(c, open(HERE / "configs" / f"arm_{n}.yaml", "w"), sort_keys=False)
    ok, L = verify(runs)
    txt = ["# Stage 1 resolved-config diffs (flat keys). BASE per pair = that start's own c3rep resolved config; yaml in configs/arm_<run>.yaml"] + L + [f"\nALL EXPECTED DIFFS/VALUES EXACT: {ok}"]
    (HERE / "config_diffs_stage1.md").write_text("\n".join(txt) + "\n"); print("\n".join(txt))
    if "--forced-fail" in sys.argv: print("\n=== FORCED FAIL ==="); sys.exit(0 if forced_fail(runs) and ok else 3)
    sys.exit(0 if ok else 2)

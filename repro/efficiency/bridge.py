"""4090 efficiency BRIDGE (commit BR): scheduling + comparison only. No numerics, no scorer, no leg_worker change, no timing is ever written or quoted.

  launcher.py bridge plan    --primary-file efficiency_primary.txt [--legs B1,B2[,B3,B4]]
  launcher.py bridge run     --primary-file F --token T --out DIR [--legs ...] [--lock PATH]      first pass: legs serial (N=1), then COMPARE
  launcher.py bridge run     ... --repeat-binding     exactly ONE binding repeat of the same legs after a FAILED first pass; both results are recorded
  launcher.py bridge compare --out DIR                re-evaluate finished legs

Legs: B1 = RAMEN S0@fp16 (reference), B2 = RAMEN PRIMARY@fp16 (test), optional B3 = Metric-VDA-S S0@fp16, B4 = Metric-VDA-S PRIMARY toggles@fp16.
PRIMARY comes ONLY from the explicit file (key=value lines; `stack=S4`, optional `exclude=T8,T7`, `exclude` may contain T11 = numpy TAG in B2); absent file = refuse.
Gate: fp16 per-item set of the launcher (depth/TAE/flicker 5e-3 rel, tOF 1.5%, PSNR 0.02 dB, SSIM 3e-4); PRIMARY = X -> BITWISE depth on every unit.
RAMEN+TAG rows are GATED separately per setting: B1 (numpy reference TAG) vs B2 (V3 TAG if the primary has T11/S4, else numpy). A failed gate leaves the 4090 latency UNPAIRED;
nothing is retuned (no gate/tolerance/stack option exists to retune). Host label '4090' in every output. Input default = the verified cache."""
from __future__ import annotations

import os
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import common as C  # noqa: E402
from repro.efficiency import launcher as LA  # noqa: E402
from repro.efficiency import legs as L  # noqa: E402

HOST = "4090"
LABEL = "efficiency bridge, DEV, RTX 4090, N=1; PRIMARY@fp16 vs S0@fp16"
DEFAULT_PRIMARY_FILE = Path(__file__).with_name("efficiency_primary.txt")
DEFAULT_LOCK = Path(os.environ.get("RAMEN_GPU_LOCK", Path(__file__).resolve().parents[2] / "GPU_LOCK"))
LOCK_TAG = "bridge-r1"
BRIDGE_IDS = ("B1", "B2", "B3", "B4")
TAG_SETTINGS = {"agf0.2t0.06": "(0.2, 0.06)", "agf0.5t0.12": "(0.5, 0.12)"}
PRIMARY_STACKS = ("X", "S1", "S2", "S3", "S4", "C", "AZ")


# ------------------------------------------------------------------ primary file
def read_primary(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"refusing: PRIMARY file {path} is absent (efficiency_primary.txt must be written explicitly after the Auditor acks the resolved stack)")
    kv = {}
    for ln in path.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        if "=" not in ln:
            raise SystemExit(f"{path}: bad line {ln!r} (expected key=value)")
        k, v = ln.split("=", 1)
        kv[k.strip()] = v.strip()
    if set(kv) - {"stack", "exclude"}:
        raise SystemExit(f"{path}: unknown keys {sorted(set(kv) - {'stack', 'exclude'})}")
    stack = kv.get("stack")
    if stack not in PRIMARY_STACKS:
        raise SystemExit(f"{path}: stack must be one of {PRIMARY_STACKS} (S0 = nothing to pair), got {stack!r}")
    exclude = tuple(x for x in kv.get("exclude", "").replace(" ", "").split(",") if x)
    bad = [x for x in exclude if x not in L.TOGGLE_FLAGS and x != "T11"]
    if bad:
        raise SystemExit(f"{path}: unknown toggle(s) in exclude: {bad}")
    return {"stack": stack, "exclude": list(exclude), "file": str(path), "file_sha256": C.sha256_file(path)}


def build_bridge_legs(primary: dict, repeat: int = 0, which=("B1", "B2")) -> dict:
    """{B1..B4: Leg}. Ids carry the bridge name; repeat=1 -> '-rep1' suffix. B3/B4 only exist for a primary with Metric-VDA-S toggles (not X)."""
    st, ex = primary["stack"], tuple(primary["exclude"])
    sfx = f"-rep{repeat}" if repeat else ""
    out = {}

    def mk(name, model, stack, purpose):
        l = L.make_leg(model, stack, "fp16", repeat=repeat, exclude=ex, purpose=purpose)
        l.id = f"{name}_{model}-{stack}-fp16{sfx}"
        if model == "ramen" and stack == "S4" and "T11" in ex:
            l.agf_backend = "cpu"                         # primary without T11: numpy TAG in B2 as well
        return l

    out["B1"] = mk("B1", "ramen", "S0", "bridge reference: RAMEN S0 fp16 (numpy reference TAG)")
    ramen_stack = st
    out["B2"] = mk("B2", "ramen", ramen_stack, f"bridge test: RAMEN PRIMARY={st} fp16")
    if st != "X":
        out["B3"] = mk("B3", "mvda", "S0", "optional: Metric-VDA-S S0 fp16")
        out["B4"] = mk("B4", "mvda", L.MVDA_MATCH.get(st, "S3" if st == "S4" else st), f"optional: Metric-VDA-S PRIMARY toggles ({st}) fp16")
    sel = [w for w in which]
    missing = [w for w in sel if w not in out]
    if missing:
        raise SystemExit(f"--legs: {missing} not available for PRIMARY stack {st} (known: {sorted(out)})")
    if "B2" in sel and "B1" not in sel or "B1" in sel and "B2" not in sel:
        raise SystemExit("--legs: B1 and B2 are the required pair and go together")
    return {k: out[k] for k in BRIDGE_IDS if k in sel}


def parse_which(spec) -> tuple:
    w = tuple(x for x in (spec or "B1,B2").replace(" ", "").split(",") if x)
    bad = [x for x in w if x not in BRIDGE_IDS]
    if bad or not {"B1", "B2"} <= set(w):
        raise SystemExit(f"--legs for bridge must be a subset of {BRIDGE_IDS} containing B1,B2 (got {spec!r})")
    if ("B3" in w) != ("B4" in w):
        raise SystemExit("--legs: B3 and B4 go together")
    return w


# ------------------------------------------------------------------ comparison
def _only_variants(leg: dict, variants: set) -> dict:
    return {**leg, "macro": {**leg["macro"], "macro": {u: v for u, v in leg["macro"]["macro"].items() if u.rsplit("|", 1)[-1] in variants}},
            "rows": [r for r in leg["rows"] if r["variant"] in variants]}


def compare_bridge_pair(test: dict, ref: dict, gate: str, variants: set) -> dict:
    t, r = _only_variants(test, variants), _only_variants(ref, variants)
    res = LA.compare_pair(t, r, gate)
    tu, ru = set(t["macro"]["macro"]), set(r["macro"]["macro"])
    res["n_units"] = [len(tu), len(ru)]
    res["unit_set_mismatch"] = sorted(tu ^ ru)[:50]
    res["PASS"] = bool(res["PASS"] and tu and tu == ru)
    return res


def bridge_comparisons(legs: dict, primary: dict) -> list:
    gate = "bitwise" if primary["stack"] == "X" else "fp16"
    v3 = primary["stack"] == "S4" and "T11" not in primary["exclude"]
    c = []
    if "B1" in legs and "B2" in legs:
        t, r = legs["B2"].id, legs["B1"].id
        c.append({"kind": "bridge", "name": "bridge_raw_B2_vs_B1", "test": t, "ref": r, "gate": gate, "variants": ["none"]})
        for v in TAG_SETTINGS:
            c.append({"kind": "bridge_tag", "name": f"bridge_tag_{v}_B2_vs_B1", "test": t, "ref": r, "gate": gate, "variants": [v],
                      "tag_backends": {"ref": "numpy_reference", "test": "fast_triton_v3" if v3 else "numpy_reference"}})
    if "B3" in legs and "B4" in legs:
        c.append({"kind": "bridge", "name": "bridge_raw_B4_vs_B3", "test": legs["B4"].id, "ref": legs["B3"].id, "gate": gate, "variants": ["none"]})
    return c


def evaluate(out: Path, comps: list, label_suffix: str = "") -> list:
    rep = []
    for c in comps:
        try:
            r = compare_bridge_pair(LA.load_leg(out, c["test"]), LA.load_leg(out, c["ref"]), c["gate"], set(c["variants"]))
        except FileNotFoundError as e:
            r = {"PASS": False, "error": f"missing leg output: {e}"}
        rep.append({**c, **r, "host": HOST, "label": LABEL + label_suffix})
    return rep


def verdict(rep: list) -> dict:
    ok = bool(rep) and all(r["PASS"] for r in rep)
    return {"host": HOST, "label": LABEL, "PASS": ok, "paired": ok,
            "latency_4090": "PAIRED (bridge gates pass)" if ok else "UNPAIRED (a bridge gate failed; no retune, report to user)",
            "failed": [r["name"] for r in rep if not r["PASS"]]}


# ------------------------------------------------------------------ GPU_LOCK
def lock_acquire(path: Path, tag: str = LOCK_TAG) -> None:
    path = Path(path)
    cur = path.read_text().strip() if path.exists() else ""
    if cur:
        raise SystemExit(f"GPU_LOCK held: {cur[:200]}")
    path.write_text(f"{tag} {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")


def lock_release(path: Path, tag: str = LOCK_TAG) -> None:
    path = Path(path)
    if path.exists() and path.read_text().startswith(tag):
        path.write_text("")


# ------------------------------------------------------------------ commands
def _record(out: Path, name: str, obj) -> None:
    C.write_json(out / name, {"host": HOST, "label": LABEL, **obj} if isinstance(obj, dict) else obj)


def _run_legs(a, out: Path, legs: dict, suffix: str) -> dict:
    a.workers = 1                                            # N = 1, serial
    a.source = "cache"
    a.patches_dir = getattr(a, "patches_dir", None)
    before = C.untouched_record("BEFORE")
    _record(out, f"BEFORE{suffix}.json", before)
    status = LA.run_phase(list(legs.values()), out, a, 1, "cache")
    after = C.untouched_record("AFTER")
    _record(out, f"AFTER{suffix}.json", after)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("host", "label")}  # noqa: E731
    same = strip(before) == strip(after) if isinstance(before, dict) else before == after
    return {"status": status, "untouched_equal": bool(same), "all_done": len(status) == len(legs) and all(v == "DONE" for v in status.values())}


def cmd_bridge_plan(a) -> int:
    primary = read_primary(a.primary_file)
    legs = build_bridge_legs(primary, which=parse_which(a.legs))
    print(f"host={HOST}  {LABEL}\nprimary={primary}")
    print(LA.leg_table(list(legs.values()), 1))
    print("comparisons:", json.dumps([f"{c['name']} ({c['gate']})" for c in bridge_comparisons(legs, primary)]))
    return 0


def cmd_bridge_run(a) -> int:
    out = a.out
    if not a.token:
        raise SystemExit("refusing to start without --token (AUD-EFF-BRIDGE-4090-...)")
    primary = read_primary(a.primary_file)
    which = parse_which(a.legs)
    vr = a.cache / "verify_report.json"
    if a.source != "cache" or not vr.exists() or not json.loads(vr.read_text())["summary"]["PASS"]:
        raise SystemExit("bridge input source is the verified cache with a PASSing verify_report.json")
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "BRIDGE_RESULT.json"
    first = json.loads(res_path.read_text()) if res_path.exists() else None
    rep_n = 0
    if a.repeat_binding:
        if first is None or first["verdict"]["PASS"]:
            raise SystemExit("--repeat-binding needs a FAILED first pass in --out (BRIDGE_RESULT.json)")
        if (out / "BRIDGE_FINAL.json").exists():
            raise SystemExit("the binding repeat was already run (exactly one is allowed)")
        if first["primary"] != primary or first["which"] != list(which):
            raise SystemExit("--repeat-binding must use the same PRIMARY file content and legs as the first pass")
        rep_n = 1
    elif first is not None:
        raise SystemExit("first pass already recorded in --out; only --repeat-binding may follow")
    legs = build_bridge_legs(primary, repeat=rep_n, which=which)
    comps = bridge_comparisons(legs, primary)
    lock = Path(a.lock)
    lock_acquire(lock)
    try:
        busy = LA.gpu_other_procs()
        if busy:
            raise SystemExit(f"card not idle, other GPU processes present: {busy}")
        _record(out, f"BRIDGE_PLAN{'_rep1' if rep_n else ''}.json", {"token": a.token, "primary": primary, "which": list(which), "N": 1, "source": "cache", "cache": str(a.cache),
                                                                   "legs": [l.to_json() for l in legs.values()], "comparisons": comps})
        run = _run_legs(a, out, legs, "_rep1" if rep_n else "")
    finally:
        lock_release(lock)
    rep = evaluate(out, comps, " [binding repeat]" if rep_n else "")
    v = verdict(rep)
    if not (run["all_done"] and run["untouched_equal"]):
        v.update({"PASS": False, "paired": False, "latency_4090": "UNPAIRED (leg failure or BEFORE/AFTER untouched-record difference)"})
    obj = {"primary": primary, "which": list(which), "run": run, "comparisons": rep, "verdict": v}
    if not rep_n:
        _record(out, "BRIDGE_RESULT.json", obj)
    else:
        _record(out, "BRIDGE_REPEAT_RESULT.json", obj)
        final = {"first_pass": first, "binding_repeat": {**obj, "verdict": v}, "verdict": v, "note": "both runs reported; the binding repeat decides"}
        _record(out, "BRIDGE_FINAL.json", final)
    C.write_sha256sums(out)
    print(f"{'PASS' if v['PASS'] else 'FAIL'} host={HOST} {v['latency_4090']}")
    return 0 if v["PASS"] else 1


def cmd_bridge_compare(a) -> int:
    plan_name = "BRIDGE_PLAN_rep1.json" if (a.out / "BRIDGE_PLAN_rep1.json").exists() and a.repeat_binding else "BRIDGE_PLAN.json"
    plan = json.loads((a.out / plan_name).read_text())
    rep = evaluate(a.out, plan["comparisons"])
    v = verdict(rep)
    _record(a.out, "BRIDGE_COMPARE.json", {"comparisons": rep, "verdict": v})
    for r in rep:
        print(("PASS " if r["PASS"] else "FAIL ") + f"{r['name']} ({r['gate']})")
    return 0 if v["PASS"] else 1


def add_bridge_parser(sp) -> None:
    b = sp.add_parser("bridge")
    bs = b.add_subparsers(dest="bridge_cmd", required=True)
    for name in ("plan", "run", "compare"):
        s = bs.add_parser(name)
        s.add_argument("--primary-file", type=Path, default=DEFAULT_PRIMARY_FILE)
        s.add_argument("--legs", default="B1,B2", help="B1,B2[,B3,B4]")
        s.add_argument("--cache", type=Path, default=C.DEFAULT_CACHE)
        s.add_argument("--out", type=Path, default=C.DEFAULT_OUT / "bridge")
        s.add_argument("--source", choices=["cache"], default="cache")
        s.add_argument("--token", default=None)
        s.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="GPU_LOCK file (must be empty to start)")
        s.add_argument("--repeat-binding", action="store_true", help="the single binding repeat after a failed first pass; records both runs")


def main_bridge(a) -> int:
    return {"plan": cmd_bridge_plan, "run": cmd_bridge_run, "compare": cmd_bridge_compare}[a.bridge_cmd](a)

"""NEW launcher for efficiency (b) (Amendment 1 / Add.29 (i)-(iv)). Runs the stack-increment legs of the plan as N FIXED worker processes (one leg per process, each with its
own output directory, no shared mutable state), reading the verified input cache; T6 (cudnn.benchmark) legs run ALONE afterwards. Never writes timing. A failed or OOM leg is
re-run and logged (failures.jsonl), never dropped; a leg that still fails makes the whole run FAIL (a missing row is a FAIL).

  launcher.py plan   --dry-run [--workers N]        print the leg table + estimated wall (planning only; no process is started, no GPU is touched)
  launcher.py run    --workers N --cache DIR --out DIR     (needs an Auditor token; not started by this file's author)
  launcher.py control --out DIR                      end-to-end control: ONE S0-fp32 RAMEN leg on the fly and from the cache, then compare depth + metric JSONs (bitwise)
  launcher.py compare --out DIR [--primary S3]       gate evaluation from finished legs (increments, noise floors, C32/C16)
Before and after every `run`/`control` the untouched records are written (eval clone HEAD+status, VDA HEAD+status, FocusMamba uv-freeze sha): BEFORE.json / AFTER.json."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.efficiency import common as C  # noqa: E402
from repro.efficiency import legs as L  # noqa: E402

PY = str(C.FOCUSMAMBA_VENV / "bin" / "python")
WORKER = str(Path(__file__).with_name("leg_worker.py"))
MAX_ATTEMPTS = 3

# gates (AUDITOR_CONDITIONS_EFFICIENCY_E1E3 (b)): fp32 = same-host R3, fp16 = cross-host set
GATES = {"fp32": {"macro_rel": 1e-4, "clip_rel": 1e-3, "psnr_abs": 0.005, "ssim_abs": 5e-5, "tof_rel": 1e-4},
         "fp16": {"macro_rel": 5e-3, "clip_rel": None, "psnr_abs": 0.02, "ssim_abs": 3e-4, "tof_rel": 1.5e-2}}
PSNR_KEYS = ("psnr", "psnr_native")
SSIM_KEYS = ("ssim", "ssim_native")
TOF_KEYS = ("tof", "tof_native")
MAIN_KEYS = ("abs_rel", "abs_rel_aligned", "d1", "delta1", "tae", "tae_gtvalid", "flicker", "abs_rel_static")


def gpu_other_procs(min_mb: int = 1024) -> list:
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True)
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e}"]
    return [l for l in o.strip().splitlines() if l and int(l.rsplit(",", 1)[1]) >= min_mb]


def leg_table(legs, N: int) -> str:
    rows = [f"{'id':26} {'model':6} {'stack':5} {'prec':5} {'rep':3} {'phase':7} {'agf':15} flags"]
    for l in legs:
        rows.append(f"{l.id:26} {l.model:6} {l.stack:5} {l.precision:5} {l.repeat:<3} {'ALONE' if l.alone else 'shared':7} {l.agf_backend:15} {','.join(l.flags) or '-'}")
    return "\n".join(rows)


def cmd_plan(a) -> int:
    legs = L.select_legs(L.build_legs(tuple(a.exclude), tuple(a.models)), a.legs)
    est = L.estimate(legs, a.workers, a.est_min_per_leg, a.cache_saving)
    print(leg_table(legs, a.workers))
    print(f"\nN (fixed) = {a.workers};  shared-phase legs {est['shared_phase_legs']}, alone-phase (T6) legs {est['alone_phase_legs']}")
    print(f"estimated wall (planning only, not a timing): {est['est_wall_h']} h  [{est['assumptions']}]")
    print("comparisons:", json.dumps([f"{c['kind']}:{c['test']}~{c['ref']}" for c in L.comparisons(legs, a.primary)]))
    if a.dry_run:
        print("--dry-run: nothing started")
    return 0


def start_leg(leg: L.Leg, out: Path, a, wid: int, source: str):
    d = out / "legs" / leg.id
    d.mkdir(parents=True, exist_ok=True)
    lj = d / "leg_spec.json"
    C.write_json(lj, leg.to_json())
    cmd = [PY, "-B", WORKER, "--leg-json", str(lj), "--out-dir", str(d), "--cache-dir", str(a.cache), "--source", source, "--worker-id", str(wid),
           "--n-workers", str(a.workers)] + (["--patches-dir", a.patches_dir] if a.patches_dir else []) + (["--save-depth"] if getattr(a, "save_depth", False) else [])
    log = open(d / "worker.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(C.SCORER_CLONE), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True", "EFF_GATE_DIR": str(out / ".gpu_gate")})   # fix 1 (eff_b): allocator-only, no numerics effect; pinned fp32 KITTI peak ~16 GiB OOMed with fragmentation next to the 2.9 GB desktop


def run_phase(legs, out: Path, a, n: int, source: str) -> dict:
    """At most n concurrent worker processes, one leg each. A leg that exits non-zero is re-queued (up to MAX_ATTEMPTS, logged); still failing -> FAIL."""
    queue = [(l, 0) for l in legs]
    running: dict = {}
    status: dict = {}
    wid_free = list(range(n))
    while queue or running:
        while queue and len(running) < n:
            l, att = queue.pop(0)
            wid = wid_free.pop(0)
            running[l.id] = (start_leg(l, out, a, wid, source), l, att, wid)
        time.sleep(5)
        for lid, (proc, l, att, wid) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[lid]
            wid_free.append(wid)
            ok = rc == 0 and (out / "legs" / lid / "DONE.json").exists()
            with open(out / "launcher_log.jsonl", "a") as f:
                f.write(json.dumps({"leg": lid, "attempt": att, "rc": rc, "ok": ok}) + "\n")
            if ok:
                status[lid] = "DONE"
            elif att + 1 < MAX_ATTEMPTS:
                (out / "legs" / lid / "FAILED.json").unlink(missing_ok=True)
                queue.append((l, att + 1))                      # re-run (resumes per clip); never dropped
            else:
                status[lid] = "FAIL"
    return status


def cmd_run(a) -> int:
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if not a.token:
        raise SystemExit("refusing to start scoring without --token (the Auditor token string, e.g. AUD-EFF-B-20261013)")
    vr = a.cache / "verify_report.json"
    if not vr.exists() or not json.loads(vr.read_text())["summary"]["PASS"]:
        raise SystemExit("input cache has no PASSing verify_report.json; route dropped for (b)")
    legs = L.select_legs(L.build_legs(tuple(a.exclude), tuple(a.models)), a.legs)
    C.write_json(out / "PLAN.json", {"N": a.workers, "token": a.token, "legs": [l.to_json() for l in legs], "exclude": a.exclude, "comparisons": L.comparisons(legs, a.primary)})
    C.write_json(out / "BEFORE.json", C.untouched_record("BEFORE"))
    status = {}
    shared = [l for l in legs if not l.alone]
    alone = [l for l in legs if l.alone]
    status.update(run_phase(shared, out, a, a.workers, a.source))
    if alone:
        busy = gpu_other_procs()
        if busy:
            raise SystemExit(f"T6 legs run alone on the card; other GPU processes present: {busy}")
        status.update(run_phase(alone, out, a, 1, a.source))
    C.write_json(out / "AFTER.json", C.untouched_record("AFTER"))
    C.write_json(out / "STATUS.json", status)
    C.write_sha256sums(out)
    return 0 if all(v == "DONE" for v in status.values()) and len(status) == len(legs) else 1


# ----------------------------------------------------------------------------- end-to-end control + compare
def load_leg(out: Path, lid: str) -> dict:
    d = out / "legs" / lid
    return {"macro": json.loads((d / "macro.json").read_text()), "shas": json.loads((d / "depth_shas.json").read_text()),
            "rows": list(csv.DictReader(open(d / "per_clip.csv"))), "toggles": json.loads((d / "toggles.json").read_text()), "dir": d}


def identical_control(fly: dict, cache: dict) -> dict:
    """Depth shas (and saved arrays when present) + per-clip metrics + macro must be IDENTICAL (bitwise on the serialised values)."""
    out = {"depth_sha_equal": {k: fly["shas"][k]["depth_sha256"] == cache["shas"].get(k, {}).get("depth_sha256") for k in fly["shas"]}}
    out["input_sha_equal"] = {k: fly["shas"][k]["input_sha256"] == cache["shas"].get(k, {}).get("input_sha256") for k in fly["shas"]}
    out["same_keys"] = set(fly["shas"]) == set(cache["shas"])
    out["metrics_rows_identical"] = fly["rows"] == cache["rows"] or [{k: v for k, v in r.items() if k != "leg"} for r in fly["rows"]] == [{k: v for k, v in r.items() if k != "leg"} for r in cache["rows"]]
    out["macro_identical"] = fly["macro"]["macro"] == cache["macro"]["macro"]
    eq = True
    for pf in sorted((fly["dir"] / "depth").rglob("*.npy")) if (fly["dir"] / "depth").exists() else []:
        pc = cache["dir"] / pf.relative_to(fly["dir"])
        eq &= pc.exists() and np.array_equal(np.load(pf), np.load(pc), equal_nan=True)
    out["saved_depth_arrays_equal"] = eq
    out["PASS"] = bool(out["same_keys"] and all(out["depth_sha_equal"].values()) and all(out["input_sha_equal"].values()) and out["metrics_rows_identical"] and out["macro_identical"] and eq)
    return out


def cmd_control(a) -> int:
    if not a.token:
        raise SystemExit("refusing to start scoring without --token")
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "BEFORE.json", C.untouched_record("BEFORE"))
    base = L.make_leg("ramen", "S0", "fp32", purpose="end-to-end control")
    st = {}
    for src in ("fly", "cache"):
        leg = L.Leg(**{**base.to_json(), "id": f"control-{src}"})
        a.save_depth = True
        st[src] = run_phase([leg], out, a, 1, src)             # sequential, same N=1 for both
    C.write_json(out / "AFTER.json", C.untouched_record("AFTER"))
    rep = identical_control(load_leg(out, "control-fly"), load_leg(out, "control-cache")) if all(v["control-" + k] == "DONE" for k, v in st.items()) else {"PASS": False, "status": st}
    C.write_json(out / "CONTROL_RESULT.json", rep)
    C.write_sha256sums(out)
    print(json.dumps({k: v for k, v in rep.items() if k in ("PASS", "same_keys", "metrics_rows_identical", "macro_identical", "saved_depth_arrays_equal")}))
    return 0 if rep["PASS"] else 1


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def compare_pair(test: dict, ref: dict, gate: str) -> dict:
    """Metric gates between two finished legs. gate: 'bitwise' (depth sha equal on every clip x cell), 'fp32', 'fp16', 'descriptive' (report only)."""
    res = {"missing_or_extra_rows": sorted(set(test["shas"]) ^ set(ref["shas"])), "nan_or_mask_change": [k for k in test["shas"] if k in ref["shas"] and (not test["shas"][k]["depth_all_finite"] or test["shas"][k]["depth_finite_mask_sha256"] != ref["shas"][k]["depth_finite_mask_sha256"])]}
    if gate == "bitwise":
        res["bitwise_depth_all_clips"] = all(test["shas"][k]["depth_sha256"] == ref["shas"][k]["depth_sha256"] for k in test["shas"] if k in ref["shas"])
    g = GATES.get(gate)
    worst = {"macro": {}, "clip": {}}
    fails = []
    tm, rm = test["macro"]["macro"], ref["macro"]["macro"]
    for unit in sorted(set(tm) & set(rm)):
        for k, v in tm[unit]["video_macro"].items():
            r = rm[unit]["video_macro"].get(k)
            if r is None or not np.isfinite(v) or not np.isfinite(r):
                continue
            if k in PSNR_KEYS or k in SSIM_KEYS:
                d, lim = abs(v - r), (g or {}).get("psnr_abs" if k in PSNR_KEYS else "ssim_abs")
            else:
                d, lim = _rel(v, r), (g or {}).get("tof_rel" if k in TOF_KEYS else "macro_rel")
            worst["macro"][k] = max(worst["macro"].get(k, 0.0), d)
            if g and lim is not None and d > lim:
                fails.append(f"macro {unit} {k} {d:.3g}>{lim}")
    if g and g["clip_rel"]:
        rr = {(r["dataset"], r["cell"], r["variant"], r["clip"]): r for r in ref["rows"]}
        for r in test["rows"]:
            q = rr.get((r["dataset"], r["cell"], r["variant"], r["clip"]))
            if not q:
                continue
            for k in MAIN_KEYS + TOF_KEYS:
                if k in r and r[k] not in ("", None) and q.get(k) not in ("", None):
                    d = _rel(float(r[k]), float(q[k]))
                    worst["clip"][k] = max(worst["clip"].get(k, 0.0), d)
                    if d > g["clip_rel"]:
                        fails.append(f"clip {r['dataset']} {r['cell']} {r['variant']} {r['clip']} {k} {d:.3g}>{g['clip_rel']}")
    res.update({"worst": worst, "gate": gate, "gate_fails": fails[:200], "n_gate_fails": len(fails)})
    res["PASS"] = bool(not res["missing_or_extra_rows"] and not res["nan_or_mask_change"] and res.get("bitwise_depth_all_clips", True) and (gate == "descriptive" or not fails))
    return res


def cmd_compare(a) -> int:
    plan = json.loads((a.out / "PLAN.json").read_text())
    legs = [L.Leg(**d) for d in plan["legs"]]
    report = []
    for c in L.comparisons(legs, a.primary):
        try:
            r = compare_pair(load_leg(a.out, c["test"]), load_leg(a.out, c["ref"]), c["gate"])
        except FileNotFoundError as e:
            r = {"PASS": False, "error": f"missing leg output: {e}"}
        report.append({**c, **r})
    C.write_json(a.out / "COMPARE.json", report)
    for r in report:
        print(("PASS " if r["PASS"] else "FAIL ") + f"{r['kind']:22} {r['test']} ~ {r['ref']} ({r['gate']})")
    return 0 if all(r["PASS"] or r["gate"] == "descriptive" for r in report) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "control", "compare"):
        s = sp.add_parser(name)
        s.add_argument("--workers", type=int, default=4, help="N, fixed and recorded before the first leg")
        s.add_argument("--cache", type=Path, default=C.DEFAULT_CACHE)
        s.add_argument("--out", type=Path, default=C.DEFAULT_OUT)
        s.add_argument("--source", choices=["cache", "fly"], default="cache")
        s.add_argument("--exclude", nargs="*", default=[], help="toggle ids removed from all stacks (bisect result), e.g. T8 T7")
        s.add_argument("--models", nargs="+", default=["ramen", "mvda"], choices=["ramen", "mvda"])
        s.add_argument("--primary", default="S3")
        s.add_argument("--legs", default=None, help="leg-subset filter: comma-separated leg ids (e.g. ramen-S0-fp16,ramen-S4-fp16); default all")
        s.add_argument("--patches-dir", default=None)
        s.add_argument("--token", default=None)
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--est-min-per-leg", type=float, default=42.0)
        s.add_argument("--cache-saving", type=float, default=0.30)
    from repro.efficiency import bridge as BR     # bridge subcommands (commit BR): scheduling/compare only
    BR.add_bridge_parser(sp)
    a = ap.parse_args(argv)
    if a.cmd == "bridge":
        return BR.main_bridge(a)
    if a.cmd == "run" and a.dry_run:
        a.cmd = "plan"
    return {"plan": cmd_plan, "run": cmd_run, "control": cmd_control, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

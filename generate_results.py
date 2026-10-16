#!/usr/bin/env python3
"""One command -> every paper number and figure.

    python generate_results.py                         # all items -> paper_results/
    python generate_results.py --dry-run               # plan + GPU-time estimate, runs nothing
    python generate_results.py --only tab:static fig:qual
    python generate_results.py --smoke                 # 1 clip per dataset -> paper_results_smoke/
    python generate_results.py --verify <ref_dir>      # compare paper_results/ with a reference (previous run or released results_v3 layout)

Config: configs/paper_results.yaml (schema repro/SCHEMA.md). The scorer (bench_joint.py, metrics*.py, evaluation/) is vendored unchanged and only shelled out to.
Layout of the output folder: <item>.csv/.json, per_clip/ (raw scorer JSON), qualitative/, MANIFEST.json (+ verify_report.json after --verify).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from repro import aggregate as A  # noqa: E402
from repro import config as C  # noqa: E402
from repro import execute as X  # noqa: E402
from repro import manifest as M  # noqa: E402
from repro import plan as P  # noqa: E402
from repro import readings as R  # noqa: E402
from repro import verify as V  # noqa: E402
from repro.util import item_stem, jdump, sha256_file, sha_matches  # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "paper_results.yaml"
SCORED = ("bench", "night", "fnoise", "latency")


def build_results(cfg: dict, items: list[str], units: list[P.Unit], out: Path) -> dict[str, dict]:
    """Aggregate whatever result files exist under out/per_clip for each scored item."""
    res = {}
    for iid in items:
        it = cfg["items"][iid]
        if it["kind"] not in SCORED:
            continue
        us = [u for u in units if iid in u.items]
        got = A.collect_item(it, us, out)
        if it["kind"] == "latency":
            it = {**it, "metrics": it.get("metrics") or ["per_frame_ms_p50", "per_frame_ms_p95"]}
        res[iid] = A.aggregate_item(iid, it, got)
        res[iid]["expected_rows"] = len(it["harness"]["rows"]) if it.get("harness") else len(us)
        R.apply_reading(it, res[iid])
        if it.get("host_slots"):                            # MacBook slot: reserved, never measured here
            res[iid]["host_slots"] = it["host_slots"]
            for r in res[iid]["rows"]:
                r["host_slots"] = {k: v["value"] for k, v in it["host_slots"].items()}                       # pre-declared verdicts (tab:clean, tab:static_diag), computed, never hand-written
        if it["kind"] == "latency":                         # every timed pass of every row must be stored (30 in the protocol run)
            res[iid]["samples_complete"] = bool(res[iid]["rows"]) and all(_n_samples(r.get("samples")) == r.get("protocol_iters") for r in res[iid]["rows"])
    return res


def _n_samples(sm):
    """Per-row timed samples: all runs of a bracket row have `iters` each; None when not stored."""
    if not sm:
        return None
    if "per_run_n_samples" in sm:
        return sm["per_run_n_samples"][0] if len(set(sm["per_run_n_samples"])) == 1 else -1
    return sm.get("n_samples")


def run_qualitative(cfg: dict, iid: str, out: Path, smoke: bool, scorer_clone: Path | None = None, tag_backend: str = "reference") -> dict:
    stem = item_stem(iid)
    try:
        import repro.qualitative as Q
    except ImportError as e:
        return {"status": "skipped", "why": f"repro.qualitative not importable: {e}"}
    fn = getattr(Q, f"render_{stem}", None)
    if fn is None:
        return {"status": "skipped", "why": f"repro.qualitative.render_{stem} not defined"}
    d = Path(out) / "qualitative" / stem
    d.mkdir(parents=True, exist_ok=True)
    it0 = cfg["items"][iid]
    entry = {**it0, **({"params": {**(it0.get("params") or {}), **it0["smoke_params"]}} if smoke and it0.get("smoke_params") else {}), "_id": iid, "_smoke": smoke, "_scorer_clone": str(scorer_clone) if scorer_clone else None, "_tag_backend": tag_backend, "_models": {m: C.model_spec(cfg, m) for m in cfg["items"][iid].get("models", [])}}
    try:
        files = fn(entry, d)
    except Exception as e:  # noqa: BLE001 - recorded, run continues
        return {"status": "failed", "why": repr(e)}
    return {"status": "ok", **({"label": it0["label"]} if it0.get("label") else {}), "files": [str(Path(f).relative_to(out)) if Path(f).is_absolute() else str(f) for f in (files or [])]}


HOST_LABEL = {"gh200": "Isambard-AI GH200", "4090": "one RTX 4090"}


def run_label(host: str, commit: str, here: str | None, smoke: bool) -> str:
    """Per-host run label (Auditor ruling 2): accuracy tables on the GH200, clean/p1/latency on the 4090; never mixed in one table."""
    h = here if host == "any" else host
    where = HOST_LABEL.get(h or "", "an unidentified host")
    jobs = os.environ.get("SLURM_JOB_ID")
    jobtxt = f" (Slurm job(s) {jobs}, {time.strftime('%Y-%m-%d', time.gmtime())})" if h == "gh200" else f" ({time.strftime('%Y-%m-%d', time.gmtime())})"
    return f"regenerated with generate_results.py at commit {commit[:8]} on {where}{jobtxt}" + ("; SMOKE: 1 clip per dataset, not a result" if smoke else "")


def detect_host() -> str | None:
    g = (M.gpu_name() or "").lower()
    return "4090" if "4090" in g else ("gh200" if "gh200" in g else None)


def qual_present(cfg: dict, items: list[str]) -> dict:
    try:
        import repro.qualitative as Q
    except ImportError:
        return {i: False for i in P.qualitative_items(cfg, items)}
    return {i: hasattr(Q, f"render_{item_stem(i)}") for i in P.qualitative_items(cfg, items)}


def do_verify(cfg: dict, items: list[str], units: list[P.Unit], out: Path, ref: Path | None, same_host: bool) -> int:
    """ref=None -> each item's configured `reference` (per model); ref given -> that dir for every row."""
    cand = build_results(cfg, items, units, out)
    report, worst = {}, 0
    for iid, c in cand.items():
        it = cfg["items"][iid]
        stem = item_stem(iid)
        us = [u for u in units if iid in u.items]
        if it.get("by_hand_reference"):                                # released P1 read: a table, not bench JSONs -> generated cross-host comparison table
            bh = it["by_hand_reference"]
            f = C.resolve_path(cfg, bh["file"])
            if not f.is_file():
                report[iid] = {"status": "SKIP", "why": f"by-hand reference {f} not on this host (run --verify on the host that holds it)", "fails": [], "flags": [], "n_compared": 0}
                print(f"verify {iid:<20} SKIP  (by-hand reference not on this host)")
                continue
            if not sha_matches(sha256_file(f), bh["sha256"]):
                report[iid] = {"status": "FAIL", "why": f"{f.name} sha256 differs from the pin", "fails": [{"row": [iid], "why": "reference sha mismatch"}], "flags": [], "n_compared": 0}
                print(f"verify {iid:<20} FAIL  (reference sha mismatch)")
                worst = 2
                continue
            rep = V.compare_p1(c, V.parse_p1_table(f.read_text()), False)             # always the cross-host class (regenerated on the GH200, released on the 4090)
            report[iid] = {**rep, "reference": str(f), "reference_sha256": bh["sha256"], "class": "cross-host (by-hand reference)"}
            import csv as _csv
            with open(out / f"{item_stem(iid)}_vs_released_crosshost.csv", "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["variant", "dataset", "arm", "metric", "cell", "regenerated", "released_P1_TABLE", "rel_dev", "within_crosshost_tol"])
                for r_ in rep["rows"]:
                    w.writerow([r_["variant"], r_["dataset"], r_["arm"], r_["metric"], r_["cell"], r_["cand"], r_["released"], r_["dev"], r_["ok"]])
            worst = max(worst, {"PASS": 0, "FLAG": 1, "FAIL": 2}[rep["status"]])
            print(f"verify {iid:<20} {rep['status']:<5} compared={rep['n_compared']} fails={len(rep['fails'])}  (cross-host, by-hand reference {f.name})")
            continue
        if it.get("harness"):                                          # latency ladder: the released session dir (rows/<stack>__<id>.json); flag >10%, never a fail
            h = P.ladder_harness(cfg, iid)
            rd = ref if ref is not None else C.reference_dir(cfg, iid, "")
            got = A.ladder_rows(rd, h["stack"], h["rows"]) if rd is not None else {}
            if not got:
                report[iid] = {"status": "SKIP", "why": "no released ladder reference configured / found", "fails": [], "flags": [], "n_compared": 0}
                print(f"verify {iid:<20} SKIP  (no reference)")
                continue
            r = A.aggregate_item(iid, {**it, "metrics": it.get("metrics") or c["metrics"]}, got)
            rep = V.compare_results(c, r, same_host, kind="latency")
            report[iid] = rep
            worst = max(worst, {"PASS": 0, "FLAG": 1, "FAIL": 2}[rep["status"]])
            print(f"verify {iid:<20} {rep['status']:<5} compared={rep['n_compared']} fails={len(rep['fails'])} flags={len(rep['flags'])}  (reference {rd})")
            for f in rep["fails"][:5]:
                print("    FAIL", f)
            for f in rep["flags"][:5]:
                print("    FLAG", f)
            continue
        if ref is not None and (ref / f"{stem}.json").exists():
            r = json.loads((ref / f"{stem}.json").read_text())
        else:
            got = {}
            for mid in dict.fromkeys(u.model_id for u in us):
                rd = ref if ref is not None else C.reference_dir(cfg, iid, mid)
                if rd is None:
                    continue
                got.update(V.ref_rows_from_released(rd, [u for u in us if u.model_id == mid], {mid: C.reference_spec(cfg, mid)}, it))
            if not got:
                report[iid] = {"status": "SKIP", "why": "no reference configured / found", "fails": [], "flags": [], "n_compared": 0}
                print(f"verify {iid:<20} SKIP  (no reference)")
                continue
            r = A.aggregate_item(iid, {**it, "metrics": it.get("metrics") or c["metrics"]}, got)
            have = {x["model"] for x in r["rows"] + r["mean"]}
            c = {**c, "rows": [x for x in c["rows"] if x["model"] in have], "mean": [x for x in c["mean"] if x["model"] in have]}
        rep = V.compare_results(c, r, same_host and it.get("verify_class") != "cross_host", kind="latency" if it["kind"] == "latency" else "bench")
        report[iid] = rep
        worst = max(worst, {"PASS": 0, "FLAG": 1, "FAIL": 2}[rep["status"]])
        print(f"verify {iid:<20} {rep['status']:<5} compared={rep['n_compared']} fails={len(rep['fails'])} flags={len(rep['flags'])}")
        for f in rep["fails"][:5]:
            print("    FAIL", f)
        for f in rep["flags"][:5]:
            print("    FLAG", f)
    for iid in items:
        it = cfg["items"][iid]
        if it["kind"] != "qualitative" or not isinstance(it.get("reference"), str):
            continue
        rd = ref if ref is not None else C.reference_dir(cfg, iid, "")
        if rd is None or not (rd / "SHA256SUMS").exists():
            continue
        rep = V.compare_sha_list(rd, out / "qualitative" / item_stem(iid))
        report[iid] = rep
        worst = max(worst, 2 if rep["status"] == "FAIL" else 0)
        print(f"verify {iid:<20} {rep['status']:<5} ref files={rep['n_ref']} byte-identical={len(rep['byte_identical'])} differ={len(rep['bytes_differ'])} missing={len(rep['missing'])}")
    jdump({"reference": str(ref) if ref else "per-item config reference", "same_host": same_host, "items": report}, out / "verify_report.json")
    print({0: "VERIFY PASS", 1: "VERIFY FLAG (review)", 2: "VERIFY FAIL"}[worst])
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None, help="default paper_results/ (paper_results_smoke/ with --smoke)")
    ap.add_argument("--only", nargs="+", default=None, help="item ids, e.g. tab:static fig:qual (tab_static also accepted)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 clip per dataset (recorded in every result; never resumes a full run)")
    ap.add_argument("--verify", metavar="REF_DIR", nargs="?", const="config", default=None, help="compare with REF_DIR (a prior paper_results/ or the released results_v3 layout); no value = each item's configured reference(s)")
    ap.add_argument("--tag-backend", choices=C.TAG_BACKENDS, default=None, help="reference (numpy, every released number) | fast (device TAG; exploratory, cross-host tolerances, planned on the 4090)")
    ap.add_argument("--host", choices=("auto", "gh200", "4090", "any"), default="auto", help="run only units planned for this host (auto = detect from nvidia-smi; no GPU detected = run all)")
    ap.add_argument("--scorer-clone", default=os.environ.get("RAMEN_SCORER_CLONE"), help="where the pristine scorer-commit clone for vendored wrappers/figures lives (default ~/.cache/ramen_scorer/<commit8>)")
    ap.add_argument("--same-host", action="store_true", help="--verify with the tight same-host tolerances")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--data-root", default=None, help="local DATA_ROOT override (4090 / local); also disables the GH200 per-key map. Default: host_paths.DATA_ROOT = /mnt/DATA/DegradedDepth")
    ap.add_argument("--allow-test", action="store_true", help="required (with RAMEN_AUDITOR_TOKEN) for any unit that reads TEST clips: labelled 'exposed exploratory'")
    ap.add_argument("--bvi-root", default=os.environ.get("BVI_ROOT"), help="BVI-RLV frames root (default: host_paths.bvi_root resolved through the GH200 map on a GH200, else under DATA_ROOT)")
    ap.add_argument("--skip-hash", action="store_true", help="do not sha256 checkpoints (existence only)")
    ap.add_argument("--plan-json", metavar="PATH", default=None, help="write the plan (units, hosts, labels, estimates, commands) as JSON for the Auditor package; implies --dry-run")
    ap.add_argument("--aggregate-only", action="store_true", help="skip running units; aggregate what is already in --out")
    a = ap.parse_args(argv)

    if a.data_root:
        os.environ["DATA_ROOT"] = a.data_root
    out = Path(a.out or ("paper_results_smoke" if a.smoke else "paper_results")).resolve()
    cfg = C.load(a.config)
    if not a.bvi_root and (cfg.get("host_paths") or {}).get("bvi_root"):
        try:
            a.bvi_root = str(C.resolve_path(cfg, cfg["host_paths"]["bvi_root"]))          # default from the (GH200) map, else DATA_ROOT
        except C.UnmappedPath as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 3
    frozen = M.frozen_state(Path(a.config))
    real_run = not (a.dry_run or a.plan_json or a.verify)
    if real_run and not frozen["ok"]:
        print("REFUSED: the config is not a frozen one: " + "; ".join(frozen["header_problems"] + ([] if frozen["tracked"] else ["config is not tracked by git"]) + (["config is modified vs HEAD"] if frozen["modified"] else [])
                                                           + ([frozen["why"]] if frozen.get("why") else [])) +
              "\n  The freeze is the commit itself: run from the committed config whose header says 'FROZEN (this commit)'.", file=sys.stderr)
        return 3
    pstats: dict = {}
    try:
        units, items = P.plan(cfg, a.only, smoke=a.smoke, tag_backend=a.tag_backend, stats=pstats)
    except C.UnmappedPath as e:                                        # a unit whose path key is unmapped on the GH200 is refused
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    for note in (pstats.get("conditions") or {}).values():
        print(note, file=sys.stderr)
    tb_eff = a.tag_backend or cfg.get("tag_backend", "reference")
    if real_run and any(u.tag_backend == "fast" and u.kind != "latency" for u in units):
        print("REFUSED: --tag-backend fast is for latency rows and the bridge only; every table value uses the reference (numpy) TAG backend.", file=sys.stderr)
        return 3
    if real_run and a.skip_hash and not a.smoke:
        print("REFUSED: --skip-hash is only allowed with --smoke (every checkpoint is re-hashed against its pin before scoring).", file=sys.stderr)
        return 3
    started = time.time()

    if a.dry_run or a.plan_json:
        print(P.format_plan(units, items, cfg, out, smoke=a.smoke, qual_present=qual_present(cfg, items), stats=pstats))
        _, problems = X.check_inputs(cfg, units, hash_files=False)
        for p in problems + X.verify_vendored():
            print("  MISSING INPUT:", p)
        gi0 = M.git_state(out)
        print(f"\nfreeze: header {'OK (FROZEN (this commit))' if not frozen['header_problems'] else 'NOT frozen: ' + '; '.join(frozen['header_problems'])}; config {'tracked, unmodified at HEAD' if frozen['ok'] or (frozen['tracked'] and not frozen['modified']) else 'tracked=%s modified=%s' % (frozen['tracked'], frozen['modified'])}; commit {frozen['commit'][:8]}; data_root {C.data_root(cfg)}; path resolution {'GH200 per-key map' if C.gh200_map_active(cfg) else 'DATA_ROOT'}")
        for iid, note in (pstats.get("conditions") or {}).items():
            print(f"condition: {note}")
        hosts_by_item = {i: sorted({u.host for u in units if i in u.items}) for i in items if cfg["items"][i]["kind"] != "qualitative"}
        print("run labels (one host per table): " + "; ".join(f"{i}={'/'.join(h)}" for i, h in hosts_by_item.items()))
        if a.plan_json:
            import hashlib
            doc = P.plan_json(units, items, cfg, out, a.smoke, pstats, {"git": gi0, "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "tag_backend": a.tag_backend or cfg.get("tag_backend", "reference"),
                                                                       "missing_inputs": problems})
            jdump(doc, Path(a.plan_json))
            print(f"\nplan written to {a.plan_json}")
        print(f"\ngit: {gi0['commit'][:8]} {'DIRTY (a real run needs --allow-dirty)' if gi0['dirty'] else 'clean'}; tag_backend: {a.tag_backend or cfg.get('tag_backend', 'reference')}")
        g = P.test_gate(units, a.allow_test, os.environ)
        print(f"\nTEST gate: {g or 'open'}")
        print("env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        return 0

    if a.verify:
        return do_verify(cfg, items, units, out, None if a.verify == "config" else Path(a.verify), a.same_host)

    gi = M.git_state(out)
    if gi["dirty"] and not a.allow_dirty:
        print("REFUSED: the git working tree is dirty, so the MANIFEST commit would not describe the code that produced the numbers.\n"
              "  Commit or stash these changes (or pass --allow-dirty, which is recorded in MANIFEST.json and is not acceptable for the submission run):\n  " + "\n  ".join(gi["dirty_files"][:20]), file=sys.stderr)
        return 3
    gate = P.test_gate(units, a.allow_test, os.environ)
    if gate and not a.aggregate_only:
        print(f"REFUSED: {gate}", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)

    ckinfo, problems = ({}, [])
    scorer_clone = None
    if not a.aggregate_only:
        vp = X.verify_vendored()
        if vp:
            print("REFUSED: vendored wrapper scripts differ from the Auditor-cleared originals:\n  " + "\n  ".join(vp), file=sys.stderr)
            return 3
        if any(u.arm or u.runner for u in units) or P.qualitative_items(cfg, items):
            try:
                scorer_clone = X.ensure_scorer_clone(Path(a.scorer_clone or Path.home() / ".cache" / "ramen_scorer" / cfg.get("scorer_commit", C.DEFAULT_SCORER_COMMIT)[:8]), cfg.get("scorer_commit", C.DEFAULT_SCORER_COMMIT))
            except RuntimeError as e:
                print(f"REFUSED: cannot set up the scorer clone: {e}", file=sys.stderr)
                return 3
    here = detect_host() if a.host == "auto" else (None if a.host == "any" else a.host)
    if not a.aggregate_only:
        ckinfo, problems = X.check_inputs(cfg, units, hash_files=not a.skip_hash)
        mism = X.mismatches(ckinfo)
        if mism:
            print("ABORT: checkpoint sha256 differs from the config pin (no exemption in smoke, resume or skip):\n  " + "\n  ".join(mism), file=sys.stderr)
            return 3
        todo = [u for u in units if not X.finished(u, out, a.smoke)[0]]
        needed = {u.model_id for u in todo}
        blocking = [p for p in problems if p.split(":")[0] in needed]
        if a.smoke and blocking:                                      # smoke on a small box: units whose checkpoint is not local are skipped, not fatal (e.g. FT n47/n48 live on Isambard)
            miss = {p.split(":")[0] for p in blocking}
            print("smoke: skipping units of models with a missing checkpoint: " + ", ".join(sorted(miss)), file=sys.stderr)
            units = [u for u in units if u.model_id not in miss]
            blocking = []
        if blocking:
            print("REFUSED: inputs missing/mismatched for units still to run:\n  " + "\n  ".join(blocking), file=sys.stderr)
            return 3

    actual_clip_sha = {}
    for d, dc in cfg["datasets"].items():
        if dc.get("clip_list"):
            f = C.resolve_path(cfg, dc["clip_list"])
            if f.is_file():
                actual_clip_sha[d] = sha256_file(f)
                if not a.aggregate_only and dc.get("clip_list_sha256") and not sha_matches(actual_clip_sha[d], dc["clip_list_sha256"]):
                    print(f"ABORT: clip list of {d} ({f}) sha256 {actual_clip_sha[d][:12]} != pinned {dc['clip_list_sha256'][:12]}", file=sys.stderr)
                    return 3
    bvi_rec = {"clip_list_sha256_pinned": cfg["datasets"].get("bvi_rlv_test", {}).get("clip_list_sha256"), "clip_list_sha256_actual": sha256_file(P.REGISTRATION) if P.REGISTRATION.is_file() else None, "per_file_sha256": None}
    if any(u.kind == "fnoise" for u in units) and not a.aggregate_only:
        if not a.bvi_root:
            print("REFUSED: BVI-RLV rows are planned: pass --bvi-root (or $BVI_ROOT) so every frame is hashed into the MANIFEST.", file=sys.stderr)
            return 3
        if not sha_matches(bvi_rec["clip_list_sha256_actual"] or "", bvi_rec["clip_list_sha256_pinned"] or "x"):
            print("ABORT: BVI registration (clip list) sha256 differs from the pin.", file=sys.stderr)
            return 3
        bvi_rec["per_file_sha256"] = M.bvi_file_shas(Path(a.bvi_root), P.REGISTRATION)

    status: dict = {}
    nfail = 0
    if not a.aggregate_only:
        for i, u in enumerate(units, 1):
            done, why = X.finished(u, out, a.smoke)
            if u.run_with:
                status[u.key] = "resumed" if u.out_path(out).exists() else f"grouped(with {u.run_with[1].split('/')[-1][:40]})"
                continue
            if done and not u.derive_from:
                status[u.key] = why
                continue
            if u.derive_from:
                status[u.key] = "derived" if done else "base_missing"
                continue
            if u.host == "mac":                                          # MacBook slot: no executor exists; the table shows 'not measured'
                status[u.key] = "not measured (host mac: no executor)"
                continue
            if here and u.host not in ("any", here):
                status[u.key] = f"skipped(planned for {u.host}, this host is {here})"
                continue
            print(f"[{i}/{len(units)}] {u.host} {u.kind} {u.label()}", flush=True)
            status[u.key] = X.run_unit(u, out, a.smoke, a.allow_test, a.bvi_root, out / "logs", scorer_clone)
            if status[u.key].startswith("failed"):
                nfail += 1
                print("   ", status[u.key], file=sys.stderr)
            else:
                art = json.loads(u.out_path(out).read_text())
                for w in X.check_result_pins(cfg, u, art):
                    print("   WARN pin mismatch:", w, file=sys.stderr)

    results = build_results(cfg, items, units, out)
    written = []
    run_labels = {}
    for iid in items:
        hs = sorted({u.host for u in units if iid in u.items}) or [P.unit_host(cfg["items"][iid], a.smoke, C.tag_backend(cfg, iid, a.tag_backend))]
        run_labels[iid] = run_label(hs[0], gi["commit"], here, a.smoke)
    for iid, r in results.items():
        bad = [[x["model"], x["dataset"], x["cell"], x["macro_mismatch"]] for x in r["rows"] if x.get("macro_mismatch")]
        if bad:                                                              # stored macro != per-clip recomputation: the item is not produced correctly (Auditor: macro_mismatch must FAIL)
            nfail += 1
            print(f"FAIL {iid}: macro_mismatch on {len(bad)} row(s): {bad[:3]}", file=sys.stderr)
        r["macro_mismatch_rows"] = bad
        r["run_label"] = run_labels[iid]
        r["host"] = sorted({u.host for u in units if iid in u.items})
        if r["expected_rows"] and len(r["rows"]) < r["expected_rows"]:
            print(f"WARN {iid}: {len(r['rows'])}/{r['expected_rows']} rows available", file=sys.stderr)
        written += A.write_item(r, out, item_stem(iid))
    qual = {}
    for iid in items:
        if cfg["items"][iid]["kind"] == "qualitative":
            qh = P.unit_host(cfg["items"][iid], a.smoke, C.tag_backend(cfg, iid, a.tag_backend))
            if here and qh not in ("any", here):
                qual[iid] = {"status": "skipped", "why": f"planned for {qh}, this host is {here}"}
                continue
            qual[iid] = run_qualitative(cfg, iid, out, a.smoke, scorer_clone, C.tag_backend(cfg, iid, a.tag_backend))
            print(f"qualitative {iid}: {qual[iid]['status']}")
    clip_lists = {d: {"clip_list_sha256": cfg["datasets"][d].get("clip_list_sha256"), "manifest_sha256": cfg["datasets"][d].get("manifest_sha256"), "clip_list_actual_sha256": actual_clip_sha.get(d)} for d in cfg["datasets"]}
    jdump({"qualitative": qual}, out / "qualitative_status.json")
    mp = M.write_manifest(out, cfg_path=Path(a.config), items=items, units=units, unit_status=status, checkpoints=ckinfo, clip_lists=clip_lists,
                          git_info=gi, started=started, smoke=a.smoke, allow_test=a.allow_test, argv=sys.argv[1:] if argv is None else argv,
                          extra={"frozen": {**frozen, "freeze": "the commit itself (header: FROZEN (this commit)); MANIFEST.git.commit + config.sha256 identify the frozen bytes"},
                                 "data_root": str(C.data_root(cfg)), "path_resolution": "gh200 per-key map (host_paths.gh200)" if C.gh200_map_active(cfg) else "DATA_ROOT", "tag_backend": tb_eff, "run_labels": run_labels, "conditional_items": pstats.get("conditions") or {},
                                 "bvi": bvi_rec, "latency_samples_complete": {i: r.get("samples_complete") for i, r in results.items() if cfg["items"][i]["kind"] == "latency"}})
    print(f"wrote {len(written)} files + {mp}; failed units: {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())

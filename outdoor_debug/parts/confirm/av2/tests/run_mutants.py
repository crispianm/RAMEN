#!/usr/bin/env python3
"""Mutation proof for the AV2 pipeline tests. Copies the repo worktree to a scratch dir (no .git), applies ONE textual mutation to a source file (the old text must occur exactly once),
runs the av2 tests + tests/test_sky_mask.py with the given python, and records which tests FAIL. A mutant that leaves everything green is reported SURVIVED.
usage: run_mutants.py REPO_ROOT SCRATCH_DIR PYTHON OUT_MD"""
import re, shutil, subprocess, sys
from pathlib import Path

AV2 = "outdoor_debug/parts/confirm/av2/"
SEG = "outdoor_debug/parts/confirm/seg/"
# (id, review-ref, file, old, new, description)
MUTANTS = [
    ("N1", "NOT caught: drop rule applied to the guards in scorer main", AV2 + "av2_scorer.py",
     "    score_frame_guard(acc, pred, gt, log, any_sky)\n    return dropped", "    if not dropped:\n        score_frame_guard(acc, pred, gt, log, any_sky)\n    return dropped", "route_frame skips the guards for a mask-dropped frame"),
    ("N2", "NOT caught: mask drop ignored for P in scorer main", AV2 + "av2_scorer.py",
     "    if not dropped:\n        sky.add(pred, certain, fid)", "    sky.add(pred, certain, fid)", "route_frame feeds a mask-dropped frame to P"),
    ("N3", "NOT caught: check_mask_meta call removed from main", AV2 + "av2_scorer.py",
     "    check_mask_meta(mmeta)\n    fs = A.sha256_file(frames_path)", "    fs = A.sha256_file(frames_path)", "load_mask_meta (called by main) no longer calls check_mask_meta"),
    ("N4", "NOT caught: scorer meta IoU threshold 0.85 -> 0.9", AV2 + "av2_scorer.py", "IOU_MIN = 0.85  ", "IOU_MIN = 0.90  ", "scorer IOU_MIN 0.9"),
    ("N5", "NOT caught: scorer meta IoU `<` -> `<=`", AV2 + "av2_scorer.py", "    if iou < IOU_MIN:", "    if iou <= IOU_MIN:", "boundary exclusive"),
    ("N6", "NOT caught: IoU pooled over non-dropped frames only", AV2 + "build_av2_masks.py",
     'for r in results.values()])\n    kept', 'for r in results.values() if not r["dropped"]])\n    kept', "assemble pools IoU after the drop"),
    ("N7", "NOT caught: certain requires only segmenter A", SEG + "make_sky_mask.py",
     "    m = (p_a >= thresh) & (p_b >= thresh) if require_both else (p_a >= thresh)", "    m = (p_a >= thresh)", "certain_sky ignores B"),
    ("N8", "NOT caught: p threshold 0.9 -> 0.8", SEG + "make_sky_mask.py", "PROB_THRESH = 0.9\n", "PROB_THRESH = 0.8\n", "threshold 0.8"),
    ("N9", "NOT caught: `any` without its p>=0.5 clause", SEG + "make_sky_mask.py", "    return (margin >= 0) | (p >= any_p)", "    return (margin >= 0)", "any_sky drops the p clause"),
    ("N10a", "NOT caught: pairing `<=` -> `<` (dataset rule)", AV2 + "av2_dataset.py", "        if dt <= MAX_PAIR_DT_NS:", "        if dt < MAX_PAIR_DT_NS:", "exclusive 25 ms in av2_dataset._select_phase"),
    ("N10b", "NOT caught: pairing `<=` -> `<` (stdlib selector)", AV2 + "select_av2_frames.py", "        if dt <= MAX_PAIR_DT_NS:", "        if dt < MAX_PAIR_DT_NS:", "exclusive 25 ms in select_av2_frames._phase"),
    ("N11", "make_sky_mask IOU_MIN 0.85 -> 0.9", SEG + "make_sky_mask.py", "IOU_MIN = 0.85  ", "IOU_MIN = 0.90  ", "builder gate 0.9"),
    ("D1", "D1: scorer-sha check disabled", AV2 + "av2_scorer.py", "    if got != expected:\n        raise AssertionError(f\"scorer sha", "    if False:\n        raise AssertionError(f\"scorer sha", "check_scorer_sha never fails"),
    ("D2", "D2: clip from a directory glob instead of the clip table (old behaviour)", AV2 + "av2_dataset.py", "        clip = clips[log][1]\n",
     "        clip = sorted(int(p.stem) for p in (Path(root) / log / 'sensors' / 'cameras' / CAMERA).glob('*.jpg'))[0::2][:MAX_LEN]\n", "clip taken from the data dir"),
    ("D4a", "D4: mask meta frames_sha compare removed", AV2 + "av2_scorer.py", '    if mmeta.get("frames_sha") != fs:', "    if False:", "frames_sha not compared"),
    ("D4b", "D4: meta_sha not read from the file", AV2 + "av2_scorer.py", "meta_sha=meta_sha, dataset_module_sha", 'meta_sha=PINS["meta_sha"], dataset_module_sha', "meta pin tautological"),
    ("D8a", "D8: NaN accepted", AV2 + "av2_scorer.py", "    if math.isnan(iou):", "    if False:", "check_mask_meta accepts NaN"),
    ("D8b", "D8: bool accepted", AV2 + "av2_scorer.py", "    if isinstance(iou, bool) or not isinstance(iou, (int, float)):", "    if not isinstance(iou, (int, float)):", "check_mask_meta accepts True"),
    ("D8c", "D8: out-of-range / inf accepted", AV2 + "av2_scorer.py", "    if iou < 0.0 or iou > 1.0:", "    if False:", "check_mask_meta accepts inf and 1.0000001"),
    ("R1", "O45: runner accepts a placeholder / non-hex sha", AV2 + "run_score.py", '    if not re.fullmatch(r"[0-9a-f]{64}", m[0]):', "    if False:", "run_score passes the placeholder on"),
    ("R2", "O45: runner passes no --expect-scorer-sha", AV2 + "run_score.py", ', "--expect-scorer-sha", sha]', "]", "scorer gets no expected sha"),
    ("D9a", "D9: safetensors presence not refused", AV2 + "build_av2_masks.py", "        if st:\n", "        if False:\n", "safetensors allowed"),
    ("D9b", "D9: use_safetensors not forwarded", SEG + "make_sky_mask.py", "kw = {} if use_safetensors is None else dict(use_safetensors=use_safetensors)", "kw = {}", "_Seg ignores use_safetensors"),
    ("D9c", "D9: weights sha not verified", AV2 + "build_av2_masks.py", "    if h != expected:", "    if False:", "verify_seg_sha never fails"),
    ("D10", "D10: Stage 2 printed with the Stage 1 note", AV2 + "av2_scorer.py", "disclosure=STAGE1_NOTE if stage == 1 else STAGE2_NOTE, d_ln", "disclosure=STAGE1_NOTE, d_ln", "compare() always prints STAGE1_NOTE"),
    ("D11a", "D11: argument order not checked", AV2 + "av2_scorer.py", '    if h["meta"].get("arm") != "H300" or t["meta"].get("arm") != "TS2":', "    if False:", "swapped arms accepted"),
    ("D11b", "D11: same checkpoint accepted", AV2 + "av2_scorer.py", '    if not h["meta"].get("ckpt_sha") or h["meta"]["ckpt_sha"] == t["meta"].get("ckpt_sha"):', "    if False:", "equal ckpt sha accepted"),
    ("D11c", "D11: kept frame ids / px counts not compared", AV2 + "av2_scorer.py", '    for k in ("kept_frame_ids", "dropped_frame_ids", "frame_mask_px"):', "    for k in ():", "frame sets not compared"),
    ("D12a", "D12: exclusion capped at 200 m", AV2 + "av2_dataset.py", "    ok = front & np.isfinite(u) & np.isfinite(v)\n    return", "    ok = front & np.isfinite(u) & np.isfinite(v) & (z <= 200)\n    return", "project_all_returns caps range"),
    ("D12b", "D12: builder uses the GT-capped map", AV2 + "build_av2_masks.py", "L.all_returns_uv(cam, lid))", "np.argwhere(np.isfinite(L.sparse_gt(cam, lid)))[:, ::-1].astype(np.float64))", "old lidar_uv_from_gt behaviour"),
    ("D12c", "D12: out-of-image returns ignored", SEG + "make_sky_mask.py", "    pad = int(np.ceil(radius))", "    pad = 0", "no canvas padding"),
    ("D14a", "D14: step 14000 not asserted", AV2 + "av2_scorer.py", '    if ck_meta.get("step") != EXPECT_STEP:', "    if False:", "any step accepted"),
    ("D14b", "D14: host GPU not asserted", AV2 + "av2_scorer.py", "    if HOST_GPU not in str(gpu_name):", "    if False:", "any host accepted"),
    ("D14c", "D14: non-cuda autocast accepted", AV2 + "av2_scorer.py", '    if not str(dev).startswith("cuda"):', "    if False:", "cuda assert removed"),
    ("D17", "D17: empty union passes the IoU gate", SEG + "make_sky_mask.py", "    if not v == v:", "    if False:", "empty union passes"),
    ("D16", "D16: dropped frames removed from the store again", AV2 + "build_av2_masks.py", "kept = {f: (r[\"certain\"], r[\"any\"]) for f, r in results.items()}", "kept = {f: (r[\"certain\"], r[\"any\"]) for f, r in results.items() if not r[\"dropped\"]}", "store without dropped frames"),
]


def run(py, cwd):
    p = subprocess.run([py, "-m", "pytest", "-q", "-rfE", "--tb=line", "-p", "no:cacheprovider", AV2 + "tests", "tests/test_sky_mask.py"], cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main(repo, scratch, py, out_md):
    repo, scratch = Path(repo), Path(scratch)
    rows = []
    rc0, out0 = None, None
    for mid, ref, rel, old, new, desc in [("CONTROL", "unmutated copy", None, None, None, "all tests must pass")] + MUTANTS:
        tgt = scratch / "w"
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(repo, tgt, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        if rel:
            f = tgt / rel; src = f.read_text()
            assert src.count(old) == 1, f"{mid}: old text occurs {src.count(old)}x in {rel}"
            f.write_text(src.replace(old, new))
        rc, out = run(py, tgt)
        failed = re.findall(r"^(?:FAILED|ERROR) (\S+?)(?: - (.*))?$", out, re.M)
        summ = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed)", l)][-1:]
        rows.append((mid, ref, rel, desc, rc, failed, summ[0] if summ else "", out))
        print(mid, "KILLED" if rc else ("OK" if mid == "CONTROL" else "SURVIVED"), len(failed), summ, flush=True)
    with open(out_md, "w") as f:
        f.write("# Mutant table (generated by tests/run_mutants.py; every mutation applied to a scratch copy of the worktree, ONE at a time)\n\n")
        f.write("Run with ~/.venvs/ramen_seg2/bin/python (pyarrow, torch, transformers present, so the end-to-end test is not skipped). Each row: the mutation, the tests that FAIL under it, the first observed failure line.\n\n")
        f.write("| id | what it was | file | mutation | result | failing tests (up to 3 of N) | observed failure |\n|---|---|---|---|---|---|---|\n")
        for mid, ref, rel, desc, rc, failed, summ, out in rows:
            res = "control: " + summ if mid == "CONTROL" else ("KILLED" if rc else "**SURVIVED**")
            ft = "<br>".join(x[0].split("::", 1)[-1] if "::" in x[0] else x[0] for x in failed[:3]) + (f"<br>(N={len(failed)})" if failed else "")
            first = [l for l in out.splitlines() if re.match(r"^(/\S+:\d+: |E  )", l)]
            obs = re.sub(r"^/\S+/(\S+:\d+: )", r"\1", first[0] if first else (failed[0][1] if failed and failed[0][1] else "")).replace("|", "\\|").replace("\n", " ")[:170]
            f.write(f"| {mid} | {ref} | {rel or ''} | {desc} | {res} | {ft} | {obs} |\n")
        surv = [r[0] for r in rows if r[0] != "CONTROL" and r[4] == 0]
        f.write(f"\nSurvivors: {surv if surv else 'none'}. Control (unmutated): {rows[0][6]}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))

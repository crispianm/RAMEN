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
    ("R2", "O45: runner passes no --expect-scorer-sha", AV2 + "run_score.py", ', "--expect-scorer-sha", sha, "--registry"', ', "--registry"', "scorer gets no expected sha"),
    ("R3", "N1: runner passes no --registry", AV2 + "run_score.py", ', "--registry", str(Path(argv[0]).resolve())]', "]", "scorer gets no registry"),
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
    # ---- re-review of 15150ca: the 10 survivors P6 P8 P9 P11-P14 P17-P19 (reviewer's own mutation texts) ...
    ("P6", "survivor: paired sweep missing from objects.tsv not refused", AV2 + "av2_dataset.py", "            if k not in keys:\n                raise AssertionError(f\"{k} not in objects.tsv\")\n        for t in clip:", "            if False:\n                raise AssertionError(f\"{k} not in objects.tsv\")\n        for t in clip:", "sweep-in-objects check off"),
    ("P8", "survivor: frames-tsv dt > 25 ms not refused", AV2 + "av2_dataset.py", "            if dt > MAX_PAIR_DT_NS:", "            if False:", "rows dt>25ms check off"),
    ("P9", "survivor: clip sortedness (match ascending)", AV2 + "av2_dataset.py", "        assert clip == sorted(set(clip)), f\"{log}: clip not strictly ascending\"", "        pass", "clip sortedness assert off"),
    ("P11", "survivor: check_pair_runs dropped ids", AV2 + "av2_scorer.py", "    for k in (\"kept_frame_ids\", \"dropped_frame_ids\", \"frame_mask_px\"):", "    for k in (\"kept_frame_ids\", \"frame_mask_px\"):", "dropped ids not compared"),
    ("P12", "survivor: check_pair_runs n_scored_frames", AV2 + "av2_scorer.py", "    for k in (\"n_scored_frames\", \"n_lidar_px_valid_before_sky\", \"n_lidar_px_sky_excluded\"):", "    for k in (\"n_lidar_px_valid_before_sky\", \"n_lidar_px_sky_excluded\"):", "n_scored_frames not compared"),
    ("P13", "survivor: check_pair_runs stage", AV2 + "av2_scorer.py", " or h[\"meta\"].get(\"stage\") != t[\"meta\"].get(\"stage\"):", ":", "stage not compared"),
    ("P14", "survivor: compare_pairs per-run stage/label", AV2 + "av2_scorer.py", "            assert r[\"meta\"].get(\"stage\") == 1 and r[\"meta\"].get(\"seed_label\") == k, f\"pair {k}: runs must carry stage 1 and the pair label\"", "            pass", "per-run stage/label assert off"),
    ("P17", "survivor: decode-sanity GT range", AV2 + "decode_sanity.py", "    assert v.size > 0 and bool((v > 0).all()) and bool((v <= A.MAX_DEPTH_M).all()), f\"{log}:{cam}: GT valid-range check failed\"", "    pass", "GT range check off"),
    ("P18", "survivor: builder image shape", AV2 + "build_av2_masks.py", "            assert img.shape[:2] == A.NATIVE_HW, f\"{log}:{cam}: image {img.shape} != native {A.NATIVE_HW}\"", "            pass", "builder native-shape assert off"),
    ("P19", "survivor: builder main without verify_segmenters", AV2 + "build_av2_masks.py", "    segmeta = verify_segmenters(a.seg_a, a.seg_b)\n", "    segmeta = dict(segmenter_a_sha=SEG_A['bin_sha'], segmenter_b_sha=SEG_B['bin_sha'], segmenter_a=dict(SEG_A), segmenter_b=dict(SEG_B))\n", "main skips verify_segmenters"),
    # ---- new code of this round (N1-N4, decode-sanity log assert)
    ("N1a", "N1: cfg check not called in main", AV2 + "av2_scorer.py", "        facts.update(assert_ckpt_cfg(load_ckpt_cfg(a.ckpt), a.pair, a.arm))", "        facts.update(dict(cfg_checked=True, cfg_sky_hinge_weight=None, cfg_sky_hinge_min_depth=None, cfg_data_seed=None))", "TS2 ckpt scored as H300 accepted"),
    ("N1b", "N1: registered-sha check not called in main", AV2 + "av2_scorer.py", "        assert_registered_ckpt(ckreg, a.pair, a.arm, facts[\"ckpt_sha\"])", "        pass", "ckpt sha not compared with the registry"),
    ("N1c", "N1: H300 hinge facts not asserted", AV2 + "av2_scorer.py", "        if loss.get(\"sky_hinge_weight\") != HINGE_W or loss.get(\"sky_hinge_min_depth\") != HINGE_MIN:", "        if False:", "H300 cfg unchecked"),
    ("N1d", "N1: TS2 control may carry the hinge", AV2 + "av2_scorer.py", "        if loss.get(\"sky_hinge_weight\", 0) not in (0, 0.0) or \"sky_hinge_min_depth\" in loss:", "        if False:", "TS2 cfg unchecked"),
    ("N1e", "N1: seed/reseed not asserted", AV2 + "av2_scorer.py", "    if data.get(\"seed\") != seed or tr.get(\"resume_reseed\") != seed:", "    if False:", "seed unchecked"),
    ("N1f", "N1: resume_from not asserted", AV2 + "av2_scorer.py", "    if tr.get(\"resume_from\") != want_start:", "    if False:", "start ckpt unchecked"),
    ("N1g", "N1: UNSET registry entries accepted", AV2 + "av2_scorer.py", "    if unset:\n", "    if False:\n", "unset list accepted"),
    ("N1h", "N1: registry sha not compared", AV2 + "av2_scorer.py", "    if reg[(pair, arm)] != ckpt_sha:", "    if False:", "wrong-pair/swapped sha accepted"),
    ("N1i", "N1: registry distinctness", AV2 + "av2_scorer.py", "    if len(set(reg.values())) != len(reg):", "    if False:", "duplicate registered shas accepted"),
    ("N2a", "N2: compare_pairs pins/scorer/registry identity", AV2 + "av2_scorer.py", "        if len(vals) != 1 or \"null\" in vals:", "        if False:", "differing pins accepted"),
    ("N2b", "N2: six distinct checkpoint shas", AV2 + "av2_scorer.py", "    if len(set(shas)) != 6 or not all(shas):", "    if False:", "one ckpt under three labels accepted"),
    ("N2c", "N2: check_pair_runs pins/scorer_sha", AV2 + "av2_scorer.py", "        if not h[\"meta\"].get(k) or h[\"meta\"].get(k) != t[\"meta\"].get(k):", "        if False:", "arms with different scorer sha accepted"),
    ("N2d", "N2: --expect-scorer-sha not required for compare", AV2 + "av2_scorer.py", "    if a.compare_pairs or a.compare:\n        check_scorer_sha(a.expect_scorer_sha)", "    if a.compare_pairs or a.compare:\n        pass", "compare without expected sha"),
    ("N2e", "N2: registry not applied in compare-pairs main", AV2 + "av2_scorer.py", "        check_runs_against_registry(js, reg)", "        pass", "unregistered runs compared"),
    ("N3", "N3: SHA256SUMS read from the old implicit location", AV2 + "av2_scorer.py", "sums_sha=A.sha256_file(a.sums))", "sums_sha=A.sha256_file(Path(a.root).parent / \"SHA256SUMS\"))", "--sums ignored"),
    ("N4a", "N4: mask meta rule not compared", AV2 + "av2_scorer.py", "    if m.get(\"rule\") != want:", "    if False:", "tampered rule accepted"),
    ("N4b", "N4: recorded code shas not compared with the files", AV2 + "av2_scorer.py", "        if mmeta[k] != v:", "        if False:", "mask built by other code accepted"),
    ("N4c", "N4: builder does not record the code shas", AV2 + "build_av2_masks.py", "lidar_exclusion=\"all returns with z>0, no range cap\", **code_provenance()))", "lidar_exclusion=\"all returns with z>0, no range cap\"))", "meta without code shas"),
    ("N4d", "N4: builder_sha pin wrong", AV2 + "av2_scorer.py", "builder_sha=\"718be614", "builder_sha=\"0000000", "pin differs from the file"),
    ("N6", "decode-sanity logs not compared with the registered list", AV2 + "decode_sanity.py", "    assert logs == registered,", "    assert True or logs == registered,", "logs 1-5 unchecked"),
]


def run(py, cwd):
    p = subprocess.run([py, "-m", "pytest", "-q", "-rfE", "--tb=line", "-p", "no:cacheprovider", AV2 + "tests", "tests/test_sky_mask.py"], cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main(repo, scratch, py, out_md):
    repo, scratch = Path(repo), Path(scratch)
    rows = []
    rc0, out0 = None, None
    import os
    only = [x for x in os.environ.get("RUN_ONLY", "").split(",") if x]                  # quick re-check of single mutants (the committed table is always a full run)
    for mid, ref, rel, old, new, desc in [("CONTROL", "unmutated copy", None, None, None, "all tests must pass")] + [m for m in MUTANTS if not only or m[0] in only]:
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

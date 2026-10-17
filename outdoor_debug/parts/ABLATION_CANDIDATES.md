# Candidate arms for the pre-declared fine-tune screen (draft, nothing run)
Start: c3_n46 step12000. Light-track rules (Add. O2): log full arm list, N tried / K better, DEV only, no compute before user + Auditor clearance. One change per arm.

| Arm | Change | Extra cost per step |
|---|---|---|
| A0 | continuation baseline (no change) | 0 |
| H80 | one-sided hinge 0.1 @ 80 m on sky | ~0 |
| H300 | same hinge, floor 300 m (see H80 rationale below) | ~0 |
| TS | TartanAir sky-threshold fix (loader rule d>1e4) | 0 |
| T-a | teacher output hinge: log-depth, pred >= teacher, on sky + pixels beyond 80 m only (teacher = frozen released Metric-VDA-S on the CLEAN frames) | +1 frozen VDA-S forward/clip, no grad (~+25-35% step time est., to be measured) |
| T-b | teacher ranking/ordering term on sky + far pixels | same teacher forward + pair sampling (small) |
| T-c | cosine feature distillation at path_3 only | same teacher forward; path_3 features of student already computed, so no extra student cost; cheapest of the teacher arms if the teacher forward is accepted |

Notes
- Teacher already has the target behaviour: KITTI DEV sky 479 m (released) vs 155-191 m ours.
- Caveat: T-a/T-b/T-c inherit the teacher's own far/sky errors and use no real far label; they test "match release", not "match truth". Pair with VKITTI far pred:gt (release 0.995).
- Trajectory result: compression is present by step 6000 and plateaus, so a step-12000 screen tests recovery, not origin. A from-init arm is a later question.

## Corrections and decision rules after Auditor notes (2026-10-17; not a clearance, user has not cleared training)
1. H80 vs H300 rationale (corrected). The earlier "80 m equals the training cap" referred to the old recipe. Under C3 (effective_c3_seed45.yaml) `silog_max_depth: 655` applies to TartanAir and both PointOdyssey sources (lines 53/85/95), `vkitti_max_depth: 655` (line 73), and the global `max_depth: 80.0` (line 156) plus per-source `max_depth: 80.0` apply to KITTI/Spring/IRS (lines 111/119). So 80 m is the valid-depth limit only for KITTI/Spring/IRS; on the 655 m sources a hinge at 80 m is active on genuinely valid far pixels. H80 tests "does any floor help"; H300 tests a floor beyond the KITTI valid range. Existing key: `sky_hinge_weight: 0.0` (line 249), `sky_vda_zero_disparity: true` (251); the old hinge was 0.05 @ 400 m.
2. Teacher arms: teacher sees CLEAN frames (privileged input) and is the released model we are measured against. Decision rule is GT-based only: VKITTI far pred:gt, KITTI/VKITTI AbsRel by range, TartanAir far. "Closer to release sky median" is circular for T-a/b/c and is NOT a target (reported as description only). If any teacher arm heads for the paper: disclose distillation from released weights on clean frames.
3. TS (TartanAir sky-threshold fix) fixes a confirmed labelling issue but is a new recipe (O2 item 5). It needs a regression test with per-env finite-sky cases on the branch before use.
4. Held fixed across all arms: start checkpoint sha, steps, seed, data-stream order, eval DEV set. A0 continuation is the in-sweep baseline.
5. N tried = 7 from the first run (A0, H80, H300, TS, T-a, T-b, T-c); any arm added later counts. Exploration log (arms, N tried incl. discarded, baseline, DEV set, shas) is appended BEFORE any result is seen.

## USER DECISION (Addendum 42, relayed by Manager 2026-10-17): supersedes the table above
1. Fix labels first: the TartanAir task (seg download, parity, per-env sky audit, TS v2 rule + regression tests) completes BEFORE any screen training. Hinge arms run on corrected TartanAir sky.
2. No teacher arms: T-a, T-b, T-c are DROPPED by the user before any run. They were never tried, so they do not count in N tried. (Rows and the teacher caveats above are kept only as a record.)
Final arms (nested, one change per step): A0 continuation (C3 recipe, old loader); TS2 = A0 + TS v2 loader; TS2+H80 = TS2 + hinge 0.1 @ 80 m; TS2+H300 = TS2 + hinge 0.1 @ 300 m. N tried = 4, all pre-declared here, DEV only. Hinge-on-TS-v2 arms are the hinge arms themselves; no further arms without an addition to this list before results are seen.

## Add. 44 (2026-10-17, user via Manager): sixth arm TS2+VK655 added BEFORE any run
Change vs TS2: VKITTI sky sentinel (65535 = 655.35 m) becomes valid metric depth (mask=1, sky=0, no hinge); VKITTI caps raised 655 -> 656 for this arm only. Motivation: inference that Metric-VDA trained VKITTI sky as a ~655 m target (unverified; VDA has no sky loss, training code unpublished). N tried: 5 -> 6. Nothing has run.

## Submission log (2026-10-17T18:11Z, Manager approval, package 2a72223 code = 0034ac7)
Clone /projects/b5dh/RAMEN_outdoor_screen_2a72223 (clean, 2a72223c); sbatch files in /projects/b5dh/RAMEN_outdoor_screen_2a72223_submit (sha-matched to outdoor_debug/parts/submit/). Smoke jobs (50 steps): outdoor_smoke_A0 = 7504811, outdoor_smoke_TS2_VK655 = 7504812. Full-run job ids are appended below before any result is read.
Smokes 7504811/7504812 COMPLETED 0:0 (3:31 each); check_first_run.sh -s: items 1-7 ok for both (VK655 bound 656, A0 655; ts2 sha f19d0b83; loss ratio VK655/A0 = 1.01). Fulls submitted 2026-10-17T18:17Z (before any result): A0 7504894, A0b 7504897, TS2 7504898, TS2_H80 7504900, TS2_H300 7504902, TS2_VK655 7504904.
Scorer pinned: outdoor_debug/parts/scorer/scorer.py sha256 fbd2c9992617b0e7e2f45de0ed4b92c9d6e28c902f3ccc146d5df83bbeaa95f1 at 840a77d (Auditor O20 S1 checked). TartanAir far DROPPED from the screen scoring (recorded 2026-10-17).
2026-10-17T18:37Z first-run stop check on the 6 fulls: one checker false positive (A0b item 7: the [data] line carries the declared seed 49 vs 45; clip counts and sources identical). Checker fixed to ignore seed in the compare (check_first_run.sh); re-run: no FAIL. Runs untouched. A0b's hinge/LR/bounds/sha items ok.

### Stop-check FAIL record (before any result was read)
- Raised 2026-10-17T18:35Z by check_first_run.sh on the 6 full logs: "[A0b] item 7: FAIL differs from reference: ['[data] train clips=94581 val clips=3 world_size=4 device=cuda:0 seed=49000']" (RESULT: FAIL (scancel)). All other items ok.
- Evidence: the only difference is seed=49000 vs A0's 45000, the declared A0b key (data.seed 49; Auditor O18); clip counts (94581 train / 3 val), world_size 4 and sources are identical.
- Manager decision (via cross-session message, received before 2026-10-17T18:24:41Z): 'continue, false positive'; no scancel. R+D had not cancelled (asked first); Manager process note: in future cancel first unless the FAIL is obviously the checker's own fault and the runs are cheap to keep.
- Checker fix (separate commit): item 7 compares the [data] line with the seed value masked; re-run on all six logs at 2026-10-17T18:24:41Z: outputs in outdoor_debug/parts/submit/stop_check_all6_2026-10-17T18:24:41Z.txt, RESULT: RESULT: no FAIL (WAIT/WARN lines may remain).

### Redo of the vacuous checker fix (Auditor O21, Manager rule: negative controls)
- 00cb04b masked the seed for EVERY arm (same value on both branches of the conditional) so its 'no FAIL' re-run (c9e8841) proved nothing; withdrawn.
- Redone checker (this commit): item 7 asserts seed=45000 for every arm except A0b = 49000 and compares [data] lines with the seed masked; item 4 counts [ts2] OCCURRENCES (4 per TS2-based arm, each with env_table_sha256 f19d0b83; 0 in A0/A0b).
- Positive run on all six real logs (2026-10-17T18:27:01Z): outdoor_debug/parts/submit/stop_check_all6_redo_2026-10-17T18:27:01Z.txt, RESULT: no FAIL.
- Negative controls (must FAIL, all did): outdoor_debug/parts/submit/negative_controls_2026-10-17T18:27:01Z.txt (A0 log as A0b -> item 7 FAIL; TS2 log with one [ts2] removed -> item 4 FAIL (3 occurrences); A0 log as TS2 -> item 4 FAIL; VK655 log as TS2 -> item 6 FAIL; A0b log as TS2_H80 -> item 7 FAIL).
- Scorer forced-fail control: outdoor_debug/parts/scorer/forced_fail_sentinel.py (+_output.txt): on the real VKITTI DEV GT, far_mask with the sentinel on gives pooled median 0.84200, with the sentinel mask off 0.45777 -> the mask is active.
- Scorer pin now: scorer.py sha256 a8b0731e7270b911f92d56eade55b413338b12fd4373aac2b93ae1a7da8171df at e0f6702 (Auditor O22 OK); replaces fbd2c999.

### S3 regression (4090, fp16, scorer a8b0731e, bench_joint 5d3d048c, pins_asserted true), 2026-10-17
Start checkpoint c3_n46 step12000 (sha f5c7c8f9...): 13 of 14 compared rows reproduce the earlier T_trajectory/D_baseline numbers (VKITTI far legacy-subsample median 0.841702 vs 0.841730, rel 3.4e-5; VKITTI AbsRel by range, KITTI/ScanNet++/NYUv2 pooled AbsRel at ~1e-16; sky and horse medians within tolerance). ONE DIFF: the exact pooled median over all 15,188,340 far px is 0.863392, 2.6% above the legacy value. Cause (to be confirmed by the Auditor): NOT sampling error but WEIGHTING: legacy = 10 clips x 100k seeded px (equal clip weight, 1M px); exact pooled weights clips by far-px count. Other forms on the same data: per-clip-median mean 0.8446, ratio of medians 0.9159, mean 0.8841. Output: outdoor_debug/parts/scorer/regression_c3_n46_step12000_20261017T182513Z/.

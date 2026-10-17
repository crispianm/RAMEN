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

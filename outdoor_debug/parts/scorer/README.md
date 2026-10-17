# Outdoor-debug by-range scorer

Files: `scorer.py` (CLI + pure functions), `tests/test_outdoor_scorer.py` (CPU unit tests, `~/.venvs/ramen/bin/python -m pytest tests -q`), `run_regression.sh` + `compare_regression.py` (regression on the start checkpoint; run by the lead on the GPU under GPU_LOCK), this README. Nothing here was run on a GPU; arm outputs are never read.

CLI: `python -I scorer.py --ckpt <resumable .pt> --out <result.json> [--datasets kitti,nyuv2,scannetpp,vkitti,horse] [--device cuda] [--precision fp16|fp32] [--nyuv2-frames 160] [--limit-clips N --max-frames M (CPU tests; pins NOT asserted)] [--repo <root>] [--horse-dir ..]`. Refuses an existing `--out`.

## Pins asserted in the scorer (S2) and logged in the JSON (`meta`)
- `bench_joint.py` sha256 must equal `5d3d048c...e6e7ef` (R3_PIN) else abort; the sha, `ramen_commit` (bj.git_state), scorer sha, checkpoint sha256 (+ `weights_key`, step) are in `meta`.
- Clip lists = `bj.select_clips(ds, "dev")` (unchanged); per-dataset key sha256 (`"\n".join(keys)`, no trailing newline; verified here to reproduce the R3 values) asserted == R3: kitti e187a28b (7), nyuv2 49adc069 (4), scannetpp 010a310b (31), vkitti 2f57ab48 (10); combined (`ds:key` lines, kitti,nyuv2,scannetpp,vkitti) asserted == aca96bbd... Keys are logged in `meta.clip_lists`. Assertions are skipped only when `--limit-clips/--max-frames` is given (and `meta.pins_asserted` says so).
- Checkpoint must contain `trainable_ema` (load via `bench_joint.build_pipeline("ours:<ckpt>")` -> `load_ours`, i.e. build_model(cfg) + `trainable_ema` strict load; the same path as the SCORE sbatch/bench_joint). The scorer aborts if only `trainable` exists.

## Where each number comes from (S5)
Guards come from UNCHANGED bench_joint functions called as `cmd_run` calls them for cell `b0s0`, split dev, with `--temporal` minus the tOF net: `build_pipeline`, `run_pipeline` (+`degrade` identity), `_vda5.score_slice`, `camera` (KITTI OXTS poses, ScanNet++ manifest K/pose), `score_clip(... sky_m / sky_mask)`. `guards[ds].video_macro` = nanmean over clips of the per-clip values, as bench_joint's `video_macro`:
- **AbsRel UNALIGNED** = `abs_rel_unaligned` (metric model; pred clamped to [1e-3, max_depth], valid min_depth < gt < max_depth with max 80 / 10 / 10 m for KITTI / NYUv2 / ScanNet++, per-frame mean, then mean over frames, then over clips).
- **TAE** = `tae_gtvalid` (TAE without sky/no-GT pixels; user ruling 10-10).
Also stored per clip: `abs_rel_aligned`, `delta1_*`, `tae`, KITTI/VKITTI `sky_*` fields.
The by-range / pred:gt / sky statistics are NOT in bench_joint; they are computed in the scorer from the same prediction, resized to GT size with cv2 (`get_infer` resize, as score_clip) and kept float32 like run_baseline/run_traj_eval.

## Definitions
**PRIMARY (S4)** `primary.vkitti_far_80_655` on the 10 VKITTI DEV clips (64 frames each):
- mask: `isfinite(gt) & 80 <= gt < 655` (m). Same as the earlier code (`(gt >= 80) & (gt < sky_depth_m=655)`), i.e. [80, 655), not (80, 655]; a pixel at exactly 80.0 m is the only difference to a literal (80,655] and is negligible. The sentinel (655.35 m = 65535 cm; bench_joint's sky label is gt >= 655) is excluded on the GT side inside the scorer, for all arms, never via loader flags.
- statistic: **`pooled_median` = median over ALL pixels of all clips of pred/gt (pooled median of per-pixel ratios, "median of ratios")**, exactly as T_trajectory `pred_over_gt_median_far80-655`. NOT the ratio of medians (also reported as `ratio_of_medians`, plus `mean`, `per_clip_median_mean`, `bin_absrel`).
- exactness: the earlier code took a seeded random subsample (<=100k px per clip, `torch.randperm` with generator seeded by `md5(key)[:8]`, drawn after the sky draw and the near<30 draw). The scorer reports the EXACT pooled median (`pooled_median`, used by the success rule) AND replicates the subsample (`legacy_subsample_median`) so the regression can match 0.84173 to 4 digits; the exact value differs from the subsample by sampling error only (expected < 0.003).
- success-rule quantity: `primary.success_rule_quantity_abs_1_minus_pred_gt` = |1 - pooled_median|.

**Truncation (S4)**: all clips are scored with the protocol frame counts (`max_len`: KITTI 110, VKITTI 64, ScanNet++ 90) EXCEPT NYUv2, which is truncated to the first 160 frames (`--nyuv2-frames 160`, default; D_baseline did this because 500-frame clips OOM a 24 GB card). The same flag value MUST be used for all 6 arms and the regression (recorded in `meta.nyuv2_frames`, `guards.nyuv2.frames_per_clip_cap`). `--nyuv2-frames 0` = all 500 (GH200 should fit; do not mix).

**GUARDS**: KITTI, NYUv2, ScanNet++ `guards.<ds>.video_macro.abs_rel_unaligned` and `.tae_gtvalid` (see above). No scale/shift alignment is applied to them.

**By range** (`by_range.kitti|vkitti`): pixel-pooled AbsRel of the raw (unclamped, unaligned) prediction on `gt>0.1 & ~sky` with bins (0.1,10], (10,30], (30,80], (80,inf) as run_baseline/run_traj_eval (VKITTI >80 excludes the sentinel because sky = gt >= 655 is removed from `base`; KITTI sky = the colour rule `kitti_sky_masks`). `pooled_absrel_raw.<ds>` = same mask, all ranges (no 80 m cap) = the earlier "absrel_raw".

**SECONDARY (description only)**:
- `secondary.kitti_sky`: prediction on the KITTI colour-rule sky mask, frames with >=50 sky px: `pooled_median`, `median_of_frame_medians` (the "KITTI per-frame sky median", T_trajectory), `framebalanced_median` (<=3000 px/frame, seeded; approximates D_baseline's unseeded 3000-px sampling), `frac_gt80`, `frac_gt200`.
- `secondary.vkitti_sky` (gt >= 655 px): `pooled_median` (frames with >=50 sky px), `exact_pooled_median_all_px`, `legacy_subsample_median` (200k/clip, = T_trajectory `sky_median_m`).
- `secondary.vkitti_near_lt30_pred_over_gt`: pooled median pred/gt, 0.1<gt<30, not sky.
- `secondary.horse`: /mnt/DATA/Horse/test_resized, Resize 518 lower_bound cubic as T_trajectory; median over frames of the top 15% rows (sky), rows 38-46% (tree line), bottom 10% (ground), frame-25 variants, clip max.
- `secondary.tartanair_far` = **null, NOT IMPLEMENTED** (open question 1).

## Earlier numbers (c3_n46 step12000 = RAMEN n46) and how they were masked
Host of the earlier numbers: the local workstation (hostname IT101668 / bench_joint "crispwork" paths, RTX 4090 24 GB), fp16 autocast, bench_joint pipeline, DEV only, one run, clean cell, NYUv2 first 160 frames. NOT the Isambard GH200 on which the arms will be scored. Sources: `parts/T_trajectory/trajectory.csv` (T), `outdoor_debug/baseline_metrics.csv` = `parts/D_baseline/baseline_metrics.csv` (D).
| quantity | earlier value | source | earlier mask | new scorer key |
|---|---|---|---|---|
| VKITTI far pred:gt (80-655) | 0.84173 (rounded 0.842) | T `pred_over_gt_median_far80-655` | (gt>=80)&(gt<655), median of seeded <=100k px/clip subsample pooled | `legacy_subsample_median` (match), `pooled_median` (exact) |
| VKITTI near<30 pred:gt | 1.10016 | T | base(gt>0.1,~sky)&gt<30, subsample | `vkitti_near_lt30...legacy_subsample_median` |
| VKITTI AbsRel raw 0-10 / 10-30 / 30-80 / >80 | 0.14398 / 0.16972 / 0.20188 / 0.21134 | T `absrel_raw_*`, D | base=gt>0.1&~sky(gt>=655), pooled, unclamped | `by_range.vkitti` |
| VKITTI sky median | 323.76 m | T `vkitti_dev sky_median_m` | gt>=655, 200k subsample/clip | `vkitti_sky.legacy_subsample_median` |
| VKITTI sky median (D) | 320.62 m | D `sky_median_pred_sky_m` | frames with sky>=50 px & scene px, 3000 px/frame unseeded | `framebalanced_median` (approx only) |
| KITTI AbsRel raw (pooled) | 0.14774 (rounded 0.148) | T `kitti_dev absrel_raw`; D all = 0.14774 | base=gt>0.1&~sky(colour rule), pixel-pooled, NO 80 m cap, unclamped | `pooled_absrel_raw.kitti` |
| KITTI cap-80 frame-macro AbsRel | 0.14867 | D `cap_perframe_macro` | 0.1<gt<80, pred clamped, per-frame mean then mean over frames (all clips pooled by frame) | closest = `guards.kitti.video_macro.abs_rel_unaligned` (clip-macro, differs slightly) |
| KITTI aligned-scale AbsRel | 0.08276 | T | median log-ratio scale per clip | not scored (guards are unaligned) |
| KITTI sky median: frame medians / pooled | 160.50 / 322.25 m | T | colour-rule sky, >=50 px frames | `kitti_sky.median_of_frame_medians` / `pooled_median` |
| KITTI sky (D, equal-frame weight) | 155.05 m | D | 3000 px/frame unseeded | `framebalanced_median` (approx) |
| ScanNet++ AbsRel raw | 0.12083 (T, D all); cap 0.12082; frame-macro 0.12066 | T, D | gt>0.1, pooled | `pooled_absrel_raw.scannetpp`; guard = `video_macro.abs_rel_unaligned` |
| NYUv2 AbsRel raw (160 fr) | 0.09485 (pooled), frame-macro 0.09295 | D | gt>0.1 | `pooled_absrel_raw.nyuv2`; guard = `video_macro.abs_rel_unaligned` |
| horse sky_top15 / treeline / ground | 199.59 / 81.55 / 3.74 m | T | median over band, all frames | `secondary.horse` |
| TAE (tae_gtvalid) | no earlier number for c3_n46 in these files | - | - | `guards.*.video_macro.tae_gtvalid` (regression sets the reference) |
| TartanAir far | none on eval clips; train-batch median pred/gt far 80-655 = 0.682 (20 batches, 14 with far px) | T `train_batches` | training batches incl. augmentation | not implemented |

### Masks that differ from the earlier numbers (answer to "same GT masks?")
- **Sentinel**: the earlier code already excluded it for the VKITTI far ratio (gt<655) and for ">80" AbsRel (sky = gt>=655 removed from base). So VKITTI numbers should reproduce with no sentinel effect. Where the sentinel WOULD matter: any statistic computed on `gt>0.1` without removing sky (e.g. bj's own `abs_rel_unaligned` for VKITTI is capped at gt<80 so is unaffected; D `all` range for VKITTI also removed sky). If sentinel px (gt=655.35, ~14% of VKITTI px per `sky_frac`) were left in the far ratio they would drag the pooled median far below ~0.84 (pred on sky ~300 m vs gt 655 -> ratio ~0.5); the unit test `test_sentinel_does_not_move_the_ratio` demonstrates the mechanism. Expected scorer-vs-earlier difference from the sentinel: 0 for the primary.
- **Pooling**: earlier KITTI/ScanNet++/NYUv2 "AbsRel raw" are pixel-pooled over all valid px; the guard in the scorer is bj's `abs_rel_unaligned` (per-frame mean, clip macro, 80/10/10 m cap, pred clamped). Compare like for like: scorer `pooled_absrel_raw` vs T/D pooled (should match to 1e-4 same host), scorer `video_macro.abs_rel_unaligned` vs nothing earlier (D `cap_perframe_macro` is a frame-macro over all clips: expect agreement to ~1e-3, not exact). Arms are compared with each other on the guard (same definition), so only the regression row needs the like-for-like columns.
- **Subsample**: exact vs subsample median of the far ratio (see above).
- **Precision/host**: earlier = 4090 fp16 autocast; the SCORE sbatch for bench_joint uses fp32 (cmd_run does not autocast). The new scorer defaults to fp16 autocast to match the earlier numbers; for GH200 either keep fp16 (same recipe on every host, recommended) or use fp32 for all arms AND the reference row; never mix. fp16 vs fp32 changed horse sky medians by <=0.2 m (D FINDINGS).

## Regression procedure (S3)
1. Lead holds GPU_LOCK, runs `bash run_regression.sh [OUT]` (checkpoint sha f5c7c8f9... and bench_joint sha asserted; fp16, NYUv2 160 frames). It writes `scores.json`, `scorer.log`, `compare.txt` (OK/DIFF per row against the table above), SHA256SUMS.
2. Same host type (4090): every OK row within tolerance (1e-4 relative for the pooled/legacy rows, 1e-3 for sky/horse medians). The far ratio `legacy_subsample_median` must equal 0.84173 to 4 digits; `pooled_median` within 0.003.
3. Cross-host (Isambard GH200 vs the 4090 numbers): declared tolerance 1% relative on pooled AbsRel / ratios and 2% on sky/horse medians (= the proposal's 1% NF floor; fp16 kernels/cuDNN algorithms differ). The authoritative reference row for the screen is the SAME scorer on c3_n46 step12000 run ON THE GH200 with the same flags as the arms, plus a 4090 row for the cross-host column. If the 4090 row misses the same-host tolerance, stop and report.
4. The arms are scored only after the scorer file, README and tests are committed, their sha256 pinned and the Auditor has checked them; do not run it on any arm output before then.

## Open questions
1. TartanAir far: no earlier eval-clip definition exists (only the train-batch statistic via `train._build_multisource`, needs the TartanAir train data, TS2 loader flags and a seed; arms differ in loader -> a number would not be like for like unless the same fixed batches/seed are used). Needs a decision: drop, or define a fixed list of training-env clips with GT depth, far px = 80<=d<655 excluding sky, same statistic.
2. The SCORE_*.sbatch run bench_joint only; the scorer needs its own sbatch step (single GPU) writing to the same score dir, and Isambard paths for horse (`--horse-dir`, data not on the cluster) and the 4 pinned DEV sets (present via bench_joint PATHS).
3. NYUv2 160-frame truncation (kept for like-for-like with D_baseline) versus all 500 frames on the GH200: needs a one-line ruling; either way identical for all arms.
4. Success-rule sign: |1-pred:gt| treats over- and under-prediction alike; the start model is at 0.158 (under).
5. KITTI TAE needs OXTS zips + calib on the host (local present; on Isambard `kitti_raw_zips`/`kitti_calib` paths in bench_joint PATHS); the scorer warns and omits tae for a clip if `camera()` fails (a missing TAE must be treated as a failed score).

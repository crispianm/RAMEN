# Long-range LiDAR DEV set: DDAD val (report-only), REGISTRATION
Filed: 2026-10-17T16:48:22Z (from `date -u`). Nothing but the tar listing (file names/sizes) and ddad.json (14 KB split file, public metadata) has been opened. No image, point cloud, GT value, depth statistic or model prediction has been read. Changing dataset, bins, metrics or frame list after any output is seen = a new registration.

## Status
DEV, exposed from its first score, report-only: no arm is chosen on it, it never becomes a held-out test without an Auditor ruling; any paper use is labelled "exposed exploratory DEV, N=5 arms disclosed". Scored for all 5 arms (A0, A0b, TS2, TS2+H80, TS2+H300) with the same pinned bench_joint commit on the same host type.

## Dataset and split
- DDAD (TRI), validation split. ddad.json `scene_splits`: "0" = 150 train scenes, "1" = 50 val scenes 000150..000199 (3950 samples). Val = split "1" only. (Argoverse 2 not needed: DDAD obtained.)
- Camera CAMERA_01 (front, 1936x1216) + aggregated Luminar-H2 LIDAR (250 m range) projected into the camera by the official dgp convention (units: metres). Evaluation cap 200 m (packnet-sfm eval_ddad.yaml, camera_01, max depth 200).
- Licence: CC BY-NC-SA 4.0 (non-commercial use only; not redistributed).

## Source and integrity
- URL: https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar (275,266,242,560 B per HEAD; downloaded parts total 275,266,254,848 B incl. tar padding; no sha256 published).
- Published md5 c0da97967f76da80f86d6f97d0d98904 (README). VERIFIED on Isambard: md5 over the 129 parts concatenated in byte order = c0da97967f76da80f86d6f97d0d98904 (job 7497679, rc=0, ended 2026-10-17T10:54:18Z). Parts in /projects/b5dh/scratch_ddad_20261017/parts/.
- Files: tar_listing.txt (119,255 entries) sha256 ef1c81e2f63eea27abac5a9077931ecb004756ab0b14b49c5a2bf7325644d603; ddad.json sha256 14c8aa06c4d262bd962a15caf03970bccf41179267cea1afcf9941e7e93f9c00 (fetched by 14,118-byte range at tar offset 19723312128+512). Per-file sha256 of the extracted subset (SHA256SUMS) is written at extraction and committed.

## Frame list (fixed by rule, from file names only)
- All 3950 CAMERA_01 frames of the 50 val scenes are run contiguously through each model as video (scene = clip; first 160 frames per clip where a window limit applies, as in the existing scorer: stated limitation if a scene exceeds it).
- Scored frames: per scene, CAMERA_01 PNG names sorted ascending (timestamp order), indices 0,10,20,...: 395 frames. List: ddad_val_frames.tsv, sha256 b8d9664fb0c28df0fab61ad71383d48660c5444c4538a8f471fd4bbc8f0e5098. The LIDAR datum for each scored frame is the one paired in the official scene_<sha>.json sample (camera and LIDAR timestamps differ); the scene json is read for pairing only. Script: make_frame_list.py.
- Extract: CAMERA_01 PNG for all 3950 frames, LIDAR npz for the 395 scored frames, scene_*.json, calibration/*.json. Estimated 10-12 GB.
- Decode sanity frames (fixed rule): index 0 of the first 5 scenes in split-file order = (000150, 15616458249936530), (000151, 15609732323309162), (000152, 15609722540809162), (000153, 15616462546436512), (000154, 15609730485809162). They stay in the list (rank 0 is also a scored frame); no frame is dropped except for a documented decode failure. Checks: units (metres), valid mask, LIDAR projection, intrinsics/extrinsics only, GT side, no model output, no depth statistics used for choices.

## Depth bins (declared now, from the LIDAR spec and the 80 m training cap, not from data)
GT valid = projected LIDAR depth > 0 and <= 200 m. Bins (m): [0,20), [20,50), [50,80), [80,120), [120,200]. Per bin report the pixel count and the number of frames contributing.

## Metrics (unaligned: no scale/shift alignment)
Per bin, pooled over scored frames: AbsRel = mean(|pred - gt| / gt); pred:gt = median(pred / gt). Also overall AbsRel over [0,200]. Reported per arm next to the screen decision rule, which does NOT change (primary: VKITTI 80-655 m |1 - pred:gt| reduction). No NF margin is applied to this set; no arm is chosen on it.

## Pinned scorer
bench_joint commit and DEV clip-list sha for the five-arm screen are logged together before the first training run (separate entry, same host type as these scores); a DDAD loader with a regression test (decode, units, masks) is added on outdoor-debug and its commit recorded here before scoring.

## Amendment A1 (2026-10-17T17:06:17Z, from `date -u`; before any model prediction; Auditor O11 R1/R2)
- Resolution rule (frozen): the model prediction for a scored frame is resized BILINEARLY to the native 1936x1216 camera resolution and sampled at the projected LIDAR pixels (integer pixel coordinates of the z-buffered projection); GT is never resized. Metric depth in metres, no scale/shift alignment.
- LIDAR pairing: official scene_<sha>.json samples (camera datum CAMERA_01 <-> LIDAR datum of the same sample). 395 of 395 scored frames paired, 0 missing on disk. Pairs file ddad_val_pairs.tsv sha256 962d499612c40bca7d314f8ff16ce1d6c7b03962ae78938f472a101766150245.
- Extract job: sbatch 7503602, sacct Submit 2026-10-17T16:48:50, Start 2026-10-17T16:48:51 (>= registration filing 16:48:22Z), End 16:53:27, COMPLETED. Extracted 8000 files (3950 CAMERA_01 PNG + 3950 LIDAR npz + 50 scene json + 50 calibration json); sha256sum -c on the cluster 8000/8000 OK. SUMS committed as val_subset_SHA256SUMS.gz (sha256 of the uncompressed file: b12fce8b50ba72c4acecddd194a4b9fdce276c59a7978b68ca18d2815bbbfcea). Local copy under data/DEV_LR/ddad_val/ holds the pruned subset (all 3950 PNGs, the 395 paired LIDAR npz, jsons).
- Earlier job 7497679 (md5 + tar listing only, Start 2026-10-17T10:45:28, End 10:54:21) opened no payload files beyond streaming the tar for md5/listing.
- Decode checks (loader regression test) use only the 5 sanity frames listed above, and cover units, valid mask, projection, intrinsics/extrinsics only.

## R1b loader commit (2026-10-17T17:36:12Z)
DDAD loader ddad_lr_dataset.py + tests/test_ddad_lr_dataset.py committed at dc166c4 (10 tests: 4 synthetic, 6 real incl. the 5 sanity frames chosen by the fixed rule; no GT statistic other than the 0<z<=200 units check was computed). Resolution rule as registered: bilinear prediction -> 1936x1216, sampled at projected LIDAR px, GT never resized. OPEN: scorer short_side (provisional 378 copied from KITTI in DDAD_LR_SPEC) must be fixed in R3 before the first score. Local pull check (R4) pending.

## R4 local pull verification (2026-10-17T18:48:47Z)
data/DEV_LR/ddad_val/ vs the committed val_subset SHA256SUMS: sha256sum -c = 4445 OK, 0 FAILED (mismatch); the other 3555 of the 8000 SUMS lines are non-pruned files that were never pulled (No such file, expected). 4445 = 3950 PNG + 395 LIDAR + 50 + 50. 280 files were re-pulled after rsync exit-255 chunk failures. Cluster parts dir /projects/b5dh/scratch_ddad_20261017/parts (129 files, 257G) deleted after this verification (exact path, per Manager). Output: scratch LR_DEV/verify_local_R4.out. R4 MET; DDAD scoring is still report-only, short_side TBD.

## Amendment A2 (2026-10-17T18:50:33Z): DDAD scorer short_side = 378 (declared BEFORE any DDAD prediction; no model has been run on any DDAD frame)
Model-input resolution for DDAD LR_DEV scoring: bench_joint-style resize (Resize lower_bound, ensure_multiple_of 14, INTER_CUBIC) with short_side = 378, i.e. 1216x1936 -> 378x602 (rounded to multiples of 14 by the same Resize), identical to the KITTI and VKITTI rows (short_side 378) whose far-range behaviour the screen decides on, so the real-data DDAD far bins and the VKITTI primary share one input scale. Prediction resolution rule unchanged (A1: bilinear to 1936x1216, sampled at projected LIDAR px, GT never resized). The loader's DDAD_LR_SPEC short_side=378 was provisional; it is now FIXED. max_len 160, 200 m cap, frames per A1 unchanged. Report-only, labelled 'exploratory DEV'; the decision rule is unchanged. Needs Auditor check before the first DDAD prediction.

## DDAD scorer pin (2026-10-17T18:55:15Z, BEFORE any DDAD prediction)
ddad_scorer.py sha256 1d1f7eea796e01c47ef25fcab02d9b7d00d0bd9f41e95bd86bee0c580ed7eaa4 (committed 556de6e), tests/test_ddad_scorer.py c7f644d7..., 8 synthetic tests pass (pred=0.9xGT -> 0.9/0.1 in every bin; GT-resize and per-frame alignment variants detectably differ; bin edges; pooling; frame-list/pairs sha assertions). Implements O11 (bins [0,20) [20,50) [50,80) [80,120) [120,200] m, pooled unaligned AbsRel, pooled median pred:gt, px/frame/scene counts, overall AbsRel) with A1+A2; asserts frame-list b8d9664f and pairs 962d4996. The model path (model_input/predict/main) has NOT been executed; no DDAD prediction exists. Auditor check required before the first DDAD prediction. Report-only, exploratory DEV.

## DDAD scorer pin UPDATE (2026-10-17T18:59:46Z): replaces the earlier pin line (Auditor O27 accepted)
ddad_scorer.py sha256 17646e01b11a53086e8dfd81ac38a0dffa943d829adbbad8f10900f162a5ecca at e0ca9d54e9dc0514131f30be064abd308d0e04ef (= 1d1f7eea + only the Auditor O26(1) shape assert/log). First DDAD run: job 7505514, start checkpoint c3_n46 step12000 only, GH200; Auditor checks its output before any arm runs on DDAD.

## Amendment A3 (2026-10-17T19:07:11Z), filed BEFORE the job: released Metric-VDA-S DDAD row (Manager request, Auditor O29)

Purpose: score the released Metric-VDA-S on the identical DDAD val LR_DEV frames so the C3 reference has a released-model comparison row. DEBUG ONLY, not paper. Exploratory DEV, report-only; never enters a decision.

- Branch: `ddad_scorer.py --spec metric_vda_s --out <json>` (mutually exclusive with `--ckpt`; the `ours:` path is behaviourally unchanged: trainable_ema assert + ckpt meta; only the shape assert and the checkpoint meta were factored into `assert_same_hw` / `model_meta`).
- bench_joint key: `metric_vda_s` -> `DEPTH_MODELS["metric_vda_s"] = _vda_released(dev, metric=True)` (bench_joint.py pinned sha 5d3d048c...). VideoDepthAnything(vits, features 64, out_channels [48,96,192,384], metric=True), strict load.
- Weights: `/projects/b5dh/benchmark/weights/metric_video_depth_anything_vits.pth` (= `bench_joint.PATHS["metric_vda_s"]` on Isambard), 116444063 bytes, sha256 `3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33` (computed on the cluster, read-only; identical to the crispwork copy `/mnt/DATA/DegradedDepth/RAMEN/checkpoints/metric_video_depth_anything_vits.pth`). The scorer asserts this sha before running and records key, path, sha256, size in meta (`release_weights`).
- Metric depth: the released model has a metric head (metric=True); `fwd` returns metres, `vda_windows` uses scale=1, shift=0 for metric models (no alignment), so the output is used directly as the unaligned metric depth, exactly like the `ours:` rows. No inverse-depth conversion exists on this path.
- Label (exact, in meta and in the result `label`): "released Metric-VDA-S at the registered 378 input (not its default 518), fp16, exploratory DEV, report-only". Asserted against the literal in `release_meta`.
- Unchanged from A1/A2/O11/O26: 395 frames / 50 scenes (asserted on a full run), ddad_val_frames.tsv and ddad_val_pairs.tsv sha pins, bins [0,20) [20,50) [50,80) [80,120) [120,200] m, pixel-pooled unaligned AbsRel + median pred:gt, A1 bilinear pred -> native 1936x1216 sampled at projected LIDAR px, GT never resized, model input = bench_joint Resize lower_bound / multiple 14 / cubic / short side 378, fp16 autocast, out_hw = model resolution with the shape assert (O26 (1)).
- New ddad_scorer.py sha256: `80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f` (previous `17646e01b11a53086e8dfd81ac38a0dffa943d829adbbad8f10900f162a5ecca`). Diff: `A3_diff.patch`. Tests: tests/test_ddad_scorer.py 14 passed (8 existing + 6 new A3).
- Job (one GH200): `outdoor_debug/parts/submit/score/DDAD_RELEASE_MVDA.sbatch` (sha256 548df4fb...ad41e at drafting), `DDAD_COMMIT` literal UNSET until submit-time sed; command inside: `python -I outdoor_debug/parts/LR_DEV/ddad_scorer.py --spec metric_vda_s --out $SCORE_ROOT/DDAD_RELEASE_MetricVDA_S/ddad_scores.json --device cuda`. Preconditions checked by the script: clean tree at DDAD_COMMIT, scorer sha, bench_joint sha, pairs sha, weights sha, output dir absent. Output dir: `/projects/b5dh/RAMEN_outdoor_screen_scores/DDAD_RELEASE_MetricVDA_S`.
- Consequence: any other job using the new commit with the `ours:` path (e.g. DDAD_REF.sbatch, DDAD_SCORER_SHA 17646e01...) must have its scorer-sha pin updated to the A3 sha, or run at the older commit.
- NOTE: DDAD_REF (job 7505514) ran at e0ca9d54 with ddad_scorer 17646e01; the ours: path in 80916c4f is behaviourally unchanged (tests); the Auditor rules whether arm DDAD rows use 80916c4f and whether the reference row needs a re-run at the new sha.

## Amendment A4 (2026-10-17T22:34:17Z, from `date -u`), filed BEFORE the job: the RELEASED Metric-VDA-S on DDAD LR_DEV at its default 518 input, report-only (user Addendum 49/50; Auditor O34 L-3)

**What is scored.** The released Metric-VDA-S (bench_joint key `metric_vda_s`, weights sha256 3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33, same as A3) on the identical DDAD val LR_DEV frames, but with model input short side 518 instead of the registered 378. DEBUG ONLY, not paper; exploratory DEV, report-only.

**Exact flag.** `ddad_scorer.py --spec metric_vda_s --release-short-side 518 --out <json> --device cuda`. Absent flag or `378` = the registered 378 path (unchanged behaviour; A3 row semantics). 

**Label (exact, asserted in meta and result):** "released Metric-VDA-S at its default 518 input (not the registered 378), fp16, exploratory DEV, report-only". Stated in words: released Metric-VDA-S at 518 input, report-only; ours is 378.

**Refusal rules (all raise ValueError in `resolve_short_side` before pins, model load or output).** `--release-short-side` other than 378/518 is refused (e.g. 448, 0); any size other than 378 is refused when no registered release `--spec` is given (every `--ckpt` run: ours at 518 is NOT proposed and cannot be run); an unregistered `--spec` with a size other than 378 is refused. `--spec` + `--ckpt` stays refused (A3). The existing O26(1) same-HxW assert is unchanged and must hold at 518.

**Unchanged (A1/A2/A3/O11/O26):** frames b8d9664f.., pairs 962d4996.., bench_joint 5d3d048c.., 395 frames / 50 scenes, bins [0,20) [20,50) [50,80) [80,120) [120,200] m, pooled unaligned AbsRel + median pred:gt, A1 bilinear prediction -> 1936x1216 sampled at projected LIDAR px (GT never resized), Resize lower_bound / multiple 14 / cubic, fp16, max_len 160. Only the Resize short side differs (518 -> model input 518x826 for 1216x1936 frames; 378 -> 378x602).

**Pins.** ddad_scorer.py sha256 `8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503` (previous 80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f); tests/test_ddad_scorer.py sha in the A4 commit message; job `outdoor_debug/parts/submit/score/DDAD_RELEASE_MVDA_518.sbatch` (output dir .../RAMEN_outdoor_screen_scores/DDAD_RELEASE_MetricVDA_S_518, DDAD_COMMIT literal UNSET until submit-time sed); checker `far_diag/a4_final/check_a4_518.py` (PASS/FAIL names only). Existing rows (ours and the A3 release row at 378) were produced by scorer sha 17646e01/80916c4f and stay valid: with the flag absent the new scorer produces a byte-identical result file (tested on a fake pipeline, old vs new).

**First-518-run requirement.** The log must contain "first clip: model input (N, 3, 518, 826) -> depth (N, 518, 826) (same HxW, identity resample)"; the sbatch aborts otherwise, and check_a4_518.py (must be ALL PASS) additionally compares every other meta field and the per-bin GT pixel/frame/scene counts with the A3 378 row.

**What the result may be used for.** Reporting the release's five-bin median pred:gt and pooled AbsRel at 518 beside the release at 378 (statement form: "the release's 120-200 m median pred:gt is X at 518 vs 0.854 at 378"), to bound how much of the DDAD release-vs-ours far gap is an input-scale effect. **May not:** be compared to ours as an equal-input row (ours is 378 only); enter any arm decision, rule, margin or the paper; support any statement about ours at 518; trigger a retraining or a new 518 run of any other model without a new amendment.

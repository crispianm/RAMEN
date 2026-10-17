# Arm checks (O15 / V2 V3 V6), repo outdoor-debug HEAD 9f1b554, CPU only, no checkpoint inference
Scripts (run from repo root with ~/.venvs/ramen/bin/python): arms.py (check 1, writes configs/arm_*.yaml + config_diffs.md), classes.py (1b, classes.json),
smoke.py (checks 2-5, smoke_results.json; stdout of the final run is summarised below). Local roots: /mnt/DATA/DegradedDepth/data/TRAIN/* (real data), seg zips
claude_outputs/scratch/outdoor_debug_20261016/TA_seg. The arm YAMLs keep the cluster paths (ts2_seg_root=/projects/b5dh/data/train/tartanair_seg/); classes.py/smoke.py remap roots in memory only.

## 1. Overrides and resolved-config diffs (config_diffs.md): ALL EXACT, no defect
A0 == effective_c3_seed45.yaml. A0b: data.seed 45->46 only (split_seed 43, seed_scheme untouched; data.seed is the only seed key the trainer reads, train.py:2300,536,1793).
TS2: tartanair ts2_sky=true, ts2_seg_root (no ts2_env_table). H80/H300 vs TS2: loss.sky_hinge_weight 0->0.1 and loss.sky_hinge_min_depth (absent->80.0/300.0), nothing else (key name from train.py:1171,1198). VK655 vs TS2: vkitti_sky_as_valid_depth=true, vkitti_max_depth 655->656, nothing else.
Every arm vs A0 shows only the declared keys. NOTE (not a diff defect): the base yaml still carries C3 run fields (training.checkpoint_dir=runs/dn01_v3_bfix_C3_B, resume_from=.../step6000.pt, max_steps=12000, lr_decay_steps=12000); the screen's run name / checkpoint dir / resume from c3_n46 step12000 / 2000 steps are NOT in my arm yamls; the submit step must set them identically across arms.

## 1b. Classes actually instantiated (real train._build_multisource -> _make_source, recorder wrapper; all 6 arms, degrade true; local roots)
tartanair -> VDASemanticsDegradedTartanAirDataset (instance attr ts2 = TS2Labeller in TS2, H80, H300, VK655; None in A0, A0b), vkitti -> VDASemanticsDegradedVKITTIDataset (vkitti_max_depth 655.0 in A0..H300; 656.0 and sky_as_valid_depth=True ONLY in TS2_VK655),
pointodyssey x2 -> VDASemanticsDegradedPointOdysseyDataset, irs -> VDASemanticsDegradedIRSDataset, spring -> DegradedSpringDataset, kitti_raw -> DegradedKITTIRawTrainDataset, nyu_train -> DegradedNYUTrainDataset.
scannetpp_train could NOT be built locally (splits/sem_test.txt missing in the local tree, FileNotFoundError); its class is by dispatch with_degradation(ScanNetPPTrainDataset), not instantiated here. Local tartanair has 9781 train clips (28 envs), not the 74-env mix.

## 2. TS2 real Downtown clip (Downtown/P008, start 0, clip idx 0 of 576; picked as max released-sky clip of ~40 scanned; seeds 0/1 matched between arms, crop box asserted identical)
| degrade | class (A0 / TS2) | A0 sky% / mask% | TS2 sky% / mask% | mask,sky max abs diff vs ts_v2_guarded->released resize/crop | depth, frames identical A0 vs TS2 |
|---|---|---|---|---|---|
| false | VDASemanticsTartanAirDataset (both; TS2 has .ts2=TS2Labeller, A0 None) | 0.00 / 100.00 | 1.612 / 98.388 | 0.0 / 0.0 | yes / yes |
| true, seed1 (non-static) | VDASemanticsDegradedTartanAirDataset | 0.00 / 100.00 | 1.612 / 98.388 | 0.0 / 0.0 | yes / yes |
| true, seed0 (STATIC clip: middle frame x T, reference handled the same) | same | 0.00 / 100.00 | 0.036 / 99.964 | 0.0 / 0.0 | yes / yes |
They DIFFER, as required. Caveat: the released rule gives 0% sky on every ~40 sampled Downtown clips' first frames (Downtown sky is finite in this tree), so A0's 0% is real, not a no-op; TS2's seg-sky is what adds the 1.6%.

## 3. VKITTI Scene20/30-deg-left/Camera_0 (clip idx 1176, raw first-frame sentinel 34.7%), same RNG, frames identical
| degrade | arm | class | sentinel px % (d>655.2) | mask % | sky % | depth max | valid px in 80-655.4 m (% of valid) | valid depth max |
|---|---|---|---|---|---|---|---|---|
| false | TS2 (cap 655) | VDASemanticsVKITTIDataset | 0.00 | 57.24 | 42.76 | 655.00 | 17.21 | 654.75 |
| false | VK655 (flag, 656) | same | 42.76 | 100.00 | 0.00 | 655.35 | 52.61 | 655.35 |
| true | TS2 | VDASemanticsDegradedVKITTIDataset | 0.00 | 57.28 | 42.72 | 655.00 | 17.30 | 651.35 |
| true | VK655 | same | 42.72 | 100.00 | 0.00 | 655.35 | 52.63 | 655.35 |
effective_depth_bound (train.py:395): TS2 655.0, VK655 656.0. Flag is live on the recipe class (the earlier no-op is not reproduced).

## 4. V3: constants that could bind a 655.35 m target
| file:line | what | binds 655.35? |
|---|---|---|
| Video-Depth-Anything video_depth_anything/video_depth.py:67 (+dpt.py output_conv2 ReLU) | head output F.relu, no upper bound (metric branch: raw depth) | no |
| models/vda_net.py (forward ~l.396-403) | no clamp on the metric output | no |
| train.py:1236,1529 with loss.metric_depth_clamp [0.1,1e5] (C3 yaml) | pred clamp before SiLog / disparity | no (1e5) |
| train.py:1219-1223 silog_ceil | per-source SiLog ceiling: tartanair/PO 655.0, VKITTI +inf (train.py:477-480), measured silog_ceil=inf on the VKITTI batch | no for VKITTI; (TartanAir/PO exclude d>=655 from SiLog by design) |
| train.py:1225 loss.metric_silog_max_depth | SiLog-only ceiling | absent in C3, no |
| data/vkitti_dataset.py:44-45,186-194; data/vda_semantics_dataset.py:150-166 | _SKY_DEPTH_M=655 strict mask / clip; flag path uses 655.35 clip and recomputed valid | TS2: yes (excludes it, by design); VK655: no (measured depth max 655.35, mask 100%) |
| train.py:436,576 common data.max_depth 80 | passed to VKITTI ctor but the VDA-semantics VKITTI getitem never reads self.max_depth | no (measured) |
| train.py:1073 scale_anchor td<=80 | teacher valid band | not active (loss.scale_anchor_weight absent in C3); would bind |
| train.py:1541 METRIC_EPS_MAX_DEPTH 80 | joint-temporal metric eval range | joint_temporal_weight 0.0 in C3: inactive |
SiLog via real train._compute_loss_vda (arm cfg, log_terms), VKITTI degraded batch, pred = k x GT:
| pred | TS2 silog | VK655 silog | TS2 SiLog mask px | VK655 SiLog mask px (sentinel px in it) |
|---|---|---|---|---|
| 0.5x | 0.4901 | 0.4901 | 4,918,400 (0) | 8,586,368 (3,667,968) |
| 1x | 1e-6 | 1e-6 | same | same |
| 2x | 0.4901 | 0.4901 | same | same |
| min(GT,80) (saturating model) | 0.3969 | 1.2242 | same | same |
Constant-ratio preds give equal SiLog by construction (shifts cancel); the saturating pred shows the sentinel px enter SiLog under VK655 only (log err mean 0.156 vs 0.988); grad abs-sum on sentinel px 0 under TS2, 9.05e-3 under VK655 (0.5x: 9.2e-4). Under TS2 the sentinel px are only in the SSI/TGM sky-zero-disparity mask (total_loss 0.5224 vs silog 0.4901 at 0.5x); VK655 total == silog (sky=0, SSI scale invariant).

## 5. Hinge (TS2 TartanAir Downtown degraded batch, static clip, 3072 sky px outside mask; arm yamls read via train.load_config + check_sky_loss; floor = cfg["loss"]["sky_hinge_min_depth"])
| arm | weight | floor | sky_hinge_weighted pred=10 m | pred=100 m | pred=1000 m |
|---|---|---|---|---|---|
| A0, A0b, TS2, TS2+VK655 | 0.0 | (80 default, unused) | not logged (0) | not logged | not logged |
| TS2+H80 | 0.1 | 80 | 0.20794 (=0.1 ln(80/10)) | 0.0 | 0.0 |
| TS2+H300 | 0.1 | 300 | 0.34012 (=0.1 ln(30)) | 0.10986 | 0.0 |
Total loss differs between H arms and others only by that term at 10 m; equal at 1000 m. Proves: the hinge branch of _compute_loss_vda reads these two keys from the arm config and fires only in H80/H300, with the right floor, on TS2-labelled sky px. Does NOT prove: the run loop passes this cfg (train.py:1841 does, not exercised), effect on training dynamics, or behaviour on a mixed multi-source batch (hinge here is not metric_only-scoped, so it applies to all sky px of all sources; only TartanAir has sky under TS2).

## O15 items
(1) done above for degrade false and true (max abs diff 0.0). (2) done: only declared keys, run name/ckpt dir not touched by me. (3) default table used (no ts2_env_table); data.tartanair_ts2.table_sha256() = f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76. DEFECT: train.py does not log it at startup: train.py:302 builds TS2Labeller (which does not keep the table path or hash) and no table_sha256 call exists anywhere in train.py, so the run would NOT log it (the hash can only be verified from the repo file). (4) done in 5.

## Not verified
ScanNet++ source build; the 74-env/Isambard trees and the real 74-env seg set; model forward/real training step; that data-stream order is identical across arms (not run); mixed-batch collate with all 9 sources; SiLog etc. on TartanAir TS2 batch.

## Follow-up (R+D, defect fix)
Defect "TS2 table hash not logged at startup" fixed in train.py (tartanair ts2 branch): prints class, seg_root and env_table_sha256 (expect f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76). Run submit step must also set the screen run name / checkpoint dir / resume_from (c3_n46 step12000) / max_steps 2000 identically in all arms (the base yaml carries C3 run fields).

# Confirmatory proposal: TS2+H300 sky placement (Track 1 STRICT)
STATUS: DRAFT for the user. Nothing is submitted, downloaded, scored or predicted. Needs: user approval, Auditor token (O34 S-1..S-6), per-job Manager OK, registration filed BEFORE any payload is opened or any prediction made.
Marked [UNVERIFIED] = from memory or not checked in this session. Verified items cite a file.

## 0. Disclosures (Auditor O34 S-1, S-6)
- TS2+H300 was picked AFTER the screen, on a DESCRIPTIVE secondary (KITTI frame-balanced sky median 173.8 (TS2) -> 491.9 (H300); VKITTI 301.6 -> 561.0). On the screen primary (VKITTI 80-655 m, |1-x| legacy median) it was NOT BETTER: 0.13626 vs A0 0.13721, NF 0.00586 (ABLATION_CANDIDATES final entry). The screen's own N tried = 6; this run is a selection from them.
- Reading guards carried over (Auditor O32): one A0/A0b pair understates seed variance (NYUv2 AbsRel 0.0974 vs 0.0767; KITTI sky median 154.1 vs 141.4); the continuation alone moved the primary; hinge was active (nonzero in all 200 logged rows).
- A recipe change is a NEW recipe. The paper is FT-only; any paper route for a sky-hinge recipe is the user's decision, not implied by this run.

## 1. Question
Does the one-sided sky hinge (weight 0.1, floor 300 m) on top of TS2 raise predicted SKY depth on real-world frames never scored by us, versus TS2 trained with the same seeds, without hurting near-field depth or temporal stability?
Why H300: it is the only hinge arm whose sky medians moved (VK655 also moved, but is by-construction and hurt VKITTI AbsRel). Honest caveat: the hinge floor (300 m) caps what the loss can ask for; true sky is effectively unbounded; the release predicts ~576-674 m. "Higher" is declared the good direction only as sky placement toward the release/physical far plane, not as a depth-accuracy claim (no sky GT exists).
Exploratory vs confirmatory: the screen numbers above are exploratory. Only the new-set primary (section 3) is confirmatory.

## 2. Design
Held fixed: TS2 loader rule (sha in SCREEN_PROPOSAL; ts2_env_table f19d0b83...), C3 recipe, scorer pins, host type GH200, 4 GPUs/run, batch 16 clips/step, S3 continuation LR schedule if (a).
Arms (Auditor/Manager binding): CONTROL = TS2 at the SAME fresh seeds; TREATMENT = TS2+H300 (only `sky_hinge_weight 0.1`, floor 300 m differs). OPTIONAL description-only reference = A0 at the same seeds (shows the TS2 effect itself; not in the decision).
Fresh seeds: 50, 51, 52 (not 45-49). Seed sets both data.seed and resume_reseed (as A0b did). Control and treatment share the seed => identical batch order and noise draws; paired difference per seed.

### Option (a): 2000-step continuation (the screen's schedule)
Start c3_n46 step12000 (sha in checkpoints_v3/eval_20261012_c3rep SUMS), S3 LR (100-step warmup to peak from step 12000, cosine to 0 at 14000, opt-in lr_restart keys), steps 12000-14000. Measured: 5228-5451 s/arm = ~1.45-1.51 h on 1 node (4 GH200); screen total incl. scoring 9.43 NHR for 6 arms.
Weakness (stated in the screen): compression appears by step 6000 of the 12k run, so this tests recovery on a settled model, not prevention. Single start checkpoint: seeds cover continuation noise only. Optional: use n47/n48 step12000 as starts for seeds 51/52 (they exist as c3rep replicates; availability on Isambard [UNVERIFIED]); this adds start-model variance and is the better design, but changes "same start" across seeds (still paired within a seed).

### Option (b): from-init 12k
"From-init" in the C3 recipe (C2_config_train/FINDINGS.md): init = released Metric-VDA-S (ViT-S encoder frozen, DPT head + motion modules trainable), 12000 steps, AdamW 1e-4, warmup 500, cosine to ~0, 16 clips/step; TS2 loader and (treatment) the hinge active from step 0. Measured: job 7226119, 08:36:42 elapsed on 4 GH200 = 8.61 NHR/run. Tests prevention, not recovery; closer to the paper recipe; 5.7x the cost of (a) per run.

### Costs (NHR = node-hours; scoring ~0.05 NHR/run incl. DDAD, from the screen; runs include scoring)
| Seeds/arm | Runs (TS2 + H300) | (a) NHR | (b) NHR | + optional A0 same seeds: (a) / (b) |
|---|---|---|---|---|
| 1 | 2 | 3.1 | 17.3 | 4.7 / 26.0 |
| 2 | 4 | 6.2 | 34.6 | 9.3 / 51.9 |
| 3 | 6 | 9.3 | 51.9 | 14.0 / 77.9 |
Wall time per run: (a) ~1.5 h, (b) ~8.7 h; runs on separate nodes in parallel if the allocation allows [UNVERIFIED], else sequential: (a) 2 seeds = 6 h; (b) 2 seeds = 35 h.

### Recommendation
(a) with 3 seeds per arm, TS2 control + H300 (9.3 NHR), starts n46/n47/n48 if available, A0 optional (+4.7). Reason: 1 seed cannot size noise (the screen's A0/A0b pair shows 26% spread on a guard); 2 seeds gives only sign agreement; 3 allows "all three same sign and magnitude > spread". (b) is the more faithful test of the recipe but 52 NHR for 3 seeds; recommended only as a follow-up IF (a) confirms (single seed pair of (b), 17.3 NHR, then). Minimum acceptable per Auditor S-4: 2 seeds with (a) (6.2 NHR); 1 seed needs a reasoned case and gives sign only.
User picks: option, seeds, A0 yes/no, n46-only vs n46/47/48 starts.

## 3. Primary on a NEW real-world set (S-2, S-3)
Prediction-only (no GT depth needed). DDAD, KITTI DEV, VKITTI, TartanAir are EXCLUDED from the primary (DDAD reported only as a guard).
Statistic P (declared now): for each frame, median predicted depth over its certain-sky pixels; set value = median over frames of P (frame-balanced, frames with >= 2000 certain sky px at 378-short-side input; count of frames dropped is reported). Per seed s: d_s = ln(P_H300,s / P_TS2,s). Secondary description: frac of sky px > 200 m and > 300 m.
Direction/rule (proposal, user/Auditor may change before filing; margin is chosen from the screen noise, not from H300's screen effect): CONFIRMED iff d_s >= ln(1.5) = +0.405 for EVERY seed AND mean(d_s) >= ln(1.5) AND all guards in section 4 pass. Basis: screen A0 vs A0b gave |ln ratio| = 0.09 (KITTI), so 0.405 is ~4.5x that; screen H300/TS2 on KITTI was ln 2.83 = 1.04, so a real effect should clear it. With 2 seeds: both seeds. Sign positive but < margin on any seed = NOT CONFIRMED (inconclusive). Negative on any seed = REFUTED.
No claim about depth accuracy of sky; claim only "predicted sky depth is >= 1.5x higher".

### Candidate survey (facts marked; disk = checked read-only)
| Set | On disk | Licence / access | Far LiDAR GT | Sky mask source | Size / frames | Domain vs KITTI/VKITTI/TA/DDAD |
|---|---|---|---|---|---|---|
| nuScenes val (CAM_FRONT + LIDAR_TOP) | YES: /mnt/DATA/DegradedDepth/data/TEST/nuscenes (trainval keyframe bundles + v1.0-trainval meta, 98 GB dir; ramen_eval PROTOCOL doc) | CC BY-NC-SA 4.0 (nuScenes terms) [UNVERIFIED]; public AWS mirror | LiDAR 32-beam, sparse; far (>80 m) returns few [UNVERIFIED]; weak far guard | none; needs segmenter | 150 val scenes (official split), ~40 keyframes/scene at 2 Hz [UNVERIFIED]; 1600x900 | Boston + Singapore, different rig/FoV from KITTI (1242x375), DDAD (1936x1216); real, includes rain |
| Argoverse 2 sensor val | not on disk | CC BY-NC-SA 4.0 [UNVERIFIED] | 2x32-beam, ~200 m [UNVERIFIED]: better far guard | none; segmenter | 150 val logs [UNVERIFIED]; ring cams ~2048x1550 (portrait front) [UNVERIFIED]; ~1 TB dataset, subset by log possible [UNVERIFIED] | 6 US cities; new |
| Cityscapes val | not on disk | registration + approval, non-commercial [UNVERIFIED] | no LiDAR; SGM disparity, far noisy | HUMAN sky labels (trainId 10): no segmenter needed | 500 annotated frames, 2048x1024; video snippets are a large separate package [UNVERIFIED]; single frames = 1-frame clips (deviation: temporal modules inactive) | German cities, daytime |
| Waymo Open val | no | custom non-commercial licence, sign-in [UNVERIFIED] | 75 m top LiDAR [UNVERIFIED] | sparse human panoptic subset only | 202 segments, hundreds of GB | US; too heavy |
| KITTI-360 | no | CC BY-NC-SA 3.0 [UNVERIFIED] | yes | human sky labels | Karlsruhe = KITTI's city/rig: NOT new evidence | reject |
| ONCE | no | non-commercial [UNVERIFIED] | 40-beam | none | big | China; not needed |
RECOMMENDED PRIMARY: nuScenes val, scenes NOT in nuscenes_night_v1 and not otherwise scored. Verified: that manifest uses 30 val scenes (15 night + 15 day control; scene-0100.. and scene-1059..1073), so ~120 val scenes were never selected by that benchmark. Whether ANY model ever ran on the others (e.g. other benchmark jobs) is NOT verified: the registration must include a grep of results/logs for nuScenes scene names and an Auditor check. Day scenes only (official scene description without "night"); rain kept. No download needed.
FALLBACK: Argoverse 2 sensor val (real ~200 m LiDAR gives a usable far guard; new domain). Plan: DOWNLOAD ONLY AFTER APPROVAL; ring_front_center images + lidar for a rule-selected subset of val logs (e.g. every 5th log in sorted uuid order, ~30 logs) via the public S3 listing (s5cmd/aws --no-sign-request) [UNVERIFIED]; sizes from a listing (HEAD) before pulling; no sha published [UNVERIFIED] so record per-file sha256 and sizes at pull and verify on the second host; frame list from file names only. Cityscapes is the alternative if the user wants human sky labels (removes segmenter risk) at the cost of approval wait and 1-frame clips.

### Sky mask (S-2; no prediction yet)
No segmenter is on disk (checked ~/.cache/huggingface/hub listing, torch hub, find by name: none). Plan, download only after approval, small weights: (i) SegFormer ADE20K (nvidia, licence "other"/NVIDIA research [UNVERIFIED]), sky = ADE20K-150 index 2 (0-based; "sky") [UNVERIFIED id]; (ii) Mask2Former or OneFormer Cityscapes (licence [UNVERIFIED]), sky = Cityscapes trainId 10. Rule (declared before any run): sky pixel = both models say sky AND softmax prob >= 0.9 in each AND not within 8 px (at 1600x900) of any non-sky class (erosion) AND no projected LiDAR return within 5 px AND above the horizon row [if horizon is known from calibration; else dropped]. Everything else near the boundary = uncertain, excluded. Mask computed on the original RGB (not on any model output of ours), at native resolution, nearest-neighbour resize to the prediction grid. Pin both checkpoint shas. Class-id lookup is asserted by name from the checkpoint's id2label, not hard-coded.
Mask checks with negative controls: (1) hand-inspect a rule-selected 12-frame panel (frames from the list by fixed stride, drawn BEFORE seeing any prediction); (2) forced-fail: mask shifted 50 px / flipped vertically must trip the "no LiDAR return in sky" check and the sky-fraction-by-image-row check; (3) night/rain frames: sky-fraction distribution reported; (4) agreement IoU of the two segmenters reported, threshold declared in the registration (proposal 0.85) else the set is rejected.

## 4. Guards (named, each with its own rule)
All unaligned, pinned bench_joint 5d3d048c and scorer a8b0731e at e0f6702 (screen pins), GH200, step 14000 only, same-seed TS2 as reference, 5% relative tolerance on near-field AbsRel (proposal; the screen's seed spread was up to 26% on NYUv2 so 1% is not usable; the paired same-seed difference should be smaller, but the user/Auditor must confirm 5%).
| Guard | Rule |
|---|---|
| KITTI, NYUv2 (first 160 frames), ScanNet++ DEV AbsRel | H300 not worse than TS2 same seed by > 5% rel, every seed |
| TAE (video_macro.tae_gtvalid) KITTI + ScanNet++ | same 5% rule |
| New-set ground/near depth vs LiDAR (nuScenes, bins 0-20, 20-50, 50-80 m, pooled AbsRel) | not worse by > 5% rel (fallback AV2 adds 80-200 m) |
| DDAD (exposed, report-only) bins incl. [80,120) [120,200] | reported, no rule; large deterioration is stated |
| Hinge activity | sky_hinge_weighted nonzero in logged rows (as in the screen) |
| VKITTI far-ground primary of the screen | reported only; it did not move in the screen and is NOT the decision |

## 5. Pre-registration structure (copy of LR_DEV)
File outdoor_debug/parts/confirm/REGISTRATION.md, filed with `date -u` BEFORE any payload is opened or prediction made (tar/metadata listing and docs only), containing: dataset, split rule, scene list from names (official val split minus the 30 night_v1 scenes minus night-described scenes; ordered by sha256 of "sky-confirm-v1" NUL scene name; take all or first K, K declared), frame rule (all keyframes of the scene as one clip, scored frames list), list sha, training-overlap check (nuScenes is not in any TRAIN source [TRAIN listing: bvi_aom, irs, kitti_raw, nyuv2, point_odyssey, scannetpp, spring, tartanair, vkitti, youtube_vos]; pre-training data of the released Metric-VDA-S/DAv2 [UNVERIFIED], stated as a limitation), licence, resolution rule (short side 378, multiple of 14, bilinear to native for GT sampling), statistic P and rule/margin (section 3), guards (section 4), scorer pin (sha, tests), reference row, forced-fail controls, negative control for EVERY check (shifted/flipped mask, scrambled frame-list sha, wrong-checkpoint sha, empty-sky frame, synthetic prediction with known sky depth through the full scorer returns exactly that depth), and a second reviewer for checks written by subagents.
Sealing: arm outputs (scores.json, sky stats) sealed until (1) check script passes 100% with forced-fail inputs failing, (2) pins asserted (scorer sha, bj sha, ckpt step 14000 + sha, host GH200), (3) reference row (start c3_n46 step12000 through the same scorer) reproduced, (4) the Auditor is told. Validity is pass/fail only before the seal lifts; the decision script (like readout.py) is committed before it is run on any arm output.
Auditor token needs: (i) registration + rule + margin approval; (ii) scorer/segmenter pins and mask panel; (iii) arm configs (arm_checks pattern: config diff = exactly the hinge keys vs TS2, seed check, LR table); (iv) per-job submission OK (Manager); (v) readout audit; (vi) any paper use.
Exploratory: the A0 reference arm, DDAD, VKITTI, the sky frac>200/300 m secondaries, any (b) follow-up, anything not in the rule. Confirmatory: only the P rule + guards above.

## 6. Risks and kill criteria
- Seed noise: the screen's seed spread is large; mitigated by same-seed pairing and 3 seeds; if the three d_s straddle zero, report NOT CONFIRMED.
- Hinge cap: floor 300 m; the loss asks only for >= 300 m, so H300 medians near 300-560 m are the floor's signature, not a calibrated far plane. A "confirmation" means the hinge works as designed on real sky, not that sky is metric-correct.
- Hinge acts on sky labels: in the screen the TS2 sky mask is TartanAir seg-sky and the VKITTI sentinel; transfer to nuScenes depends on the model generalising "sky-like" regions; the segmenter mask may differ from the training sky concept (clouds, bright overcast, buildings/haze). Report P by weather where descriptions allow.
- Segmenter error biases P (non-sky contaminants lower P). Same mask for both arms, so the bias cancels in d_s but not in absolute P.
- Continuation (a) may understate or overstate what a from-init run does; (b) is the follow-up.
- Near-field cost: raising far sky may shift near scale (screen guards were within NF at seed 45 only).
- Selection: S-1 above; the result is a test of one selected arm, N tried stays disclosed (6 screen + 1 confirmatory pair).
KILL: stop and report if (1) the first-run stop check (logged per-group LR vs table) mismatches; (2) loss spike > 2x A0 step-12000 loss in the smoke; (3) any non-finite or checkpoint-sha mismatch; (4) the mask validity checks fail (set rejected, fallback considered under a NEW registration); (5) the reference row does not reproduce.
Outcomes: CONFIRMED (rule met on all seeds + guards) = "H300 raises real-scene sky depth >=1.5x versus TS2 at same seeds on nuScenes val-unscored", nothing about accuracy. NOT CONFIRMED / REFUTED = report as such; no second try on the same set without a new registration (the set is then exposed).

## 7. Open questions
User: option (a)/(b); seeds 2 or 3; A0 reference (+1.55 NHR per seed in (a)); 1.5x margin and 5% guard tolerance; nuScenes primary with AV2 fallback or Cityscapes; n46-only vs n46/47/48 starts; whether a from-init follow-up is wanted.
Manager: Isambard availability of c3rep n47/n48 step12000; nuScenes copy to Isambard (a rule-selected scene subset only; size of that subset unmeasured); download approvals (segmenter weights, small).
Auditor: ruling on the segmenter pair and thresholds; whether nuScenes val-unselected counts as "never scored by us" (needs the log grep); who is the second reviewer for the mask/scorer code; margin and tolerance.

## Auditor O35 integration (draft, 2026-10-17)
- FRAMING (put first in the user-facing summary): the primary P measures what the H300 hinge enforces by design, so a PASS is near-built-in and shows the transfer of the hinge's effect to new data, NOT accuracy. The GUARDS carry the information.
- Q1 dataset: nuScenes val-unselected only if (i) the scene list comes from the official split + descriptions only and (ii) a read-only grep of every selected scene name AND token across results/, claude_outputs/, qualitative/, audit/, session_briefs/, paper_cvpr2027/ and all benchmark/job logs returns zero hits (output committed; a hit drops that scene). The held-out TEST area stays closed until the Auditor and the user clear it for this purpose. AUDITOR RECOMMENDATION (the user chooses): Argoverse 2 sensor val as the primary set: never downloaded, ~200 m LiDAR allows an 80-200 m AbsRel guard (nuScenes far returns are sparse). With AV2, add an 80-200 m AbsRel guard at 5% relative to same-seed TS2. Download only after approval; registration first.
- Q2 segmenters accepted as written, plus: both checkpoint shas and licences in the registration; the 12-frame stride panel inspected and committed BEFORE any prediction; dropped-frame count reported per arm; ONE frozen mask file (sha) shared by both arms.
- Q3 accepted: ln 1.5 on every seed AND on the mean; guards 5% relative per seed vs same-seed TS2.
- Q4 second reviewer: a fresh-context agent that did not write the code (Manager-spawned; receives only the registration, the code and the forced-fail spec), reporting pass/fail with evidence; the Auditor audits that report plus own check and is not the second reviewer.
- Starts: n46/n47/n48 step12000, one per seed (data.seed 50/51/52), is the better design. VERIFIED read-only by R+D 2026-10-17: all three exist on Isambard (/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n4{6,7,8}/step12000.pt); n47 sha256 2ee98b7e... and n48 bf053bd3... on the cluster equal the c3rep SUMS_local entries (n46 f5c7c8f9... verified earlier); local copies load with optimizer, rng, ema, trainable_ema, step 12000, micro 48000, epoch 2 and param-group initial_lr [1e-4, 1e-4, 1e-3] (resumable, R-b per start). The cluster-side resumability read is the same file as the local one by sha.

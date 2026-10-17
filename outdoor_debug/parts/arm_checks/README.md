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
Total loss differs between H arms and others only by that term at 10 m; equal at 1000 m. Proves: the hinge branch of _compute_loss_vda reads these two keys from the arm config and fires only in H80/H300, with the right floor, on TS2-labelled sky px. Does NOT prove: the run loop passes this cfg (train.py:1841 does, not exercised), effect on training dynamics, or behaviour on a mixed multi-source batch (hinge here is not metric_only-scoped, so it applies to all sky px of all sources; [CORRECTED per Auditor O20: in A0/TS2/H80/H300 the VKITTI sentinel is ALSO sky, so the hinge acts on VKITTI sky too; under VK655 VKITTI has no sky] TartanAir has TS2 sky).

## O15 items
(1) done above for degrade false and true (max abs diff 0.0). (2) done: only declared keys, run name/ckpt dir not touched by me. (3) default table used (no ts2_env_table); data.tartanair_ts2.table_sha256() = f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76. DEFECT: train.py does not log it at startup: train.py:302 builds TS2Labeller (which does not keep the table path or hash) and no table_sha256 call exists anywhere in train.py, so the run would NOT log it (the hash can only be verified from the repo file). (4) done in 5.

## Not verified
ScanNet++ source build; the 74-env/Isambard trees and the real 74-env seg set; model forward/real training step; that data-stream order is identical across arms (not run); mixed-batch collate with all 9 sources; SiLog etc. on TartanAir TS2 batch.

## Follow-up (R+D, defect fix)
Defect "TS2 table hash not logged at startup" fixed in train.py (tartanair ts2 branch): prints class, seg_root and env_table_sha256 (expect f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76). Run submit step must also set the screen run name / checkpoint dir / resume_from (c3_n46 step12000) / max_steps 2000 identically in all arms (the base yaml carries C3 run fields).

# ADDENDUM 2026-10-17 (Add.45 / Auditor O17): continuation LR blocker, resumability, seeds, stop check
Files: train.py (`make_lr_scale`), tests/test_lr_restart.py (6 tests), arms.py (regenerates configs/*, config_diffs.md), lr_tables.py/.txt, seed_check.py/.txt/.json, first_run_stop_check.md. No commit made.

## LR blocker: confirmed and fixed (opt-in keys)
With the C3 yaml (warmup 500, lr_decay_steps 12000) and resume at gstep 12000, the real schedule gives lr_scale = 0.0 for every step 12000..13999 (table S0 below): every arm would have been a no-op. Fix: `train.make_lr_scale(t, warmup, total_steps)` (the old closure moved to module level, unchanged formula) + opt-in keys `training.lr_restart_step` R, `lr_restart_warmup` W, `lr_restart_peak` P (all three or none, else ValueError; also ValueError unless W>=1, 0<P<=1, lr_decay_steps > R+W). Keys absent = the old formula (test: exact float equality old vs new on steps 0..14000 for the C3 config, and for lr_decay_steps 14000 / 1e9 / absent).
S3 formula (step >= R): W-warmup `P*(step-R+1)/W` for step-R < W, then `P*0.5*(1+cos(pi*(step-R-W)/(lr_decay_steps-R-W)))` clamped at prog 1. OFF-BY-ONE CONVENTION: same as the base warmup, `(k+1)/W`; lr_scale(gstep) is evaluated BEFORE `gstep += 1`, so the first update after resume (gstep 12000) uses 0.01 (not 0) and gstep 12099 is the first at 1.0. The stdout `lr=` printed at step G is lr_scale(G-1).
Arms (Add.45) bake S3: lr_decay_steps 14000, restart 12000 / 100 / peak 1.0, max_steps 14000 (1900 cosine steps from 12100). configs/lr_S1.yaml, lr_S2.yaml, lr_S3.yaml are overlays (S3 = what the arms contain).
Optimizer groups, read from the step-12000 checkpoint (`lr_tables.py <ckpt>`): 3 param groups, initial_lr [1e-4 (188 tensors), 1e-4 (64), 1e-3 (2 tensors = gates g,g2, gate_lr_scale 10)]; saved current lr = 1.87e-12 / 1.87e-11 (scale at step 11999). Matches expectation (1e-4 main, 1e-3 gate); the trainer overwrites `g['lr']=g['initial_lr']*sc` every update, so stale saved lr is irrelevant.
Tables (scale; per-group LR = initial_lr x scale), real code, max_steps 14000:
| schedule | 12000 | 12001 | 12050 | 12100 | 13000 | 13050 | 13999 |
|---|---|---|---|---|---|---|---|
| S0 as-is C3 yaml | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1 lr_decay 14000 | 0.05318 | 0.05313 | 0.05060 | 0.04808 | 0.01348 | 0.01217 | 1.35e-08 |
| S2 lr_decay 1e9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| S3 peak 0.5 | 0.005 | 0.01 | 0.255 | 0.5 | 0.2706 | 0.25 | 3.42e-07 |
| S3 peak 1.0 (SELECTED) | 0.01 | 0.02 | 0.51 | 1.0 | 0.5413 | 0.5 | 6.83e-07 |
Selected S3 LR (main groups 1e-4 x scale / gate group 1e-3 x scale): 12000: 1.0e-06 / 1.0e-05; 12049: 5.0e-05 / 5.0e-04; 12100: 1.0e-04 / 1.0e-03; 13050: 5.0e-05 / 5.0e-04; 13999: 6.8e-11 / 6.8e-10. Full table: lr_tables.txt. Test (arm yaml -> real lr_scale): 12000 0.01, 12049 0.5, 12099 1.0, 12100 1.0, 13050 0.5, 13999 ~6.8e-7.
Optimizer state stays resumed (train.py loads it; Adam moments from step 12000 reused). Not tested: a real GPU resume run.

## R-a Arm YAMLs (configs/arm_*.yaml, regenerated by arms.py)
Common to all six: resume_from `/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt` (exists on Isambard, 137,093,822 B, mtime Oct 10 20:01), max_steps 14000, checkpoint_at_steps [12500,13000,13500,14000] (checkpoint_every_steps 500 still writes last.pt), val_every_steps 2000 (one val at 14000; unchanged), loss.log_terms true (already) + log_terms_every 10, S3 LR keys; checkpoint_dir `runs/outdoor_screen_<ARM>` per arm (the trainer has no separate run-name key; checkpoint_dir is the run identity). Resolved diffs (config_diffs.md): every pair differs only by its declared keys + checkpoint_dir; A0 vs the base yaml differs only by the common continuation fields; (L5) the 7 LR/length keys identical across all six arms: OK; log_terms/log_terms_every true/10 in all six. Unverified: the arm yamls were not launched (no GPU); the ts2_seg_root cluster path is unchanged.

## R-b Step-12000 checkpoint is resumable
Local: /mnt/DATA/DegradedDepth/checkpoints_v3/eval_20261012_c3rep/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt, sha256 f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af = SHA256SUMS_local entry = Isambard copy (`ssh b5dh.aip2.isambard sha256sum`, size 137093822). torch.load(cpu, weights_only=False) keys: cfg(5 sections), epoch=2, step=12000, micro=48000, best=0.34847, metrics(13), optimizer{state: 250 entries, param_groups: 3}, ema (340 tensors), rng{python,numpy,torch,cuda}, trainable (340), trainable_ema (340). No `model` key: this is a frozen-backbone checkpoint, trainer path `_load_resume_trainable` (uses optimizer + ema + trainable), as for the C3 run itself. Resume start_epoch = 3.

## R-c Seeds (seed_check.py, real code; output seed_check.txt/.json)
What fixes the post-resume stream, from the code:
1. Sampler (train.py `_build_multisource` l.~533-541): for world_size > 1 (the 4-GPU screen) a WeightedRandomSampler with its OWN generator seeded `data.seed*1000 + rank (+250 under seed_scheme x1000)` at construction; NOT part of the checkpoint rng and not touched by resume, so the post-resume order is that stream from its start, determined by data.seed alone. For world_size 1 the generator is None and the sampler draws from the global torch RNG (restored from the checkpoint, then reseeded if resume_reseed).
2. Dataset ctor `seed=data.seed` (only used for val clips / per-clip val noise; train noise is not seeded from it).
3. Noise/degrade/static draws: `seeded_noise: true` -> `torch.randint` from the (worker-seeded) torch RNG; DataLoader worker seeds come from the main-process GLOBAL torch RNG at iterator creation. Under resume the global RNG = `_restore_rng_state(checkpoint rng)` (identical for any data.seed: `seed_everything(data.seed)` at startup is overwritten), then `_start_state` applies `training.resume_reseed` -> `seed_everything(resume_reseed, rank, scheme)`.
So: data.seed changes the sampled clip order only when world_size > 1 and changes nothing about noise under resume; resume_reseed changes noise (and, at world_size 1 only, the sampler); A0b needs BOTH. (Side note, existing behaviour: without resume_reseed every rank restores rank 0's saved global rng, so ranks share noise seeds; same in all arms.)
Experiment (first 200 sampler indices -> clip ids (source, first frame path, start); fake resume rng identical for all arms; num_workers 0; scannetpp_train dropped because its split list is not local; noise = real dataset `__getitem__` frames of the same 2 clips under each variant's post-`_start_state` global rng):
| variant vs A0 | world 4: identical sampler idx / clip ids | world 4: same noise | world 1: identical idx / clip ids | world 1: same noise |
|---|---|---|---|---|
| TS2 | 200/200, 200/200 | yes (frames identical) | 200/200, 200/200 | yes |
| A0b data.seed 46 only | 0/200, 1/200 | yes (identical) | 200/200 | yes |
| A0b resume_reseed 46 only | 200/200 | NO (differs) | 0/200 | NO |
| A0b both (data.seed 46 + resume_reseed 46) | 0/200, 1/200 | NO | 0/200 | NO |
(i) A0 == TS2: first 200 clips identical (TS2 changes only labels/ sky mask; noise identical). (ii) A0b[both] differs in clips and noise at both world sizes.
RECOMMENDATION: A0 and all other arms keep data.seed 45 with no resume_reseed (identical sampler + noise across the five label/loss arms, which is what the nested comparison needs). A0b = `data.seed: 46` AND `training.resume_reseed: 46` (the yaml is now so). CAVEAT: c3_n46's own training used data.seed 46, so its sampler stream (from 6000 on, a fresh generator at resume) started from the same stream start: A0b at world 4 replays the SAME clip order that n46 saw right after its step-6000 resume (clips already seen; A0's seed-45 stream does not). If the user wants A0b to be unseen data, use an unused value (e.g. data.seed 49; 46-48 were the c3rep replicates). Not verified: DataLoader worker processes (num_workers 8) were not run; the claim about worker seeds comes from torch's DataLoader contract and the code reading; 4-rank behaviour was exercised through rank 0 of a world-size-4 sampler only.

## R-d First-run stop check: first_run_stop_check.md
Key facts: only group-0 LR is printed (every 20 steps; `lr=` at step G = initial_lr*scale(G-1)); `sky_hinge_weighted` is only emitted when loss.sky_hinge_weight > 0 (absent, not 0, in non-H arms) so the check is presence-based, and an all-zero value inside H arms is a WARN (hinge is one-sided: 0 if pred >= floor). Per-group LRs for groups 1-2 are not logged (L4 limitation, stated); expected group-0 prints are tabled.

## Tests
tests/test_lr_restart.py 6 passed; all 20 test files that import train (incl. test_resume_exactness, test_checkpoint_at_steps_warning, test_dn01_v3_wp18b, test_compute_loss_*): 145 passed, 2 skipped (run after the final train.py change).

## R+D decision on A0b seed (supersedes the seed-46 mention above)
A0b uses data.seed 49 + training.resume_reseed 49 (a seed not used by c3rep replicates 46/47/48, so the 4-GPU sampler order is not a replay of n46's post-resume clips). All other arms keep data.seed 45, no resume_reseed. Config diffs re-run: ALL EXPECTED DIFFS EXACT.

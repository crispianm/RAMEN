# C2_config_train: Phase 1 (config), 9, 10 static parts (subagent report, saved by the lead; the subagent could not write the file)
Files here: effective_{c3_seed45,ramen_n46,ft_n46}.yaml (train.load_config with the full extends chain), diff_table.md, cfg_vs_ckpt.py, mix_stats.*, src_depth_stats.*, ta_*.{py,csv,txt}, param_change*.{py,csv}, gradscale.py, c3_headline_loss_terms.csv. CPU only.
## Effective config
- The config embedded in each step12000 checkpoint equals the resolved effective config, key for key (0 differing keys, all three).
- Commits (from ledgers; checkpoints store no git hash): headline s45 c1abf81d61baa8ca3b337c1f79c9b02a29929144 (B-FIX job 7159803); n46 pair (jobs 7226119, 7226122) 0699ee4fac9fb3b71839c8d7ed0b62c2874dc5aa. The parent leg (r50_B step 6000) commit is not recorded in what was read.
- Differences: headline s45 resumes synreal_r50_B at step 6000 and continues to 12000 (first 6000 steps = old B recipe: TA/PO metric supervision off, VKITTI capped 80 m, no sky loss); the n46 pair is 12000 steps of C3 from the released Metric-VDA-S. FT drops the restorer (no shared_temporal, enh_weight 0, no gate_lr_scale). Sources, sampling masses, loss weights, sky settings, ceilings, LRs, freezing are identical across the three => the outdoor question is a SHARED-recipe question.
- Shared recipe: ViT-S encoder fully frozen (lora_rank 32 dormant); init = released Metric-VDA-S; output = ReLU depth in metres, no upper bound; DPT head + motion modules trainable (RAMEN also restorer + 2 gates). Loss: VideoDepthLoss on 1/clamp(pred,0.1,1e5) + metric SiLog (w 1.0, lambda 0.5); sky enters VDA loss at disparity 0 (sky_vda_zero_disparity true), sky_hinge_weight 0.0, sky excluded from the SiLog mask; silog_max_depth 655 for TartanAir and both PointOdyssey sources only; depth bounds TartanAir 10000, VKITTI 655, PointOdyssey 999, IRS/Spring/KITTI 80, NYU/ScanNet++ 10. AdamW lr 1e-4, wd 0, warmup 500, cosine to ~1.9e-12 at 12000; clip 1.0 per module; grad_accum 4; batch 1/GPU x 4 GPUs (16 clips/step); bf16; EMA decay 0 (EMA bit-identical to live weights); head_lr_mult 1.0; gate lr 1e-3; 80% of clips degraded (awgn_blur + lowlight).
## Sampling (Phase 9)
sampling_mass is a per-source total, P(source)=mass/sum; per-clip weight = mass/len, with replacement (train.py:499-525); WeightedRandomSampler simulated on 4 ranks matches the config; sampler runs in the main process.
| source | P(source) | clips | repeats per clip over 12k steps |
|---|---|---|---|
| TartanAir | 27.9% | 9781 | 5.5 |
| VKITTI | 3.6% | 1280 | 5.4 |
| PointOdyssey train | 7.9% | 7204 | 2.1 |
| PointOdyssey val (used as training source) | 1.2% | 1059 | 2.1 |
| IRS | 9.0% | 3191 | 5.4 |
| Spring | 0.4% | 134 | 6.3 |
| KITTI | 27.0% | 1277 | 40.6 |
| NYU | 15.4% | 1307 | 22.5 |
| ScanNet++ | 7.7% | 54585 | 0.27 |
Mix: indoor-only sources (IRS, NYU, ScanNet++) 32.0% of clips; KITTI+VKITTI 30.6%; ~75% of TartanAir frames outdoor-type (21.1% of all clips); 99.6% of clips metric-supervised; ~51.7% metric-supervised outdoor; sky-supervised clips 31.5% (TartanAir + VKITTI only), sky pixels 2.95% of all pixels, no real-world sky supervised anywhere; valid far pixels 5.7% at >=30 m, 2.85% at >=80 m (TartanAir supplies 2.59 of 2.85 points), 1.96% at >=200 m; KITTI LiDAR covers 15% of pixels, 1.3% at >=30 m, 0% at >=80 m, no sky; PointOdyssey has no sky label and 14% of train pixels are invalid and unsupervised. (Pixel statistics measured on local train data; indoor-only sources not measured and treated as 0 sky/0 far.)
## Optimisation (Phase 10)
Metric head trainable and in the optimiser for RAMEN and FT (64 tensors in group 1, lr 1e-4; train.py asserts optimiser params == trainable params); all 6 output_conv tensors changed; only 4 tensors in the unused refinenet4.resConfUnit1 never get gradients. Relative L2 change vs released Metric-VDA-S in all three: projects 0.17-0.18, resize_layers 0.20, scratch 0.15, motion modules 0.07-0.08, final 1x1 output weight 0.12-0.14. Scheduler matches 12000 steps; EMA == live weights; resume restores optimiser/Adam/RNG at the correct step; post-clip gradient RMS 0.92-1.0 of the 1.0 clip (from Adam second moments) => clip active on almost every step (pre-clip norms never logged); validation is Spring only (metric-unsupervised, 20 clips), final step used; only outdoor monitor is in-sample VKITTI median depth (ratio 0.98-1.05); headline loss terms VDA total ~0.39-0.43, SiLog 0.26 -> 0.18, 397/400 clips metric-supervised.
## Findings
| ID | issue | file:line | severity | status | fix / next |
|---|---|---|---|---|---|
| C2-01 | No metric sky term: sky_hinge_weight 0.0; sky enters only the scale/shift-invariant VDA loss; SiLog mask uses m not m\|sky | train.py:1155-1159,1201 | high (config fact) | CONFIRMED config; outdoor impact UNRESOLVED | test sky_hinge_weight (+ sky_hinge_metric_only, code exists, opt-in) |
| C2-02 | Real far and sky supervision is nil: far/sky metric targets synthetic only (VKITTI 3.6%, TartanAir 27.9% of clips); GT >=80 m is 2.85% of mix pixels | configs; data/kitti_raw_train.py | high | CONFIRMED (measurement) | dense far data / upweight VKITTI far |
| C2-03 | KITTI clips drawn 40.6x each, NYU 22.5x each over 12k steps | train.py:499-525 | medium-high | CONFIRMED (arithmetic + simulation); harm UNRESOLVED | cut kitti/nyu mass, raise VKITTI, short ablations |
| C2-04 | DPT head moves 15-20% from the release at full lr (head_lr_mult 1.0); the release wins outdoor TAE => forgetting possible | train.py:115; cfg | medium | movement CONFIRMED; harm UNRESOLVED | head_lr_mult 0.1 (needs training) |
| C2-05 | Grad clip 1.0 active on almost every step | train.py:1540,2852 | low-medium | UNRESOLVED (pre-clip norms needed) | log pre-clip norms; try clip 5-10 |
| C2-06 | Headline s45 = 6000 steps old B recipe + 6000 steps C3, differs from the n46 pair | cfg resume_from | medium (attribution) | CONFIRMED | use the n46 pair for attribution |
| C2-07 | On resume the sampler generator is re-seeded, not restored: second leg replays first leg's draw order | train.py:520-522 | low | CONFIRMED by reading | save generator state |
| C2-08 | Validation/monitoring blind to far and sky | cfg val_sources, eval.outdoor_monitor | medium (process) | CONFIRMED | add a held-out far/sky monitor |
| C2-09 | TartanAir exactly 10000 m neither valid (<1e4) nor sky (>1e4) (0.00-0.01%); 5.1% of TA valid pixels lie at 2000-9999 m where they act like sky | data/vda_semantics_dataset.py:99-101 | low | CONFIRMED, negligible | use >=1e4 for sky |
| C2-10 | PointOdyssey: no sky label, 14% pixels invalid | data/pointodyssey_dataset.py:145 | low-medium | CONFIRMED | sky mask from depth >=998 |
| C2-11 | A non-finite micro-batch skips backward on one rank only (DDP desync risk) | train.py:2837-2841 | low | RULED OUT in practice (logged skips 0) | allreduce the finite flag |
| C2-12..14 | frozen/dead output layer; config resolve/inheritance error; EMA/scheduler/weight-decay/loss-scaling mismatch | n/a | none | RULED OUT | none |
## Ranking (plausibly hurts outdoor/far/sky, RAMEN and FT): 1. C2-02 with C2-01; 2. C2-03; 3. C2-04; then C2-06, C2-08, C2-05, C2-09/10/07.
## Prior work: the A_sky REPORT said the sky premise is not supported for the TAE question; this audit does not contradict it, it adds that the metric sky term is off and sky is only 2.95% of pixels. METRICFIX/SKYSCALE opt-in keys already exist in train.py; C3 uses silog_max_depth and sky_vda_zero_disparity but leaves the hinge at 0.
## Needs GPU/real batches/decision: pre-clip per-module and per-dataset/loss gradient norms (Phase 11); short ablations (sky hinge metric-only; head_lr_mult 0.1; kitti/nyu mass cut + VKITTI up); decision to use the pure-C3 n46 pair, not the s45 hybrid, for attribution.

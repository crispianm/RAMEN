# RAMEN outdoor metric depth debugging plan

Branch `outdoor-debug` is a temporary research branch on RAMEN, created at the user's request on 2026-10-16.
Before the public release it must be either merged (confirmed fixes with regression tests only) or deleted.
RAMEN `main` stays reproducibility-only.

Plan text below is the user's, verbatim.

```text
RAMEN OUTDOOR METRIC DEPTH DEBUGGING PLAN

OBJECTIVE
Find and fix training bugs causing poor metric depth predictions in outdoor scenes, especially distant geometry, the horizon, and sky.

IMPLEMENTATION RULES
- Work through the phases in order.
- Verify the actual code before assuming a bug exists.
- Make one functional change at a time.
- Keep the current training configuration as the baseline.
- Use short, reproducible experiments before full training runs.
- Record code changes, configuration changes, loss values, gradient statistics, and evaluation results.
- Do not treat invalid depth, distant valid geometry, and sky as equivalent.
- Do not silently clamp invalid pixels into valid training targets.
- For every confirmed bug, report the exact file, line, root cause, fix, and regression test.
- Do not implement speculative fixes without evidence.

============================================================
PHASE 1: ESTABLISH THE BASELINE
============================================================

[ ] Identify the exact Git commit, training configuration, parent configurations, and checkpoint being investigated.
[ ] Resolve all inherited configuration values and save the complete effective configuration.
[ ] Identify the model's actual metric-depth output activation, scaling, minimum depth, and maximum depth.
[ ] Record all training datasets, sampling probabilities, loss weights, augmentations, and depth limits.
[ ] Run inference on representative indoor, outdoor, and sky-heavy samples.
[ ] Save RGB, predicted depth, ground-truth depth, validity masks, and sky masks.
[ ] Calculate AbsRel, RMSE, delta1, and scale-invariant log error separately for indoor and outdoor data.
[ ] Calculate metrics separately for 0-10 m, 10-30 m, 30-80 m, and greater than 80 m, wherever ground truth supports those ranges.
[ ] Measure the percentage of predictions near the model's minimum and maximum output values.
[ ] Record median predicted sky depth and predicted horizon depth.
[ ] Compare raw metric predictions with predictions after optimal scale alignment.
[ ] Save the baseline results before changing anything.

============================================================
PHASE 2: AUDIT THE MODEL AND OUTPUT REPRESENTATION
============================================================

[ ] Inspect the upstream VideoDepthAnything metric head implementation.
[ ] Verify whether metric predictions are depth, inverse depth, normalized depth, or scaled depth.
[ ] Check for sigmoid, ReLU, softplus, exponential, or other final output activations.
[ ] Check for hardcoded 10 m, 80 m, 100 m, or other output limits.
[ ] Check whether the model's maximum depth is configurable.
[ ] Check whether the training configuration actually passes the intended maximum depth into the model.
[ ] Check whether checkpoint loading overwrites or preserves output scaling.
[ ] Check whether a metric checkpoint is loaded into a compatible metric architecture.
[ ] Check for missing, unexpected, or silently ignored checkpoint keys.
[ ] Verify that the prediction-is-disparity flag matches the actual output representation.
[ ] Verify every depth-to-disparity and disparity-to-depth conversion.
[ ] Check reciprocal operations for zeros, NaNs, infinities, and negative values.
[ ] Check all depth clamps and their effect on gradients.
[ ] Check whether output resizing or interpolation changes metric scale.
[ ] Check whether any model component assumes a specific depth range.
[ ] Verify that inference, training, validation, and checkpoint evaluation interpret predictions identically.
[ ] Run synthetic unit tests with known depths of 1, 5, 10, 30, 80, 200, and 600 m.
[ ] Confirm that the output head can represent every depth range being supervised.
[ ] If output range is restricted, test a correctly initialized wider-range or unbounded head.

============================================================
PHASE 3: AUDIT DATASET DECODING AND DEPTH UNITS
============================================================

For every dataset:

[ ] Verify depth file format and original encoding.
[ ] Verify correct depth decoding and scaling into metres.
[ ] Verify handling of uint16, uint32, floating-point, and compressed depth formats.
[ ] Verify handling of zero, negative, NaN, infinity, and sentinel values.
[ ] Verify interpretation of sky and missing measurements.
[ ] Verify validity mask before resizing and augmentation.
[ ] Verify depth range and clipping behaviour.
[ ] Verify correspondence between RGB frames and depth frames.
[ ] Verify camera selection, sequence indexing, and temporal ordering.
[ ] Verify train/validation split without unintended leakage.
[ ] Verify camera-coordinate depth versus Euclidean range, where applicable.
[ ] Verify that no obsolete depth normalization is inherited from older training code.

Dataset-specific checks:

[ ] KITTI: Verify 16-bit PNG scaling by 256 where that encoding is used.
[ ] KITTI: Verify that sparse zero-depth pixels are invalid.
[ ] KITTI: Verify valid LiDAR returns remain correctly positioned after resizing.
[ ] KITTI: Verify the actual maximum available measured depth.
[ ] VKITTI2: Verify PNG scaling by 100.
[ ] VKITTI2: Verify that 65535 is treated as a sentinel, not a real 655.35 m measurement.
[ ] VKITTI2: Verify that valid distant geometry is not confused with the sentinel.
[ ] VKITTI2: Verify the sky mask against RGB images.
[ ] TartanAir: Verify whether sky uses infinity, a large finite value, or another encoding.
[ ] TartanAir: Verify legitimate far geometry is not classified as sky.
[ ] TartanAir: Verify the consequences of the configured 10,000 m depth limit.
[ ] PointOdyssey: Verify depth units, maximum-depth handling, and invalid-depth encoding.
[ ] IRS: Verify whether depth represents metric distance, disparity, or another quantity.
[ ] Spring: Verify depth interpretation and whether metric supervision is intentionally disabled.
[ ] NYU: Verify metre scaling and indoor depth range.
[ ] ScanNet++: Verify metre scaling, invalid-depth handling, and alignment.
[ ] Compare depth distributions across datasets after preprocessing.

============================================================
PHASE 4: AUDIT RESIZING, CROPPING, AND AUGMENTATION
============================================================

[ ] Trace RGB, depth, validity mask, and sky mask through every transform.
[ ] Verify that all four use identical spatial crop coordinates.
[ ] Verify RGB and depth have matching dimensions after preprocessing.
[ ] Use nearest-neighbour interpolation for sparse depth and discrete masks.
[ ] Check whether dense depth is resized using bilinear interpolation.
[ ] Check whether interpolation mixes valid geometry with invalid pixels.
[ ] Check whether sky sentinel values contaminate interpolated depth.
[ ] Test masked interpolation for dense depth maps.
[ ] Verify validity masks are not interpolated as continuous values without appropriate thresholding.
[ ] Verify sky masks remain binary after resizing.
[ ] Check whether random crops disproportionately remove sky or distant geometry.
[ ] Check whether square crops remove important outdoor context.
[ ] Check whether aspect-ratio changes distort depth cues.
[ ] Verify horizontal flips preserve RGB-depth alignment.
[ ] Verify camera intrinsics are updated where geometric transformations require it.
[ ] Check whether depth values are incorrectly scaled during image resizing.
[ ] Check whether augmentation is applied consistently across video frames.
[ ] Check whether photometric augmentation changes only RGB and not depth or masks.
[ ] Check whether strong brightness, contrast, blur, or noise augmentation damages sky appearance.
[ ] Save before-and-after visualizations for each dataset.
[ ] Temporarily disable all augmentation and compare training behaviour.

============================================================
PHASE 5: AUDIT VALIDITY MASKS AND SKY HANDLING
============================================================

[ ] Identify every place where validity masks are created, modified, or combined.
[ ] Identify every place where sky masks are created, modified, or combined.
[ ] Verify zero-depth invalid pixels never become valid supervision.
[ ] Verify NaN and infinity never enter ordinary metric losses.
[ ] Verify sky sentinels never become ordinary metric targets.
[ ] Verify genuine distant geometry remains valid where intended.
[ ] Check inconsistent strict versus inclusive depth thresholds.
[ ] Check whether masks are applied before or after clamping.
[ ] Check whether masks are applied before or after inverse-depth conversion.
[ ] Check whether missing sky masks silently disable sky supervision.
[ ] Check whether sky masks have correct batch, temporal, and spatial dimensions.
[ ] Check whether sky masks accidentally include foreground pixels.
[ ] Check whether valid geometry overlaps the sky mask.
[ ] Check whether semantic sky labels are reliable across weather conditions.
[ ] Check whether sky supervision is enabled in the effective training configuration.
[ ] Check whether sky supervision applies to all relevant datasets or only a subset.
[ ] Check whether sky supervision is unintentionally disabled for sparse datasets.
[ ] Check whether sky loss produces finite, nonzero gradients.
[ ] Check whether sky gradients reach the metric head.
[ ] Check whether sky loss gradients conflict with valid distant-geometry gradients.
[ ] Log the percentage of valid, invalid, sky, and distant pixels per dataset.
[ ] Log how many sky pixels actually contribute to each loss.
[ ] Visualize sky masks overlaid on RGB for at least 100 randomly sampled outdoor frames.

============================================================
PHASE 6: AUDIT THE VDA LOSS IMPLEMENTATION
============================================================

[ ] Read the complete implementation of the underlying VDA loss.
[ ] Verify the loss expects disparity rather than metric depth.
[ ] Verify prediction and target are converted consistently.
[ ] Verify target disparity is zero only for explicitly labelled sky pixels.
[ ] Verify invalid depth pixels do not produce large reciprocal values.
[ ] Check robust scale-and-shift normalization.
[ ] Check normalization when most pixels are invalid.
[ ] Check normalization when most pixels are sky.
[ ] Check whether sky pixels affect normalization statistics.
[ ] Check whether adding sky pixels changes effective loss scale.
[ ] Check whether normalization is performed per frame, per clip, or per batch.
[ ] Check whether normalization eliminates useful metric-scale gradients.
[ ] Check for sign or ordering errors.
[ ] Check whether loss terms are weighted by valid-pixel count or sample count.
[ ] Check whether sparse and dense samples have comparable loss magnitudes.
[ ] Check whether sky-heavy clips dominate the objective.
[ ] Check whether reductions produce different weighting across image resolutions.
[ ] Check whether temporal and spatial terms use compatible masks.
[ ] Check whether temporal losses require valid depth at both corresponding locations.
[ ] Check whether temporal losses penalize legitimate occlusions or disocclusions.
[ ] Check whether temporal consistency encourages stable but incorrect sky depth.
[ ] Test VDA loss on synthetic perfect predictions and deliberately incorrect predictions.
[ ] Numerically check gradients using small synthetic tensors.

============================================================
PHASE 7: AUDIT METRIC SILOG SUPERVISION
============================================================

[ ] Verify exact SiLog formula and sign.
[ ] Verify configured SiLog weight.
[ ] Verify configured SiLog lambda.
[ ] Verify SiLog is active for all intended metric-supervised datasets.
[ ] Verify metric_sup flags survive batching and collation.
[ ] Verify SiLog excludes datasets without valid metric ground truth.
[ ] Verify SiLog uses raw metric predictions, not normalized disparity.
[ ] Verify SiLog uses correctly scaled metre-valued targets.
[ ] Check whether SiLog mask excludes too much distant geometry.
[ ] Inspect per-source silog_max_depth settings.
[ ] Inspect global metric_silog_max_depth setting.
[ ] Verify per-source ceilings propagate correctly into batches.
[ ] Verify missing ceilings do not accidentally become zero.
[ ] Check whether loss silently disappears when a batch has no eligible pixels.
[ ] Check whether zero predictions are clamped before logarithms.
[ ] Measure how often prediction clamping produces zero gradients.
[ ] Check whether predictions above configured clamps receive zero gradients.
[ ] Check for NaN, infinity, and overflow in logarithmic operations.
[ ] Compare SiLog error and gradients by depth range.
[ ] Check whether scale-invariant training leaves systematic metric-scale bias.
[ ] Compare existing SiLog implementation against a trusted reference.
[ ] Test alternative SiLog weights and lambda values.

============================================================
PHASE 8: AUDIT SPARSE KITTI SUPERVISION
============================================================

[ ] Verify KITTI batches are correctly identified as sparse.
[ ] Verify sparse loss applies only to sparse samples.
[ ] Verify dense samples in mixed batches still receive intended dense loss.
[ ] Check minimum-valid-pixel threshold for sparse frames.
[ ] Log how many KITTI frames fail that threshold.
[ ] Check sparse normalization stability with few LiDAR points.
[ ] Check whether per-frame normalization removes useful scale information.
[ ] Check whether SiLog restores metric-scale supervision for KITTI.
[ ] Verify sparse and dense loss contributions are correctly weighted.
[ ] Check whether mixed-batch weighting unintentionally suppresses KITTI gradients.
[ ] Check whether KITTI's lack of dense sky supervision is compensated by other datasets.
[ ] Check whether sparse supervision disproportionately covers roads and nearby objects.
[ ] Check whether the model learns sky predictions from KITTI despite having no sky targets.
[ ] Compare KITTI-only and dense-outdoor-only training experiments.
[ ] Test an additional sparse metric loss without scale-and-shift normalization.
[ ] Test valid-point Huber or log-depth loss on KITTI.
[ ] Keep invalid LiDAR locations excluded from ordinary metric losses.

============================================================
PHASE 9: AUDIT DATASET SAMPLING AND MIXING
============================================================

[ ] Calculate effective dataset probabilities from sampling masses.
[ ] Verify actual sampled frequencies match configuration.
[ ] Verify sampling masses are interpreted correctly.
[ ] Check source selection across workers and distributed ranks.
[ ] Check whether datasets repeat more often than intended.
[ ] Check whether some datasets are effectively absent.
[ ] Check whether PointOdyssey validation data is intentionally used as a training source.
[ ] Check indoor/outdoor mixing proportions.
[ ] Calculate percentage of metric-supervised outdoor clips.
[ ] Calculate percentage of sky-supervised clips.
[ ] Calculate percentage of valid far-depth pixels.
[ ] Compare per-dataset loss magnitudes.
[ ] Compare per-dataset gradient norms.
[ ] Check whether indoor datasets dominate metric-scale learning.
[ ] Check whether synthetic outdoor datasets dominate appearance learning.
[ ] Check whether sparse KITTI contributes less gradient than expected.
[ ] Check whether large weakly supervised datasets dominate optimization.
[ ] Test increasing VKITTI sampling.
[ ] Test increasing outdoor metric-supervised sampling.
[ ] Test reducing indoor sampling.
[ ] Test balancing batches by indoor versus outdoor category.
[ ] Test balancing by depth distribution rather than dataset size.
[ ] Test balancing by sky-pixel coverage.
[ ] Test short outdoor-only fine-tuning.

============================================================
PHASE 10: AUDIT OPTIMIZATION AND TRAINING MECHANICS
============================================================

[ ] Verify all intended metric-head parameters have requires_grad=True.
[ ] Verify intended parameters are included in optimizer.
[ ] Verify correct parameter groups and learning rates.
[ ] Check accidentally frozen backbone, decoder, or output layers.
[ ] Check checkpoint restoration of optimizer and scheduler.
[ ] Verify resumed training uses intended learning rate.
[ ] Verify configured steps match scheduler.
[ ] Check whether learning rate decays too early.
[ ] Check weight decay exclusions.
[ ] Check gradient accumulation.
[ ] Check mixed-precision loss scaling.
[ ] Check excessive gradient clipping.
[ ] Log total and per-module gradient norms.
[ ] Check NaN and infinite gradients.
[ ] Check zero-gradient parameters.
[ ] Check metric head receives gradients from intended losses.
[ ] Check whether reconstruction or auxiliary objectives conflict with metric depth.
[ ] Check whether loss weighting changes with batch composition.
[ ] Check training/validation preprocessing consistency.
[ ] Check stateful layers between training and evaluation.
[ ] Check distributed loss normalization.
[ ] Check random seeds and worker reproducibility.
[ ] Run one-batch overfitting test.
[ ] Run ten-batch overfitting test.
[ ] Confirm model can fit distant geometry and sky in a tiny controlled dataset.

============================================================
PHASE 11: MEASURE GRADIENTS AND IDENTIFY FAILING OBJECTIVES
============================================================

Create a diagnostic script that:

[ ] Loads a real training batch.
[ ] Runs a forward pass.
[ ] Calculates each loss component separately.
[ ] Measures each component's gradient norm on the metric head.
[ ] Measures gradients for near, middle, and far geometry.
[ ] Measures sky gradients separately.
[ ] Records fraction of zero gradients.
[ ] Records fraction of invalid or nonfinite gradients.
[ ] Compares gradient directions between VDA and SiLog.
[ ] Compares gradient directions between sky and distant geometry.
[ ] Measures whether sky loss improves or worsens metric depth on valid pixels.
[ ] Measures each dataset's contribution to total gradients.
[ ] Saves results in CSV for checkpoint comparisons.

Run diagnostics on initial, intermediate, and final checkpoints.

============================================================
PHASE 12: RUN CONTROLLED LOSS ABLATIONS
============================================================

Use the same initialization, dataset samples, training duration,
and evaluation procedure wherever possible.

[ ] Baseline: Current combined objective.
[ ] Experiment A: SiLog only.
[ ] Experiment B: VDA loss only.
[ ] Experiment C: VDA + SiLog without sky supervision.
[ ] Experiment D: VDA + SiLog with zero-disparity sky supervision.
[ ] Experiment E: VDA + SiLog with one-sided metric sky hinge.
[ ] Experiment F: VDA + SiLog with both sky objectives.
[ ] Experiment G: Increase SiLog weight.
[ ] Experiment H: Decrease SiLog weight.
[ ] Experiment I: Direct metric log-depth or Huber loss for valid geometry.
[ ] Experiment J: Depth-range-balanced metric loss.
[ ] Experiment K: One-sided lower-bound loss for valid geometry beyond supported range.
[ ] Experiment L: Exclude extreme synthetic far-depth values from exact metric supervision.
[ ] Experiment M: Increase weight of valid distant geometry.
[ ] Experiment N: Exclude sky from VDA normalization statistics while retaining sky supervision.
[ ] Experiment O: Apply sky loss only to datasets with reliable sky labels.

For each experiment compare:
- Sky depth
- Horizon behaviour
- Far-depth metrics
- Indoor metrics
- Gradient statistics
- Stability
- Runtime

============================================================
PHASE 13: TEST ALTERNATIVE SKY SUPERVISION
============================================================

[ ] Test zero-disparity supervision for sky.
[ ] Test one-sided minimum-depth hinge for sky.
[ ] Test different minimum sky-depth thresholds.
[ ] Test bounded far-depth targets if architecture has a hard maximum.
[ ] Test sky supervision using semantic masks.
[ ] Test sky supervision using frozen relative-depth teacher.
[ ] Test teacher-based ranking instead of exact pseudo-depth.
[ ] Test sky-versus-foreground depth ordering constraints.
[ ] Test sky loss requiring sky to be farther than nearby valid geometry.
[ ] Test ignoring uncertain sky boundaries.
[ ] Test excluding thin structures, tree branches, and horizon boundaries.
[ ] Test weak-to-strong sky loss weights.
[ ] Test whether sky improvements degrade indoor performance.
[ ] Test sky prediction stability across adjacent frames.
[ ] Test whether sky gradients remain useful after disparity normalization.

Do not use arbitrary finite sky distances as exact ground truth
unless explicitly treating them as approximations or bounds.

============================================================
PHASE 14: TEST ALTERNATIVE DEPTH REPRESENTATIONS
============================================================

[ ] Compare current metric head against unbounded positive-depth head.
[ ] Test wider bounded-depth head if current head saturates.
[ ] Test log-depth prediction.
[ ] Test inverse-depth prediction with metric-scale calibration.
[ ] Test hybrid metric-depth and disparity objective.
[ ] Test direct metric supervision without inverse-depth normalization.
[ ] Test separate scale prediction branch if needed.
[ ] Test different initialization for expanded output head.
[ ] Check numerical stability at large distances.
[ ] Check whether alternatives improve sky without reducing near-depth accuracy.

============================================================
PHASE 15: RUN TARGETED DATASET ABLATIONS
============================================================

[ ] VKITTI only.
[ ] TartanAir only.
[ ] KITTI only.
[ ] PointOdyssey only.
[ ] All synthetic outdoor datasets only.
[ ] All real outdoor datasets only.
[ ] All outdoor datasets combined.
[ ] All datasets without indoor data.
[ ] All datasets without VKITTI.
[ ] All datasets without TartanAir.
[ ] All datasets without KITTI.
[ ] All datasets without PointOdyssey.
[ ] Outdoor-first training followed by mixed training.
[ ] Existing mixed training followed by outdoor fine-tuning.

Use short runs initially. Expand only promising experiments.

============================================================
PHASE 16: VALIDATE THE FINAL FIX
============================================================

[ ] Re-run all unit tests.
[ ] Re-run dataset and mask visualizations.
[ ] Confirm no invalid depth enters metric losses.
[ ] Confirm distant valid geometry receives intended supervision.
[ ] Confirm sky receives intended supervision.
[ ] Confirm model output range supports training targets.
[ ] Confirm gradients are finite and nonzero where expected.
[ ] Evaluate held-out indoor and outdoor sequences.
[ ] Evaluate synthetic and real outdoor scenes separately.
[ ] Evaluate by depth range.
[ ] Evaluate sky depth and horizon behaviour.
[ ] Evaluate temporal consistency on outdoor videos.
[ ] Compare against original baseline checkpoint.
[ ] Repeat promising experiments with at least three random seeds.
[ ] Check improvements persist after full-length training.
[ ] Remove experimental code and unused configuration options.
[ ] Add regression tests for confirmed bugs.
[ ] Document root causes, fixes, and quantitative improvements.

============================================================
REQUIRED OUTPUTS
============================================================

1. audit.md
   - Suspected bugs
   - Exact files and lines
   - Evidence
   - Severity
   - Confirmed, ruled out, or unresolved status
   - Recommended fixes

2. dataset_diagnostics/
   - RGB images
   - Ground-truth depth
   - Validity masks
   - Sky masks
   - Before/after augmentation visualizations

3. loss_diagnostics.csv
   - Per-loss gradient measurements
   - Per-depth-range gradient measurements
   - Per-dataset contributions
   - Sky gradient measurements

4. baseline_metrics.csv
   - Original model evaluation results
   - Indoor/outdoor breakdown
   - Depth-range breakdown
   - Sky/horizon diagnostics

5. experiments.csv
   - Experiment identifier
   - Git commit
   - Configuration
   - Checkpoint
   - Changes made
   - Evaluation metrics
   - Conclusions

6. fixes/
   - Minimal independently reviewable patches
   - Regression tests for each confirmed bug

7. final_report.md
   - Confirmed root causes
   - Permanent fixes
   - Quantitative improvements
   - Remaining limitations
   - Recommended final training configuration

============================================================
EXECUTION PRIORITY
============================================================

PRIORITY 0: CRITICAL CHECKS
- Verify metric head maximum output depth.
- Verify native metric-depth versus disparity representation.
- Verify sky and validity masks.
- Verify depth units and sentinel handling.
- Verify actual effective loss configuration.
- Verify VDA and SiLog gradients on distant geometry and sky.

PRIORITY 1: LIKELY TRAINING ISSUES
- Investigate SiLog depth ceilings.
- Investigate scale-and-shift normalization.
- Investigate sparse KITTI loss weighting.
- Investigate dataset sampling proportions.
- Investigate invalid-pixel interpolation.
- Investigate missing or ineffective sky supervision.
- Investigate checkpoint and optimizer initialization.

PRIORITY 2: TARGETED EXPERIMENTS
- Test sky hinge loss.
- Test direct metric log-depth loss.
- Test increased outdoor sampling.
- Test depth-range-balanced supervision.
- Test outdoor-only fine-tuning.
- Test expanded metric-head output range.

PRIORITY 3: ADVANCED CHANGES
- Alternative depth representations.
- Teacher-based sky supervision.
- Ranking constraints.
- Architecture modifications.
- Larger dataset ablations.

STOP CONDITIONS
- Do not start expensive ablations until critical checks are complete.
- Do not launch full-length training until a short run shows measurable improvement.
- Do not merge a fix without a regression test.
- Do not declare a bug confirmed solely because a configuration looks suspicious.
- Do not accept improved sky appearance if valid outdoor metric accuracy deteriorates substantially.
- Do not modify multiple unrelated components in a single experiment.

FINAL SUCCESS CRITERIA
- Outdoor metric depth accuracy improves on held-out scenes.
- Far-depth predictions improve without destroying near-depth accuracy.
- Sky predictions are consistently distant and temporally stable.
- Invalid pixels do not contribute incorrect supervision.
- Indoor performance remains acceptable.
- Improvements are reproducible across random seeds.
- Every confirmed bug has a documented root cause and regression test.
```

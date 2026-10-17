# FAR_DIAG: why is real far geometry compressed (Track 2, LIGHT, Add. O2 / user Add. 46)
Date 2026-10-17. Diagnosis only: CPU, no training, no cluster compute, nothing committed or applied. Everything below is labelled MEASURED (computed here or read from a committed output, source named) or INFERRED (reasoning, not tested). Scripts and raw outputs are in this directory; large scratch in /mnt/DATA/DegradedDepth/claude_outputs/scratch/outdoor_debug_20261017_fardiag/.

## 0. The symptom, restated with numbers (MEASURED, committed outputs)
DDAD val LR_DEV (exposed, report-only, scorer 80916c4f, 378 input, unaligned): median pred:gt per GT bin.
| model | [0,20) | [20,50) | [50,80) | [80,120) | [120,200] | log-slope of ratio vs depth |
|---|---|---|---|---|---|---|
| ours start (c3_n46 12k) | 1.201 | 1.123 | 0.943 | 0.742 | 0.574 | -0.26 |
| released Metric-VDA-S @378 | 1.187 | 1.133 | 1.022 | 0.907 | 0.854 | -0.12 |
| A0 / A0b | 1.193 / 1.209 | 1.109 / 1.142 | 0.926 / 0.958 | 0.722 / 0.754 | 0.548 / 0.582 | -0.27 / -0.25 |
| TS2 / H80 / H300 / VK655 | 1.19-1.20 | 1.11-1.12 | 0.92-0.93 | 0.72 | 0.542-0.548 | -0.27 |
Both models overshoot near range by ~1.2 (a common offset, likely camera/FOV, INFERRED); the DEPTH-DEPENDENT part differs: ours/release = 1.01, 0.99, 0.92, 0.82, 0.67 from near to far. Per-bin n_px: 4.75M, 3.23M, 0.86M, 0.42M, 0.27M (the far bins are 7% of scored px).
Training-domain evidence (MEASURED, T_trajectory/trajectory.csv): VKITTI DEV (a TRAINED source) pred:gt in the 80-655 m GT bin: release 0.995 -> ours 0.84-0.86 (steps 6000-12000), near (<30 m) 1.095 -> 1.10 unchanged. Train batches (mixed sources): far-valid 80-655 m 0.848 -> 0.68, near <10 m 1.416 -> 1.09, mid 10-80 m 0.93 -> 0.94. So the far compression is not only a real-vs-synthetic transfer gap: it is already present, smaller, on trained data, appears by step 6000 and plateaus.

## 1. Per-depth-bin accounting through the real loaders (MEASURED; perbin_accounting.py, perbin_tables.md, perbin_raw.json)
Method: real C3 data block (configs/experiments/dn01_c3rep_20261009_c3_n46.yaml), train._build_multisource, real dataset classes (518 VDA resize+crop, RGB degradation neutralised with clean_prob 1; GT never degraded), 8-frame clips, 40 clips per source, seeded random.Random(45), all 9 sources (NYU: 3 of 12 local shards; ScanNet++: local subset140 scenes; as in E2). SiLog-eligible = mask & metric_sup & depth < silog_ceil (the real per-source ceilings: 655 on TartanAir and both PointOdyssey, none elsewhere). The mix is formed by weighting each source's fractions with mass_s / sum(mass) (sampling scheme, train.py:499-525: P(clip) = mass_s/sum(mass) for a source, uniform within the source, with replacement; the earlier simulation matched this). The stratified draw (40 clips/source) gives per-source precision; it is not the realised seed-45 draw.
## Table A: per source, % of ALL pixels of the sampled clips (8-frame clips at 518, 40 clips/source, seed 45)

| source | mass share | clips | sky % | valid % | 0-20 | 20-50 | 50-80 | 80-120 | 120-200 | 200-655 | >=655 | SiLog-eligible % (all bins) | SiLog-elig >=80 m % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tartanair | 27.9% | 40 | 7.00 | 93.0 | 75.600 | 8.172 | 1.725 | 1.092 | 0.753 | 0.723 | 4.927 | 88.1 | 2.568 |
| vkitti | 3.6% | 40 | 17.53 | 82.5 | 49.235 | 19.838 | 5.185 | 3.150 | 3.260 | 1.799 | 0.000 | 82.5 | 8.209 |
| pointodyssey_train | 7.9% | 40 | 0.00 | 86.3 | 83.981 | 2.168 | 0.192 | 0.008 | 0.000 | 0.000 | 0.000 | 86.3 | 0.008 |
| pointodyssey_val | 1.2% | 40 | 0.00 | 88.8 | 88.021 | 0.816 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 88.8 | 0.000 |
| irs | 9.0% | 40 | 0.00 | 97.5 | 97.351 | 0.163 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 97.5 | 0.000 |
| spring | 0.4% | 40 | 0.00 | 87.6 | 67.243 | 17.966 | 2.395 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 |
| kitti_raw | 27.0% | 40 | 0.00 | 17.3 | 13.228 | 3.573 | 0.503 | 0.000 | 0.000 | 0.000 | 0.000 | 17.3 | 0.000 |
| nyu_train | 15.3% | 40 | 0.00 | 96.4 | 96.354 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 96.4 | 0.000 |
| scannetpp_train | 7.7% | 40 | 0.00 | 99.9 | 99.935 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 99.9 | 0.000 |

## Table B: mass-weighted MIX (expected fraction of all pixels in a sampled batch; weight = mass_s / sum mass)

| quantity | 0-20 | 20-50 | 50-80 | 80-120 | 120-200 | 200-655 | >=655 | total |
|---|---|---|---|---|---|---|---|---|
| valid GT px (VDA loss mask) % | 65.599 | 4.235 | 0.830 | 0.419 | 0.328 | 0.267 | 1.376 | 73.06 |
| SiLog-eligible px % | 65.306 | 4.157 | 0.820 | 0.419 | 0.328 | 0.267 | 0.000 | 71.30 |
| share of SiLog-eligible px | 91.60 | 5.83 | 1.15 | 0.59 | 0.46 | 0.37 | 0.00 | 100 |

sky px (VDA zero-disparity target only) = 2.59% of all px

## Table C: far (>=80 m) SiLog pixel share of the mix by source

- tartanair: 0.718% of all px = 70.8% of far SiLog px;  120-655 m: 0.412%
- vkitti: 0.296% of all px = 29.2% of far SiLog px;  120-655 m: 0.182%
- pointodyssey_train: 0.001% of all px = 0.1% of far SiLog px;  120-655 m: 0.000%
- pointodyssey_val: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- irs: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- spring: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- kitti_raw: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- nyu_train: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- scannetpp_train: 0.000% of all px = 0.0% of far SiLog px;  120-655 m: 0.000%
- mix total >=80 m SiLog-eligible: 1.014% of all px = 1.42% of SiLog px

Reading (MEASURED):
- 91.6% of all SiLog-eligible pixels are < 20 m; 80-655 m is 1.42% of SiLog pixels (1.01% of ALL mix pixels; of that 70.8% TartanAir, 29.2% VKITTI; every other source 0.00%). 120-655 m is 0.80% of SiLog pixels (0.59% of all pixels). The earlier "2.85% of pixels >= 80 m" counted TartanAir's valid >= 655 m tail too (here 1.38% of mix pixels: valid for the VDA loss, removed from SiLog by the 655 ceiling) and used a different sample.
- Caps: KITTI keeps 17.3% of its pixels (sparse LiDAR), 0 beyond 80 m (raw LiDAR has 0.02% of valid px > 80 m, B1 kitti_check); only 0.5% of KITTI pixels are at 50-80 m. Spring, PO, IRS, NYU and ScanNet++ supply nothing beyond 80 m (PO-train 0.008%). VKITTI: 8.2% of its pixels are 80-655 m, 3.3% at 120-200 m, 1.8% at 200-655 m; the 655 cap removes only the sky sentinel. TartanAir: 2.6% of pixels 80-655 m SiLog-eligible, 4.9% valid > 655 m (VDA mask only), 7.0% sky.
- Sky (zero-disparity VDA target only, no SiLog): 2.59% of mix pixels.
- Far-range supervision therefore comes from two synthetic sources, ~1% of pixels; KITTI (27% of the sampling mass, 40x repeated clips) supplies 0 far pixels and 5.3x fewer valid pixels per clip than TartanAir.

## 2. Per-bin loss weight / gradient accounting (MEASURED; grad_accounting.py -> grad_accounting_output.txt)
Method: the real train._compute_loss_vda on CPU (effective C3/FT loss block: SSI + 0.5 x multi-scale gradient matching + 10 x TGM on disparity 1/clamp(pred), SiLog w 1.0 lambda 0.5 on metric depth). Synthetic clip, T=4, 8 equal-area bands at GT 5,10,20,40,80,150,300,500 m, one band at a time carries a log error (pred = GT x 0.80 or x 0.55), the rest perfect; metric = mean |dL/dlog(pred)| per pixel inside the perturbed band (x1e-6, loss pooled over the whole clip). Terms separated by linearity (VideoDepthLoss reads only gradient_weight and tgm_weight; ssi_weight is ignored, train.py:833-836).
| term (x1e-6 per px) | 5 m | 20 m | 80 m | 150 m | 500 m | ratio 20 m / 150 m | 20 m / 500 m |
|---|---|---|---|---|---|---|---|
| SSI, err x0.55 | 442 | 364 | 81 | 68 | 20 | 5.3 | 18 |
| gradient matching, x0.55 | 888 | 493 | 157 | 88 | 17 | 5.6 | 29 |
| TGM, x0.55 | 162 | 70 | 18 | 9.9 | 3.0 | 7.1 | 24 |
| VDA total, x0.55 | 1016 | 851 | 186 | 156 | 36 | 5.5 | 23 |
| VDA total, x0.80 | 1081 | 679 | 504* | 105 | 25 | 6.5 | 27 |
| SiLog, any depth (x0.55 or x0.80) | 334 | 334 | 334 | 334 | 334 | 1.0 | 1.0 |
(*) the 80 m value for x0.80 is non-monotone because the per-frame median/MAD normaliser puts that band near the frame's median disparity; ratios use 20 m and 150 m/500 m only.
- The "~6.7x weaker VDA pull at far" claim is CONFIRMED: per-pixel gradient w.r.t. log depth at 150 m is 1/5.3 to 1/7.4 of that at 20 m for every VDA term (disparity = 1/d predicts 7.5x; the E_gradients real-batch figure was 6.7x), and 1/18 to 1/29 at 500 m. In per-metre units it is 100-250x (C1-06).
- SiLog is uniform in log space (ratio exactly 1.0 for equal log error): it is the ONLY term that supervises metric scale (the VDA loss is invariant to a global scale, C1-01). Its effective weight is 1.0 divided by the batch value sqrt(mean(d^2) - 0.5 mean(d)^2) (d = log GT - log pred): a rare far error gets a LARGER per-pixel gradient (SiLog pooled over eligible px, far band error x0.55, rest perfect): far share 1.4% -> 1618e-6 per far px, 3% -> 1087, 5% -> 820, 10% -> 542, 50% -> 199; total far pull (sum over far px) rises with the far share: 118e-3 (1.4%), 172e-3 (3%), 221e-3 (5%), 612e-3 (50%). At the mix's 1.4% far share, SiLog pulls a far px with a x0.55 error ~10x harder than VDA does at 150 m (1618 vs 156), but the far pixels' total share of the loss is small.
- INFERRED: at the trained operating point (final SiLog ~0.18, i.e. mean-square log error ~0.03), far pixels at the observed train-batch ratio 0.68 (d=0.39) contribute 0.014*0.15 = 0.002 (~6%) of it; at the DDAD-like 0.55 it would be ~16%. So the far error is cheap to leave in place in the objective as it stands; there is no strong restoring force. This is arithmetic from measured shares, not a tested equilibrium.
- Degenerate row, do not interpret: the 'same error in all bands' VDA row in the output is the L1 subgradient at an exactly-zero residual (a global scale is invisible to VDA), i.e. numerical noise; only its SiLog row (86e-6 flat) is meaningful.
- C1-04 (MEASURED earlier): sky at zero-disparity target puts a small push toward nearer on valid far pixels (+4.9e-5 per log, vs SiLog 3.3e-4 per px at a 0.2 log error); size on real data unresolved; the screen's TS2/H80/H300 arms left the DDAD far bins unchanged (readout), which argues it is not the main driver (INFERRED).

## 3. What the DDAD far error looks like, and why the release may be closer (MEASURED + INFERRED)
MEASURED: table in section 0. The ours-vs-release gap grows monotonically with depth (log ratio 0.01, -0.01, -0.08, -0.20, -0.40), starting around 50-80 m, which is also where KITTI's LiDAR ends (p99 55 m, cap 80 m) and where VKITTI/TartanAir are the only supervisors. AbsRel at far is worse in ours at 120-200 m (0.42-0.45 vs 0.337) and the median bias is the main part of it. No arm moves the DDAD far bins beyond A0-vs-A0b spread (0.548/0.582 at 120-200 m); TS2+VK655 (VKITTI far + sentinel at 655 m directly supervised) gets the VKITTI primary to |1-x| 0.093 but DDAD 120-200 m stays 0.543: more synthetic-VKITTI far supervision does not transfer to this real set.
Possible reasons the release is closer at far (all INFERRED, none isolated; the release was trained on Virtual KITTI + IRS only, VDA README `--metric` help):
1. No KITTI-raw (27% mass, 40x repeats, real images whose depth never exceeds 80 m and mostly < 55 m): a real-image prior "depth <= ~80 m" our model has and the release lacks.
2. No indoor 10 m-capped metric sources (NYU, ScanNet++ = 22% of mass) and no TartanAir / PO / Spring: a smaller range prior; the release keeps VKITTI's 655 m range.
3. Head drift: 15-20% relative L2 change of the DPT head from the release at full lr (C2-04); the far behaviour of the release is a property of the original head weights that a recipe without a far anchor does not preserve (trajectory: lost by step 6000, never regained).
4. Far px are 1.4% of SiLog px with 70% of them synthetic TartanAir (domain gap, saturated fog-like depths, 4.9% > 655 m valid in VDA only) so the far fit is weak even in-domain (VKITTI 0.84).
5. Input scale: DDAD at 378 input makes far objects tiny; the release is evaluated outside its native 518 (A4 would bound this for the release only); camera FOV/focal differences between DDAD and the training cameras (not checked here) are a candidate for the common ~1.2 near-range offset in both models.
6. The release may have been trained with a different loss/crop pipeline (training code unpublished; Add. 44 provenance: the sentinel-as-valid hypothesis is unverified).

## 4. Which candidate causes are measured
| candidate | status |
|---|---|
| (i) far pixels get ~6.7x weaker VDA pull | MEASURED, confirmed: 5.3-7.4x at 150 m vs 20 m per term; but VDA is scale-invariant so it cannot set metric scale at any range; effect on far scale = nil by construction |
| (ii) far GT only from TartanAir + VKITTI | MEASURED: 1.0% of mix pixels / 1.4% of SiLog px; KITTI 0 beyond 80 m |
| (iii) SiLog form | MEASURED: uniform per log-error; self-normalising; it is the only scale anchor; lambda 0.5 leaves a global bias half-penalised (C1-01); no sign of a bug |
| (iv) head range / activation | RULED OUT earlier (A_model: ReLU, unbounded, 228-674 m reached) |
| (v) mix weights | MEASURED shares above; causal effect NOT tested (only a training arm can) |
| in-domain compression exists | MEASURED (VKITTI 0.84 vs release 0.995) |
| cause of the compression | NOT isolated; ranked hypotheses in section 3 |

## 5. Candidate light-screen arms
See LIGHT_SCREEN_DRAFT.md (question, 5+1 arms with real config keys, primary and independent metrics, pre-declared rule, cost). A4 (release at 518): A4_draft.md, A4_diff.patch, A4_test_additions.py.

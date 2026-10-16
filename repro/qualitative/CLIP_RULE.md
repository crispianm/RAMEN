# CLIP RULE, REVISION ftn46 (original pre-declared 2026-10-12 BEFORE any clip was selected and before any render; this revision written 2026-10-12 per R7 BEFORE `clip_rule.py` was re-run; original kept as CLIP_RULE.pre_ftn46.md, its selection kept as CLIP_SELECTION.pre_ftn46.json)

**Revision note (R7):** the only change is R1's FT: user decision via Manager (Addendum 12 / figures decision 2) FT panel = `c3rep ft_n46`; Auditor correction: write this new rule with a new pinned sha, then recompute. R2-R6 are unchanged; the MEDIANS are by C3 seed-45 raw and therefore unchanged (asserted by the script against the pre_ftn46 selection); only the FAILURE case is recomputed against ft_n46 (R5).

Source of the ask: USER_DECISIONS_20261012.md Addendum 11 (sha 2224cac9), user answer "Render on 4090 after latency": C3 (seed 45) vs seed-matched FT vs released VDA-S, side-by-sides + GIFs on DEV clips only, one median clip per dataset + one failure case (Bonn or ScanNet++), teaser re-rendered under the same rule. The Auditor approves this rule before any GPU work.

## R0. What the rule may read
Only ALREADY-SCORED per-clip result JSONs (CPU, no render, no model run). DEV membership = the benchmark's own definition (`bench_joint.load_clips`: manifest order, ScanNet `eval_scenes` truncation first, then the even-indexed clips `clips[0::2]`). TEST clips (`clips[1::2]`) are never selected, never rendered, never shown. The selection script recomputes DEV from the manifest files (no `allow_test`), asserts equality with `bj.load_clips(ds, "dev")`, and refuses if a selected clip is not in it.

## R1. Models and rows
- C3 = seed 45, `bfix_C3_B` step12000, sha256 c9decd90a4b4c2746ca1a389d84680a15a4a9f1572cb2ddb5ba9a73ee19edac9, RAW (no post-filter). Rows: `results_v3/bfix_s12/ours-...C3_B...step12000/bench/*__<ds>__llvar__all__v3.json` (GH200, scorer commit 368b5c53).
- FT = `c3rep ft_n46`: the raw fine-tuned VDA-S trained with the C3 recipe, from scratch, SEED 46 (D6: raw, no TAG). **There is no FT at seed 45 in the C3 recipe**, so the FT panel is a different seed and lineage from the seed-45 C3; every caption says: 'FT: C3 recipe, seed 46, from scratch (different seed/lineage from the seed-45 C3)'. Checkpoint: Isambard `/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_ft_n46/step12000.pt`, sha256 18c9d8caeab1ab88d52b92782e307bcb8d82897e8a8017aed172f06c26af92a0 (= `ckpt_sha256` in the c3rep ft_n46 RUN_MANIFEST/result rows), pulled to `checkpoints_v3/eval_20261012_ft_n46/` with sha verified at both ends. Rows: `results_v3/c3rep_s12/metric_vda_s_ft:...c3rep_20261009_ft_n46_step12000` (GH200, scorer 368b5c53), cell `llvar`, `per_video[<clip>]['abs_rel_aligned']`. The earlier default (pre-C3-recipe `synreal_r50_mvdas_ft`, seed 45, sha 37e5c076...) is NOT used any more; its selection file is kept and disclosed.
- Released = Metric-VDA-S (`metric_video_depth_anything_vits.pth`, sha256 3c28432b...), the project's released comparator, raw.

## R2. Headline metric and cell
`abs_rel_aligned` (VDA-protocol AbsRel after one per-clip least-squares scale+shift on disparity; lower is better) of C3 RAW, cell `llvar` (the headline moving-video low-light cell), read from `per_video[<clip>]["abs_rel_aligned"]`. (Temporal metric for the teaser: see R6.)

## R3. Datasets
`kitti`, `bonn`, `scannetpp`, `nyuv2`, `sintel_rgbfix` (the five GT datasets of the headline table). nuScenes-night / BVI have no depth GT and are not used. KITTI excludes drive 2011_10_03_drive_0047 (the model trained on its VKITTI replica); the exclusion is applied before ranking.

## R4. One MEDIAN clip per dataset
Candidates = the DEV clips of that dataset (after R3's exclusion) that have a finite metric. Sort ascending by (metric, clip key as a string). With n candidates the median clip is index (n-1)//2 (the lower median). Deterministic; no manual override.

## R5. One FAILURE case
Candidates = the DEV clips of Bonn and ScanNet++ (`bonn`, `scannetpp`) with finite `abs_rel_aligned` for BOTH C3 raw and the R1 ft_n46 row, cell `llvar`. Ratio r = abs_rel_aligned(C3 raw) / abs_rel_aligned(FT raw). The failure case = the clip with the largest r (ties: clip key ascending). If that clip is already that dataset's median clip, take the next largest r. If the largest r < 1 (C3 better than FT on every candidate) the item is labelled "weakest relative to FT" and NOT "failure"; the caption must say so. Chosen by this rule, not by eye.

## R6. Frames, panels, teaser
- Stills use frame index floor(N/2) of the rendered clip (N = frames after the benchmark's own slicing); GIFs use all rendered frames (stride 1, at most 64), 10 fps.
- Teaser (same rule): the Bonn median clip of R4 rendered in the STATIC low-light cell (`st_llvar`: the clip's middle frame repeated 64x, an independent degradation per frame), panels: degraded input, released depth, C3 + TAG(agf0.2t0.06) depth, and the depth-change maps (|D_t - D_{t-1}|/D_t, 0-3 %) of both. TAG = agf0.2t0.06, the canonical headline setting (not the TAG-grid substitution).
- Comparison panels per selected clip (cell `llvar`, per-clip severity from `bench_joint.llvar_draw`, same noise stream as the scored row): degraded input | released | FT raw | C3 raw | C3 + TAG(agf0.2t0.06), depth row and per-pixel AbsRel row (GT-aligned with the benchmark's per-clip fit; valid-GT pixels only). Number-free renders: names only, no digits on any image; colour ranges are in sidecars. Depth colour range = the clip's GT 2nd-98th percentile (log, turbo), identical for every panel, so no model sets it.

## R7. What this rule does NOT do
No selection by looking at renders. No re-pick after rendering. Any change (FT_CHOICE, metric, datasets) needs the Auditor and a new CLIP_RULE.md sha BEFORE `clip_rule.py` is re-run; the old selection files are kept.

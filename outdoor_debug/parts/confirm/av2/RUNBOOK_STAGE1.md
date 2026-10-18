# RUNBOOK, AV2 Stage 1 (code and pins only; no rule is changed here; rules live in AV2_REGISTRATION.md incl. AMENDMENT R1 and 'Stage structure')

Python for every step: `~/.venvs/ramen_seg2/bin/python -I` on the machine that has the data (pyarrow, pandas, torch, cv2, scipy, transformers present). The scorer itself needs a GH200 and CUDA.
Never glob the data dir; the clip and the scored frames come ONLY from the three registered tables below.

## 0. Literals that must be filled / checked before any real data is opened
EXPECT_SCORER_SHA = <FILLED_AFTER_FINAL_COMMIT>

`EXPECT_SCORER_SHA` must be replaced by the literal sha256 of the final committed `av2_scorer.py` (`git show <final commit>:outdoor_debug/parts/confirm/av2/av2_scorer.py | sha256sum`), committed in this file BEFORE data is opened. The runner (`run_score.py`) reads this line (it refuses a placeholder or anything that is not 64 hex characters), passes it as `--expect-scorer-sha`, and never hashes the scorer itself; the scorer compares it with the sha of the file it is running from and refuses on mismatch. The scorer's own sha is NOT in its in-file PINS (a file cannot contain its own hash).

| pin | value | where checked |
|---|---|---|
| stage-1 log list `av2_val_logs_stage1_50.txt` | f7f508f958e8e9d173772677d239c7a53165a0fb67520a68a4bfcf134592e370 | scorer PINS.logs_sha |
| `av2_frames.tsv` (792 scored frames) | 1207c93b73bcf2967643445e1a27d479e415a581b7aa08bfcce14f26db604e59 | `A.REGISTERED_SHAS` (mask builder + scorer), scorer PINS.frames_sha, mask meta.frames_sha == sha(frames) |
| `av2_clips.tsv` (pinned clip table, one row per log) | 3f57db92806fec2d7e559584af1234fbf290a51ea322750c007df334cb05c1ee | same |
| `objects.tsv` (R1 download list) | 9d577f318630b1056b08b109c079b54859971d2cbcaa4a0e21f484465e909f64 | same; every clip jpg and every paired sweep must be listed in it |
| SegFormer-B5 ADE20K `pytorch_model.bin` | a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df @ 739f5d4692954e4a185eac280dec1ba5a7d52f1d | constants `SEG_A` in build_av2_masks.py (no CLI sha), scorer PINS.segmenter_a_sha (read back from the mask meta) |
| OneFormer Cityscapes Swin-L `pytorch_model.bin` | 876855252aa674c7ad6e920c268d6722add811a927806372811faa724f19856c @ 8ed649db2cc1756b8bc36f0cad00b0812e333027 | `SEG_B`, PINS.segmenter_b_sha; weight dirs must hold NO `*.safetensors`; models are loaded with use_safetensors=False |
| bench_joint.py | 5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef | PINS.bench_joint_sha |
| `av2_dataset.py` | 52b3eb9c3d63479cb8efe5a61f15078fbbcb68b349c2771a2b2609f7a9fd31ae | scorer PINS.dataset_module_sha (= the mask meta's `dataset_module_sha`; test_pins_equal_actual_file_shas) |
| `build_av2_masks.py` | 718be61453e894356f3c10b3111af688840f5bdb1911cc0e2729cf1cad5ae2bc | PINS.builder_sha = mask meta `builder_sha` (N4) |
| `seg/make_sky_mask.py` | 31b18f51aed452f575019eee1ad463ee3b6d277ba9945ee505a1baaea16f29e1 | PINS.make_sky_mask_sha = mask meta `make_sky_mask_sha` (N4) |
| `av2_stage1_ckpts.txt` (six CKPT_SHA lines, filled in pin fill 1/3 after the N7 dry run on the real files) | registry_sha 1289688ffe4ad75a2545bfd40bd4cb36832e501f266220cb71c387d1bc93bb0b (pin fill 2/3) | scorer PINS.registry_sha, asserted at scoring and at compare |
| `outdoor_debug/parts/LR_DEV/ddad_scorer.py` | 8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503 | PINS.ddad_scorer_sha |
| `sky_masks.npz` / `sky_masks.meta.json` / `SHA256SUMS` (FROZEN mask set, Auditor O57; filled in pin fill 2/3) | mask_sha de2e9ca4177691ee6e3b904481e4177346192afdf617f24eb2c147c93258f1b1; meta_sha 10285355e791c9e9032ecb5b00d51c412065e7741b698f0f19a493966e4fa073; sums_sha (meta_R1/SHA256SUMS) 4ecb4581fd8d7b96f451fcdc234e953c139a8163842ca8afe3eb4706d777a963 | scorer PINS (mask_sha, meta_sha, sums_sha) |

The scorer also asserts, from the mask meta: `rule` == the registered RULE dict (prob 0.9, erosion 8 px @ 900 ref, LiDAR 5 px @ 900 ref, drop 0.01, any_p 0.5, IoU 0.85, work short 512, mult 32, seed 0, ADE sky id 2, Cityscapes sky id 10) and the recorded `builder_sha` / `make_sky_mask_sha` / `dataset_module_sha` equal both the PINS and the files on disk (the git commit is recorded too, informational).

### Registered Stage 1 checkpoints (N1, D1; Auditor). A CLI label never decides d.
The six `CKPT_SHA <pair> <arm> = <sha256 | UNSET>` lines live in their OWN file `av2_stage1_ckpts.txt` (UNSET for now), whose sha256 is pinned in the scorer as `PINS.registry_sha` (UNSET until the registry is filled; the scorer refuses while it is UNSET). `run_score.py` passes that file as `--registry`. The scorer refuses while ANY line is `UNSET`, when the six are not distinct, when the registry file's sha is not `PINS.registry_sha` (at scoring AND at compare), or when the run's checkpoint sha is not the registered one for its `--pair` + `--arm`. It also reads the checkpoint's own `cfg` and asserts: H300 `loss.sky_hinge_weight` 0.1 and `sky_hinge_min_depth` 300 (TS2: weight 0 or absent, no min_depth); `loss.sky_hinge_max_depth` absent or null and `loss.sky_vda_zero_disparity` true (declared in all six stage1 yamls and the start checkpoints) in both arms; `data.seed` and `training.resume_reseed` = 50/51/52 for s50/s51/s52; `training.resume_from` = the declared start `.../dn01_c3rep_20261009_c3_n46|n47|n48/step12000.pt`; `max_steps` 14000; TartanAir `ts2_sky` true. The full stored cfg and its sha256 (`cfg_sha256`) go into each run json; compare re-derives the arm from the stored cfg and requires that the two cfgs of a pair differ ONLY in `loss.sky_hinge_weight`, `loss.sky_hinge_min_depth`, `training.checkpoint_dir` (recursive key diff).

**Order of filling (acyclic, train -> registry -> PINS -> scorer sha -> EXPECT_SCORER_SHA; do not reorder):**
1. train the six Stage 1 runs (step-14000 checkpoints exist);
2. **required before step 3 (N7):** repeat the dry run on the six REAL step-14000 checkpoints: `python -I dryrun_ckpt_facts.py <dir with the six step14000.pt>` (the version committed here was only run on the c3_n46/47/48 step-12000 START checkpoints, because the arm checkpoints do not exist yet; see DRYRUN_CKPT_FACTS.txt). All six must pass `assert_ckpt_facts` + `assert_ckpt_cfg` and every pair must pass `check_pair_cfgs`; otherwise stop;
3. fill the six `CKPT_SHA` lines in `av2_stage1_ckpts.txt` with the sha256 of the six checkpoints, commit;
4. set `PINS.registry_sha` in `av2_scorer.py` = sha256 of `av2_stage1_ckpts.txt`, with the other UNSET pins (section 6), commit;
5. the scorer file is now final: put its sha256 in `EXPECT_SCORER_SHA` (section 0), commit WITHOUT touching `av2_scorer.py` or the registry file.

## 1. Tests (must be 100% green; synthetic only)
`~/.venvs/ramen_seg2/bin/python -m pytest -q outdoor_debug/parts/confirm/av2/tests tests/test_sky_mask.py` (the end-to-end test is skipped without pyarrow). Mutation proof: `tests/run_mutants.py`, result in MUTANT_TABLE.md.

## 2. Download and verify (existing `download_av2_val.sh`, objects.tsv above). Verify = listed sizes + SHA256SUMS; record sha(SHA256SUMS) as PINS.sums_sha.
The download writes the sums file to `$DEST/meta_R1/SHA256SUMS` (NOT next to the data dir). The scorer takes it ONLY through the required `--sums $DEST/meta_R1/SHA256SUMS` argument (N3) and pins its sha256; there is no implicit location.

## 3. Decode sanity (registration rule 6, O9a-style; GT side only; pass/fail only; no depth statistic used for any choice)
`python -I decode_sanity.py <root> av2_frames.tsv 5 av2_val_logs_selected50.txt` runs, for clip_idx 0 of the first 5 logs (sorted) of the frames tsv, which it asserts are logs 1-5 of the registered list (`av2_val_logs_selected50.txt` == `av2_val_logs_stage1_50.txt`), EXACTLY: (1) jpg decodes to uint8 (2048,1550,3), not constant; (2) intrinsics image size == (2048,1550) (asserted in A.Log), K finite, fx,fy>0, principal point inside the image; (3) paired sweep reads and the pose chain (exact camera and lidar timestamps) gives finite camera-frame points; dtype of x,y,z recorded; (4) sparse GT is float32 (2048,1550), has >= 1 return, all valid values in (0,200] m, no inf. Any failure raises. No model output is used.

## 4. Mask build (R1 rules; one store shared by both arms)
`python -I build_av2_masks.py --root <data> --frames av2_frames.tsv --clips av2_clips.tsv --objects objects.tsv --out sky_masks.npz --seg-a <dir with the SegFormer pytorch_model.bin> --seg-b <dir with the OneFormer pytorch_model.bin> --device cuda`.
Refuses an existing output, wrong table shas, a missing clip/sweep, a `.safetensors` file, a wrong weight sha. Pooled IoU of the two segmenters' `any` masks over ALL candidate frames (before any drop) must be >= 0.85, else MaskRejected and NO store is written (verdict NOT CONFIRMED, stop). The LiDAR exclusion uses EVERY return in front of the camera (no 200 m cap). The meta (`sky_masks.meta.json`) records status, pooled_iou, dropped list, segmenter repo+revision+sha, frames/clips/objects shas and the sweep xyz dtypes. Fill PINS.mask_sha and PINS.meta_sha from the two files.

## 5. 12-frame mask panel (committed BEFORE any prediction)
`python -I ../seg/make_panel.py --root <data> --frames av2_frames.tsv --masks sky_masks.npz --out mask_panel.png`. Frames = round(linspace(0,n-1,12)) over the sorted frame ids of the frames tsv (fixed rule); certain-sky red, any-only blue, P-dropped frames tagged. Looked at only for gross mask failure (O35 Q2); the thresholds are not tuned from it.

## 6. Fill the remaining PINS, commit, then fill EXPECT_SCORER_SHA (section 0) and commit again WITHOUT touching av2_scorer.py (the expected sha must be the sha of the committed scorer).

## 7. Scoring (per checkpoint; GH200 + CUDA)
`python -I run_score.py RUNBOOK_STAGE1.md -- --ckpt <ckpt.pt> --pair s50|s51|s52 --arm H300|TS2 --out <r.json> --root <data> --mask sky_masks.npz --frames av2_frames.tsv --clips av2_clips.tsv --objects objects.tsv --logs av2_val_logs_stage1_50.txt --sums $DEST/meta_R1/SHA256SUMS` (the runner appends `--expect-scorer-sha` and `--registry av2_stage1_ckpts.txt`; reference row: `--spec metric_vda_s` instead of `--ckpt/--pair/--arm`).
Asserted by the scorer before the first prediction: scorer sha (outside pin), all PINS, mask meta (status OK, IoU >= 0.85, dropped list, frames_sha == sha(frames tsv)), checkpoint step == 14000 (read from the checkpoint file), host GPU name contains GH200, CUDA fp16 autocast, the checkpoint's own cfg against the pair/arm labels and its sha against the registered table (section 0; N1), the mask meta rule and builder shas (N4). The seed label (s50/s51/s52 -> 50/51/52) is written to the meta as expected_seed. The checkpoint sha, the arm, the cfg facts, the full stored cfg + `cfg_sha256` and sha256(av2_stage1_ckpts.txt) as `registry_sha` are written to the meta. **The run json's sha256 is printed and written to `<out>.sha256` (D3): after each run, append that line (`<sha256>  <name>.json`) to `av2_stage1_run_shas.txt` and COMMIT the six lines before any compare.**
Each run records the kept / dropped sky-frame ids and the per-frame certain-sky px counts.

## 8. Decision
`python -I av2_scorer.py --compare-pairs H300_s50 TS2_s50 H300_s51 TS2_s51 H300_s52 TS2_s52 --registry av2_stage1_ckpts.txt --expect-scorer-sha <EXPECT_SCORER_SHA> --run-shas av2_stage1_run_shas.txt --out verdict.json` (order is enforced: first argument of each pair is the H300 run). `--registry`, `--expect-scorer-sha` and `--run-shas` are required (N2, D3). Refuses a registry file that is not the pinned one (D1), a run json whose sha256 is not in the committed run-shas file (D3), runs whose `meta.scorer_sha` is not the expected sha or whose `meta.pins` differ from the current PINS (D4), runs whose stored cfg contradicts their arm label or whose pair cfgs differ outside the declared keys (D2/D3), a guard table that is not exactly the five registered bins (D6), and swapped arms, equal checkpoints, differing kept frame ids / px counts, runs whose pins / scorer_sha / registry_sha differ, fewer than six distinct checkpoint shas, and any checkpoint sha that is not the registered one for its pair+arm. Prints the Stage 1 disclosure string.
**Stage 2** (single seed; only if Stage 1 is CONFIRMED and the user clears it) is NOT runnable with this scorer version: it needs a new scorer version (its own declared cfg facts: from-scratch start, seed 50, no start checkpoint) and therefore a new scorer sha and a new registration entry. `--pair stage2` is refused at the cfg check and `--compare` refuses non-Stage-1 labels.

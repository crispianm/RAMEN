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
| `sky_masks.npz` / `sky_masks.meta.json` / `SHA256SUMS` / `av2_dataset.py` / `ddad_scorer.py` | filled after the mask build / download (PINS.mask_sha, meta_sha, sums_sha, dataset_module_sha, ddad_scorer_sha are `UNSET` until then; the scorer refuses while any is UNSET) | scorer PINS |

## 1. Tests (must be 100% green; synthetic only)
`~/.venvs/ramen_seg2/bin/python -m pytest -q outdoor_debug/parts/confirm/av2/tests tests/test_sky_mask.py` (the end-to-end test is skipped without pyarrow). Mutation proof: `tests/run_mutants.py`, result in MUTANT_TABLE.md.

## 2. Download and verify (existing `download_av2_val.sh`, objects.tsv above). Verify = listed sizes + SHA256SUMS; record sha(SHA256SUMS) as PINS.sums_sha.

## 3. Decode sanity (registration rule 6, O9a-style; GT side only; pass/fail only; no depth statistic used for any choice)
`python -I decode_sanity.py <root> av2_frames.tsv 5` runs, for clip_idx 0 of the first 5 logs (sorted) of the frames tsv, EXACTLY: (1) jpg decodes to uint8 (2048,1550,3), not constant; (2) intrinsics image size == (2048,1550) (asserted in A.Log), K finite, fx,fy>0, principal point inside the image; (3) paired sweep reads and the pose chain (exact camera and lidar timestamps) gives finite camera-frame points; dtype of x,y,z recorded; (4) sparse GT is float32 (2048,1550), has >= 1 return, all valid values in (0,200] m, no inf. Any failure raises. No model output is used.

## 4. Mask build (R1 rules; one store shared by both arms)
`python -I build_av2_masks.py --root <data> --frames av2_frames.tsv --clips av2_clips.tsv --objects objects.tsv --out sky_masks.npz --seg-a <dir with the SegFormer pytorch_model.bin> --seg-b <dir with the OneFormer pytorch_model.bin> --device cuda`.
Refuses an existing output, wrong table shas, a missing clip/sweep, a `.safetensors` file, a wrong weight sha. Pooled IoU of the two segmenters' `any` masks over ALL candidate frames (before any drop) must be >= 0.85, else MaskRejected and NO store is written (verdict NOT CONFIRMED, stop). The LiDAR exclusion uses EVERY return in front of the camera (no 200 m cap). The meta (`sky_masks.meta.json`) records status, pooled_iou, dropped list, segmenter repo+revision+sha, frames/clips/objects shas and the sweep xyz dtypes. Fill PINS.mask_sha and PINS.meta_sha from the two files.

## 5. 12-frame mask panel (committed BEFORE any prediction)
`python -I ../seg/make_panel.py --root <data> --frames av2_frames.tsv --masks sky_masks.npz --out mask_panel.png`. Frames = round(linspace(0,n-1,12)) over the sorted frame ids of the frames tsv (fixed rule); certain-sky red, any-only blue, P-dropped frames tagged. Looked at only for gross mask failure (O35 Q2); the thresholds are not tuned from it.

## 6. Fill the remaining PINS, commit, then fill EXPECT_SCORER_SHA (section 0) and commit again WITHOUT touching av2_scorer.py (the expected sha must be the sha of the committed scorer).

## 7. Scoring (per checkpoint; GH200 + CUDA)
`python -I run_score.py RUNBOOK_STAGE1.md -- --ckpt <ckpt.pt> --pair s50|s51|s52 --arm H300|TS2 --out <r.json> --root <data> --mask sky_masks.npz --frames av2_frames.tsv --clips av2_clips.tsv --objects objects.tsv --logs av2_val_logs_stage1_50.txt` (reference row: `--spec metric_vda_s` instead of `--ckpt/--pair/--arm`).
Asserted by the scorer before the first prediction: scorer sha (outside pin), all PINS, mask meta (status OK, IoU >= 0.85, dropped list, frames_sha == sha(frames tsv)), checkpoint step == 14000 (read from the checkpoint file), host GPU name contains GH200, CUDA fp16 autocast. The seed is carried by the pair label (s50/s51/s52 -> 50/51/52, written to the meta as expected_seed); it is NOT cross-checked against the checkpoint's own cfg (limitation, disclosed). The checkpoint sha and the arm are written to the meta.
Each run records the kept / dropped sky-frame ids and the per-frame certain-sky px counts.

## 8. Decision
`python -I av2_scorer.py --compare-pairs H300_s50 TS2_s50 H300_s51 TS2_s51 H300_s52 TS2_s52 --out verdict.json` (order is enforced: first argument of each pair is the H300 run). Refuses swapped arms, equal checkpoints, differing kept frame ids / px counts. Prints the Stage 1 disclosure string. Stage 2 (single seed, only if Stage 1 is CONFIRMED and the user clears it): `--compare H300.json TS2.json` with runs made with `--pair stage2`; prints the Stage 2 single-seed string.

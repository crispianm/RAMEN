# ddad_scorer.py (DDAD val LR_DEV)
EXPLORATORY DEV, REPORT-ONLY. Registered in REGISTRATION.md (A1 resolution, A2 short_side 378), metrics per Auditor O11. No arm is chosen on it; never a held-out test.

## Definitions (O11, unaligned, no alignment of any kind)
- GT valid: projected sparse LIDAR depth with 0 < z <= 200 m, finite (NaN, 0, negative, > 200 excluded). GT is native 1936x1216 and is never resized (a non-native GT raises).
- Pred: model output (metres) at model resolution, bilinearly resized (align_corners=False) to 1936x1216 and sampled at the valid LIDAR px (`ddad_lr_dataset.sample_prediction`). The scorer calls `run_pipeline` with out_hw = model resolution so bench_joint's own align_corners=True upsample is NOT applied (A1 would be violated otherwise). Pred is not clamped.
- Bins by GT distance: [0,20) [20,50) [50,80) [80,120) [120,200] m (half-open, last closed; 20.0 -> [20,50), 120.0 and 200.0 -> [120,200]).
- Per bin: AbsRel = sum(|p-g|/g)/n pooled over all valid px of all scored frames; median pred:gt = pooled median of p/g; n_px; n_frames and n_scenes contributing. Overall AbsRel over (0,200]. Also n_scored_frames, n_scenes, n_nonfinite_pred_px (non-finite pred is NOT hidden: it propagates into the numbers).
- Scored frames = the 395 of ddad_val_frames.tsv; the other frames of each scene (all CAMERA_01 frames, timestamp order, windowed to max_len 160) are model context.

## Asserted at start
frame list sha256 b8d9664f...5098, pairs sha256 962d4996...0245, bench_joint.py sha 5d3d048c...ef, checkpoint has trainable_ema (sha, step, arch recorded), fp16 autocast on cuda, short_side 378, full run = 395 frames / 50 scenes. Meta records host, gpu, torch, ddad_scorer/dataset-module/scorer.py shas, ramen commit, per-scene scored counts, label. `--limit-scenes N` is allowed only as a smoke: meta.status = PARTIAL, limit_scenes recorded, pins_asserted false. Output file is never overwritten.

## Run
GH200 (Isambard, same host type for all 5 arms): `~/.venvs/ramen/bin/python -I outdoor_debug/parts/LR_DEV/ddad_scorer.py --ckpt <RAMEN .pt> --out <result.json> --repo <repo root with bench_joint.py at the pinned sha>`; set `DDAD_LR_ROOT` to the dir holding `ddad_val/` and `ddad_val_pairs.tsv`. Same command on the 4090 later (GPU_LOCK protocol, `--device cuda`).
No-data check: `python -I ddad_scorer.py --selftest` (synthetic only). Tests: `python -I -m pytest tests/test_ddad_scorer.py` (8, synthetic only).

## Not verified here (no model/data run)
The full model path (`model_input`, `predict`, `main`) has not been executed; first execution is the first DDAD prediction, after Auditor check.

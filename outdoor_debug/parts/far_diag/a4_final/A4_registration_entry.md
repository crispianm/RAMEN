
## Amendment A4 (2026-10-17T22:34:17Z, from `date -u`), filed BEFORE the job: the RELEASED Metric-VDA-S on DDAD LR_DEV at its default 518 input, report-only (user Addendum 49/50; Auditor O34 L-3)

**What is scored.** The released Metric-VDA-S (bench_joint key `metric_vda_s`, weights sha256 3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33, same as A3) on the identical DDAD val LR_DEV frames, but with model input short side 518 instead of the registered 378. DEBUG ONLY, not paper; exploratory DEV, report-only.

**Exact flag.** `ddad_scorer.py --spec metric_vda_s --release-short-side 518 --out <json> --device cuda`. Absent flag or `378` = the registered 378 path (unchanged behaviour; A3 row semantics). 

**Label (exact, asserted in meta and result):** "released Metric-VDA-S at its default 518 input (not the registered 378), fp16, exploratory DEV, report-only". Stated in words: released Metric-VDA-S at 518 input, report-only; ours is 378.

**Refusal rules (all raise ValueError in `resolve_short_side` before pins, model load or output).** `--release-short-side` other than 378/518 is refused (e.g. 448, 0); any size other than 378 is refused when no registered release `--spec` is given (every `--ckpt` run: ours at 518 is NOT proposed and cannot be run); an unregistered `--spec` with a size other than 378 is refused. `--spec` + `--ckpt` stays refused (A3). The existing O26(1) same-HxW assert is unchanged and must hold at 518.

**Unchanged (A1/A2/A3/O11/O26):** frames b8d9664f.., pairs 962d4996.., bench_joint 5d3d048c.., 395 frames / 50 scenes, bins [0,20) [20,50) [50,80) [80,120) [120,200] m, pooled unaligned AbsRel + median pred:gt, A1 bilinear prediction -> 1936x1216 sampled at projected LIDAR px (GT never resized), Resize lower_bound / multiple 14 / cubic, fp16, max_len 160. Only the Resize short side differs (518 -> model input 518x826 for 1216x1936 frames; 378 -> 378x602).

**Pins.** ddad_scorer.py sha256 `8e71511b1192a1a15797a71482c40b547bac7d905641b7d2f145ad1da079a503` (previous 80916c4f3b2ab0b8999b07e142847f8df157db5da756192f02f6df5dea1ed09f); tests/test_ddad_scorer.py sha in the A4 commit message; job `outdoor_debug/parts/submit/score/DDAD_RELEASE_MVDA_518.sbatch` (output dir .../RAMEN_outdoor_screen_scores/DDAD_RELEASE_MetricVDA_S_518, DDAD_COMMIT literal UNSET until submit-time sed); checker `far_diag/a4_final/check_a4_518.py` (PASS/FAIL names only). Existing rows (ours and the A3 release row at 378) were produced by scorer sha 17646e01/80916c4f and stay valid: with the flag absent the new scorer produces a byte-identical result file (tested on a fake pipeline, old vs new).

**First-518-run requirement.** The log must contain "first clip: model input (N, 3, 518, 826) -> depth (N, 518, 826) (same HxW, identity resample)"; the sbatch aborts otherwise, and check_a4_518.py (must be ALL PASS) additionally compares every other meta field and the per-bin GT pixel/frame/scene counts with the A3 378 row.

**What the result may be used for.** Reporting the release's five-bin median pred:gt and pooled AbsRel at 518 beside the release at 378 (statement form: "the release's 120-200 m median pred:gt is X at 518 vs 0.854 at 378"), to bound how much of the DDAD release-vs-ours far gap is an input-scale effect. **May not:** be compared to ours as an equal-input row (ours is 378 only); enter any arm decision, rule, margin or the paper; support any statement about ours at 518; trigger a retraining or a new 518 run of any other model without a new amendment.

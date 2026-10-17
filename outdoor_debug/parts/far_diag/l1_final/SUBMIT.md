# L1 SUBMIT (DRAFT; nothing submitted). Exact commands: `submit_l1.sh` (starts with `exit 99`; run the blocks by hand after approval). Full design/approval list: `L1_FINAL.md`.

## Order
0. Clone at the pinned commit; sbatch copies with the pin substituted OUTSIDE the clone (the clone must stay clean; each job checks HEAD == pin, clean tree, config / base / release-weights sha256).
0b. S6 literal: `export S6SHA=298cac3b...` (the Auditor's cleared sha, pasted; never derived from the clone) and `s6_sha_gate.sh` refuses a placeholder or a patch with another sha.
1. Smokes (120 steps, 4 GPU, 25 min limit): S1 and S5 (`SMOKE=120`; 3 changed config lines asserted). **GATE 1 (explicit step; no sbatch dependency from any smoke to any full job):** both COMPLETED 0:0, `check_first_run_l1.sh -F -S 120 ...` no FAIL, `check_optimizer_lr.py ... step120.pt --smoke 120 --expect-step 120 -F` no FAIL (--smoke applies the sbatch's own 3 substitutions to the arm yaml). Only then Block 2 is submitted.
2. BASELINES FIRST (submitted only after GATE 1, NOT `afterok` the smokes): B0 and B0b at 2000 steps. First-run stop check at step >= 200 (B0 first); optimizer probe on `last.pt` after step 520; final `-F` pass at the end.
3. Score B0/B0b (`SCORE_*`, one GH200 each: by-range scorer + DDAD, ~12 min) `afterok` their training.
4. Read G0 ONLY from `l1_readout.py --stage 2000 --ddad-gate A --arms B0,B0b`: rc 0 -> Block 5; rc 10 -> Block 6 (automatic 4000-step baseline retry; arms NOT run at 2000).
5. Arms S1-S5 at 2000 (all at once, stop check each, score `afterok`); **S6 separately**: only after the Auditor's diff check of `S6_diff.patch` (job refuses without `S6_CLEARED_PATCH_SHA` = the cleared sha256 and `train.py == 8e1d9ae + patch`), with its own smoke (SM6) and an explicit GATE 2 (stop check -F -S 120 + probe --smoke 120) BEFORE J_S6 is submitted; J_S6 has no dependency on SM6. rc 4/5 of the readout: score step/sha integrity failure / a declared arm missing.
6. Retry: B0/B0b at 4000 (`*_k4000`), score, `l1_readout.py --stage 4000`: rc 0 -> S1-S5 (+ gated S6) at 4000; rc 11 -> STOP and report.

## Costs (NHR = node-hours; measured 1.45-1.51 per 2000-step run, 4000-step extrapolated 2x; scoring ~0.12; smoke ~0.1)
| Scenario | expected | limit-based max |
|---|---|---|
| A: G0 passes at 2000 (3 smokes, 8 x 2000-step trains, 8 scores) | **13.3** | 29.3 (3 h x 8 + smokes 25 min x 3 + scores 30 min x 8) |
| B: G0 fails at 2000, passes at 4000 (2000 baselines wasted, 2 x 4000 baselines, 6 arms at 4000) | **28.6** | 56.7 (4000-step limit 5.5 h) |
| C: fails at 2000 and 4000 -> STOP | **9.7** | 19.8 |
Cap 45 NHR actual: Manager sums `sacct` after each block; worst expected (B) 28.6; the limit-based B figure exceeds 45 only if jobs ran to their limits. Baseline retry alone = +6.5 NHR (not ~+3).

## Files the jobs need in the pinned clone
`outdoor_debug/parts/far_diag/l1_final/{configs/k2000,configs/k4000,sbatch,S6_diff.patch,check_first_run_l1.sh,check_optimizer_lr.py,l1_readout.py}`, `outdoor_debug/parts/confirm/stage1/base/effective_ramen_n46.yaml`, `outdoor_debug/parts/scorer/scorer.py`, `outdoor_debug/parts/submit/score/check_scores.py`, `outdoor_debug/parts/LR_DEV/{ddad_scorer.py,check_ddad_output.py,ddad_ref2/ddad_scores.json}`, `bench_joint.py`, `train.py`. Regenerate `sbatch/` with `gen_l1.sh` after ANY yaml/template change (the yaml sha256 is baked into each TRAIN file).

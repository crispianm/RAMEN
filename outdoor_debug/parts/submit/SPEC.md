# Outdoor-debug 6-arm screen: Isambard job spec (DRAFT, nothing submitted)
Files here: `train_template.sbatch` + `score_template.sbatch` (edit these), `gen_train.sh` (regenerates `TRAIN_<ARM>.sbatch`, `SCORE_<ARM>.sbatch`), `submit.sh` (the commands; refuses to run as a script), `check_first_run.sh` (stop check items 1-7 + spike rule).

## Pins
- Code: crispianm/RAMEN branch `outdoor-debug`, commit `2a72223c068d5ba0bde869a7c27d0e90d21abb5d` (text-only successor of `0034ac711866d704a1cabe0b8ea38ed1d0b022ac`; `git diff --stat` = SCREEN_PROPOSAL.md only; both accepted by the scripts, the used one is printed in the log). Fresh clone dir `/projects/b5dh/RAMEN_outdoor_screen_2a72223` (does not exist yet; checked read-only).
- Start checkpoint `/projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_c3_n46/step12000.pt` sha256 `f5c7c8f98fe0c9f97bcac3586e577d04e914fbbe1e2ac41dbdbd8ce0501db1af` (re-verified read-only on the cluster 2026-10-17: match). Every arm YAML's `resume_from` equals this path.
- bench_joint.py sha256 `5d3d048ce7b26fb53e1e6a5aadf8265cb92c19dafc7340bc2e04cc6bcdf6e7ef` (R3_PIN.md); DEV combined clip-list sha `aca96bbd894855cc99b8d97dbdf6e957d1c1f4efb0d49c77ff6108d8dccd7de6`.
- ts2 env table sha256 `f19d0b838252c9a2409498b35f46738d99b3646afba984d6a82ffea3cd610b76` (TS2 arms; checked from the committed file and printed by train.py); ts2_seg_root `/projects/b5dh/data/train/tartanair_seg/` exists (75 entries, read-only check).

## Arms (config = `outdoor_debug/parts/arm_checks/configs/arm_<ARM>.yaml`; the sbatch aborts on sha mismatch)
| arm | yaml sha256 | checkpoint_dir (in the clone) |
|---|---|---|
| A0 | d347010a94e914d3fcb9a616c160fc796cd1b1ceaf0088d50592347ff9846cf0 | runs/outdoor_screen_A0 |
| A0b | a0c15c71cb0b142126279ff92c351c6cc9c429b5ff09d364f094c39d955096cd | runs/outdoor_screen_A0b |
| TS2 | 0e04c9c61f0a115a1c7ddfb225300500d54ab4f6f77c30c4bd0b976191567ead | runs/outdoor_screen_TS2 |
| TS2_H80 | 7c7d980e1500d1178fb45ce2f4a830a7d34fddde798337ea446a61ecb51e4009 | runs/outdoor_screen_TS2_H80 |
| TS2_H300 | 30131c56829b28770c4e6bf19b5447e8698468567edcc03de21ea411aeb461c6 | runs/outdoor_screen_TS2_H300 |
| TS2_VK655 | dd9c8f4cdc769f83c49b8b1a963a821fc7c8abc0198f39ed508650f689ad4763 | runs/outdoor_screen_TS2_VK655 |
Checkpoints are written at 12500/13000/13500/14000 (+ every 500 steps, + final) only under the arm's own dir; the script refuses to start if that dir exists. Job name `outdoor_screen_<ARM>`, logs `logs/outdoor_screen_<ARM>_<jobid>.{out,err}` in the clone.

## Resources and cost (launch lines copied from jobs 7226119 / 7243692: workq, 1 node, gres gpu:4, 72 cpus, 400G, FocusMamba venv, `torchrun --standalone --nproc_per_node=4 train.py --config ...`, no account line, no module loads)
- Per arm: expected ~1.45 h wall (0.72 h/1000 steps measured) = ~5.8 GH200-h; limit 3 h (2x margin; the 4000-step job ran 2:52 of 8:00). 6 arms: expected ~8.7 NHR (1 node-hour = 1 NHR; observed charging ~1.46x GPU share, so the portal may show up to ~12), limit-based max 6 x 3 h x 1 node = 18 NHR. Smokes: 2 x 25 min x 1 node = 0.83 NHR max. Scoring (single GPU each): 6 x 6 h x 0.25 = 9 NHR max (not measured; DEV-only should be well below). Whole screen limit-based max ~27.8 NHR vs the Auditor's 17.8 NHR figure (which counts expected scoring); tell the Auditor the limit-based number before submit.
- Submit all six fulls at once behind the smokes (earlier practice: 6 fulls `afterok` 2 smokes; QoS 256 running / 512 submitted; no GrpTRES cap). No local GPU_LOCK on the cluster. Nodes are not shared (1 full node each).

## Recommended procedure (submission needs Manager approval, parent submits)
0. Block 0 of submit.sh: fresh clone at the pinned sha; `git status --porcelain` empty. Do not touch existing clones.
1. Smoke: A0 and TS2_VK655 with `SMOKE=1` (50 steps: max_steps 12050; checkpoint_dir smoke50; everything else identical; the script diffs exactly 2 changed lines). Gate = both COMPLETED 0:0 AND `check_first_run.sh -s ...` has no FAIL: LR rows 12020/12040 match the table (12060+ are WAIT), resume line step=12000, hinge key absent, TS2 line + env sha in TS2_VK655 only, bound `vkitti=656` in VK655 vs `655` in A0, identical `[data]` lines, finite. Loss-spike rule: report to the parent BEFORE the fulls if the resumed loss mean > 2x A0's (the script's ratio line compares against the first listed log, i.e. A0; the proposal's reference is A0's step-12000 loss, which you can also read from the c3_n46 metrics). The fulls are `afterok` the smokes, so a failed smoke cancels them.
2. Fulls (Block 2). When each log reaches step >= 12050 run `check_first_run.sh` with A0's log first; any FAIL = `scancel` that job and report. WARN on H80/H300 (hinge exactly 0) = report as 'hinge inert at this start point'.
3. Score each arm (Block 3, `afterok` its train job): `bench_joint.py run --model ours:<arm>/step14000.pt --dataset kitti nyuv2 scannetpp vkitti --cells b0s0 --split dev --temporal`, same clone/sha/host for all arms, output `/projects/b5dh/RAMEN_outdoor_screen_2a72223_scores/<ARM>/` (refuses an existing dir). DEV only; no `--allow-test`.
4. Results are 'exploratory, unaudited' (O18); success rule = SCREEN_PROPOSAL.md.

## Must be approved before anything runs
Manager (debug-track job approval, O2 item 4): the clone creation (Block 0), the 2 smokes, the 6 fulls, the 6 score jobs, and the limit-based NHR figure above. Auditor token only if a result later heads for paper/merge/retrain.

## Open items
- The by-range pooled metrics (VKITTI 80-655 m pred:gt, KITTI/VKITTI AbsRel by range, TartanAir far) are NOT produced by bench_joint; they came from `D_baseline/run_baseline.py` (local paths, 4 hard-coded models). SCORE_*.sbatch has a TODO; a parametrised copy needs writing and an Auditor check before it can score arms. DDAD scoring is a marked TODO (data not on cluster).
- GitHub reachability from Isambard and the exact clone mechanism are unverified (earlier clones were cluster-local; option B bundle given).
- Not verified here: `jobs/run_ramen_v2.sh` and `scripts/preflight_experiment.py` are not in the public repo, so the sbatch calls torchrun directly (no preflight step); the smoke YAML is written to `/projects/b5dh/.torch_cache/tmp` outside the clone (relative paths in the YAML resolve from cwd = clone, as for in-repo configs; not exercised); no job was run, only `bash -n`, the smoke `sed`/diff, and `check_first_run.sh` on synthetic logs.

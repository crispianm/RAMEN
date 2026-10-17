# Pre-declared screen proposal (DRAFT for user + Auditor clearance; no training until cleared)
Status: light track (Add. O2), DEV only, TEST closed. Not a paper claim; anything heading for paper/merge/retrain needs a separate pre-declaration + Auditor token.

## Question
Does correcting TartanAir sky labels (TS v2), and then adding a one-sided sky hinge, move far/sky depth toward GT when applied from the c3_n46 step-12000 checkpoint? Note from the trajectory: compression appears by step 6000 and plateaus, so this screen tests recovery from a settled model, not origin.

## Arms (N tried = 4 primary + A0b noise arm = 5; exploration log appended before results)
| Arm | Change vs previous | Config |
|---|---|---|
| A0 | none (in-sweep baseline) | C3 recipe, c1abf81 loader, `sky_hinge_weight: 0.0` |
| TS2 | TartanAir loader sky rule = TS v2 (seg-sky first, saturation fallback, uncertain px excluded from sky and valid) | A0 + new loader flag (new recipe, own regression test) |
| TS2+H80 | + one-sided sky hinge, weight 0.1, floor 80 m | TS2 + `sky_hinge_weight: 0.1`, floor 80 m |
| TS2+H300 | hinge floor 300 m instead | as above, floor 300 m |
Dropped by user before any run: teacher arms T-a/b/c (not counted).
Held fixed across arms: start checkpoint sha (c3_n46 step12000, SUMS in checkpoints_v3/eval_20261012_c3rep), steps, seed, data-stream order, eval DEV set, optimizer settings.

## Steps and cost
- 2000 fine-tune steps per arm (1/6 of the 12k recipe), same 16 clips/step, lr 1e-4 as recipe. 4 arms = 8000 steps total, about 2/3 of ONE 12k run.
- GPU-hours: NOT measured. I have no per-step wall time on record, so any number would be a guess. Proposed: take the c3rep job's Slurm elapsed (read-only sacct, needs ssh, I will not contact Isambard without go) and scale by 2000/12000 per arm. The 4090 (24 GB) cannot run the 16-clip recipe at the training config unmodified; batch size would have to change, which makes it a different recipe, so I recommend Isambard GH200 for the arms.
- Host: Isambard (74 envs, seg required for all 74 training envs; the 28 local envs are insufficient for the real sampling mix). Needs the 74-env seg pull first (job spec to Manager) plus a loader that reads seg.
- Eval: bench_joint.py on the audited DEV clip lists; metrics GT-based only: VKITTI far pred:gt, KITTI and VKITTI AbsRel by range, TartanAir far. Release-sky-median closeness is reported as description only, not as the decision rule.
- Success rule (pre-declared): an arm counts as better only if VKITTI far pred:gt moves toward 1.0 AND near-field AbsRel (KITTI, NYU, ScanNet++ DEV) does not worsen by more than the A0 repeat noise; with one seed the noise is unknown, so I propose repeating A0 with a second seed to size it. DECLARED SET (fixed now, per Add. O6b): 5 arms = A0, A0b (A0 with a second seed, noise sizing only), TS2, TS2+H80, TS2+H300; N tried = 5. No arm or combination is added after any result is seen; a later combination is a new sweep with its own log entry and counts in N tried. User may strike A0b before clearing (then N = 4).
- Labelling: Downtown, Hospital, ForestEnv, OldBrickHouseNight are TRAINING envs; any TartanAir number on them is "training environments, not held out".

## Gates
Prereqs before any run: (1) TartanAir task complete: parity test and per-env sky audit (O5) in AND checked by the Auditor before TS v2 starts (O6c); (2) regression tests (M-03 8000 m dome, B2-03 10000.0 edge, per-env finite-sky) pass on the branch; (3) Isambard seg pull; (4) user + Auditor clear this proposal; (5) Isambard job needs the Auditor token and Manager approval.

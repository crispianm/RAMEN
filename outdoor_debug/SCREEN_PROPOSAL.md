# Pre-declared screen proposal (DRAFT for user + Auditor clearance; no training until cleared)
Status: light track (Add. O2), DEV only, TEST closed. Not a paper claim; anything heading for paper/merge/retrain needs a separate pre-declaration + Auditor token.

## Question
Does correcting TartanAir sky labels (TS v2), and then adding a one-sided sky hinge, move far/sky depth toward GT when applied from the c3_n46 step-12000 checkpoint? Note from the trajectory: compression appears by step 6000 and plateaus, so this screen tests recovery from a settled model, not origin. A negative result at 2000 steps therefore does NOT rule out applying the fix from init.
Caveat on the target: VKITTI trains on all 5 scenes and the TartanAir envs are training envs, so the primary far/sky metrics ("training scenes") measure fit to the corrected labels, not generalisation. DEV has no held-out far/sky GT. KITTI DEV unaligned AbsRel is the real-scene guard; horse/KITTI sky medians are description only. (A real-world far-GT DEV set is under discussion with the user; not part of this screen unless added before clearance.)

## Deviation from Add.42
Add.42 listed A0, TS v2, H80, H300. This draft: hinge arms are nested on TS v2 (TS2+H80, TS2+H300; standalone H80/H300 on old labels dropped) and A0b (second-seed A0, noise sizing only) is added. The user should accept or strike this explicitly.

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
- GPU-hours (measured from Isambard sacct, read-only, 2026-10-17): the c3rep n46 12k-step job 7226119 took 08:36:42 elapsed on 4 GH200 (includes startup/val/checkpointing); the continuation job dn01_jt_ctrl_cont8to12 (4000 steps) took 02:52:12 on 4 GPUs. That is ~0.72 h per 1000 steps, so a 2000-step arm is ~1.45 h wall = ~5.8 GH200-hours (4 GPUs). 5 arms: ~7.2 h wall, ~29 GPU-h of training. Eval is separate: single-GPU bench jobs for 12k models ran ~5.5-6 h each, but those covered the full DEV+TEST-class cells, so DEV-only scoring will be cheaper (not measured); budget at most 6 GPU-h/arm = 30 GPU-h. NHR limit-based maximum: training 5 arms x 3 h limit x 4 GPUs = 60 + eval 5 x 6 h x 1 GPU = 30, total <= 90 NHR. Expected use is well under that.
- Host: Isambard (74 envs, seg required for all 74 training envs; the 28 local envs are insufficient for the real sampling mix). Needs the 74-env seg pull first (job spec to Manager) plus a loader that reads seg.
- Eval: bench_joint.py on the audited DEV clip lists; metrics GT-based only: VKITTI far pred:gt, KITTI and VKITTI AbsRel by range, TartanAir far. Release-sky-median closeness is reported as description only, not as the decision rule.
- Success rule (pre-declared numbers): per metric, noise floor NF = max(|A0 - A0b|, 1% relative). One A0/A0b pair understates seed variance (stated limitation). An arm is BETTER only if (target change > NF) AND (every near-field guard is not worse than A0 by more than NF). Target (labelled "training scenes"): VKITTI far pred:gt toward 1.0; TartanAir far (training envs). Guards: KITTI, NYUv2, ScanNet++ DEV AbsRel (unaligned) plus TAE on the same clips. Description only: horse and KITTI sky medians. DEV clip lists: bj.select_clips(ds,"dev") for each dataset; their key list sha256 is computed and written into the exploration log before the first run (not yet computed).
- Declared set (Add. O6b): 5 arms = A0, A0b, TS2, TS2+H80, TS2+H300; N tried = 5; nothing added after any result; a later combination is a new sweep with its own log entry and counts in N tried. User may strike A0b (N = 4).
- Labelling: Downtown, Hospital, ForestEnv, OldBrickHouseNight are TRAINING envs; any TartanAir number on them is "training environments, not held out".

## Gates
Prereqs before any run: (1) TartanAir task complete: parity test and per-env sky audit (O5) in AND checked by the Auditor before TS v2 starts (O6c); (2) the exact TS v2 rule is written here (per-env seg class ids, saturation-fallback threshold, definition of 'uncertain') after the O5 audit, and the regression tests (M-03 8000 m dome, B2-03 10000.0 edge, per-env finite-sky) pinning it pass on the branch; (3) Isambard seg pull; (4) user + Auditor clear this proposal; (5) Isambard job needs Manager approval (debug track, O2 item 4). An Auditor token is needed only if a result later heads for the paper, a merge or a retrain.

## TS v2 exact rule
PENDING: to be filled from the per-env sky audit (parts/TA_sky). This proposal is not clearable until this section is complete.

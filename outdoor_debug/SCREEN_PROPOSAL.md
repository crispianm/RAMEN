# Pre-declared screen proposal
STATUS: USER CLEARED (Addendum 43, 2026-10-17: "Clear, keep A0b"); the Add.42 deviation (nested hinge arms, A0b added) is ACCEPTED. Clearance is conditional: do not submit until (1) TS v2 rule filled from the audit and Auditor-checked (if materially different from 'seg-sky first, saturation fallback, uncertain excluded', e.g. whole envs excluded, tell the Manager first, the user may need to re-clear); (2) DEV clip-list sha logged; (3) regression tests pass; (4) Isambard 74-env seg pull job approved by Manager and completed. Real far GT (DDAD/AV2) is report-only, exploratory DEV, and does NOT change the decision rule.
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
- GPU-hours (measured from Isambard sacct, read-only, 2026-10-17): the c3rep n46 12k-step job 7226119 took 08:36:42 elapsed on 4 GH200 (includes startup/val/checkpointing); the continuation job dn01_jt_ctrl_cont8to12 (4000 steps) took 02:52:12 on 4 GPUs. That is ~0.72 h per 1000 steps, so a 2000-step arm is ~1.45 h wall = ~5.8 GH200-hours (4 GPUs). 5 arms: ~7.2 h wall, ~29 GPU-h of training. Eval is separate: single-GPU bench jobs for 12k models ran ~5.5-6 h each, but those covered the full DEV+TEST-class cells, so DEV-only scoring will be cheaper (not measured); budget at most 6 GPU-h/arm = 30 GPU-h. Units: GPU-h above are GPU-hours. NHR = node-hours (1 node = 4 GH200; 1 GPU = 0.25 node). Limit-based maximum: training 5 arms x 3 h limit x 1 node = 15 NHR, eval 5 x 6 h x 0.25 node = 7.5 NHR, total <= 22.5 NHR. Expected: ~7.2 NHR training (5 x 1.45 h x 1 node) plus DEV eval (not measured). In GPU-h: ~29 training, <=30 eval.
- Host: Isambard (74 envs, seg required for all 74 training envs; the 28 local envs are insufficient for the real sampling mix). Needs the 74-env seg pull first (job spec to Manager) plus a loader that reads seg.
- Eval: bench_joint.py on the audited DEV clip lists; metrics GT-based only: VKITTI far pred:gt, KITTI and VKITTI AbsRel by range, TartanAir far. Release-sky-median closeness is reported as description only, not as the decision rule.
- Success rule (pre-declared numbers): per metric, noise floor NF = max(|A0 - A0b|, 1% relative). One A0/A0b pair understates seed variance (stated limitation).
  - PRIMARY TARGET (the one leg for BETTER, "training scenes"): change = reduction of |1 - pred:gt| in the VKITTI 80-655 m bin vs A0. BETTER requires change > NF.
  - GUARDS: every near-field guard not worse than A0 by more than NF: KITTI, NYUv2, ScanNet++ DEV AbsRel (unaligned), and TAE on the same clips = video_macro.tae_gtvalid (user ruling 10-10) for every TAE guard.
  - SECONDARY / description only: TartanAir far (training envs, "training environments, not held out"), horse and KITTI sky medians.
  - Scoring: all arms scored on the same host type with one pinned bench_joint commit. The DEV clip-list sha (from bj.select_clips(ds,"dev") keys) and that commit are computed and logged together before the first training run (not yet computed).
- Declared set (Add. O6b): 5 arms = A0, A0b, TS2, TS2+H80, TS2+H300; N tried = 5; nothing added after any result; a later combination is a new sweep with its own log entry and counts in N tried. User may strike A0b (N = 4).
- Labelling: Downtown, Hospital, ForestEnv, OldBrickHouseNight are TRAINING envs; any TartanAir number on them is "training environments, not held out".

## Gates
Prereqs before any run: (1) TartanAir task complete: parity test and per-env sky audit (O5) in AND checked by the Auditor before TS v2 starts (O6c); (2) the exact TS v2 rule is written here (per-env seg class ids, saturation-fallback threshold, definition of 'uncertain') after the O5 audit, and the regression tests (M-03 8000 m dome, B2-03 10000.0 edge, per-env finite-sky) pinning it pass on the branch; (3) Isambard seg pull; (4) user + Auditor clear this proposal; (5) Isambard job needs Manager approval (debug track, O2 item 4). An Auditor token is needed only if a result later heads for the paper, a merge or a retrain.

## TS v2 exact rule (DRAFT from the O5 audit; awaiting Auditor check; parts/TA_sky/TS_v2_PROPOSAL.md + ts_v2_draft.py)
- saturated = ~isfinite(d) | d >= 1e4.
- sky = seg-sky class (per-env ids from official seg_label_map.json; skysphere ids included) at ANY depth.
- uncertain = saturated AND seg says non-sky (or env has no usable sky label): excluded from both sky and valid mask.
- valid mask = isfinite(d) & d > 0 & d < max_depth & ~sky & ~uncertain.
- Envs with no usable sky label (CoalMine, HongKong, Ocean, OldBrickHouseNight; ArchVizTinyHouseNight skysphere id absent in sample): saturated px are UNCERTAIN by default (sky=0, mask=0); opt-in flag `fallback_sat_is_sky` off in the screen. No whole env is excluded.
- Regression tests (tests/test_ts_v2_sky_draft.py, 5 pass): 8000 m dome -> sky=1, mask=0; 10000.0 edge; 65504/inf/NaN; tree quirk; unlabelled-env default; non-positive depth. Official-reader parity: tests/test_tartanair_official_parity.py (118 pass; 117 frames, all 28 local envs, incl. Downtown, GreatMarsh, HongKong, and the 65504 envs).
- Needs the seg masks at train time on whichever host trains (Isambard: 74-env seg pull, pending).
